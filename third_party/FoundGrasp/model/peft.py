# peft.py
# Drop-in PEFT modules for your GraspPro (Depth-Anything first).
# - LoRA: patches backbone Linear layers (attn qkv/proj by default)
# - AdaptFormer: per-layer bottleneck MLP residual on patch tokens
# - ViT-Adapter: SPM + deformable-attention interaction adapted as a PEFT baseline
# - Ours (Geo-Conditioned AdaptFormer): AdaptFormer update gated by depth_feat (geometry prior)

import math
import re
import types
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth per-sample (applied at residual addition, as in AdaptFormer)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = torch.floor(noise + keep_prob)
        return x / keep_prob * noise


# =========================================================
# 0) Token utilities (works even if extra tokens exist)
# =========================================================
def get_patch_tokens(x: torch.Tensor, feat_h: int, feat_w: int):
    """
    x: (B, S, C) token sequence
    returns:
      prefix: (B, S-HW, C), patch: (B, HW, C), patch_start: int
    """
    hw = feat_h * feat_w
    patch_start = x.shape[1] - hw
    if patch_start < 0:
        raise RuntimeError(f"Token sequence too short: S={x.shape[1]} < HW={hw}")
    prefix = x[:, :patch_start, :]
    patch = x[:, patch_start:, :]
    return prefix, patch, patch_start


def merge_patch_tokens(prefix: torch.Tensor, patch: torch.Tensor):
    return torch.cat([prefix, patch], dim=1)


def tokens_to_map(patch: torch.Tensor, feat_h: int, feat_w: int):
    """patch: (B, HW, C) -> (B, C, H, W)"""
    B, HW, C = patch.shape
    assert HW == feat_h * feat_w
    return patch.transpose(1, 2).reshape(B, C, feat_h, feat_w)


def map_to_tokens(feat: torch.Tensor):
    """feat: (B, C, H, W) -> (B, HW, C)"""
    B, C, H, W = feat.shape
    return feat.reshape(B, C, H * W).transpose(1, 2)


def _normalize_layer_spec(layer_spec):
    if isinstance(layer_spec, str):
        return layer_spec.lower().replace("-", "_")
    return layer_spec


def _resolve_layers_to_adapt(num_layers: int, layer_spec="all", decoder_out_indices=None):
    layer_spec = _normalize_layer_spec(layer_spec)

    if layer_spec in (None, "all"):
        return set(range(num_layers))

    if layer_spec == "last_half":
        return set(range(num_layers // 2, num_layers))

    if layer_spec in ("decoder_aligned_late", "decoder_late", "tap_aligned_late"):
        if decoder_out_indices is not None and len(decoder_out_indices) >= 2:
            start = int(decoder_out_indices[-2])
        else:
            start = num_layers // 2
        return set(range(start, num_layers))

    if layer_spec is not None and not isinstance(layer_spec, str):
        try:
            return {int(idx) for idx in layer_spec}
        except TypeError:
            pass

    raise ValueError(
        f"Unsupported layers spec: {layer_spec!r}. "
        "Use 'all', 'last_half', 'decoder_aligned_late', or an explicit list of indices."
    )


# =========================================================
# 0b) Pure PyTorch multi-scale deformable attention
#     Mirrors the ViT-Adapter / Deformable-DETR MSDeformAttn API, but uses
#     grid_sample so this repo does not depend on the compiled CUDA extension.
# =========================================================
def _is_power_of_2(n):
    if not isinstance(n, int) or n <= 0:
        return False
    return (n & (n - 1)) == 0


def get_reference_points(spatial_shapes, device, dtype=torch.float32):
    refs = []
    for h, w in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype=dtype, device=device),
            torch.linspace(0.5, w - 0.5, w, dtype=dtype, device=device),
            indexing='ij',
        )
        ref_y = ref_y.reshape(-1)[None] / max(h, 1)
        ref_x = ref_x.reshape(-1)[None] / max(w, 1)
        refs.append(torch.stack((ref_x, ref_y), -1))
    return torch.cat(refs, dim=1)[:, :, None, :]


def make_deform_inputs(query_shapes, value_shapes, device, dtype=torch.float32):
    if isinstance(query_shapes, tuple):
        query_shapes = [query_shapes]
    value_spatial_shapes = torch.as_tensor(value_shapes, dtype=torch.long, device=device)
    level_start_index = torch.cat((
        value_spatial_shapes.new_zeros((1,)),
        value_spatial_shapes.prod(1).cumsum(0)[:-1],
    ))
    reference_points = get_reference_points(query_shapes, device=device, dtype=dtype)
    return reference_points, value_spatial_shapes, level_start_index


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # value: (B, S, heads, head_dim)
    # sampling_locations: (B, Lq, heads, levels, points, 2), normalized [0, 1]
    B, _, num_heads, head_dim = value.shape
    _, len_q, _, num_levels, num_points, _ = sampling_locations.shape
    value_splits = [int(h) * int(w) for h, w in value_spatial_shapes.tolist()]
    value_list = value.split(value_splits, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampled_values = []

    for level, (h, w) in enumerate(value_spatial_shapes.tolist()):
        h, w = int(h), int(w)
        value_l = value_list[level].flatten(2).transpose(1, 2)
        value_l = value_l.reshape(B * num_heads, head_dim, h, w)
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled_l = F.grid_sample(
            value_l,
            grid_l,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        sampled_values.append(sampled_l)

    attention_weights = attention_weights.transpose(1, 2)
    attention_weights = attention_weights.reshape(B * num_heads, 1, len_q, num_levels * num_points)
    output = (
        torch.stack(sampled_values, dim=-2).flatten(-2) * attention_weights
    ).sum(-1)
    output = output.view(B, num_heads * head_dim, len_q)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        dim_per_head = d_model // n_heads
        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "MSDeformAttn is most efficient when d_model / n_heads is a power of 2.",
                stacklevel=2,
            )

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        inner_dim = int(d_model * ratio)
        if inner_dim % n_heads != 0:
            raise ValueError(f"d_model * ratio={inner_dim} must be divisible by n_heads={n_heads}")

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, inner_dim)
        self.output_proj = nn.Linear(inner_dim, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        B, len_q, _ = query.shape
        _, len_in, _ = input_flatten.shape
        expected_len = int((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum().item())
        if expected_len != len_in:
            raise RuntimeError(f"MSDeformAttn input length mismatch: got {len_in}, expected {expected_len}")
        if reference_points.shape[0] == 1 and B != 1:
            reference_points = reference_points.expand(B, -1, -1, -1)

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        head_dim = value.shape[-1] // self.n_heads
        value = value.view(B, len_in, self.n_heads, head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            B, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            B, len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[:, 1], input_spatial_shapes[:, 0]], -1
            ).to(dtype=query.dtype, device=query.device)
            sampling_locations = (
                reference_points[:, :, None, :, None, :].to(dtype=query.dtype)
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2].to(dtype=query.dtype)
                + sampling_offsets / self.n_points
                * reference_points[:, :, None, :, None, 2:].to(dtype=query.dtype)
                * 0.5
            )
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        core_dtype = value.dtype
        if core_dtype in (torch.float16, torch.bfloat16):
            output = ms_deform_attn_core_pytorch(
                value.float(),
                input_spatial_shapes,
                sampling_locations.float(),
                attention_weights.float(),
            ).to(core_dtype)
        else:
            output = ms_deform_attn_core_pytorch(
                value,
                input_spatial_shapes,
                sampling_locations,
                attention_weights,
            )
        return self.output_proj(output)


# =========================================================
# 1) LoRA
# =========================================================
class LoRALinear(nn.Module):
    """
    y = xW^T + b + scale * ( (dropout(x) A^T) B^T )
    - Original W,b frozen
    - A: (r, in), B: (out, r)
    """
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = (self.alpha / self.r) if self.r > 0 else 0.0

        # frozen base
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # low-rank
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.drop = nn.Identity()

    def forward(self, x):
        # x: (..., in_features)
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            l = F.linear(self.drop(x), self.lora_A)   # (..., r)
            l = F.linear(l, self.lora_B)              # (..., out)
            out = out + self.scale * l
        return out


def _set_module(root: nn.Module, name: str, new_module: nn.Module):
    """Set module by dotted name."""
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def apply_lora(backbone, r=8, alpha=16, dropout=0.0,
               target_name_regex=r"(attention\.attention\.(query|key|value)|attention\.output\.dense)",
               verbose=True):
    pat = re.compile(target_name_regex)
    targets = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Linear) and pat.search(name):
            targets.append(name)

    patched = []
    for name in targets:
        module = dict(backbone.named_modules())[name]
        lora = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
        _set_module(backbone, name, lora)
        patched.append(name)

    if verbose:
        print(f"[LoRA] Patched {len(patched)} Linear layers with regex={target_name_regex}")
        if len(patched) == 0:
            print("[LoRA][WARN] No layers matched. Check regex.")
    return patched


def mark_only_lora_trainable(model: nn.Module):
    """Call after freezing backbone to re-enable LoRA params."""
    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            p.requires_grad = True


# =========================================================
# 2) AdaptFormer (baseline)
# =========================================================
class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.0, use_ln: bool = True):
        super().__init__()
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # x: (B, HW, C)
        h = self.ln(x)
        h = self.down(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.up(h)
        return h


# =========================================================
# 3a) RGBD-AdaptFormer: additive depth conditioning at the bottleneck
#     delta = W_up(GELU(W_down(x) + depth_proj(depth_tokens)))
#     Depth adds signal additively; no multiplicative gate.
#     Comparison baseline for Geo-AdaptFormer (gating vs. additive).
# =========================================================
class RGBDAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        depth_dim: int,
        bottleneck: int = 64,
        dropout: float = 0.0,
        use_ln: bool = True,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.down = nn.Linear(dim, bottleneck)
        self.depth_proj = nn.Linear(depth_dim, bottleneck)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        # Zero-init depth_proj: starts as plain AdaptFormer at init
        nn.init.zeros_(self.depth_proj.weight)
        nn.init.zeros_(self.depth_proj.bias)

    def forward(self, patch_tokens, feat_h, feat_w, depth_feat):
        """
        patch_tokens: (B, HW, C)
        depth_feat:   (B, depth_dim, h', w')
        """
        h = self.ln(patch_tokens)
        h = self.down(h)                                                      # (B, HW, bottleneck)

        if depth_feat is not None:
            d = F.interpolate(depth_feat, size=(feat_h, feat_w), mode="bilinear", align_corners=False)
            B, D, Hf, Wf = d.shape
            d = d.reshape(B, D, Hf * Wf).transpose(1, 2)                     # (B, HW, depth_dim)
            h = h + self.depth_proj(d)                                        # additive conditioning

        h = self.act(h)
        h = self.drop(h)
        h = self.up(h)
        return h


# PEFT types that use the bottleneck adapter pattern (parallel-to-MLP placement
# via install()). Other types (LoRA, ViT-Adapter family) handle integration
# differently and are not listed here.
BOTTLENECK_PEFT_TYPES = (
    "adaptformer",
    "geoformer",
    "grip",
)


def _make_zero_init_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.0,
    final_bias: float = 0.0,
):
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(hidden_dim, out_dim),
    )
    nn.init.kaiming_uniform_(mlp[0].weight, a=5**0.5)
    nn.init.zeros_(mlp[0].bias)
    nn.init.zeros_(mlp[-1].weight)
    nn.init.constant_(mlp[-1].bias, final_bias)
    return mlp


def _apply_gate_activation(raw_gate: torch.Tensor, gate_act: str) -> torch.Tensor:
    """
    Activation for the gate α. With 'tanh', α ∈ (-1, 1) and α=0 at zero raw input,
    so the gated path vanishes at initialization. 'sigmoid' kept for ablation.
    """
    gate_act = (gate_act or "tanh").lower()
    if gate_act == "tanh":
        return torch.tanh(raw_gate)
    if gate_act == "sigmoid":
        return torch.sigmoid(raw_gate)
    raise ValueError(f"Unsupported gate_act={gate_act!r}. Use tanh or sigmoid.")


# =========================================================
# 5) GRIP-Adapter (Geometry Rescaled by Image Prior) — Ours
# =========================================================
class GRIPAdapter(nn.Module):
    """
    GRIP-Adapter: Geometry Rescaled by Image Prior.

    Per-token output (the adapter residual added in parallel to the layer's MLP):

        Δx = W_up( GELU( v + z + α · r(z) ) )

    Components:
        v    = W_v · x                          visual bottleneck
        z    = W_d · d                          depth bottleneck
        u    = sg(v)                            RGB anchor, detached
        α    = tanh( MLP_g([z, u]) )            anchor-conditioned channel-wise gate
        r(z) = z                                depth residual content (identity by default)

    With `use_value_mlp=False` (default), r(z) = z, so the adapter reduces to
        Δx = W_up( GELU( v + (1+α) · z ) ),
    i.e., depth contribution scaled by an anchor-conditioned channel-wise gain
    in (0, 2).

    Initialization. The final layer of `MLP_g` and `W_up` are zero-initialized,
    so α ≡ 0 and Δx ≡ 0 at the start of training. The depth projection `W_d`
    is also zero-initialized so the visual-only baseline is recovered when
    depth is unavailable.

    LayerNorm placement. With `use_ln=False` (default, AdaptFormer pattern), the
    adapter assumes the input is already block-norm'd (i.e., supplied by the
    transformer block's own norm2 via the install() monkey-patch). No internal
    LN is applied to x. This avoids a second LN-affine on the visual stream and
    keeps parity with the AdaptFormer / RGBD-AdaptFormer baselines.

    Ablation flags:
        use_value_mlp:    r(z) = z if False; r(z) = MLP_r(z) if True.
        separate_anchor:  u = sg(v) if False; u = W_u · sg(x) if True.
        stop_gradient:    apply sg(·) to the anchor input if True (default).
        gate_act:         'tanh' (default; α ∈ (-1, 1)) or 'sigmoid'.
        gate_type:        'channel' (default; α per-token per-channel),
                          'scalar' (per-token, channel-uniform),
                          'channel_only' (per-channel, spatially-uniform).
        symmetric_gate:   If True, adds a mirrored visual gate β:
                              β = gate_act( MLP_β([v, sg(z)]) )
                              Δx = W_up( GELU( (1+β)·v + z + α·r(z) ) )
                          MLP_β has the same structure and zero-init as MLP_α,
                          so β=0 at initialization and behaviour matches the
                          asymmetric form. sg(·) is applied to z when
                          stop_gradient=True (mirror of the α path).
    """

    def __init__(
        self,
        dim: int,
        depth_dim: int,
        bottleneck: int = 96,
        dropout: float = 0.0,
        use_ln: bool = False,
        use_value_mlp: bool = False,
        separate_anchor: bool = False,
        anchor_bottleneck: int = None,
        stop_gradient: bool = True,
        gate_act: str = "tanh",
        gate_init_bias: float = 0.0,
        gate_type: str = "channel",
        hidden_dim: int = None,
        symmetric_gate: bool = False,
    ):
        super().__init__()
        if gate_type not in ("channel", "scalar", "channel_only"):
            raise ValueError(
                f"Unsupported gate_type={gate_type!r}. Use channel, scalar, or channel_only."
            )

        self.use_value_mlp = use_value_mlp
        self.separate_anchor = separate_anchor
        self.stop_gradient = stop_gradient
        self.gate_act = gate_act
        self.gate_type = gate_type
        self.symmetric_gate = symmetric_gate
        self.anchor_bottleneck = anchor_bottleneck or bottleneck

        # Base projections (visual + depth, plus output up-projection).
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.down = nn.Linear(dim, bottleneck)                # W_v
        self.depth_proj = nn.Linear(depth_dim, bottleneck)    # W_d
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck, dim)                  # W_up
        # Depth projection zero-init → model starts visual-only when depth is
        # the first signal to enter the bottleneck. Up projection zero-init
        # → Δx = 0 at initialization, frozen backbone unperturbed.
        nn.init.zeros_(self.depth_proj.weight)
        nn.init.zeros_(self.depth_proj.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # Optional separate anchor projection (W_u distinct from W_v).
        if separate_anchor:
            self.anchor_ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
            self.anchor_down = nn.Linear(dim, self.anchor_bottleneck)  # W_u
            nn.init.kaiming_uniform_(self.anchor_down.weight, a=5 ** 0.5)
            nn.init.zeros_(self.anchor_down.bias)

        # Anchor-conditioned gate MLP. Final layer zero-init so α = 0 at start.
        hidden = hidden_dim or bottleneck
        gate_out_dim = 1 if gate_type == "scalar" else bottleneck
        self.gate_mlp = _make_zero_init_mlp(
            bottleneck + self.anchor_bottleneck,
            hidden,
            gate_out_dim,
            dropout=dropout,
            final_bias=gate_init_bias,
        )

        # Symmetric visual gate β (mirror of α, conditioned on sg(z)). β=0 at init.
        if symmetric_gate:
            self.beta_mlp = _make_zero_init_mlp(
                2 * bottleneck,
                hidden,
                gate_out_dim,
                dropout=dropout,
                final_bias=gate_init_bias,
            )

        # Optional non-linear residual content r(z) = MLP_r(z). Default r(z) = z.
        if use_value_mlp:
            self.value_mlp = _make_zero_init_mlp(
                bottleneck, hidden, bottleneck, dropout=dropout,
            )
        else:
            self.value_mlp = None

    def _anchor(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """RGB anchor u, optionally detached and/or from a separate projection."""
        if self.separate_anchor:
            x_in = x.detach() if self.stop_gradient else x
            return self.anchor_down(self.anchor_ln(x_in))
        return v.detach() if self.stop_gradient else v

    def _compute_gate(self, mlp: nn.Module, primary: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Channel-wise gate from primary bottleneck and anchor (stop-gradient applied by caller)."""
        if self.gate_type == "channel_only":
            p_in = primary.mean(dim=1, keepdim=True)
            a_in = anchor.mean(dim=1, keepdim=True)
        else:
            p_in, a_in = primary, anchor
        raw = mlp(torch.cat([p_in, a_in], dim=-1))
        return _apply_gate_activation(raw, self.gate_act)

    def forward(self, x, feat_h, feat_w, depth_feat):
        """
        Args:
            x:          (B, HW, C)  patch tokens. When use_ln=False, expected
                        to be the output of the block's norm2 (supplied via
                        install()).
            feat_h, feat_w: spatial shape of the patch grid.
            depth_feat: (B, D, h', w')  depth encoder feature map.

        Returns:
            (B, HW, C) adapter residual to add to the layer output.
        """
        v = self.down(self.ln(x))                                  # (B, HW, B_b)

        if depth_feat is None:
            h = v
        else:
            # Align depth to the patch grid (H/4 → feat_h × feat_w).
            d = F.interpolate(
                depth_feat, size=(feat_h, feat_w),
                mode="bilinear", align_corners=False,
            )
            B, D, _, _ = d.shape
            d = d.reshape(B, D, feat_h * feat_w).transpose(1, 2)   # (B, HW, D)
            z = self.depth_proj(d)                                  # (B, HW, B_b)

            u = self._anchor(x, v)                                  # (B, HW, B_a)
            alpha = self._compute_gate(self.gate_mlp, z, u)        # (B, *, gate_out)
            r_z = self.value_mlp(z) if self.use_value_mlp else z   # (B, HW, B_b)

            if self.symmetric_gate:
                z_anchor = z.detach() if self.stop_gradient else z
                beta = self._compute_gate(self.beta_mlp, v, z_anchor)
                h = (1 + beta) * v + z + alpha * r_z
            else:
                h = v + z + alpha * r_z

        h = self.act(h)
        h = self.drop(h)
        return self.up(h)


# =========================================================
# 6) ViT-Adapter proper (MSDeformAttn architecture, pure PyTorch backend)
#    Reference: arXiv:2205.08534
# =========================================================

class ViTAdapterSPM(nn.Module):
    """
    Spatial Prior Module: RGB image → multi-scale CNN features at H/8, H/16, H/32.
    Each scale projected to embed_dim. Uses GroupNorm instead of SyncBatchNorm.
    """
    def __init__(self, embed_dim: int, in_channels: int = 3, inplanes: int = 64):
        super().__init__()
        def _gn(c): return nn.GroupNorm(min(32, c), c)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, inplanes, 3, 2, 1, bias=False), _gn(inplanes), nn.ReLU(True),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False), _gn(inplanes), nn.ReLU(True),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False), _gn(inplanes), nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),
        )  # → H/4, W/4
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, 3, 2, 1, bias=False), _gn(2 * inplanes), nn.ReLU(True),
        )  # → H/8
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, 3, 2, 1, bias=False), _gn(4 * inplanes), nn.ReLU(True),
        )  # → H/16
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, 3, 2, 1, bias=False), _gn(4 * inplanes), nn.ReLU(True),
        )  # → H/32
        self.proj2 = nn.Conv2d(2 * inplanes, embed_dim, 1)
        self.proj3 = nn.Conv2d(4 * inplanes, embed_dim, 1)
        self.proj4 = nn.Conv2d(4 * inplanes, embed_dim, 1)

    def forward(self, x: torch.Tensor):
        """x: (B, in_channels, H, W) → (c2, c3, c4) each (B, embed_dim, Hi, Wi)"""
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return self.proj2(c2), self.proj3(c3), self.proj4(c4)


class GeoViTAdapterSPM(nn.Module):
    """
    Depth-feature based SPM: uses depth_feat (at H/4) and downsamples to 3 scales
    matching ViT resolution × {2, 1, 0.5}, matching the original ViT-Adapter's
    c2/c3/c4 spatial extents relative to ViT patch tokens.
    """
    def __init__(self, embed_dim: int, depth_dim: int):
        super().__init__()
        def _dsconv(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, 2, 1, groups=c, bias=False),
                nn.Conv2d(c, c, 1, bias=True),
                nn.GroupNorm(min(32, c), c),
                nn.GELU(),
            )
        self.proj = nn.Conv2d(depth_dim, embed_dim, 1, bias=False)
        self.down2 = _dsconv(embed_dim)   # 2*feat → feat
        self.down3 = _dsconv(embed_dim)   # feat → feat/2

    def forward(self, depth_feat: torch.Tensor, feat_h: int, feat_w: int):
        """
        depth_feat: (B, depth_dim, h', w') - from depth backbone at H/4.
        Returns c2, c3, c4 at 2*feat, feat, feat/2 resolution respectively.
        """
        # Upsample to 2× ViT patch resolution (matching original ViT-Adapter c2 scale)
        d = F.interpolate(depth_feat, size=(feat_h * 2, feat_w * 2),
                          mode='bilinear', align_corners=False)
        c2 = self.proj(d)    # (B, embed_dim, 2*feat_h, 2*feat_w)
        c3 = self.down2(c2)  # (B, embed_dim, feat_h, feat_w)
        c4 = self.down3(c3)  # stride-2 depth prior
        c4 = F.interpolate(
            c4,
            size=(max(1, feat_h // 2), max(1, feat_w // 2)),
            mode='bilinear',
            align_corners=False,
        )
        return c2, c3, c4


class _ViTAdapterConvFFN(nn.Module):
    """Official ViT-Adapter ConvFFN, generalized to arbitrary per-scale shapes."""
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, c_shapes: list) -> torch.Tensor:
        x = self.fc1(x)
        parts = torch.split(x, [h * w for h, w in c_shapes], dim=1)
        conv_parts = []
        for part, (h, w) in zip(parts, c_shapes):
            B, N, C = part.shape
            part = part.transpose(1, 2).reshape(B, C, h, w)
            part = self.dwconv(part)
            conv_parts.append(part.reshape(B, C, N).transpose(1, 2))
        x = torch.cat(conv_parts, dim=1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _ViTAdapterInjector(nn.Module):
    """SPM -> ViT injector from the original ViT-Adapter interaction block."""
    def __init__(self, dim: int, num_heads: int, n_points: int = 4,
                 deform_ratio: float = 1.0, init_scale: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.feat_norm = nn.LayerNorm(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=3,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )
        self.gamma = nn.Parameter(init_scale * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        attn = self.attn(
            self.query_norm(query),
            reference_points,
            self.feat_norm(feat),
            spatial_shapes,
            level_start_index,
            None,
        )
        return query + self.gamma * attn


class _ViTAdapterExtractor(nn.Module):
    """ViT -> SPM extractor from the original ViT-Adapter interaction block."""
    def __init__(self, dim: int, num_heads: int, n_points: int = 4,
                 deform_ratio: float = 1.0, with_cffn: bool = True,
                 cffn_ratio: float = 0.25, drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.feat_norm = nn.LayerNorm(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=1,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )
        self.with_cffn = with_cffn
        if with_cffn:
            hidden = max(1, int(dim * cffn_ratio))
            self.ffn = _ViTAdapterConvFFN(dim, hidden, drop=drop)
            self.ffn_norm = nn.LayerNorm(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index, c_shapes):
        attn = self.attn(
            self.query_norm(query),
            reference_points,
            self.feat_norm(feat),
            spatial_shapes,
            level_start_index,
            None,
        )
        query = query + attn
        if self.with_cffn:
            query = query + self.drop_path(self.ffn(self.ffn_norm(query), c_shapes))
        return query


class _ViTAdapterInteraction(nn.Module):
    """One ViT-Adapter interaction block: inject, then extract."""
    def __init__(self, dim: int, num_heads: int, n_points: int = 4,
                 dropout: float = 0.0, drop_path: float = 0.0,
                 with_cffn: bool = True, cffn_ratio: float = 0.25,
                 init_scale: float = 0.0, deform_ratio: float = 1.0,
                 extra_extractor: bool = False):
        super().__init__()
        self.injector = _ViTAdapterInjector(
            dim=dim,
            num_heads=num_heads,
            n_points=n_points,
            deform_ratio=deform_ratio,
            init_scale=init_scale,
        )
        self.extractor = _ViTAdapterExtractor(
            dim=dim,
            num_heads=num_heads,
            n_points=n_points,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=dropout,
            drop_path=drop_path,
        )
        if extra_extractor:
            self.extra_extractors = nn.ModuleList([
                _ViTAdapterExtractor(
                    dim=dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    deform_ratio=deform_ratio,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    drop=dropout,
                    drop_path=drop_path,
                )
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def inject(self, x: torch.Tensor, c: torch.Tensor, deform_inputs) -> torch.Tensor:
        return self.injector(x, deform_inputs[0], c, deform_inputs[1], deform_inputs[2])

    def extract(self, c: torch.Tensor, x: torch.Tensor, deform_inputs, c_shapes: list) -> torch.Tensor:
        c = self.extractor(c, deform_inputs[0], x, deform_inputs[1], deform_inputs[2], c_shapes)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(c, deform_inputs[0], x, deform_inputs[1], deform_inputs[2], c_shapes)
        return c


def _interaction_ends(num_layers: int, num_interactions: int) -> list:
    """Divide ViT layers into groups; return last layer index of each group."""
    group = num_layers // num_interactions
    ends = [min(group * (i + 1) - 1, num_layers - 1) for i in range(num_interactions)]
    ends[-1] = num_layers - 1  # ensure last group includes the final layer
    return ends


def _interaction_ends_from_layers(selected_layers, num_interactions: int) -> list:
    layers = sorted(int(idx) for idx in selected_layers)
    if not layers:
        return []
    if len(layers) <= num_interactions:
        return layers

    step = len(layers) / float(num_interactions)
    ends = []
    for i in range(num_interactions):
        pos = min(math.ceil(step * (i + 1)) - 1, len(layers) - 1)
        ends.append(layers[pos])

    deduped = []
    seen = set()
    for idx in ends:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    if deduped[-1] != layers[-1]:
        deduped[-1] = layers[-1]
    return deduped


def _init_vit_adapter_weights(module):
    if isinstance(module, MSDeformAttn):
        module._reset_parameters()
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()


class ViTAdapterPEFT(nn.Module):
    """
    Proper ViT-Adapter adapted for GraspPro's per-layer PEFT interface.

    Forward protocol (called by PeftManager):
      1. prepare(pixel_values, x, feat_h, feat_w)  — before the ViT loop.
         Runs RGB SPM and applies first injection (SPM → ViT). Returns updated x.
      2. forward(layer_idx, x, feat_h, feat_w)  — after each ViT layer.
         At interaction boundaries: extract (ViT→SPM) + inject (SPM→ViT for next group).

    Note: self._c / self._c_shapes are per-forward temporary state.
    With DDP each process has its own copy, so no cross-process conflicts.
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_interactions: int = 4,
        num_heads: int = 6,
        inplanes: int = 64,
        in_channels: int = 3,
        n_points: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        init_scale: float = 0.0,
        deform_ratio: float = 1.0,
        add_level_embed: bool = True,
        spm_pool_factor: int = 1,
        use_extra_extractor: bool = True,
        layers_to_adapt="all",
    ):
        super().__init__()
        self.spm_pool_factor = spm_pool_factor
        self.layers_to_adapt = _resolve_layers_to_adapt(num_layers, layers_to_adapt)
        self.first_selected_layer = min(self.layers_to_adapt) if self.layers_to_adapt else None
        self.interaction_ends = _interaction_ends_from_layers(self.layers_to_adapt, num_interactions)
        self.num_interactions = len(self.interaction_ends)
        self.layer_to_inter = {l: i for i, l in enumerate(self.interaction_ends)}

        self.spm = ViTAdapterSPM(embed_dim=embed_dim, in_channels=in_channels, inplanes=inplanes)
        self.add_level_embed = add_level_embed
        if add_level_embed:
            self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
            nn.init.normal_(self.level_embed)

        self.interactions = nn.ModuleList([
            _ViTAdapterInteraction(
                embed_dim,
                num_heads,
                n_points=n_points,
                dropout=dropout,
                drop_path=drop_path,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                init_scale=init_scale,
                deform_ratio=deform_ratio,
                extra_extractor=(i == self.num_interactions - 1 and use_extra_extractor),
            )
            for i in range(self.num_interactions)
        ])
        self.spm.apply(_init_vit_adapter_weights)
        self.interactions.apply(_init_vit_adapter_weights)

        # Per-forward state (not parameters)
        self._c = None
        self._c_shapes = None
        self._deform_inputs1 = None
        self._deform_inputs2 = None
        self._started = False

    def _to_tokens(self, c2, c3, c4):
        B, C, h2, w2 = c2.shape
        _, _, h3, w3 = c3.shape
        _, _, h4, w4 = c4.shape
        t2 = c2.reshape(B, C, -1).transpose(1, 2)
        t3 = c3.reshape(B, C, -1).transpose(1, 2)
        t4 = c4.reshape(B, C, -1).transpose(1, 2)
        if self.add_level_embed:
            t2 = t2 + self.level_embed[0]
            t3 = t3 + self.level_embed[1]
            t4 = t4 + self.level_embed[2]
        return torch.cat([t2, t3, t4], dim=1), [(h2, w2), (h3, w3), (h4, w4)]

    def prepare(self, pixel_values: torch.Tensor, x: torch.Tensor,
                feat_h: int, feat_w: int) -> torch.Tensor:
        """Run RGB SPM and apply first injection. Returns updated x."""
        self._c = None  # reset state for this forward pass
        self._started = False
        c2, c3, c4 = self.spm(pixel_values)
        # Align SPM spatial sizes to ViT patch resolution.
        # The CNN uses fixed strides (H/8, H/16, H/32) which only aligns with patch_size=16.
        # For patch_size=14 (DINOv2) we interpolate so c2=2*feat, c3=feat, c4=feat/2.
        c2 = F.interpolate(c2, (feat_h * 2, feat_w * 2), mode='bilinear', align_corners=False)
        c3 = F.interpolate(c3, (feat_h, feat_w), mode='bilinear', align_corners=False)
        c4h, c4w = max(1, feat_h // 2), max(1, feat_w // 2)
        c4 = F.interpolate(c4, (c4h, c4w), mode='bilinear', align_corners=False)
        # For large images, pool SPM down to avoid O(N_spm × N_vit) attention explosion.
        # spm_pool_factor=3 → c3 at feat/3 resolution, matching the 224px-equivalent scale.
        if self.spm_pool_factor > 1:
            p = self.spm_pool_factor
            c2 = F.adaptive_avg_pool2d(c2, (max(1, feat_h * 2 // p), max(1, feat_w * 2 // p)))
            c3 = F.adaptive_avg_pool2d(c3, (max(1, feat_h // p), max(1, feat_w // p)))
            c4 = F.adaptive_avg_pool2d(c4, (max(1, c4h // p), max(1, c4w // p)))
        self._c, self._c_shapes = self._to_tokens(c2, c3, c4)
        self._deform_inputs1 = make_deform_inputs(
            (feat_h, feat_w), self._c_shapes, x.device, dtype=x.dtype
        )
        self._deform_inputs2 = make_deform_inputs(
            self._c_shapes, [(feat_h, feat_w)], x.device, dtype=x.dtype
        )
        if self.first_selected_layer == 0 and self.interaction_ends:
            prefix, patch, _ = get_patch_tokens(x, feat_h, feat_w)
            patch = self.interactions[0].inject(patch, self._c, self._deform_inputs1)
            self._started = True
            return merge_patch_tokens(prefix, patch)
        return x

    def forward(self, layer_idx: int, x: torch.Tensor,
                feat_h: int, feat_w: int, depth_feat=None) -> torch.Tensor:
        if self._c is None or not self.layers_to_adapt:
            return x
        if not self._started:
            if self.first_selected_layer is not None and layer_idx == self.first_selected_layer - 1:
                prefix, patch, _ = get_patch_tokens(x, feat_h, feat_w)
                patch = self.interactions[0].inject(patch, self._c, self._deform_inputs1)
                self._started = True
                return merge_patch_tokens(prefix, patch)
            return x
        inter_idx = self.layer_to_inter.get(layer_idx, -1)
        if inter_idx < 0:
            return x
        prefix, patch, _ = get_patch_tokens(x, feat_h, feat_w)
        # Extract: update SPM features from ViT tokens
        self._c = self.interactions[inter_idx].extract(
            self._c, patch, self._deform_inputs2, self._c_shapes
        )
        # Inject: update ViT tokens from SPM (for next group)
        if inter_idx < self.num_interactions - 1:
            patch = self.interactions[inter_idx + 1].inject(patch, self._c, self._deform_inputs1)
        return merge_patch_tokens(prefix, patch)


class GeoViTAdapterPEFT(nn.Module):
    """
    Geo-ViT-Adapter: uses depth features as spatial priors instead of RGB CNN.

    Same bidirectional interaction as ViTAdapterPEFT, but the SPM is replaced
    with a depth-conditioned multi-scale prior derived from depth_feat
    (already computed by the depth backbone).
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        depth_dim: int,
        num_interactions: int = 4,
        num_heads: int = 6,
        n_points: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        init_scale: float = 0.0,
        deform_ratio: float = 1.0,
        detach_depth: bool = True,
        add_level_embed: bool = True,
        spm_pool_factor: int = 1,
        use_extra_extractor: bool = True,
        layers_to_adapt="all",
    ):
        super().__init__()
        self.spm_pool_factor = spm_pool_factor
        self.detach_depth = detach_depth
        self.layers_to_adapt = _resolve_layers_to_adapt(num_layers, layers_to_adapt)
        self.first_selected_layer = min(self.layers_to_adapt) if self.layers_to_adapt else None
        self.interaction_ends = _interaction_ends_from_layers(self.layers_to_adapt, num_interactions)
        self.num_interactions = len(self.interaction_ends)
        self.layer_to_inter = {l: i for i, l in enumerate(self.interaction_ends)}

        self.spm = GeoViTAdapterSPM(embed_dim=embed_dim, depth_dim=depth_dim)
        self.add_level_embed = add_level_embed
        if add_level_embed:
            self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
            nn.init.normal_(self.level_embed)

        self.interactions = nn.ModuleList([
            _ViTAdapterInteraction(
                embed_dim,
                num_heads,
                n_points=n_points,
                dropout=dropout,
                drop_path=drop_path,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                init_scale=init_scale,
                deform_ratio=deform_ratio,
                extra_extractor=(i == self.num_interactions - 1 and use_extra_extractor),
            )
            for i in range(self.num_interactions)
        ])
        self.spm.apply(_init_vit_adapter_weights)
        self.interactions.apply(_init_vit_adapter_weights)

        self._c = None
        self._c_shapes = None
        self._deform_inputs1 = None
        self._deform_inputs2 = None
        self._started = False

    def _to_tokens(self, c2, c3, c4):
        B, C, h2, w2 = c2.shape
        _, _, h3, w3 = c3.shape
        _, _, h4, w4 = c4.shape
        t2 = c2.reshape(B, C, -1).transpose(1, 2)
        t3 = c3.reshape(B, C, -1).transpose(1, 2)
        t4 = c4.reshape(B, C, -1).transpose(1, 2)
        if self.add_level_embed:
            t2 = t2 + self.level_embed[0]
            t3 = t3 + self.level_embed[1]
            t4 = t4 + self.level_embed[2]
        return torch.cat([t2, t3, t4], dim=1), [(h2, w2), (h3, w3), (h4, w4)]

    def prepare(self, depth_feat: torch.Tensor, x: torch.Tensor,
                feat_h: int, feat_w: int) -> torch.Tensor:
        """Run depth SPM and apply first injection. Returns updated x."""
        self._c = None
        self._started = False
        d = depth_feat.detach() if self.detach_depth else depth_feat
        c2, c3, c4 = self.spm(d, feat_h, feat_w)
        if self.spm_pool_factor > 1:
            p = self.spm_pool_factor
            c2h, c2w = c2.shape[2], c2.shape[3]
            c3h, c3w = c3.shape[2], c3.shape[3]
            c4h, c4w = c4.shape[2], c4.shape[3]
            c2 = F.adaptive_avg_pool2d(c2, (max(1, c2h // p), max(1, c2w // p)))
            c3 = F.adaptive_avg_pool2d(c3, (max(1, c3h // p), max(1, c3w // p)))
            c4 = F.adaptive_avg_pool2d(c4, (max(1, c4h // p), max(1, c4w // p)))
        self._c, self._c_shapes = self._to_tokens(c2, c3, c4)
        self._deform_inputs1 = make_deform_inputs(
            (feat_h, feat_w), self._c_shapes, x.device, dtype=x.dtype
        )
        self._deform_inputs2 = make_deform_inputs(
            self._c_shapes, [(feat_h, feat_w)], x.device, dtype=x.dtype
        )
        if self.first_selected_layer == 0 and self.interaction_ends:
            prefix, patch, _ = get_patch_tokens(x, feat_h, feat_w)
            patch = self.interactions[0].inject(patch, self._c, self._deform_inputs1)
            self._started = True
            return merge_patch_tokens(prefix, patch)
        return x

    def forward(self, layer_idx: int, x: torch.Tensor,
                feat_h: int, feat_w: int, depth_feat=None) -> torch.Tensor:
        if self._c is None or not self.layers_to_adapt:
            return x
        if not self._started:
            if self.first_selected_layer is not None and layer_idx == self.first_selected_layer - 1:
                prefix, patch, _ = get_patch_tokens(x, feat_h, feat_w)
                patch = self.interactions[0].inject(patch, self._c, self._deform_inputs1)
                self._started = True
                return merge_patch_tokens(prefix, patch)
            return x
        inter_idx = self.layer_to_inter.get(layer_idx, -1)
        if inter_idx < 0:
            return x
        prefix, patch, _ = get_patch_tokens(x, feat_h, feat_w)
        self._c = self.interactions[inter_idx].extract(
            self._c, patch, self._deform_inputs2, self._c_shapes
        )
        if inter_idx < self.num_interactions - 1:
            patch = self.interactions[inter_idx + 1].inject(patch, self._c, self._deform_inputs1)
        return merge_patch_tokens(prefix, patch)


# =========================================================
# 6) PEFT Manager
# =========================================================
class PeftManager(nn.Module):
    """
    Unified PEFT manager for residual adapters that operate on patch tokens.
    Supported peft_type:
      - adaptformer       — per-layer bottleneck MLP, AdaptFormer-style
      - geoformer  — additive RGB-D bottleneck adapter
      - grip              — GRIP-Adapter (Ours; flag-based ablation variants)
      - vit_adapter       — ViT-Adapter (RGB SPM + cross-attention interaction)
      - geo_vit_adapter   — ViT-Adapter with depth-based SPM

    LoRA is applied by patching backbone modules and is not handled here.
    """
    def __init__(
        self,
        peft_type: str,
        num_layers: int,
        embed_dim: int,
        depth_dim: int,
        layers_to_adapt="all",   # "all", "last_half", "decoder_aligned_late", or explicit indices
        decoder_out_indices=None,
        adapter_bottleneck: int = 64,
        adapter_dropout: float = 0.0,
        adapter_scale: float = 1.0,
        adapter_drop_path: float = 0.2,
        adapter_use_ln: bool = None,
        detach_depth: bool = True,
        # GRIP-Adapter specific
        use_value_mlp: bool = False,
        separate_anchor: bool = False,
        anchor_bottleneck: int = None,
        stop_gradient: bool = True,
        gate_act: str = "tanh",
        gate_init_bias: float = 0.0,
        gate_type: str = "channel",
        symmetric_gate: bool = False,
        # ViT-Adapter / Geo-ViT-Adapter specific
        num_interactions: int = 4,
        num_heads: int = 6,
        n_points: int = 4,
        inplanes: int = 64,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        init_scale: float = 0.0,
        deform_ratio: float = 1.0,
        add_level_embed: bool = True,
        spm_pool_factor: int = 1,
        use_extra_extractor: bool = True,
    ):
        super().__init__()
        self.peft_type = peft_type.lower()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.depth_dim = depth_dim
        self.adapter_scale = adapter_scale
        self.detach_depth = detach_depth

        self.layers_to_adapt = _resolve_layers_to_adapt(
            num_layers,
            layers_to_adapt,
            decoder_out_indices=decoder_out_indices,
        )

        self._installed = False  # True after install() monkey-patches backbone blocks
        self.drop_path = DropPath(adapter_drop_path) if adapter_drop_path > 0.0 else nn.Identity()

        # Per-step context set by GraspPro before the layer loop. Read by
        # the monkey-patched block forward via _call_adapter().
        self._ctx_feat_h = None
        self._ctx_feat_w = None
        self._ctx_depth_feat = None
        self._ctx_num_prefix = 0

        if self.peft_type == "adaptformer":
            # use_ln=False: the block's norm2 supplies the LN'd input via install().
            self.blocks = nn.ModuleList([
                BottleneckAdapter(embed_dim, bottleneck=adapter_bottleneck,
                                  dropout=adapter_dropout, use_ln=False)
                for _ in range(num_layers)
            ])
        elif self.peft_type == "geoformer":
            use_ln = True if adapter_use_ln is None else bool(adapter_use_ln)
            self.blocks = nn.ModuleList([
                RGBDAdapter(embed_dim, depth_dim, bottleneck=adapter_bottleneck,
                            dropout=adapter_dropout, use_ln=use_ln)
                for _ in range(num_layers)
            ])
        elif self.peft_type == "grip":
            # GRIP-Adapter (Ours). use_ln=False relies on the block's norm2
            # (set up by install()) for visual normalization.
            use_ln = False if adapter_use_ln is None else bool(adapter_use_ln)
            self.blocks = nn.ModuleList([
                GRIPAdapter(
                    dim=embed_dim,
                    depth_dim=depth_dim,
                    bottleneck=adapter_bottleneck,
                    dropout=adapter_dropout,
                    use_ln=use_ln,
                    use_value_mlp=use_value_mlp,
                    separate_anchor=separate_anchor,
                    anchor_bottleneck=anchor_bottleneck,
                    stop_gradient=stop_gradient,
                    gate_act=gate_act,
                    gate_init_bias=gate_init_bias,
                    gate_type=gate_type,
                    symmetric_gate=symmetric_gate,
                )
                for _ in range(num_layers)
            ])
        elif self.peft_type == "vit_adapter":
            self.adapter = ViTAdapterPEFT(
                embed_dim=embed_dim, num_layers=num_layers,
                num_interactions=num_interactions, num_heads=num_heads,
                n_points=n_points,
                inplanes=inplanes, dropout=adapter_dropout,
                drop_path=adapter_drop_path, with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, init_scale=init_scale,
                deform_ratio=deform_ratio,
                add_level_embed=add_level_embed, spm_pool_factor=spm_pool_factor,
                use_extra_extractor=use_extra_extractor,
                layers_to_adapt=self.layers_to_adapt,
            )
        elif self.peft_type == "geo_vit_adapter":
            self.adapter = GeoViTAdapterPEFT(
                embed_dim=embed_dim, num_layers=num_layers, depth_dim=depth_dim,
                num_interactions=num_interactions, num_heads=num_heads,
                n_points=n_points,
                dropout=adapter_dropout, drop_path=adapter_drop_path,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, init_scale=init_scale,
                deform_ratio=deform_ratio,
                detach_depth=detach_depth, add_level_embed=add_level_embed,
                spm_pool_factor=spm_pool_factor,
                use_extra_extractor=use_extra_extractor,
                layers_to_adapt=self.layers_to_adapt,
            )
        else:
            raise ValueError(f"Unknown peft_type={peft_type}")

    def install(self, backbone_layers):
        """Place bottleneck adapter parallel to MLP at each supported block.

        Follows AdaptFormer (Chen et al., 2022):
            x̃   = x + Attn-residual   (block-specific structure preserved)
            out = x̃ + MLP-residual    + s · DropPath(Adapter(LN2(x̃)))

        Supported block kinds (auto-detected):
          - 'da3'    — depth-anything-3 Block (norm1/attn/ls1/norm2/mlp/ls2)
          - 'dinov2' — HF Dinov2Layer (also covers DAv2)
          - 'dinov3' — HF DINOv3ViTLayer (attention returns tuple, takes position_embeddings)
          - 'vit'    — HF ViTLayer (BERT-style; MLP residual handled inside ViTOutput)
          - 'siglip' — HF SiglipEncoderLayer (pre-norm, no DropPath/LayerScale)
          - 'sam'    — HF SamVisionLayer (windowed attn, full-map MLP; (B,H,W,C))

        Unsupported blocks fall back to the post-block forward path, leaving
        backbone behavior unchanged.
        """
        if self.peft_type not in BOTTLENECK_PEFT_TYPES:
            return

        patched = 0
        for i, layer in enumerate(backbone_layers):
            if i not in self.layers_to_adapt:
                continue
            blk = getattr(layer, '_blk', layer)
            kind = self._detect_block_kind(blk)
            if kind is None:
                print(f"[PEFT/{self.peft_type}] Block {i} ({type(blk).__name__}) unsupported; falling back to post-block placement.")
                continue
            blk.forward = types.MethodType(self._make_forward(i, kind), blk)
            patched += 1

        if patched > 0:
            self._installed = True
            print(f"[PEFT/{self.peft_type}] Installed parallel-to-MLP in {patched} blocks.")

    @staticmethod
    def _detect_block_kind(blk):
        if hasattr(blk, 'ls2') and hasattr(blk, 'attn') and hasattr(blk, 'norm2') and hasattr(blk, 'mlp'):
            return 'da3'
        cls = type(blk).__name__
        if cls == 'DINOv3ViTLayer':
            return 'dinov3'
        if cls == 'Dinov2Layer':
            return 'dinov2'
        if cls == 'ViTLayer':
            return 'vit'
        if cls == 'SiglipEncoderLayer':
            return 'siglip'
        if cls == 'SamVisionLayer':
            return 'sam'
        return None

    def _make_forward(self, layer_idx, kind):
        return {
            'da3':    self._make_da3_forward,
            'dinov2': self._make_dinov2_forward,
            'dinov3': self._make_dinov3_forward,
            'vit':    self._make_vit_forward,
            'siglip': self._make_siglip_forward,
            'sam':    self._make_sam_forward,
        }[kind](layer_idx)

    def _make_da3_forward(self, layer_idx):
        """DA3 Block: x = x + LS1(Attn(LN1(x)));  x = x + LS2(MLP(LN2(x)))
        (no DropPath on residuals — DA3 sample_drop_ratio is always 0)."""
        peft, scale, dp = self, self.adapter_scale, self.drop_path

        def fwd(self_blk, x, pos=None, attn_mask=None):
            x = x + self_blk.ls1(self_blk.attn(self_blk.norm1(x), pos=pos, attn_mask=attn_mask))
            normed = self_blk.norm2(x)
            delta = peft._call_adapter(layer_idx, normed)
            delta = F.pad(delta, (0, 0, peft._ctx_num_prefix, 0))
            return x + self_blk.ls2(self_blk.mlp(normed)) + dp(scale * delta)

        return fwd

    def _make_dinov2_forward(self, layer_idx):
        """HF Dinov2Layer (also DAv2): attention sub-block then MLP sub-block,
        each with DropPath(LayerScale(...)) + residual. Adapter is added at
        the MLP residual destination, sharing LN2(x̃) with the MLP."""
        peft, scale, dp = self, self.adapter_scale, self.drop_path

        def fwd(self_blk, hidden_states):
            h = self_blk.norm1(hidden_states)
            a = self_blk.attention(h)
            if isinstance(a, tuple):
                a = a[0]
            a = self_blk.layer_scale1(a)
            hidden_states = self_blk.drop_path(a) + hidden_states

            normed = self_blk.norm2(hidden_states)
            mlp_out = self_blk.layer_scale2(self_blk.mlp(normed))
            delta = peft._call_adapter(layer_idx, normed)
            delta = F.pad(delta, (0, 0, peft._ctx_num_prefix, 0))
            return self_blk.drop_path(mlp_out) + dp(scale * delta) + hidden_states

        return fwd

    def _make_dinov3_forward(self, layer_idx):
        """HF DINOv3ViTLayer: like Dinov2 but attention returns a tuple and
        accepts attention_mask / position_embeddings (RoPE) kwargs."""
        peft, scale, dp = self, self.adapter_scale, self.drop_path

        def fwd(self_blk, hidden_states, attention_mask=None, position_embeddings=None):
            residual = hidden_states
            h = self_blk.norm1(hidden_states)
            h, _ = self_blk.attention(h, attention_mask=attention_mask, position_embeddings=position_embeddings)
            h = self_blk.layer_scale1(h)
            hidden_states = self_blk.drop_path(h) + residual

            residual = hidden_states
            normed = self_blk.norm2(hidden_states)
            mlp_out = self_blk.layer_scale2(self_blk.mlp(normed))
            delta = peft._call_adapter(layer_idx, normed)
            delta = F.pad(delta, (0, 0, peft._ctx_num_prefix, 0))
            return self_blk.drop_path(mlp_out) + dp(scale * delta) + residual

        return fwd

    def _make_vit_forward(self, layer_idx):
        """HF ViTLayer (BERT-style pre-norm). ViTOutput adds the MLP residual
        internally, so the adapter is summed on top of its output — yielding
        the parallel-to-MLP destination x' + MLP-residual + s · Adapter-residual.
        """
        peft, scale, dp = self, self.adapter_scale, self.drop_path

        def fwd(self_blk, hidden_states):
            h = self_blk.layernorm_before(hidden_states)
            a = self_blk.attention(h)
            if isinstance(a, tuple):
                a = a[0]
            hidden_states = a + hidden_states

            normed = self_blk.layernorm_after(hidden_states)
            # ViTOutput(intermediate_out, x') = dropout(dense(intermediate_out)) + x'
            mlp_with_res = self_blk.output(self_blk.intermediate(normed), hidden_states)
            delta = peft._call_adapter(layer_idx, normed)
            delta = F.pad(delta, (0, 0, peft._ctx_num_prefix, 0))
            return mlp_with_res + dp(scale * delta)

        return fwd

    def _make_siglip_forward(self, layer_idx):
        """HF SiglipEncoderLayer: pre-norm with no DropPath / no LayerScale.
        Attention returns (out, _) and requires attention_mask as a kwarg."""
        peft, scale, dp = self, self.adapter_scale, self.drop_path

        def fwd(self_blk, hidden_states, attention_mask=None, **kwargs):
            residual = hidden_states
            h = self_blk.layer_norm1(hidden_states)
            h, _ = self_blk.self_attn(hidden_states=h, attention_mask=attention_mask, **kwargs)
            hidden_states = residual + h

            residual = hidden_states
            normed = self_blk.layer_norm2(hidden_states)
            mlp_out = self_blk.mlp(normed)
            delta = peft._call_adapter(layer_idx, normed)
            delta = F.pad(delta, (0, 0, peft._ctx_num_prefix, 0))
            return residual + mlp_out + dp(scale * delta)

        return fwd

    def _make_sam_forward(self, layer_idx):
        """HF SamVisionLayer: windowed attention then a full-map MLP. layer_norm2
        and mlp run on the unpartitioned (B, H, W, C) map, so the adapter is summed
        parallel to the MLP on the same normed input (SAM has no prefix tokens)."""
        peft, scale, dp = self, self.adapter_scale, self.drop_path

        def fwd(self_blk, hidden_states):
            residual = hidden_states
            h = self_blk.layer_norm1(hidden_states)
            if self_blk.window_size > 0:
                height, width = h.shape[1], h.shape[2]
                h, padding_shape = self_blk.window_partition(h, self_blk.window_size)
            h, _ = self_blk.attn(hidden_states=h)
            if self_blk.window_size > 0:
                h = self_blk.window_unpartition(h, self_blk.window_size, padding_shape, (height, width))
            hidden_states = residual + h
            normed = self_blk.layer_norm2(hidden_states)          # (B, H, W, C)
            mlp_out = self_blk.mlp(normed)
            delta = peft._call_adapter_sam(layer_idx, normed)     # (B, H, W, C)
            return hidden_states + mlp_out + dp(scale * delta)

        return fwd

    def _call_adapter_sam(self, layer_idx, normed_map):
        """normed_map: (B, H, W, C) from SAM's layer_norm2. Flatten to patch
        tokens, run the bottleneck adapter, restore the spatial layout."""
        B, H, W, C = normed_map.shape
        tokens = normed_map.reshape(B, H * W, C)
        adapter = self.blocks[layer_idx]
        if self.peft_type == "adaptformer":
            delta = adapter(tokens)
        else:
            d = self._ctx_depth_feat
            if self.detach_depth and d is not None:
                d = d.detach()
            delta = adapter(tokens, H, W, d)
        return delta.reshape(B, H, W, C)

    def _call_adapter(self, layer_idx, normed_full):
        """normed_full: (B, T, C) including prefix tokens.
        Returns delta on patch tokens only: (B, HW, C)."""
        np_ = self._ctx_num_prefix
        normed_patches = normed_full[:, np_:, :]
        adapter = self.blocks[layer_idx]
        if self.peft_type == "adaptformer":
            return adapter(normed_patches)
        d = self._ctx_depth_feat
        if self.detach_depth and d is not None:
            d = d.detach()
        return adapter(normed_patches, self._ctx_feat_h, self._ctx_feat_w, d)

    def prepare(self, pixel_values=None, x=None, feat_h=None, feat_w=None, depth_feat=None):
        """
        Called before the ViT layer loop. For vit_adapter / geo_vit_adapter:
        runs the SPM and applies the first injection. Returns updated x.
        No-op (returns x) for all other PEFT types.
        """
        if self.peft_type == "vit_adapter":
            rgb = pixel_values[:, :3] if pixel_values is not None and pixel_values.shape[1] > 3 else pixel_values
            return self.adapter.prepare(rgb, x, feat_h, feat_w)
        elif self.peft_type == "geo_vit_adapter":
            return self.adapter.prepare(depth_feat, x, feat_h, feat_w)
        return x

    def forward(self, layer_idx: int, x: torch.Tensor, feat_h: int, feat_w: int, depth_feat=None):
        if self.peft_type in ("vit_adapter", "geo_vit_adapter"):
            return self.adapter(layer_idx, x, feat_h, feat_w, depth_feat=depth_feat)

        # If install() succeeded, the block already applied the adapter parallel
        # to MLP inside its own forward — nothing left to do here.
        if self._installed:
            return x

        if layer_idx not in self.layers_to_adapt:
            return x

        prefix, patch, patch_start = get_patch_tokens(x, feat_h, feat_w)  # patch: (B,HW,C)

        if self.peft_type == "adaptformer":
            delta = self.blocks[layer_idx](patch)
            patch = patch + self.drop_path(self.adapter_scale * delta)

        elif self.peft_type in ("geoformer", "grip"):
            d = depth_feat.detach() if (self.detach_depth and depth_feat is not None) else depth_feat
            delta = self.blocks[layer_idx](patch, feat_h, feat_w, d)
            patch = patch + self.drop_path(self.adapter_scale * delta)

        return merge_patch_tokens(prefix, patch)
