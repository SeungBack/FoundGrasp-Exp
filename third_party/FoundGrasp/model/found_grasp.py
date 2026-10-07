import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoConfig, AutoBackbone, AutoModelForDepthEstimation, ViTModel
from transformers.modeling_outputs import BaseModelOutput, BackboneOutput

# from .grasp_decoder import *

from graspnetAPI.grasp import GraspGroup, Grasp
import numpy as np

from .prompts import PROMPT_GENERATOR_REGISTRY, ViPTPromptGenerator
from .grasp_head import GRASP_HEAD_REGISTRY
from .dpt import DECODER_REGISTRY
from .utils import pixel_nms, pad_to_multiple, crop_pred_dict, get_intermediate_layer_config, pad_to_fixed_size
from .blocks import GeometricDepthStem, get_depth_input_channels
from .orientation import decode_joint_orientation_classes
from .peft import (
    PeftManager,
    apply_lora,
    mark_only_lora_trainable,
)



def _build_rotation_matrices_gpu(approaches, grippers):
    """GPU version of _build_rotation_matrices_from_dirs. Inputs: (N,3) CUDA float tensors."""
    grippers = grippers.clone()
    grippers[grippers[:, 2] > 0] *= -1
    approach_norms = approaches.norm(dim=1, keepdim=True)
    gripper_norms = grippers.norm(dim=1, keepdim=True)
    valid = (approach_norms[:, 0] > 1e-6) & (gripper_norms[:, 0] > 1e-6)
    unit_approach = F.normalize(approaches, dim=1)
    gripper_ortho = grippers - (grippers * unit_approach).sum(dim=1, keepdim=True) * unit_approach
    go_norms = gripper_ortho.norm(dim=1, keepdim=True)
    # 1e-3 threshold: go_norms near 1e-6 causes float32 relative error ~10% in unit_gripper,
    # exceeding the 1e-2 tolerance checked by GraspNetEval's configuration_from_params.
    valid = valid & (go_norms[:, 0] > 1e-3)
    unit_gripper = F.normalize(gripper_ortho, dim=1)
    unit_cross = F.normalize(torch.linalg.cross(unit_approach, unit_gripper), dim=1)
    rotations = torch.stack([unit_approach, unit_gripper, unit_cross], dim=2)  # (N, 3, 3)
    return rotations, valid


def _voxel_downsample_gpu(points, voxel_size):
    """GPU voxel downsampling: keep one point per voxel (preserves original order within voxel)."""
    if points.shape[0] == 0:
        return points
    vcoords = torch.floor(points / voxel_size).long()
    vcoords -= vcoords.min(dim=0).values
    M = vcoords.max(dim=0).values + 1
    keys = vcoords[:, 0] * M[1] * M[2] + vcoords[:, 1] * M[2] + vcoords[:, 2]
    sort_idx = torch.argsort(keys, stable=True)
    sorted_keys = keys[sort_idx]
    is_first = torch.ones(len(sorted_keys), dtype=torch.bool, device=points.device)
    is_first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return points[sort_idx[is_first]]


def _empty_grasp_keep_mask_gpu(scene_points, centers, rotations, widths, depths,
                               min_points=10, height=0.02, depth_base=0.02, chunk_size=512):
    """GPU version of _empty_grasp_keep_mask."""
    num_grasps = centers.shape[0]
    if num_grasps == 0:
        return torch.zeros((0,), dtype=torch.bool, device=centers.device)
    if scene_points.shape[0] == 0:
        return torch.zeros((num_grasps,), dtype=torch.bool, device=centers.device)
    keep_mask = torch.zeros((num_grasps,), dtype=torch.bool, device=centers.device)
    for start in range(0, num_grasps, chunk_size):
        end = min(start + chunk_size, num_grasps)
        c = centers[start:end]
        R = rotations[start:end]
        w = widths[start:end, None]
        d = depths[start:end, None]
        targets = scene_points.unsqueeze(0) - c.unsqueeze(1)
        targets = torch.einsum('bpi,bij->bpj', targets, R)
        mask1 = (targets[:, :, 2] > -height / 2) & (targets[:, :, 2] < height / 2)
        mask2 = (targets[:, :, 0] > -depth_base) & (targets[:, :, 0] < d)
        mask4 = targets[:, :, 1] < -w / 2
        mask6 = targets[:, :, 1] > w / 2
        keep_mask[start:end] = (mask1 & mask2 & (~mask4) & (~mask6)).sum(dim=1) >= int(min_points)
    return keep_mask


def _expected_angular_error_from_kappa(kappa):
    """Convert AngMF concentration to expected angular error in radians.

    We reuse the closed-form expected angular error from Bae et al. (ICCV 2021)
    as a lightweight inference-time risk proxy. Lower expected error means the
    predicted direction is more concentrated.
    """
    exp_term = torch.exp(-math.pi * kappa)
    return (2.0 * kappa / (kappa.square() + 1.0)) + (math.pi * exp_term / (1.0 + exp_term))


# =============================================================================
# DA3 (Depth Anything 3) backbone wrapper
# DA3 uses a multi-view DinoVisionTransformer (S views, (B,S,C,H,W) input).
# This wrapper adapts it for single-view (B,C,H,W) FoundGrasp usage:
#   - embeddings() tokenizes a single image and pre-computes RoPE positions
#   - layers property returns plain-Python block adapters that pass RoPE for
#     blocks >= rope_start, skipping double nn.Module registration
# =============================================================================

class _DA3Config:
    """Minimal HF-compatible config stub so FoundGrasp can read .config.*"""
    def __init__(self, vit):
        self.hidden_size = vit.embed_dim
        self.patch_size = vit.patch_size
        self.num_register_tokens = getattr(vit, 'num_register_tokens', 0)
        self.apply_layernorm = False  # DA3 norm applied explicitly via backbone.apply_norm()


class _DA3BlockAdapter:
    """Plain-Python callable wrapping one DA3 Block for (B, N, C) single-view use.
    Not an nn.Module so the block is NOT registered twice (already lives in _vit.blocks)."""
    __slots__ = ('_blk', '_idx', '_wrapper')

    def __init__(self, blk, idx, wrapper):
        self._blk = blk
        self._idx = idx
        self._wrapper = wrapper

    def __call__(self, x):
        vit = self._wrapper._vit
        rope_start = getattr(vit, 'rope_start', 999)
        if self._idx < rope_start or vit.rope is None:
            return self._blk(x)
        else:
            return self._blk(x, pos=self._wrapper._rope_pos)

    def parameters(self):
        return self._blk.parameters()

    def named_parameters(self, prefix='', recurse=True):
        return self._blk.named_parameters(prefix=prefix, recurse=recurse)

    def __repr__(self):
        return f"_DA3BlockAdapter(idx={self._idx}, block={type(self._blk).__name__})"


class DA3BackboneWrapper(nn.Module):
    """Wraps DA3's DinoVisionTransformer so FoundGrasp can treat it like an HF ViT backbone.

    Interface exposed:
      .config           — _DA3Config with hidden_size, patch_size, num_register_tokens
      .embeddings(x)    — (B,C,H,W) -> (B,N,C) tokens, also caches RoPE positions
      .layers           — plain-Python list of _DA3BlockAdapter (no double registration)
      .apply_norm(x)    — apply the trained DA3 LayerNorm to patch tokens
    """

    def __init__(self, vit):
        super().__init__()
        self._vit = vit                   # All DA3 weights registered here
        self.config = _DA3Config(vit)
        # Plain Python list: no double nn.Module registration of blocks
        self._block_adapters = [
            _DA3BlockAdapter(blk, i, self) for i, blk in enumerate(vit.blocks)
        ]
        self._rope_pos = None             # (B, N, rope_dim) — set each forward pass

    @property
    def layers(self):
        return self._block_adapters

    def embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Tokenize a single image: patch_embed + cls_token + pos_embed.
        Also caches RoPE positions for blocks that need them."""
        B, C, H, W = pixel_values.shape
        vit = self._vit
        # Pre-compute RoPE positions (non-None only when vit.rope is set)
        if vit.rope is not None:
            pos, _ = vit._prepare_rope(B, 1, H, W, pixel_values.device)
            self._rope_pos = pos[:, 0, :, :]  # (B, N_tokens, rope_dim)
        else:
            self._rope_pos = None
        # prepare_tokens_with_masks expects (B, S, C, H, W)
        x_mv = pixel_values.unsqueeze(1)               # (B, 1, C, H, W)
        x = vit.prepare_tokens_with_masks(x_mv)        # (B, 1, N, C)
        return x[:, 0, :, :]                           # (B, N, C)

    @property
    def norm(self):
        """Expose DA3's pretrained LayerNorm via .norm, same interface as HF backbones.
        @property avoids double nn.Module registration (no __setattr__ call)."""
        return self._vit.norm


class SamLayerNorm(nn.LayerNorm):
    """LayerNorm that supports channels_first format (B, C, H, W)."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__(normalized_shape, eps=eps)
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_first":
            x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            x = super().forward(x)
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        else:
            x = super().forward(x)
        return x


class SamVisionNeck(nn.Module):
    """SAM-style vision neck for feature projection and normalization."""
    def __init__(self, hidden_size, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = SamLayerNorm(output_channels, data_format="channels_first")
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = SamLayerNorm(output_channels, data_format="channels_first")

    def forward(self, hidden_states):
        # Input: (B, H, W, C) -> permute to (B, C, H, W)
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states



class FoundGrasp(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cfg = config
        self.input_type = self.cfg.input_type

        # 1. Backbone Setup
        if "da3" in self.cfg.backbone.name.lower():
            print("Using Depth-Anything-3 (DA3) Backbone")
            from depth_anything_3.api import DepthAnything3
            _da3 = DepthAnything3.from_pretrained(self.cfg.backbone.name)
            self.backbone = DA3BackboneWrapper(_da3.model.backbone.pretrained)
        elif "depth-anything" in self.cfg.backbone.name:
            print("Using Depth-Anything Backbone")
            self.backbone = AutoModelForDepthEstimation.from_pretrained(self.cfg.backbone.name).backbone
        elif "sam" in self.cfg.backbone.name.lower(): # SAM 감지
            # SAM은 전체 모델에서 vision_encoder만 떼어냅니다.
            full_model = AutoModel.from_pretrained(self.cfg.backbone.name)
            self.backbone = full_model.vision_encoder
        elif "siglip" in self.cfg.backbone.name.lower():
            # SigLIP: full model (text+vision) → extract vision_model only.
            # vision_model config has model_type='siglip_vision_model', no CLS token.
            print("Using SigLIP Vision Backbone")
            _siglip_cfg = AutoConfig.from_pretrained(self.cfg.backbone.name)
            if getattr(_siglip_cfg, 'model_type', '') == 'siglip_vision_model':
                self.backbone = AutoModel.from_pretrained(self.cfg.backbone.name)
            else:
                self.backbone = AutoModel.from_pretrained(self.cfg.backbone.name).vision_model
        else:
            _cfg = AutoConfig.from_pretrained(self.cfg.backbone.name)
            if getattr(_cfg, 'model_type', '') == 'vit':
                print("Using Standard ViT Backbone")
                self.backbone = ViTModel.from_pretrained(self.cfg.backbone.name, add_pooling_layer=False)
            else:
                self.backbone = AutoBackbone.from_pretrained(self.cfg.backbone.name)

        self.freeze_backbone = self.cfg.backbone.freeze
        # Validate that backbone has accessible layers at init time
        if not (hasattr(self.backbone, "encoder") and (hasattr(self.backbone.encoder, "layer") or hasattr(self.backbone.encoder, "layers"))) \
                and not hasattr(self.backbone, "layers") \
                and not hasattr(self.backbone, "layer"):
            raise AttributeError("Cannot find encoder layers in backbone.")
        self.is_da3 = 'da3' in self.cfg.backbone.name.lower()
        self.is_dinov3 = 'dinov3' in self.cfg.backbone.name.lower()
        self.is_sam = 'sam' in self.cfg.backbone.name.lower()
        self.is_siglip = 'siglip' in self.cfg.backbone.name.lower()
        self.is_plain_vit = getattr(self.backbone.config, 'model_type', '') == 'vit'

        self.patch_size = getattr(self.backbone.config, 'patch_size', 14)
        self.embed_dim = self.backbone.config.hidden_size

        # SAM and SigLIP have no cls_token → num_prefix = 0; all other models have at least a cls_token
        if self.is_sam or self.is_siglip:
            self.num_prefix = 0
        else:
            self.num_prefix = 1 + getattr(self.backbone.config, "num_register_tokens", 0)
        self.apply_layernorm = getattr(self.backbone.config, "apply_layernorm", False)
        # Different backbones expose their final norm under different attribute names.
        # Cache a reference so forward() doesn't need to branch.
        if self.apply_layernorm or self.is_da3:
            _norm = getattr(self.backbone, "norm", None) or getattr(self.backbone, "layernorm", None)
            if _norm is None:
                self.apply_layernorm = False
            self._backbone_norm = _norm
        else:
            self._backbone_norm = None
        print(self.num_prefix, "prefix tokens will be skipped in feature extraction.")

        # 2. Unified Prompt Generator (Compatible with all modes)
        prompt_mode = self.cfg.prompt.mode.lower()
        print(prompt_mode, PROMPT_GENERATOR_REGISTRY.keys())
        if prompt_mode in PROMPT_GENERATOR_REGISTRY:
            generator_cls = PROMPT_GENERATOR_REGISTRY[prompt_mode]
            self.prompt_generator = generator_cls(
                embed_dim=self.embed_dim,
                patch_size=self.patch_size,
                num_layers=len(self.layers), 
                num_prompts=self.cfg.prompt.num_prompts,
                use_deep=self.cfg.prompt.use_deep, # Config에 추가 필요
                input_type=self.cfg.input_type # Not used yed
            )
            self.num_prompts = self.cfg.prompt.num_prompts
        else:
            print(f"Prompt mode '{prompt_mode}' not found in PROMPT_GENERATOR_REGISTRY. No prompt generator will be used.")
            self.prompt_generator = None
            self.num_prompts = 0

     
        # 3. Decoder
        # DINOv2 Base usually has [768, 768, 768, 768] for out_indices
        dec_cfg = getattr(self.cfg, 'decoder', None)
        decoder_preset = getattr(dec_cfg, 'preset', None) if dec_cfg else None
        out_indices, out_channels, feature_dim = get_intermediate_layer_config(
            self.cfg.backbone.name,
            decoder_preset=decoder_preset,
        )
        if dec_cfg and getattr(dec_cfg, 'out_indices', None):
            out_indices = list(dec_cfg.out_indices)
        if dec_cfg and getattr(dec_cfg, 'feature_dim', None):
            feature_dim = dec_cfg.feature_dim
        if dec_cfg and getattr(dec_cfg, 'out_channels', None):
            out_channels = list(dec_cfg.out_channels)
        if len(out_indices) != 4:
            raise ValueError(f"DPT decoder expects 4 out_indices, got {out_indices}")
        self.out_indices = out_indices

        #  Depth Encoder
        depth_in = get_depth_input_channels(self.cfg.input_type)
        self.depth_encoder = GeometricDepthStem(depth_in, feature_dim, self.cfg.depth_backbone)
        decoder_class = DECODER_REGISTRY[self.cfg.decoder.name]
        self.decoder = decoder_class(
            in_channel=self.embed_dim,
            out_channels=out_channels, 
            feature_dim=feature_dim,
            patch_size=self.patch_size,
            fusion_type=self.cfg.decoder.fusion_type,
        )
        self.layernorm = None  # backbone.norm is used directly in forward when apply_layernorm=True

        # 4. Head
        head_class = GRASP_HEAD_REGISTRY[self.cfg.grasp_head.name]
        branch_type = self.cfg.grasp_head.get('branch_type', 'single')
        if branch_type != 'single':
            raise ValueError(
                f"ParallelGraspHead now supports only branch_type='single', got '{branch_type}'"
            )
        head_kwargs = {
            key: value
            for key, value in self.cfg.grasp_head.items()
            if key not in {'name', 'branch_type'}
        }
        self.head = head_class(
            feature_dim=feature_dim,
            **head_kwargs,
        )

        
        ## 5. PEFT Setup (LoRA, Adapters, etc.)
        peft_cfg = getattr(self.cfg, "peft", None)
        self.peft = None
        self.peft_type = "none"

        if peft_cfg is not None:
            self.peft_type = getattr(peft_cfg, "type", "none").lower()
        print(self.peft_type + " PEFT method will be applied." if self.peft_type != "none" else "No PEFT method will be applied.")

        # IMPORTANT:
        # - LoRA patches backbone internals (must be applied before/after freezing carefully)
        # - Residual adapters live outside backbone (PeftManager)
        if self.peft_type == "lora":
            # Patch backbone Linear modules -> LoRA modules (weights remain frozen)
            r = getattr(peft_cfg, "r", 8)
            alpha = getattr(peft_cfg, "alpha", 16)
            lora_dropout = getattr(peft_cfg, "dropout", 0.0)
            regex = getattr(peft_cfg, "target_regex", r"(qkv|query|key|value|proj|out_proj)")
            apply_lora(self.backbone, r=r, alpha=alpha, dropout=lora_dropout, target_name_regex=regex, verbose=True)

        elif self.peft_type in [
            "adaptformer", "geoformer", "grip",
            "vit_adapter", "geo_vit_adapter",
        ]:
            bottleneck = getattr(peft_cfg, "bottleneck", 64)
            adrop = getattr(peft_cfg, "dropout", 0.0)
            ascale = getattr(peft_cfg, "scale", 1.0)
            adapter_drop_path = getattr(peft_cfg, "drop_path", 0.2)
            adapter_use_ln = getattr(peft_cfg, "use_ln", None)
            layers_to_adapt = getattr(peft_cfg, "layers", "all")
            detach_depth = getattr(peft_cfg, "detach_depth", True)

            # GRIP-Adapter specific
            use_value_mlp = getattr(peft_cfg, "use_value_mlp", False)
            separate_anchor = getattr(peft_cfg, "separate_anchor", False)
            stop_gradient = getattr(peft_cfg, "stop_gradient", True)
            anchor_bottleneck = getattr(peft_cfg, "anchor_bottleneck", None)
            gate_act = getattr(peft_cfg, "gate_act", "tanh")
            gate_init_bias = getattr(peft_cfg, "gate_init_bias", 0.0)
            gate_type = getattr(peft_cfg, "gate_type", "channel")
            symmetric_gate = getattr(peft_cfg, "symmetric_gate", False)

            # ViT-Adapter / Geo-ViT-Adapter specific
            num_interactions = getattr(peft_cfg, "num_interactions", 4)
            num_heads = getattr(peft_cfg, "num_heads", 6)
            n_points = getattr(peft_cfg, "n_points", 4)
            inplanes = getattr(peft_cfg, "inplanes", 64)
            with_cffn = getattr(peft_cfg, "with_cffn", True)
            cffn_ratio = getattr(peft_cfg, "cffn_ratio", 0.25)
            init_scale = getattr(peft_cfg, "init_scale", 0.0)
            deform_ratio = getattr(peft_cfg, "deform_ratio", 1.0)
            add_level_embed = getattr(peft_cfg, "add_level_embed", True)
            spm_pool_factor = getattr(peft_cfg, "spm_pool_factor", 1)
            use_extra_extractor = getattr(peft_cfg, "use_extra_extractor", True)

            self.peft = PeftManager(
                peft_type=self.peft_type,
                num_layers=len(self.layers),
                embed_dim=self.embed_dim,
                depth_dim=feature_dim,
                layers_to_adapt=layers_to_adapt,
                decoder_out_indices=self.out_indices,
                adapter_bottleneck=bottleneck,
                adapter_dropout=adrop,
                adapter_scale=ascale,
                adapter_drop_path=adapter_drop_path,
                adapter_use_ln=adapter_use_ln,
                detach_depth=detach_depth,
                use_value_mlp=use_value_mlp,
                separate_anchor=separate_anchor,
                stop_gradient=stop_gradient,
                anchor_bottleneck=anchor_bottleneck,
                gate_act=gate_act,
                gate_init_bias=gate_init_bias,
                gate_type=gate_type,
                symmetric_gate=symmetric_gate,
                num_interactions=num_interactions,
                num_heads=num_heads,
                n_points=n_points,
                inplanes=inplanes,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                init_scale=init_scale,
                deform_ratio=deform_ratio,
                add_level_embed=add_level_embed,
                spm_pool_factor=spm_pool_factor,
                use_extra_extractor=use_extra_extractor,
            )

        elif self.peft_type != "none":
            raise ValueError(
                f"Unknown or unsupported PEFT method: {self.peft_type}. "
                "Use one of: lora, adaptformer, geoformer, grip, "
                "vit_adapter, geo_vit_adapter."
            )


        # 5. Freeze backbone if specified
        if self.freeze_backbone:
            print("Freezing backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # If LoRA, re-enable LoRA params AFTER freezing backbone
        if self.peft_type == "lora":
            print("Re-enabling LoRA parameters for training...")
            mark_only_lora_trainable(self.backbone)

        # Install parallel-to-MLP placement for all bottleneck PEFT variants.
        # Only DA3-style blocks are actually patched; other backbones fall back
        # to the post-block forward path inside PeftManager.
        if self.peft is not None:
            self.peft.install(self.layers)

    @property
    def layers(self):
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return self.backbone.encoder.layer
        elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
            return self.backbone.encoder.layers
        elif hasattr(self.backbone, "layers"):
            return self.backbone.layers
        elif hasattr(self.backbone, "layer"):
            return self.backbone.layer
        raise AttributeError("Cannot find encoder layers in backbone.")

    def load_state_dict(self, state_dict, strict=True):
        # Remap legacy checkpoints where backbone layers were registered as "layers.*"
        # instead of "backbone.encoder.layer.*" or "backbone.layers.*"
        if any(k.startswith('layers.') for k in state_dict):
            own_keys = set(super().state_dict().keys())
            if any(k.startswith('backbone.encoder.layer.') for k in own_keys):
                new_prefix = 'backbone.encoder.layer.'
            elif any(k.startswith('backbone.layers.') for k in own_keys):
                new_prefix = 'backbone.layers.'
            else:
                new_prefix = None

            if new_prefix is not None:
                remapped = {}
                for k, v in state_dict.items():
                    if k.startswith('layers.'):
                        remapped[new_prefix + k[len('layers.'):]] = v
                    else:
                        remapped[k] = v
                state_dict = remapped
                print(f"[FoundGrasp] Remapped legacy 'layers.*' → '{new_prefix}*'")

        return super().load_state_dict(state_dict, strict=strict)

    def _make_depth_values(self, data):
        if 'xyz' not in self.input_type and 'd' not in self.input_type and 'c' not in self.input_type:
            return None
        depth_valuess = []
        if 'd' in self.input_type:
            depth_valuess.append(data['depth'])
        if 'xyz' in self.input_type:
            depth_valuess.append(data['xyz'])
        if 'c' in self.input_type:
            depth_valuess.append(data['depth_mask'])  # (B, 1, H, W)
        depth_values = torch.cat(depth_valuess, dim=1)
        return depth_values

    def forward(self, data):
        # 1. Preprocessing & Padding
        depth_values = self._make_depth_values(data)
        raw_depth = data['depth']
        depth_mask = data.get('depth_mask', (raw_depth > 0).float())
        pixel_values, pad_hw = pad_to_multiple(data['pixel_values'], self.patch_size)
        depth_values, _ = pad_to_multiple(depth_values, self.patch_size)
        raw_depth, _ = pad_to_multiple(raw_depth, self.patch_size)
        depth_mask, _ = pad_to_multiple(depth_mask, self.patch_size)

        B, C, H, W = pixel_values.shape

        # 2. Get RGB Embeddings
        if self.is_sam:
            # SAM: patch_embed → (B, H, W, C); interpolate pos_embed to actual grid size
            x = self.backbone.patch_embed(pixel_values)
            pos = self.backbone.pos_embed  # (1, src_h, src_w, C)
            tgt_h, tgt_w = x.shape[1], x.shape[2]
            if (tgt_h, tgt_w) != (pos.shape[1], pos.shape[2]):
                pos = F.interpolate(
                    pos.permute(0, 3, 1, 2),
                    size=(tgt_h, tgt_w),
                    mode='bicubic',
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            x = x + pos
        elif self.is_dinov3:
            x = self.backbone.embeddings(pixel_values)
            position_embeddings = self.backbone.rope_embeddings(pixel_values)
        elif self.is_plain_vit or self.is_siglip:
            # Both use fixed absolute position embeddings needing interpolation for non-native sizes.
            # SiglipVisionEmbeddings.forward supports interpolate_pos_encoding since transformers 4.43.
            x = self.backbone.embeddings(pixel_values, interpolate_pos_encoding=True)
        else:
            x = self.backbone.embeddings(pixel_values)

        # Depth
        depth_feat = self.depth_encoder(depth_values)
        depth_feat_gated = depth_feat

        # 3. Prompts & Injections
        shallow_p, deep_p, injection, depth_state = None, None, None, None
        if self.prompt_generator is not None:
            shallow_p, deep_p, injection, depth_state = self.prompt_generator(
                depth_values, x[:, self.num_prefix:]
            )
        if injection is not None:
            updated = x[:, self.num_prefix:, :] + injection
            x = torch.cat([x[:, :self.num_prefix, :], updated], dim=1)
        if shallow_p is not None:
            x = torch.cat((x[:, :self.num_prefix, :], shallow_p, x[:, self.num_prefix:, :]), dim=1)
        
        # Pre-compute feat_h/feat_w for PEFT.
        # SAM tokens are (B, H, W, C); other backbones use (B, N, C) sequence format.
        feat_h = feat_w = None
        if self.peft is not None:
            if self.is_sam:
                # Derive spatial dims from current x shape.
                # vit_adapter / geo_vit_adapter require (B, N, C) and are not supported
                # for SAM. Bottleneck adapters (adaptformer / geoformer / grip) run
                # parallel-to-MLP inside the patched SamVisionLayer when install()
                # succeeded; otherwise they fall back to the post-block reshape path.
                feat_h, feat_w = x.shape[1], x.shape[2]
                if self.peft._installed:
                    self.peft._ctx_depth_feat = depth_feat_gated
                    self.peft._ctx_num_prefix = 0
            else:
                feat_h, feat_w = H // self.patch_size, W // self.patch_size
                # Parallel-to-MLP path: monkey-patched block forward reads context from PeftManager.
                if self.peft._installed:
                    self.peft._ctx_feat_h = feat_h
                    self.peft._ctx_feat_w = feat_w
                    self.peft._ctx_depth_feat = depth_feat_gated
                    self.peft._ctx_num_prefix = self.num_prefix
                # vit_adapter / geo_vit_adapter: run SPM + first injection before the layer loop
                if self.peft.peft_type in ("vit_adapter", "geo_vit_adapter"):
                    x = self.peft.prepare(
                        pixel_values=pixel_values,
                        x=x,
                        feat_h=feat_h,
                        feat_w=feat_w,
                        depth_feat=depth_feat_gated,
                    )

        # 4. Forward Loop (Deep Prompting)
        all_hidden_states = []
        for i, layer in enumerate(self.layers):
            # A. VPT Deep (Replacement)
            if i > 0 and deep_p is not None and not isinstance(deep_p, dict):
                curr_deep = deep_p[:, i-1] # (B, N, D)
                x = torch.cat((
                    x[:, :self.num_prefix, :],         
                    curr_deep,             
                    x[:, (self.num_prefix+self.num_prompts):, :] 
                ), dim=1)
            # B. ViPT Deep (Layer-wise Injection)
            if i > 0 and depth_state is not None:
                # 여기서도 self.num_prefix 이후의 토큰만 추출해서 넘겨야 함 (현재 구현 맞음)
                # ViPT는 토큰 개수를 늘리지 않으므로 prompt_len(VPT용)은 신경 안 써도 됨 (VPT와 섞어 쓰지 않는다면)
                start_idx = self.num_prefix 
                curr_patch_tokens = x[:, start_idx:, :]
                
                layer_injection, depth_state = self.prompt_generator.forward_layer(
                    i, curr_patch_tokens, depth_state
                )
                if layer_injection is not None:
                    # Injection 더하기 (Residual 방식)
                    modified_spatial = curr_patch_tokens + layer_injection
                    x = torch.cat([x[:, :start_idx, :], modified_spatial], dim=1)

            # Layer Forward
            if self.is_dinov3:
                layer_out = layer(x, position_embeddings=position_embeddings)
            elif self.is_siglip:
                layer_out = layer(x, None)
            else:
                layer_out = layer(x)
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            if self.peft is not None:
                if self.is_sam and self.peft.peft_type not in ("vit_adapter", "geo_vit_adapter"):
                    if not self.peft._installed:
                        # Fallback (SAM block not patched): reshape (B,H,W,C) → sequence,
                        # apply bottleneck adapter post-block, then restore spatial layout.
                        B_s, H_s, W_s, C_s = x.shape
                        x_seq = x.reshape(B_s, H_s * W_s, C_s)
                        x_seq = self.peft(i, x_seq, H_s, W_s, depth_feat=depth_feat_gated)
                        x = x_seq.reshape(B_s, H_s, W_s, C_s)
                    # else: adapter applied parallel-to-MLP inside the patched block forward
                elif not self.is_sam:
                    x = self.peft(i, x, feat_h, feat_w, depth_feat=depth_feat_gated)

            all_hidden_states.append(x)


        # 6. Feature Extraction
        # Need to handle index shift if prompts were added
        feats = []
        if self.is_sam: # already h, w and normalized
            for i, idx in enumerate(self.out_indices):
                feats.append(all_hidden_states[idx].permute(0,3,1,2).contiguous())
        else:
            feat_h, feat_w = H // self.patch_size, W // self.patch_size
            expected_spatial_tokens = feat_h * feat_w
            actual_seq_len = all_hidden_states[self.out_indices[0]].shape[1]
            dynamic_start_idx = actual_seq_len - expected_spatial_tokens
            if dynamic_start_idx < 0:
                raise RuntimeError(f"Backbone output too small: {actual_seq_len} tokens, expected at least {expected_spatial_tokens}.")
            # print(f"DEBUG: feat_h={feat_h}, feat_w={feat_w}, seq_len={actual_seq_len}, start_idx={dynamic_start_idx}")
            for i, idx in enumerate(self.out_indices):
                # start_idx 대신 dynamic_start_idx 사용
                state = all_hidden_states[idx][:, dynamic_start_idx:]
                # Apply backbone's pretrained norm (DINOv3, DA3, etc.)
                if self._backbone_norm is not None:
                    state = self._backbone_norm(state)
                feats.append(state.reshape(B, feat_h, feat_w, -1).permute(0,3,1,2).contiguous())
            
        # 7. Decoder & Head
        # NOTE: decoder always receives raw depth_feat (not gated) — gating only targets
        # the GeoAdaptFormer path. apply_to_depth_feat=true only gates PEFT's depth input,
        # not the decoder/head. This is intentional: double-gating the decoder would
        # over-suppress depth information in the reconstruction pathway.
        shared_feat = self.decoder(feats, depth_feat)
        if getattr(self.head, 'needs_xyz', False):
            preds = self.head(shared_feat, depth_feat, H, W, xyz_raw=depth_values[:, :3])
        else:
            preds = self.head(shared_feat, depth_feat_gated, H, W)
        return crop_pred_dict(preds, pad_hw)

    def _decode_joint_orientation_candidates(self, predictions, vi, ui, target_hw, chunk_size=2048):
        orientation_feat = predictions['_orientation_feat']
        if orientation_feat.dim() == 4:
            orientation_feat = orientation_feat[0]
        if orientation_feat.dim() != 3:
            raise ValueError(
                f"_orientation_feat must have shape (C,H,W), got {tuple(orientation_feat.shape)}"
            )

        _, feat_h, feat_w = orientation_feat.shape
        full_hw = predictions.get('_orientation_full_hw', None)
        if full_hw is None:
            full_h, full_w = target_hw
        else:
            if full_hw.dim() > 1:
                full_hw = full_hw[0]
            full_h, full_w = int(full_hw[0].item()), int(full_hw[1].item())
        full_h = max(int(full_h), 1)
        full_w = max(int(full_w), 1)

        full_h_t = vi.new_tensor(float(full_h), dtype=torch.float32).clamp_min(1.0)
        full_w_t = ui.new_tensor(float(full_w), dtype=torch.float32).clamp_min(1.0)
        feat_y = (vi.float() + 0.5) * float(feat_h) / full_h_t - 0.5
        feat_x = (ui.float() + 0.5) * float(feat_w) / full_w_t - 0.5
        y0 = torch.floor(feat_y).long()
        x0 = torch.floor(feat_x).long()
        y1 = y0 + 1
        x1 = x0 + 1
        wy = (feat_y - y0.float()).clamp(0.0, 1.0).unsqueeze(1)
        wx = (feat_x - x0.float()).clamp(0.0, 1.0).unsqueeze(1)
        y0 = y0.clamp(0, feat_h - 1)
        y1 = y1.clamp(0, feat_h - 1)
        x0 = x0.clamp(0, feat_w - 1)
        x1 = x1.clamp(0, feat_w - 1)

        feat_grid = orientation_feat.permute(1, 2, 0)
        f00 = feat_grid[y0, x0]
        f01 = feat_grid[y0, x1]
        f10 = feat_grid[y1, x0]
        f11 = feat_grid[y1, x1]
        feat_samples = (
            f00 * (1.0 - wy) * (1.0 - wx)
            + f01 * (1.0 - wy) * wx
            + f10 * wy * (1.0 - wx)
            + f11 * wy * wx
        ).float()

        classifier_weight = predictions['_orientation_weight']
        classifier_bias = predictions['_orientation_bias']
        if classifier_weight.dim() == 3:
            classifier_weight = classifier_weight[0]
        if classifier_bias.dim() == 2:
            classifier_bias = classifier_bias[0]
        classifier_weight = classifier_weight.float()
        classifier_bias = classifier_bias.float()

        pred_classes = []
        for start in range(0, feat_samples.shape[0], chunk_size):
            end = min(start + chunk_size, feat_samples.shape[0])
            logits = F.linear(feat_samples[start:end], classifier_weight, classifier_bias)
            pred_classes.append(torch.argmax(logits, dim=1))
        pred_classes = torch.cat(pred_classes, dim=0)

        meta = predictions.get('_orientation_meta', None)
        if meta is None:
            num_views, num_angles = 300, 12
        else:
            if meta.dim() > 1:
                meta = meta[0]
            num_views, num_angles = int(meta[0].item()), int(meta[1].item())

        approaches, grippers = decode_joint_orientation_classes(
            pred_classes,
            num_views=num_views,
            num_angles=num_angles,
        )
        return (
            approaches.to(device=feat_samples.device, dtype=feat_samples.dtype),
            grippers.to(device=feat_samples.device, dtype=feat_samples.dtype),
        )
    
    def to_grasp_group(self, predictions, xyz, graspness_threshold=0.01,
                    objectness_threshold=0.5,
                    graspness_exp=0.1, grasp_score_exp=2.0,
                    total_score_threshold=None,
                    approach_kappa_risk_weight=0.5,
                    gripper_kappa_risk_weight=1.0,
                    width_multiplier=1.2,
                    z_max=1.5,
                    empty_grasp_min_points=0,
                    segm_erosion_k=0,
                    max_grasps=2048,
                    voxel_size=0.004):
        """
        Convert pixel-wise model predictions to GraspGroup format.

        Args:
            predictions: dict containing model outputs
                - 'objectness': (1, H, W) objectness logits
                - 'graspness': (1, H, W) graspness scores in [0, 1]
                - 'grasp_score': (1, H, W) quality score in [0, 1]
                - 'approach_dir': (3, H, W) normalized approach direction
                - 'gripper_dir': (3, H, W) normalized gripper direction
                - 'grasp_width': (1, H, W) normalized grasp width
                - 'grasp_depth': (1, H, W) normalized grasp depth
            xyz: (3, H, W) 3D point cloud coordinates
            graspness_threshold: minimum graspness score to consider
            objectness_threshold: minimum objectness probability to consider
            graspness_exp: exponent for graspness in score formula (default: 0.1)
            grasp_score_exp: exponent for grasp_score in score formula (default: 2.0)
            total_score_threshold: NMS threshold applied on final ranking score
            approach_kappa_risk_weight: risk weight for approach expected angular error
            gripper_kappa_risk_weight: risk weight for gripper expected angular error
            z_max: maximum scene depth allowed for grasp centers (None disables)
            empty_grasp_min_points: minimum number of voxelized scene points
                inside the gripper closing region; 0 disables the proxy
            voxel_size: voxel NMS grid size in meters
        Returns:
            GraspGroup: collection of 3D grasps
            points_3d: (H, W, 3) 3D point cloud
        """


        # Extract predictions (no batch dimension) - keep as tensors
        segm = predictions['objectness'][0]   # (H, W)
        segm = torch.sigmoid(segm)  # Convert logits to probabilities
        graspness = predictions['graspness'][0] # (H, W)
        grasp_score = predictions['grasp_score'][0]  # (H, W)
        has_dense_dirs = 'approach_dir' in predictions and 'gripper_dir' in predictions
        has_joint_orientation = (
            '_orientation_feat' in predictions
            and '_orientation_weight' in predictions
            and '_orientation_bias' in predictions
        )
        if has_dense_dirs:
            approach_dir = predictions['approach_dir']  # (3, H, W)
            gripper_dir = predictions['gripper_dir']  # (3, H, W)
        elif not has_joint_orientation:
            raise ValueError(
                "Predictions must contain either dense approach_dir/gripper_dir "
                "or sparse joint orientation classifier outputs."
            )
        else:
            approach_dir, gripper_dir = None, None
        grasp_width = predictions['grasp_width'][0] / 10.0  # (H, W) undo scaling
        grasp_depth = predictions['grasp_depth'][0] / 25.0  # (H, W) undo scaling

        # segm gates candidates via hard mask (objectness_threshold + segm_erosion_k);
        # not included in ranking score to avoid double-counting boundary suppression.
        total_score = (graspness ** graspness_exp) * (grasp_score ** grasp_score_exp)

        if 'approach_kappa' in predictions and approach_kappa_risk_weight > 0:
            approach_kappa = predictions['approach_kappa'][0].squeeze(0)  # (H, W), kappa >= 1
            approach_risk = _expected_angular_error_from_kappa(approach_kappa)
            total_score = total_score * torch.exp(-approach_kappa_risk_weight * approach_risk)
        if 'gripper_kappa' in predictions and gripper_kappa_risk_weight > 0:
            gripper_kappa = predictions['gripper_kappa'][0].squeeze(0)  # (H, W), kappa >= 1
            gripper_risk = _expected_angular_error_from_kappa(gripper_kappa)
            total_score = total_score * torch.exp(-gripper_kappa_risk_weight * gripper_risk)
        if total_score_threshold is None:
            total_score_threshold = graspness_threshold

        # Create initial valid mask
        valid_depth_mask = torch.isfinite(xyz[2]) & (xyz[2] > 0.1)
        if z_max is not None:
            valid_depth_mask = valid_depth_mask & (xyz[2] < z_max)

        # Morphological erosion on objectness mask: suppress grasps near object boundaries
        # where segmentation predictions are unreliable (erosion = dilation of background complement).
        segm_fg = (segm > objectness_threshold).float()
        if segm_erosion_k > 0:
            bg_dilated = torch.nn.functional.max_pool2d(
                (1 - segm_fg).unsqueeze(0).unsqueeze(0),
                kernel_size=segm_erosion_k * 2 + 1, stride=1, padding=segm_erosion_k
            ).squeeze()
            segm_fg = (bg_dilated < 0.5).float()

        valid_mask = (graspness > graspness_threshold) & \
                            (segm_fg > 0.5) & \
                            valid_depth_mask

        # NMS
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # (N, 2)
        if len(valid_indices) > 0:
            v_idx, u_idx = valid_indices[:, 0], valid_indices[:, 1]
            pts = xyz[:, v_idx, u_idx].T.float()             # (N, 3) on GPU
            scores_gpu = total_score[v_idx, u_idx]           # (N,) on GPU
            voxel_size = float(voxel_size)
            if voxel_size <= 0:
                raise ValueError(f"voxel_size must be positive, got {voxel_size}")
            vcoords = torch.floor(pts / voxel_size).long()
            vcoords -= vcoords.min(dim=0).values
            M = vcoords.max(dim=0).values + 1
            keys = vcoords[:, 0] * M[1] * M[2] + vcoords[:, 1] * M[2] + vcoords[:, 2]
            # lexsort equivalent: stable sort by key; within same key, score-desc order is preserved
            score_order = torch.argsort(scores_gpu, descending=True)
            key_order = torch.argsort(keys[score_order], stable=True)
            final_order = score_order[key_order]
            sorted_keys = keys[final_order]
            is_first = torch.ones(len(sorted_keys), dtype=torch.bool, device=sorted_keys.device)
            is_first[1:] = sorted_keys[1:] != sorted_keys[:-1]
            keep = final_order[is_first]
            if len(keep) > max_grasps:
                _, top_idx = torch.topk(scores_gpu[keep], min(max_grasps, len(keep)))
                keep = keep[top_idx]
            valid_indices = valid_indices[keep]

        if len(valid_indices) == 0:
            return GraspGroup()

        # Extract all needed tensors on GPU (single batch, no intermediate CPU transfers)
        vi, ui = valid_indices[:, 0], valid_indices[:, 1]
        centers_t  = xyz[:, vi, ui].T.float()                         # (N, 3)
        if has_dense_dirs:
            approaches_t = approach_dir[:, vi, ui].T.float()          # (N, 3)
            grippers_t   = gripper_dir[:, vi, ui].T.float()           # (N, 3)
        else:
            approaches_t, grippers_t = self._decode_joint_orientation_candidates(
                predictions,
                vi,
                ui,
                target_hw=xyz.shape[-2:],
            )
        widths_t  = grasp_width[vi, ui].float() * width_multiplier    # (N,)
        depths_t  = grasp_depth[vi, ui].float()                       # (N,)
        scores_t  = total_score[vi, ui].float()                       # (N,)

        rotations_t, valid_rot = _build_rotation_matrices_gpu(approaches_t, grippers_t)
        final_mask = valid_rot  # bool tensor on GPU

        if empty_grasp_min_points > 0 and final_mask.any():
            scene_pts = xyz[:, valid_depth_mask].T.float()
            scene_pts = _voxel_downsample_gpu(scene_pts, float(0.008))
            keep_by_empty = _empty_grasp_keep_mask_gpu(
                scene_pts,
                centers_t[final_mask], rotations_t[final_mask],
                widths_t[final_mask], depths_t[final_mask],
                min_points=int(empty_grasp_min_points),
                depth_base=float(0.02),
            )
            active = torch.where(final_mask)[0]
            final_mask = final_mask.clone()
            final_mask[active[~keep_by_empty]] = False

        # Single CPU transfer at the end
        gg = GraspGroup()
        if final_mask.any():
            idx = torch.where(final_mask)[0]
            grasp_group_array = np.zeros((len(idx), 17), dtype=np.float32)
            grasp_group_array[:, 0] = scores_t[idx].detach().cpu().numpy()
            grasp_group_array[:, 1] = widths_t[idx].detach().cpu().numpy()
            grasp_group_array[:, 2] = 0.02
            grasp_group_array[:, 3] = depths_t[idx].detach().cpu().numpy()
            grasp_group_array[:, 4:13] = rotations_t[idx].reshape(-1, 9).detach().cpu().numpy()
            grasp_group_array[:, 13:16] = centers_t[idx].detach().cpu().numpy()
            gg.grasp_group_array = grasp_group_array

        return gg
