import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul

from .utils import token2feature, feature2token  # 기존 유틸 사용 가정
from .blocks import GeometricDepthStem

class BasePromptGenerator(nn.Module):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward(self, depth_values, rgb_tokens=None):
        """
        Returns:
            shallow_prompts: (B, N, D) or None
            deep_prompts: (B, L-1, N, D) or None
            injection: (B, L_seq, D) or None
        """
        raise NotImplementedError

class VPTPromptGenerator(BasePromptGenerator):
    """
    Standard VPT (Visual Prompt Tuning) Implementation
    - Supports Shallow & Deep prompting
    - Uses exact initialization from original paper
    """
    def __init__(self, embed_dim=768, patch_size=14, num_prompts=10, 
                 use_deep=True, num_layers=12, dropout=0.1, **kwargs):
        super().__init__(embed_dim, patch_size)
        self.num_prompts = num_prompts
        self.use_deep = use_deep
        self.dropout = nn.Dropout(dropout)

        # --- Initialization (Original Paper Scheme) ---
        # patch_size가 tuple이 아니면 pair로 변환
        p_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        val = math.sqrt(6. / float(3 * reduce(mul, p_size, 1) + embed_dim))

        # 1. Shallow Prompts
        self.shallow_embeddings = nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
        nn.init.uniform_(self.shallow_embeddings.data, -val, val)

        # 2. Deep Prompts
        if use_deep and num_layers > 1:
            self.deep_embeddings = nn.Parameter(torch.zeros(num_layers - 1, num_prompts, embed_dim))
            nn.init.uniform_(self.deep_embeddings.data, -val, val)
        else:
            self.deep_embeddings = None

    def forward(self, depth_values, rgb_tokens=None):
        B = depth_values.shape[0]
        
        # Shallow
        shallow = self.shallow_embeddings.expand(B, -1, -1)
        shallow = self.dropout(shallow)
        
        # Deep
        deep = None
        if self.deep_embeddings is not None:
            deep = self.deep_embeddings.unsqueeze(0).expand(B, -1, -1, -1)
            deep = self.dropout(deep)
            
        return shallow, deep, None, None




class Fovea(nn.Module):
    def __init__(self, smooth=False):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.smooth = smooth
        if smooth:
            self.smooth_param = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        x_flat = x.flatten(2) # (B, C, N)

        if self.smooth:
            mask = self.softmax(x_flat * self.smooth_param)
        else:
            mask = self.softmax(x_flat)
            
        output = mask * x_flat
        return output.view(b, c, h, w)

class ViPTPromptBlock(nn.Module):
    def __init__(self, embed_dim, hide_channel=8, smooth=True):
        super().__init__()
        self.conv0_0 = nn.Conv2d(embed_dim, hide_channel, kernel_size=1) # RGB
        self.conv0_1 = nn.Conv2d(embed_dim, hide_channel, kernel_size=1) # Depth
        self.conv1x1 = nn.Conv2d(hide_channel, embed_dim, kernel_size=1) # Out
        self.fovea = Fovea(smooth=smooth)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, rgb_feat, depth_feat):
        # rgb_feat, depth_feat: [B, C, H, W]
        x0 = self.conv0_0(rgb_feat)
        x1 = self.conv0_1(depth_feat)
        
        # Spatial Gating on RGB + Add Depth Info (Original Logic)
        out = self.fovea(x0) + x1
        return self.conv1x1(out)

class ViPTPromptGenerator(BasePromptGenerator):
    def __init__(self, embed_dim=768, patch_size=14, input_type='xyzc', 
                 use_deep=True, num_layers=12, **kwargs):
        super().__init__(embed_dim, patch_size)
        self.use_deep = use_deep
        
        # Depth Embedding
        in_channels = self._get_in_channels(input_type)
        self.patch_embed_depth = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(1, embed_dim) # 원본 ViPT는 Norm을 PatchEmbed 안에 두거나 직후에 둠
        )
        
        # Shallow (Input Level)
        self.shallow_block = ViPTPromptBlock(embed_dim, hide_channel=8, smooth=True)
        self.shallow_norm = nn.LayerNorm(embed_dim) 
        
        # Deep (Layer 1 ~ N-1)
        self.deep_blocks = nn.ModuleList()
        self.deep_norms_rgb = nn.ModuleList()  
        self.deep_norms_prompted = nn.ModuleList() 
        
        # Layer 1부터 시작하므로 num_layers-1 개가 필요함
        if use_deep and num_layers > 1:
            for _ in range(num_layers - 1):
                self.deep_blocks.append(ViPTPromptBlock(embed_dim, hide_channel=8, smooth=True))
                self.deep_norms_rgb.append(nn.LayerNorm(embed_dim))
                self.deep_norms_prompted.append(nn.LayerNorm(embed_dim))

    def _get_in_channels(self, input_type):
        c = 0
        if 'xyz' in input_type: c += 3
        if 'd' in input_type: c += 1
        if 'c' in input_type: c += 1 # mask
        return max(1, c)

    def forward(self, depth_values, rgb_tokens):
        """
        Shallow Injection (Layer 0 Input)
        rgb_tokens: [B, N_spatial, C] (Prefix 제외된 순수 토큰)
        """
        depth_emb = self.patch_embed_depth(depth_values) # [B, C, H, W]
        B, C, H, W = depth_emb.shape
        
        # [FIX] Slicing 제거. GraspPro가 맞게 준다고 가정.
        # Shape을 depth_emb 기준으로 맞춤 (Dynamic Resolution 지원)
        rgb_feat = token2feature(self.shallow_norm(rgb_tokens), shape=(H, W))
        
        # Shallow Block
        out_feat = self.shallow_block(rgb_feat, depth_emb)
        injection = feature2token(out_feat)
        
        # out_feat는 다음 레이어(Layer 1)를 위한 depth_state가 됨
        return None, None, injection, out_feat

    def forward_layer(self, layer_idx, rgb_tokens, depth_state):
        """
        Deep Injection (Layer i Input, i >= 1)
        rgb_tokens: [B, N_spatial, C]
        depth_state: [B, C, H, W] (Previous Prompt Output)
        """
        # layer_idx는 1부터 시작 (GraspPro 루프 기준)
        if not self.use_deep or (layer_idx - 1) >= len(self.deep_blocks):
            return None, depth_state
        
        idx = layer_idx - 1  # 0-based index for lists
        block = self.deep_blocks[idx]
        norm_rgb = self.deep_norms_rgb[idx]
        norm_prompted = self.deep_norms_prompted[idx]
        
        B, C, H, W = depth_state.shape
        
        # [FIX] Slicing 제거
        rgb_feat = token2feature(norm_rgb(rgb_tokens), shape=(H, W))
        
        # Depth state normalization (Channel First -> Last -> Norm -> First)
        # ViPT 원본은 Prompt State에도 Norm을 적용하고 들어감
        depth_state_normed = depth_state.permute(0, 2, 3, 1) # B H W C
        depth_state_normed = norm_prompted(depth_state_normed)
        depth_state_normed = depth_state_normed.permute(0, 3, 1, 2).contiguous() # B C H W
        
        # Block forward
        out_feat = block(rgb_feat, depth_state_normed)
        injection = feature2token(out_feat)
        
        return injection, out_feat


class GeometricPromptGenerator(BasePromptGenerator):
    """
    Ours: Geometry-Aware Prompting
    - Shallow: Generated via Cross-Attention (Prompts Q, Depth K/V)
    - Deep: Standard VPT Static Deep Prompts (Optional)
    - Injection: Gated ViPT Injection
    """
    def __init__(self, embed_dim=768, patch_size=14, num_prompts=10, 
                 use_deep=True, num_layers=12, input_type='xyzc', 
                 depth_feat_dim=32, **kwargs):
        super().__init__(embed_dim, patch_size)
        
        # 1. Geometry Encoder
        in_channels = self._get_in_channels(input_type)
        self.geo_encoder = GeometricDepthStem(in_channels, depth_feat_dim, num_blocks=1)
        self.geo_projector = nn.Sequential(
            nn.Conv2d(depth_feat_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, embed_dim),
            nn.GELU()
        )
        
        # 2. Shallow Generator (Cross Attn)
        self.base_prompts = nn.Parameter(torch.randn(1, num_prompts, embed_dim) * 0.02)
        self.geo_cross_attn = GeometricCrossAttention(embed_dim, num_heads=8)
        
        # 3. Injection Generator (Gated ViPT)
        self.patch_embed_depth = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(1, embed_dim)
        )
        self.vipt_block = ViPTPromptBlock(embed_dim, hide_channel=32, smooth=True)
        self.norm_rgb = nn.LayerNorm(embed_dim)
        self.injection_gate = nn.Parameter(torch.zeros(1) + 1e-4)

        # 4. Deep Prompts (Static Parameter like VPT)
        # "Ours"에서도 Deep Prompt는 보통 Static하게 유지하여 연산량 조절
        self.use_deep = use_deep
        if use_deep and num_layers > 1:
            p_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
            val = math.sqrt(6. / float(3 * reduce(mul, p_size, 1) + embed_dim))
            self.deep_embeddings = nn.Parameter(torch.zeros(num_layers - 1, num_prompts, embed_dim))
            nn.init.uniform_(self.deep_embeddings.data, -val, val)
        else:
            self.deep_embeddings = None

    def _get_in_channels(self, input_type):
        c = 0
        if 'xyz' in input_type: c += 3
        if 'd' in input_type: c += 1
        if 'c' in input_type: c += 1
        return max(1, c)

    def forward(self, depth_values, rgb_tokens):
        B = depth_values.shape[0]
        
        # --- A. Shallow Prompt Generation ---
        # 1. Extract Geometry Features
        geo_feat = self.geo_encoder(depth_values) 
        geo_feat = self.geo_projector(geo_feat)
        # 2. Cross Attention
        raw_prompts = self.base_prompts.expand(B, -1, -1)
        shallow_prompts = self.geo_cross_attn(raw_prompts, geo_feat)
        
        # --- B. Injection Generation ---
        # 1. Depth Patch Embed
        depth_emb = self.patch_embed_depth(depth_values)
        _, _, H, W = depth_emb.shape
        # 2. RGB Feat
        spatial_tokens = rgb_tokens[:, -(H*W):, :]
        rgb_feat = token2feature(spatial_tokens, shape=(H, W)).permute(0,2,3,1)
        rgb_feat = self.norm_rgb(rgb_feat).permute(0,3,1,2)
        # 3. Gated Injection
        local_feat = self.vipt_block(rgb_feat, depth_emb)
        injection = feature2token(local_feat) * self.injection_gate
        
        # --- C. Deep Prompts ---
        deep_prompts = None
        if self.deep_embeddings is not None:
            deep_prompts = self.deep_embeddings.unsqueeze(0).expand(B, -1, -1, -1)

        return shallow_prompts, deep_prompts, injection

# Registry for easy access
PROMPT_GENERATOR_REGISTRY = {
    'vpt': VPTPromptGenerator,
    'vipt': ViPTPromptGenerator,
    'ours': GeometricPromptGenerator,
    'geometric': GeometricPromptGenerator 
}