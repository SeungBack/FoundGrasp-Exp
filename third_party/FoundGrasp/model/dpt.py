import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import _make_scratch, _make_fusion_block

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class BasicFusion(nn.Module):
    """
    가장 기초적인 Fusion 방식.
    Concat -> Conv(3x3) -> BN -> ReLU
    """
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, depth):
        # (B, 2C, H, W) -> (B, C, H, W)
        return self.proj(torch.cat([rgb, depth], dim=1))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 채널이 너무 작아지지 않도록 최소값 보장
        hidden_planes = max(in_planes // ratio, 8)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel 축으로 Avg, Max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CBAMFusion(nn.Module):
    """
    RGB와 Depth를 Concat한 후, CBAM(Channel -> Spatial)을 통해 Refine하는 방식
    """
    def __init__(self, dim):
        super().__init__()
        # 1. 초기 융합 (Channel Reduction)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. Attention Modules
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()

    def forward(self, rgb, depth):
        # Initial Fusion
        x = self.conv_cat(torch.cat([rgb, depth], dim=1))
        
        # Channel Attention
        x = x * self.ca(x)
        
        # Spatial Attention
        x = x * self.sa(x)
        
        # Residual Connection (원본 RGB 정보 보존을 위해 선택적으로 추가 가능)
        # 여기서는 CBAM 논문대로 Refine된 Feature 자체를 반환하거나,
        # RGB에 더해주는 방식을 택할 수 있음. 
        # Fusion 관점에서는 Refined Feature 반환이 일반적.
        return x

class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation.
    Depth 정보가 RGB Feature의 Scale(gamma)과 Shift(beta)를 예측하여 변조합니다.
    수식: RGB_new = gamma(Depth) * RGB + beta(Depth)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Depth로부터 Gamma(Scale)와 Beta(Shift)를 예측
        self.cond_net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dim, 2 * dim, kernel_size=1) # Output: Gamma + Beta
        )
        
        # 최종 Projection
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.cond_net[-1].weight.data.zero_()
        self.cond_net[-1].bias.data.zero_()

    def forward(self, rgb, depth):
        # 1. Depth Condition 생성
        # (B, 2*dim, H, W)
        style = self.cond_net(depth)
        gamma, beta = torch.split(style, self.dim, dim=1)
        
        # 2. Modulation
        # Gamma는 1을 중심으로, Beta는 0을 중심으로 학습되도록 초기화하면 좋음 (여기선 생략)
        # RGB feature를 Depth 정보에 따라 강화/억제
        modulated = (1 + gamma) * rgb + beta
        
        # 3. Residual & Projection
        # 원본 정보 보존을 위해 Residual 연결 가능
        return self.proj(modulated) 

class SKFusion(nn.Module):
    """
    Selective Kernel Fusion.
    Global Context를 보고 RGB와 Depth 중 어느 쪽 채널을 더 많이 가져갈지 
    Softmax로 가중치를 조절합니다.
    """
    def __init__(self, dim, r=8):
        super().__init__()
        d = max(int(dim / r), 32)
        
        # Global Information Aggregation
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Selection Weights Generator
        self.fc = nn.Sequential(
            nn.Conv2d(dim, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, dim * 2, kernel_size=1, bias=False) # 2 sets of weights
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb, depth):
        # 1. Combine initially
        U = rgb + depth
        
        # 2. Global Context Descriptor
        S = self.gap(U) # (B, C, 1, 1)
        
        # 3. Generate Selection Weights
        Z = self.fc(S) # (B, 2*C, 1, 1)
        
        # Reshape to (B, 2, C, 1, 1) for Softmax along dim 1
        B, _, _, _ = Z.shape
        Z = Z.view(B, 2, -1, 1, 1)
        weights = self.softmax(Z) # Sum of weights = 1
        
        w_rgb = weights[:, 0, :, :, :]
        w_depth = weights[:, 1, :, :, :]
        
        # 4. Adaptive Selection
        V = w_rgb * rgb + w_depth * depth
        
        return V

class GatedFusion(nn.Module):
    """
    Lightweight Gated Fusion Module.
    RGB와 Depth Feature를 받아 Gating을 통해 유의미한 정보만 선별하여 융합합니다.
    
    Logic:
        1. RGB + Depth Concat
        2. Channel Reduction (Efficiency)
        3. Calculate Spatial Attention Maps (Gates) for RGB and Depth
        4. Weighted Sum & Projection
    """
    def __init__(self, dim):
        super().__init__()
        # 연산량 감소를 위해 내부에서 채널을 줄여서 Gate 계산
        reduced_dim = max(dim // 4, 32) 
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, reduced_dim, kernel_size=1),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, 2, kernel_size=1), # Output: 2 channels (Gate_RGB, Gate_Depth)
            nn.Sigmoid()
        )
        
        # 융합된 Feature를 원래 차원으로 복원/혼합
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, depth):
        # rgb, depth: (B, C, H, W)
        cat = torch.cat([rgb, depth], dim=1)
        
        gates = self.gate_conv(cat)  # (B, 2, H, W)
        gates = F.softmax(gates, dim=1)  # Sum = 1 (competitive)

        gate_r = gates[:, 0:1, :, :]
        gate_d = gates[:, 1:2, :, :]

        # Weighted sum (not concat)
        fused = rgb * gate_r + depth * gate_d  # (B, C, H, W)
        return self.proj(fused)  # Optional projection

class DPT(nn.Module):
    """
    Standard DPT Decoder
    - Uses RefineNets to progressively fuse multi-scale features.
    - Restores features to high resolution (Stride 1 or 4).
    - Logic extracted from HierarchicalGraspDPTv1.
    """
    def __init__(self, 
                 in_channel=768,
                 out_channels=[96, 192, 384, 768], 
                 feature_dim=256, 
                 patch_size=14,
                 use_bn=False,
                 **kwargs
                 ):
        super().__init__()
        
        # 1. Readout Projects (Channel Adjustment)
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channel, out_ch, kernel_size=1, stride=1)
            for out_ch in out_channels
        ])

        # 2. Resize Layers (Alignment)
        # Assuming input scales are [4, 8, 16, 32]
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        # 3. RefineNets (Cascaded Fusion)
        self.scratch = _make_scratch(out_channels, feature_dim, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(feature_dim, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(feature_dim, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(feature_dim, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(feature_dim, use_bn)

        # 4. Output Conv
        self.output_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim) if use_bn else nn.Identity(),
            nn.ReLU(True)
        )

        self.patch_size = patch_size

    def forward(self, features, depth_feat=None):
        """
        Args:
            features: List of 4 feature maps [H/4, H/8, H/16, H/32]
            target_h, target_w: Target resolution for final interpolation
        Returns:
            Fused feature map (B, feature_dim, target_h, target_w)
        """
        out = []
        # Pre-process features (Project & Resize)
        # Note: The original code had dynamic sizing logic inside loop.
        # Here we simplify assuming standard DPT inputs.
        for i, x in enumerate(features):
            x = self.projects[i](x) # [B, out_ch, H_i, W_i]
            x = self.resize_layers[i](x) # Resize to common scale
            out.append(x)
        layer_1, layer_2, layer_3, layer_4 = out
        
        # Prepare inputs for RefineNets
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # Cascaded Fusion (Top-down)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, size=layer_1_rn.shape[2:]) # H/4 instead of H/2
        
        out = self.output_conv(path_1) # [B, feature_dim, H//4, W//4]
        return out
    

DECODER_REGISTRY = {
    'DPT': DPT,
}
