import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(channels, max_groups=8):
    """Get number of groups for GroupNorm that divides channels evenly."""
    for g in [max_groups, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ConcatFusion(nn.Module):
    """Simple concatenation + convolution fusion"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_get_num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.fusion(x)


class AddFusion(nn.Module):
    """Element-wise RGB-D addition followed by one fusion conv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != 2 * out_channels:
            raise ValueError(
                f"AddFusion expects in_channels == 2 * out_channels, got {in_channels} and {out_channels}"
            )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_get_num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        rgb, depth = torch.chunk(x, 2, dim=1)
        return self.fusion(rgb + depth)


class FiLMFusion(nn.Module):
    """
    FiLM-based Fusion:
      y = (1 + gamma(depth)) * shared + beta(depth)  (+ optional depth residual)
    - shared: (B, C, H, W)
    - depth : (B, C, h, w)  (해상도 다르면 자동 resize)
    """
    def __init__(self, dim, cond_dim=None, hidden=None, add_depth_residual=True, norm='group'):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim or dim
        self.add_depth_residual = add_depth_residual

        hidden = hidden or max(32, dim // 4)

        # depth -> (gamma, beta) 예측
        self.cond_net = nn.Sequential(
            nn.Conv2d(self.cond_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * dim, kernel_size=1),
        )

        # ✅ 중요: 초기엔 FiLM 영향이 "0"이 되도록 (즉 y ≈ shared)
        nn.init.zeros_(self.cond_net[-1].weight)
        nn.init.zeros_(self.cond_net[-1].bias)

        # optional: depth를 feature로 더해주는 residual (안정적으로 depth 정보 주입)
        if add_depth_residual:
            self.depth_proj = nn.Conv2d(self.cond_dim, dim, kernel_size=1)
            nn.init.zeros_(self.depth_proj.weight)
            nn.init.zeros_(self.depth_proj.bias)
        else:
            self.depth_proj = None

        # output projection (정리)
        if norm == 'group':
            self.out = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.GroupNorm(8 if dim >= 8 else 1, dim),
                nn.GELU(),
            )
        elif norm == 'batch':
            self.out = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
        else:
            self.out = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
            )

    def forward(self, feature):
        shared, depth = torch.chunk(feature, 2, dim=1)
        if depth.shape[-2:] != shared.shape[-2:]:
            depth = F.interpolate(depth, size=shared.shape[-2:], mode='bilinear', align_corners=False)

        gamma_beta = self.cond_net(depth)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        y = (1.0 + gamma) * shared + beta

        if self.depth_proj is not None:
            y = y + self.depth_proj(depth)

        return self.out(y)

class ResidualFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_groups = _get_num_groups(out_channels)
        # 1. 차원 축소/매칭 (Projection)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU()
        )

        # 2. Residual Block (Conv3x3 -> GN -> GELU -> Conv3x3 -> GN)
        self.res_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, in_channels, H, W] -> Concat된 입력
        x = self.project(x)
        identity = x
        
        out = self.res_block(x)
        out += identity  # Residual Connection
        return self.act(out)



class BasicConv(nn.Module):
    """표준적인 Conv-BN-ReLU 블록"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        # torch.max는 (max_val, max_idx)를 반환하므로 [0]이 필요하지만,
        # torch.mean은 텐서를 바로 반환하므로 [0]을 쓰면 안 됩니다.
        
        # [수정 전] 에러 발생 코드
        # return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1)[0].unsqueeze(1)), dim=1)
        
        # [수정 후] Correct
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), 
            dim=1
        )
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) 
        return x * scale

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class CBAMFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        """
        in_channels: 2*feature_dim (RGB + Depth concatenated)
        out_channels: feature_dim
        """
        super().__init__()
        # 1. Projection (Dimension Reduction 2C -> C)
        self.project = BasicConv(in_channels, out_channels, kernel_size=1, relu=True)
        
        # 2. CBAM Modules
        self.ChannelGate = ChannelGate(out_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
        
        # 3. Final Refine
        self.refine = BasicConv(out_channels, out_channels, kernel_size=3, padding=1, relu=True)

    def forward(self, x):
        # x: [B, 2C, H, W]
        x = self.project(x)
        identity = x
        
        # Apply Channel Attention
        x = self.ChannelGate(x)
        # Apply Spatial Attention
        x = self.SpatialGate(x)
        
        # Residual Connection
        out = x + identity
        return self.refine(out)
    
class CrossGatedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels: 2*feature_dim (Input is concat of [RGB, Depth])
        out_channels: feature_dim
        """
        super().__init__()
        self.half_in = in_channels // 2
        
        # 각 모달리티별 특징 추출 (Feature Transformation)
        self.rgb_transform = nn.Sequential(
            nn.Conv2d(self.half_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.depth_transform = nn.Sequential(
            nn.Conv2d(self.half_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # 융합된 정보를 정리하는 레이어
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), # 1x1로 채널 믹싱
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # 3x3으로 공간 정보 정리
            nn.BatchNorm2d(out_channels),
            # 마지막에는 활성화 함수 없이 나가거나, 필요시 추가
        )
        self.final_act = nn.GELU()

    def forward(self, x):
        # x: [B, 2*C, H, W] -> RGB와 Depth가 Concat된 상태라고 가정
        
        # 1. Split Modalities
        # 입력 채널의 절반씩 나눔 (앞쪽이 RGB, 뒤쪽이 Depth라고 가정. 순서는 바뀌어도 대칭적이라 무방)
        rgb_raw, depth_raw = torch.split(x, self.half_in, dim=1)
        
        # 2. Transform Features
        feat_rgb = self.rgb_transform(rgb_raw)     # [B, out_ch, H, W]
        feat_depth = self.depth_transform(depth_raw) # [B, out_ch, H, W]
        
        # 3. Cross Gating Mechanism
        # Depth 정보로 RGB를 Gating (Depth가 뚜렷한 곳의 RGB를 강조)
        gate_rgb = torch.sigmoid(feat_depth) 
        gated_rgb = feat_rgb * gate_rgb
        
        # RGB 정보로 Depth를 Gating (RGB 텍스처가 있는 곳의 Depth를 강조)
        gate_depth = torch.sigmoid(feat_rgb)
        gated_depth = feat_depth * gate_depth
        
        # 4. Additive Fusion (Element-wise Sum)
        fused = gated_rgb + gated_depth
        
        # 5. Final Refinement
        out = self.fusion_conv(fused)
        
        return self.final_act(out)


class BasicBlockGN(nn.Module):
    """
    ResNet Basic Block with GroupNorm
    """
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        num_groups = _get_num_groups(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, planes)
        self.gelu = nn.GELU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, planes)
            )

    def forward(self, x):
        out = self.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.gelu(out)
        return out


def get_depth_input_channels(input_type):
    depth_in = 0
    if 'xyz' in input_type:
        depth_in += 3
    if 'd' in input_type:
        depth_in += 1
    if 'c' in input_type:
        depth_in += 1
    return depth_in

class GeometricDepthStem(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, backbone='resnet18'):
        """
        Depth Encoder using pretrained ResNet18/34
        Fuses Layer1 (H/4) and Layer2 (H/8) to produce rich H/4 features.
        
        Output: H/4 × W/4 feature map (out_channels)
        """
        super().__init__()

        # 1. Load Pretrained ResNet
        if backbone == 'resnet18':
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            l1_ch, l2_ch = 64, 128
        elif backbone == 'resnet34':
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            l1_ch, l2_ch = 64, 128
        elif backbone == 'resnet50':
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            l1_ch, l2_ch = 256, 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 2. Modify first conv layer
        if in_channels != 3:
            self.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            # Pretrained weight averaging for initialization
            with torch.no_grad():
                self.conv1.weight[:] = torch.mean(resnet.conv1.weight, dim=1, keepdim=True)
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 3. Extract Blocks
        self.layer1 = resnet.layer1  # H/4, 64ch
        self.layer2 = resnet.layer2  # H/8, 128ch

        # 4. Fusion Layer (1x1 Conv)
        # Layer1(64) + Upsampled Layer2(128) = 192 channels -> out_channels(128)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(l1_ch + l2_ch, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()  # ReLU -> GELU (Backbone 통일성)
        )

    def forward(self, x):
        # Stem: H -> H/4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1: H/4 (Low-level Geometry)
        feat_l1 = self.layer1(x)  # (B, 64, H/4, W/4)

        # Layer 2: H/8 (Mid-level Geometry)
        feat_l2 = self.layer2(feat_l1) # (B, 128, H/8, W/8)

        # Fusion: Upsample L2 to H/4 and Concat with L1
        feat_l2_up = F.interpolate(feat_l2, size=feat_l1.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenation (Rich Context + Fine Detail)
        feat_fused = torch.cat([feat_l1, feat_l2_up], dim=1) # (B, 192, H/4, W/4)
        
        # Projection to target dimension
        out = self.fusion_conv(feat_fused) # (B, 128, H/4, W/4)

        return out

## DPT

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.GELU(),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch




class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)

        return output
