import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoBackbone
import math
import numpy as np


def _infer_backbone_size(backbone_name):
    backbone_name = backbone_name.lower()
    if 'small' in backbone_name or 'vits' in backbone_name:
        return 'small'
    if 'base' in backbone_name or 'vitb' in backbone_name:
        return 'base'
    if 'large' in backbone_name or 'vitl' in backbone_name:
        return 'large'
    if 'giant' in backbone_name or 'vitg' in backbone_name:
        return 'giant'
    if 'huge' in backbone_name or 'vith' in backbone_name:
        return 'huge'
    if 'so400m' in backbone_name:
        return 'so400m'
    raise ValueError(f"Unknown backbone name: {backbone_name}")


_LEGACY_OUT_INDICES = {
    'small': [2, 5, 8, 11],
    'base': [2, 5, 8, 11],
    'large': [4, 11, 17, 23],
    'huge': [7, 15, 23, 31],
    'so400m': [6, 13, 20, 26],  # SigLIP SO400M: 27 layers
}

_DAV2_DECODER_SPECS = {
    # Depth Anything V2 official DPT decoder sizing by encoder family.
    'small': ([2, 5, 8, 11], [48, 96, 192, 384], 64),
    'base': ([2, 5, 8, 11], [96, 192, 384, 768], 128),
    'large': ([4, 11, 17, 23], [256, 512, 1024, 1024], 256),
    'giant': ([9, 19, 29, 39], [1536, 1536, 1536, 1536], 384),
}


def get_intermediate_layer_config(backbone_name, decoder_preset=None):
    # Keep the historical FoundGrasp default unchanged unless a preset is requested.
    size = _infer_backbone_size(backbone_name)
    preset = (decoder_preset or 'legacy').lower()

    if preset in ('legacy', 'default'):
        if size not in _LEGACY_OUT_INDICES:
            raise ValueError(f"No legacy tap config for backbone: {backbone_name}")
        out_indices = _LEGACY_OUT_INDICES[size]
        out_channels = [48, 96, 192, 384]
        feature_dim = 64
    elif preset in ('dav2', 'depth_anything_v2', 'official', 'size_scaled'):
        if size not in _DAV2_DECODER_SPECS:
            raise ValueError(
                f"decoder.preset={decoder_preset!r} is only defined for DAv2 "
                f"S/B/L/G-style backbones, got: {backbone_name}"
            )
        out_indices, out_channels, feature_dim = _DAV2_DECODER_SPECS[size]
    else:
        raise ValueError(
            f"Unknown decoder preset: {decoder_preset!r}. "
            "Use 'legacy' or 'dav2'."
        )

    return list(out_indices), list(out_channels), int(feature_dim)

# -----------------------------------------------------------------------------
# 1. ViPT Style Utilities (Internalized)
# -----------------------------------------------------------------------------

# model/utils.py

import math

def token2feature(tokens, shape=None):
    """
    Args:
        tokens: (B, L, D)
        shape: (H, W) tuple. 
    """
    B, L, D = tokens.shape
    
    if shape is not None:
        H, W = shape
        if L != H * W:
            # 안전장치: 혹시라도 개수가 안 맞으면 에러 발생
            raise ValueError(f"Token length ({L}) != Shape ({H}x{W})")
    else:
        # shape이 없을 때만 기존 방식 (정사각형 가정)
        H = W = int(math.isqrt(L))

    # view 대신 reshape 사용 권장
    x = tokens.permute(0, 2, 1).reshape(B, D, H, W).contiguous()
    return x

def feature2token(x):
    """(B, D, H, W) -> (B, L, D)"""
    B, C, H, W = x.shape
    L = H * W
    # View -> Permute
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens

# Apply pixel-wise NMS
def pixel_nms(score_map, threshold, kernel_size=3):
    """
    Local maximum filtering using max pooling
    Args:
        score_map: torch.Tensor (H, W) or numpy array
        threshold: float
        kernel_size: int
    Returns:
        torch.Tensor (H, W) boolean mask
    """
    # Convert to tensor if needed and add batch/channel dimensions
    if isinstance(score_map, np.ndarray):
        score_tensor = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0).float()
    else:
        score_tensor = score_map.unsqueeze(0).unsqueeze(0).float()

    # Apply max pooling
    max_pool = F.max_pool2d(score_tensor, kernel_size,
                            stride=1, padding=kernel_size//2)

    # Keep only local maxima above threshold
    mask = (score_tensor == max_pool) & (score_tensor > threshold)
    return mask.squeeze()

def pixel_soft_nms(score_map, valid_mask, sigma=1.0, iters=2, score_min=0.0):
    """
    Very simple pixel-space Gaussian Soft-NMS.
    - score_map: (H,W) tensor
    - valid_mask: (H,W) bool
    Returns: new_score_map (H,W)
    """
    s = (score_map * valid_mask.float()).clone()
    H, W = s.shape

    # precompute gaussian kernel (fixed 2D)
    radius = int(3 * sigma)
    k = 2 * radius + 1
    yy, xx = torch.meshgrid(torch.arange(k, device=s.device),
                            torch.arange(k, device=s.device), indexing='ij')
    yy = yy - radius
    xx = xx - radius
    g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    g = g / (g.max() + 1e-9)  # normalize peak=1
    g = g.view(1, 1, k, k)

    s4 = s.view(1, 1, H, W)

    for _ in range(iters):
        # pick current local maxima map (not strict equality)
        mx = F.max_pool2d(s4, kernel_size=k, stride=1, padding=radius)
        peak = (s4 == mx) & (s4 > score_min)

        # decay neighbors around peaks (convolution of peak mask)
        peak_f = peak.float()
        influence = F.conv2d(peak_f, g, padding=radius)
        s4 = s4 * (1.0 - influence).clamp(min=0.0)

    return s4.view(H, W)

# 유틸리티 함수: 배수 패딩 (DINO, DepthAnything용)
def pad_to_multiple(x, multiple):
    H, W = x.shape[-2:]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (pad_h, pad_w)

# 유틸리티 함수: 고정 크기 패딩 (SAM용 1024x1024)
def pad_to_fixed_size(x, target_size=1024):
    H, W = x.shape[-2:]
    pad_h = max(0, target_size - H)
    pad_w = max(0, target_size - W)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h)) # 오른쪽, 아래쪽 채움
    return x, (pad_h, pad_w)

def crop_pred_dict(pred: dict, pad_hw):
    """
    Crop predictions to remove padding.
    
    Args:
        pred: Dictionary of predictions with shape (B, C, H, W)
        pad_hw: Either a tuple (pad_h, pad_w) or a list of tuples [(pad_h, pad_w), ...]
    
    Returns:
        Dictionary with cropped predictions
    """
    # Handle batch case: pad_hw is a list of (pad_h, pad_w) tuples
    if isinstance(pad_hw, list):
        # Check if all padding values are the same
        if len(pad_hw) > 0 and all(p == pad_hw[0] for p in pad_hw):
            # All same padding - can process as single crop
            pad_h, pad_w = pad_hw[0]
        else:
            # Different padding per batch - need to process separately
            # This is more complex, for now assume all same
            # TODO: implement per-batch cropping if needed
            pad_h, pad_w = pad_hw[0]
    else:
        # Single tuple case
        pad_h, pad_w = pad_hw
    
    if pad_h == 0 and pad_w == 0:
        return pred

    out = {}
    for k, v in pred.items():
        if k.startswith('_orientation_') or k.startswith('_attribute_'):
            out[k] = v
            continue
        # v: (B, L, H, W) 가정
        if v is None:
            out[k] = None
            continue
        if v.dim() == 4:
            H, W = v.shape[-2:]
            h_end = H - pad_h if pad_h > 0 else H
            w_end = W - pad_w if pad_w > 0 else W
            out[k] = v[..., :h_end, :w_end]
        else:
            # 혹시 다른 텐서(스칼라/벡터 등)가 섞여있을 때는 그대로
            out[k] = v
    return out
