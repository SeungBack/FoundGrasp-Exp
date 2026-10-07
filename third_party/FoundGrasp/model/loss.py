import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .orientation import (
    decode_joint_orientation_classes,
    quantize_joint_orientation_vectors,
)

COS_EPS = 1e-4  # Epsilon for numerical stability in cosine similarity


def nll_vonmf(dot, pred_kappa):
    loss = - torch.log(pred_kappa) \
            - (pred_kappa * (dot - 1)) \
            + torch.log(1 - torch.exp(- 2 * pred_kappa))
    return loss

def nll_angmf(dot, pred_kappa):
    loss = - torch.log(torch.square(pred_kappa) + 1) \
            + pred_kappa * torch.acos(dot) \
            + torch.log(1 + torch.exp(-pred_kappa * np.pi))
    return loss


def align_and_expand(tensor, target):
    """
    Aligns tensor dimensions to match target and expands.
    Handles cases where tensor has extra singleton dimensions (e.g., [B, 1, H, W] vs [B, H, W]).
    """
    if tensor is None:
        return None
    while tensor.dim() > target.dim():
        tensor = tensor.squeeze(1)
    while tensor.dim() < target.dim():
        tensor = tensor.unsqueeze(1)
    return tensor.expand_as(target)


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, 'get'):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class PairwiseRankingLoss(nn.Module):
    def __init__(
        self,
        num_samples=2000,
        min_gt_diff=0.1,
        margin=None,
    ):
        """
        Pairwise fixed-margin hinge ranking loss.

        valid pair: diff_gt >= min_gt_diff
        loss = relu(margin - (pred_i - pred_j))

        Args:
            num_samples: 픽셀 샘플링 수 (all-to-all 계산 전 subsample)
            min_gt_diff: pair 선택 threshold.
                         score(step=0.1)는 0.1, graspness(연속)는 0.05 권장.
            margin: hinge margin. None이면 min_gt_diff를 사용.
        """
        super().__init__()
        self.num_samples = num_samples
        self.min_gt_diff = min_gt_diff
        self.margin = min_gt_diff if margin is None else margin

    def _compute_ranking_loss(self, p_flat, g_flat):
        """주어진 1D 예측/GT 텐서에 대해 pairwise ranking loss를 계산."""
        N = p_flat.size(0)
        if N > self.num_samples:
            perm = torch.randperm(N, device=p_flat.device)[:self.num_samples]
            p_flat = p_flat[perm]
            g_flat = g_flat[perm]

        # Pairwise difference: [i,j] = val_i - val_j
        diff_gt   = g_flat.unsqueeze(1) - g_flat.unsqueeze(0)   # (N,N)
        diff_pred = p_flat.unsqueeze(1) - p_flat.unsqueeze(0)   # (N,N)

        # min_gt_diff > 0이므로 대각(diff=0)은 자동 제외
        valid_pairs = diff_gt >= self.min_gt_diff
        if valid_pairs.sum() == 0:
            return None, 0

        loss = F.relu(self.margin - diff_pred[valid_pairs])
        return loss.sum(), loss.numel()

    def forward(self, pred, gt, valid_mask):
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if gt.dim() == 4:
            gt = gt.squeeze(1)
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(1)

        total_loss  = 0.0
        total_pairs = 0

        for b in range(pred.shape[0]):
            mask_b = valid_mask[b] > 0
            if mask_b.sum() < 2:
                continue

            loss_sum, n_pairs = self._compute_ranking_loss(pred[b][mask_b], gt[b][mask_b])
            if loss_sum is None:
                continue
            total_loss  += loss_sum
            total_pairs += n_pairs

        if total_pairs == 0:
            return pred.new_tensor(0.0)

        return total_loss / total_pairs


class ObjectWiseRankingLoss(nn.Module):
    """
    Object-wise pairwise ranking loss.

    GT가 물체별로 독립적으로 스케일링된 경우(e.g. graspness min-max),
    서로 다른 물체의 픽셀끼리는 ranking 비교가 무의미하다.
    물체별로 독립적으로 pairwise ranking을 계산한 뒤 평균한다.

    Args:
        num_samples_per_obj: 물체당 최대 샘플링 픽셀 수
        min_gt_diff: 이 값 이상 차이나는 쌍만 ranking에 포함
    """
    def __init__(
        self,
        num_samples_per_obj=400,
        min_gt_diff=0.05,
        margin=None,
    ):
        super().__init__()
        self.num_samples_per_obj = num_samples_per_obj
        self.min_gt_diff = min_gt_diff
        self.margin = min_gt_diff if margin is None else margin

    def _obj_ranking_loss(self, p_obj, g_obj):
        N = p_obj.size(0)
        if N > self.num_samples_per_obj:
            perm = torch.randperm(N, device=p_obj.device)[:self.num_samples_per_obj]
            p_obj = p_obj[perm]
            g_obj = g_obj[perm]

        diff_gt   = g_obj.unsqueeze(1) - g_obj.unsqueeze(0)
        diff_pred = p_obj.unsqueeze(1) - p_obj.unsqueeze(0)

        valid_pairs = diff_gt >= self.min_gt_diff
        if valid_pairs.sum() == 0:
            return None

        loss = F.relu(self.margin - diff_pred[valid_pairs])
        return loss.mean()

    def forward(self, pred, gt, valid_mask, segm):
        """
        Args:
            pred:       (B, 1, H, W) or (B, H, W)
            gt:         (B, 1, H, W) or (B, H, W)
            valid_mask: (B, 1, H, W) or (B, H, W)
            segm:       (B, 1, H, W) or (B, H, W) — object segmentation (0=background)
        """
        def _squeeze(t):
            return t.squeeze(1) if t.dim() == 4 else t

        pred, gt, valid_mask, segm = map(_squeeze, (pred, gt, valid_mask, segm))

        obj_losses = []

        for b in range(pred.shape[0]):
            valid_b = valid_mask[b] > 0
            segm_b  = segm[b]

            obj_ids = segm_b[valid_b].unique()
            obj_ids = obj_ids[obj_ids > 0]  # 배경(0) 제외

            for obj_id in obj_ids:
                obj_mask = valid_b & (segm_b == obj_id)
                if obj_mask.sum() < 2:
                    continue

                loss = self._obj_ranking_loss(pred[b][obj_mask], gt[b][obj_mask])
                if loss is not None:
                    obj_losses.append(loss)

        if len(obj_losses) == 0:
            return pred.new_tensor(0.0)

        return torch.stack(obj_losses).mean()

def dense_gt_propagation(
    target_maps,
    valid_mask,
    depth_img,
    objectness_mask=None,
    radius=3,
    depth_thresh=0.005,
):
    """
    GPU-based dense propagation for grasp supervision using PyTorch.
    Uses dilation-based approach for efficient nearest neighbor search.

    Args:
        target_maps: Dict of target tensors {key: (B, C, H, W)}
        valid_mask: Original valid mask (B, 1, H, W)
        depth_img: Depth image (B, 1, H, W) or (B, H, W)
        objectness_mask: Object segmentation mask (B, 1, H, W), propagation only within this region
        radius: Maximum propagation radius in pixels
        depth_thresh: Maximum depth difference for propagation

    Returns:
        updated_maps: Dict of propagated target tensors
        confidence: Confidence weights (B, 1, H, W)
        updated_valid_mask: Updated valid mask (B, 1, H, W)
    """
    device = valid_mask.device
    B, _, H, W = valid_mask.shape

    # Ensure depth_img is (B, 1, H, W)
    if depth_img.dim() == 3:
        depth_img = depth_img.unsqueeze(1)

    # Ensure objectness_mask is (B, 1, H, W) boolean
    if objectness_mask is not None:
        if objectness_mask.dim() == 3:
            objectness_mask = objectness_mask.unsqueeze(1)
        obj_mask = (objectness_mask > 0).float()  # (B, 1, H, W)
    else:
        obj_mask = torch.ones((B, 1, H, W), device=device)

    sigma_dist = radius / 3.0
    sigma_depth = depth_thresh / 3.0

    org_mask = (valid_mask > 0).float()  # (B, 1, H, W)

    # Create coordinate grids (shared across batch)
    y_coords = torch.arange(H, device=device).float().view(1, 1, H, 1).expand(B, 1, H, W)
    x_coords = torch.arange(W, device=device).float().view(1, 1, 1, W).expand(B, 1, H, W)

    # Initialize source coordinate maps with self-coordinates
    # These will store the coordinates of the nearest valid pixel
    src_y = y_coords.clone()
    src_x = x_coords.clone()

    # Initialize distance map (0 for valid, inf for invalid)
    # This tracks the actual euclidean distance to the source pixel
    best_dist = torch.where(org_mask > 0, torch.zeros_like(org_mask), torch.full_like(org_mask, float('inf')))

    # Use iterative approach: for each distance level, expand the frontier
    current_mask = org_mask.clone()  # pixels that have been assigned

    # Precompute neighbor offsets and their distances
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            step_dist = (dy**2 + dx**2) ** 0.5  # 1.0 for orthogonal, ~1.41 for diagonal
            neighbors.append((dy, dx, step_dist))

    for d in range(1, radius + 1):
        # Dilate current mask
        dilated = F.max_pool2d(current_mask, kernel_size=3, stride=1, padding=1)

        # New frontier: pixels that are newly reached
        frontier = (dilated > 0) & (current_mask == 0) & (obj_mask > 0)

        # Debug output (uncomment to enable)
        # if d == 1:
        #     print(f"[DEBUG] Iteration {d}:")
        #     print(f"  - current_mask sum: {current_mask.sum().item():.0f}")
        #     print(f"  - dilated sum: {(dilated > 0).sum().item():.0f}")
        #     print(f"  - obj_mask sum: {(obj_mask > 0).sum().item():.0f}")
        #     print(f"  - (dilated > 0) & (current_mask == 0): {((dilated > 0) & (current_mask == 0)).sum().item():.0f}")
        #     print(f"  - frontier sum: {frontier.sum().item():.0f}")

        if frontier.sum() == 0:
            break

        # For frontier pixels, find which neighbor gives the smallest total distance
        for dy, dx, step_dist in neighbors:
            # To get neighbor at (y+dy, x+dx), we need to shift the tensor
            # so that position (y, x) contains the value from (y+dy, x+dx)
            # This means we roll/shift in the opposite direction

            # Use torch.roll for simpler and correct shifting
            shifted_mask = torch.roll(current_mask, shifts=(-dy, -dx), dims=(2, 3))
            shifted_src_y = torch.roll(src_y, shifts=(-dy, -dx), dims=(2, 3))
            shifted_src_x = torch.roll(src_x, shifts=(-dy, -dx), dims=(2, 3))

            # Handle boundary: pixels that wrapped around should not be valid
            # Create boundary mask
            boundary_mask = torch.ones_like(current_mask, dtype=torch.bool)
            if dy > 0:  # Looking at neighbor above, bottom rows wrapped
                boundary_mask[:, :, -dy:, :] = False
            elif dy < 0:  # Looking at neighbor below, top rows wrapped
                boundary_mask[:, :, :-dy, :] = False
            if dx > 0:  # Looking at neighbor to the left, right cols wrapped
                boundary_mask[:, :, :, -dx:] = False
            elif dx < 0:  # Looking at neighbor to the right, left cols wrapped
                boundary_mask[:, :, :, :-dx] = False

            # Compute actual distance from current pixel to the neighbor's source
            new_dist = torch.sqrt((y_coords - shifted_src_y) ** 2 + (x_coords - shifted_src_x) ** 2)

            # Update condition: frontier pixel, neighbor is valid (assigned), within boundary, and this path is shorter
            update_mask = frontier & (shifted_mask > 0) & boundary_mask & (new_dist < best_dist)

            src_y = torch.where(update_mask, shifted_src_y, src_y)
            src_x = torch.where(update_mask, shifted_src_x, src_x)
            best_dist = torch.where(update_mask, new_dist, best_dist)
            current_mask = torch.where(update_mask, torch.ones_like(current_mask), current_mask)
            
            # Clean up intermediate tensors to prevent memory accumulation
            del shifted_mask, shifted_src_y, shifted_src_x, boundary_mask, new_dist, update_mask

    # Now src_y, src_x contain coordinates of nearest valid pixel for each propagated pixel
    # best_dist already contains the actual euclidean distance
    dist_map = best_dist

    # Get depth at source locations using grid_sample
    # Normalize coordinates to [-1, 1]
    src_y_norm = (src_y / (H - 1)) * 2 - 1
    src_x_norm = (src_x / (W - 1)) * 2 - 1
    grid = torch.stack([src_x_norm, src_y_norm], dim=-1).squeeze(1)  # (B, H, W, 2)

    src_depth = F.grid_sample(depth_img, grid, mode='nearest', align_corners=True)  # (B, 1, H, W)

    # Compute depth difference
    depth_diff = torch.abs(depth_img - src_depth)

    # Propagation mask: within radius, within depth threshold, within object, not original valid
    prop_mask = (dist_map > 0) & (dist_map <= radius) & (depth_diff <= depth_thresh) & (obj_mask > 0) & (org_mask == 0)

    # Compute confidence
    dist_w = torch.exp(-(dist_map ** 2) / (2 * sigma_dist ** 2))
    depth_w = torch.exp(-(depth_diff ** 2) / (2 * sigma_depth ** 2))
    confidence = dist_w * depth_w
    confidence = torch.where(prop_mask, confidence, torch.zeros_like(confidence))
    confidence = torch.where(org_mask > 0, torch.ones_like(confidence), confidence)

    # Propagate target values using grid_sample
    updated_maps = {}
    for key, arr in target_maps.items():
        # Use grid_sample for efficient gathering
        propagated = F.grid_sample(arr, grid, mode='nearest', align_corners=True)
        # Expand masks to match arr's channel dimension for proper broadcasting
        C = arr.shape[1]
        org_mask_exp = org_mask.expand(-1, C, -1, -1)
        prop_mask_exp = prop_mask.expand(-1, C, -1, -1)
        # Keep original values where valid, use propagated where prop_mask
        updated = torch.where(org_mask_exp > 0, arr, torch.where(prop_mask_exp, propagated, arr))
        updated_maps[key] = updated

    # Update valid mask
    updated_valid_mask = (confidence > 0).float()

    # Return debug info for visualization
    debug_info = {
        'dist_map': dist_map,
        'depth_diff': depth_diff,
        'radius': radius,
        'depth_thresh': depth_thresh,
    }

    return updated_maps, confidence, updated_valid_mask, debug_info


def _sample_valid_orientation_coords(valid_mask, max_samples):
    mask = valid_mask.squeeze(1) if valid_mask.dim() == 4 else valid_mask
    coords = torch.nonzero(mask > 0, as_tuple=False)  # (M, 3): b, y, x
    if coords.numel() == 0:
        return coords
    if max_samples is not None and coords.shape[0] > int(max_samples):
        perm = torch.randperm(coords.shape[0], device=coords.device)[:int(max_samples)]
        coords = coords[perm]
    return coords


def _orientation_full_hw(output, target_hw):
    full_hw = output.get('_orientation_full_hw', None)
    if full_hw is None:
        B = output['_orientation_feat'].shape[0]
        return torch.tensor(
            target_hw,
            device=output['_orientation_feat'].device,
            dtype=torch.long,
        ).view(1, 2).expand(B, -1)
    if full_hw.dim() == 1:
        full_hw = full_hw.view(1, 2)
    return full_hw.to(device=output['_orientation_feat'].device, dtype=torch.long)


def _gather_sparse_feature_map(feature_map, coords, full_hw):
    _, _, feat_h, feat_w = feature_map.shape
    b_idx = coords[:, 0].long()
    y_idx = coords[:, 1].long()
    x_idx = coords[:, 2].long()
    full_h = full_hw[b_idx, 0].float().clamp_min(1.0)
    full_w = full_hw[b_idx, 1].float().clamp_min(1.0)

    # Match F.interpolate(..., align_corners=False) used by the dense heads,
    # but gather only the supervised pixels instead of materializing HxW logits.
    feat_y = (y_idx.float() + 0.5) * float(feat_h) / full_h - 0.5
    feat_x = (x_idx.float() + 0.5) * float(feat_w) / full_w - 0.5
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

    feat_grid = feature_map.permute(0, 2, 3, 1)
    f00 = feat_grid[b_idx, y0, x0]
    f01 = feat_grid[b_idx, y0, x1]
    f10 = feat_grid[b_idx, y1, x0]
    f11 = feat_grid[b_idx, y1, x1]
    return (
        f00 * (1.0 - wy) * (1.0 - wx)
        + f01 * (1.0 - wy) * wx
        + f10 * wy * (1.0 - wx)
        + f11 * wy * wx
    )


def _gather_orientation_features(output, coords, target_hw):
    full_hw = _orientation_full_hw(output, target_hw)
    return _gather_sparse_feature_map(output['_orientation_feat'], coords, full_hw)


def _joint_orientation_meta(output):
    meta = output.get('_orientation_meta', None)
    if meta is None:
        return 300, 12
    if meta.dim() > 1:
        meta = meta[0]
    return int(meta[0].item()), int(meta[1].item())


def get_sparse_joint_orientation_loss(
    loss_config,
    output,
    target_app,
    target_grip,
    valid_mask,
    confidence=None,
):
    max_samples = cfg_get(loss_config, 'max_samples', 65536)
    chunk_size = int(cfg_get(loss_config, 'chunk_size', 4096))
    weight_scale = float(cfg_get(loss_config, 'weight', 1.0))
    symmetric_gripper = bool(cfg_get(loss_config, 'symmetric_gripper', True))
    normalize_by_log_classes = bool(
        cfg_get(loss_config, 'normalize_by_log_classes', True)
    )
    loss_type = cfg_get(loss_config, 'type', 'cross_entropy')
    if loss_type != 'cross_entropy':
        raise ValueError(f"Joint orientation baseline requires cross_entropy, got {loss_type!r}.")

    coords = _sample_valid_orientation_coords(valid_mask, max_samples)
    orientation_feat = output['_orientation_feat']
    if coords.numel() == 0:
        return orientation_feat.sum() * 0.0

    _, _, target_h, target_w = target_app.shape
    feat_samples = _gather_orientation_features(output, coords, (target_h, target_w))
    b_idx = coords[:, 0].long()
    y_idx = coords[:, 1].long()
    x_idx = coords[:, 2].long()

    app_samples = target_app.permute(0, 2, 3, 1)[b_idx, y_idx, x_idx]
    grip_samples = target_grip.permute(0, 2, 3, 1)[b_idx, y_idx, x_idx]
    num_views, num_angles = _joint_orientation_meta(output)
    class_target = quantize_joint_orientation_vectors(
        app_samples,
        grip_samples,
        num_views=num_views,
        num_angles=num_angles,
        symmetric_gripper=symmetric_gripper,
    ).long()

    sample_weight = None
    if confidence is not None:
        conf = confidence.squeeze(1) if confidence.dim() == 4 else confidence
        sample_weight = conf[b_idx, y_idx, x_idx].float()

    classifier_weight = output['_orientation_weight'][0].float()
    classifier_bias = output['_orientation_bias'][0].float()
    loss_sum = orientation_feat.new_tensor(0.0)
    weight_sum = orientation_feat.new_tensor(0.0)

    for start in range(0, feat_samples.shape[0], chunk_size):
        end = min(start + chunk_size, feat_samples.shape[0])
        logits = F.linear(feat_samples[start:end].float(), classifier_weight, classifier_bias)
        loss = F.cross_entropy(logits, class_target[start:end], reduction='none')
        if normalize_by_log_classes:
            loss = loss / math.log(logits.shape[1])
        if sample_weight is not None:
            w = sample_weight[start:end].to(loss.dtype)
            loss_sum = loss_sum + (loss * w).sum()
            weight_sum = weight_sum + w.sum()
        else:
            loss_sum = loss_sum + loss.sum()
            weight_sum = weight_sum + loss.new_tensor(loss.numel(), dtype=loss.dtype)

    return (loss_sum / weight_sum.clamp_min(1.0)) * weight_scale


def get_sparse_joint_orientation_loss_mse(
    loss_config,
    output,
    gt_sparse_list,
    device,
):
    """RGB Matters-style joint (V*A) multi-label orientation heatmap (MSE).

    Each valid grasp at a supervised pixel marks its quantized (view, angle)
    class as a positive (1.0); all other classes are 0.0. The orientation
    logits are regressed toward this multi-hot binary target with plain MSE
    (no positive weighting, no sigmoid), matching the RGB Matters AVN heatmap
    objective (Gou et al., ICRA'21). Inference is unchanged: argmax over the
    same logits selects the (view, angle) class.
    """
    weight_scale = float(cfg_get(loss_config, 'weight', 1.0))
    max_samples = int(cfg_get(loss_config, 'max_samples', 65536))
    chunk_size = int(cfg_get(loss_config, 'chunk_size', 4096))
    symmetric_gripper = bool(cfg_get(loss_config, 'symmetric_gripper', True))

    orientation_feat = output['_orientation_feat']
    num_views, num_angles = _joint_orientation_meta(output)
    classifier_weight = output['_orientation_weight'][0].float()
    classifier_bias = output['_orientation_bias'][0].float()
    H, W = output['objectness'].shape[-2:]

    loss_sum = orientation_feat.sum() * 0.0  # graph-connected zero
    count = orientation_feat.new_tensor(0.0)

    for b, gt_data in enumerate(gt_sparse_list):
        indices = gt_data.get('valid_pixel_indices', None)
        if indices is None or len(indices) == 0:
            continue
        v = indices[:, 0].long().to(device)
        u = indices[:, 1].long().to(device)
        N = v.shape[0]

        sel = None
        if max_samples is not None and N > max_samples:
            sel = torch.randperm(N, device=device)[:max_samples]
            v, u, N = v[sel], u[sel], sel.shape[0]

        coords = torch.stack(
            [torch.full((N,), b, device=device, dtype=torch.long), v, u], dim=1
        )
        feat_b = _gather_orientation_features(output, coords, (H, W)).float()  # (N, C)

        gt_app = gt_data['approach_dir'].to(device).float()   # (N0, K, 3)
        gt_grip = gt_data['gripper_dir'].to(device).float()   # (N0, K, 3)
        gt_score = gt_data['grasp_score'].to(device).float()  # (N0, K)
        if sel is not None:
            gt_app, gt_grip, gt_score = gt_app[sel], gt_grip[sel], gt_score[sel]
        K = gt_app.shape[1]

        valid_k = (gt_score > 0).reshape(-1)                  # (N*K,)
        cls = quantize_joint_orientation_vectors(
            gt_app.reshape(-1, 3),
            gt_grip.reshape(-1, 3),
            num_views=num_views,
            num_angles=num_angles,
            symmetric_gripper=symmetric_gripper,
        ).long()                                              # (N*K,)
        row_idx = torch.arange(N, device=device).repeat_interleave(K)
        row_idx = row_idx[valid_k]
        cls = cls[valid_k]

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            logits = F.linear(feat_b[start:end], classifier_weight, classifier_bias)
            target = torch.zeros_like(logits)
            in_chunk = (row_idx >= start) & (row_idx < end)
            if in_chunk.any():
                target[row_idx[in_chunk] - start, cls[in_chunk]] = 1.0
            loss_sum = loss_sum + F.mse_loss(logits, target, reduction='sum')
            count = count + logits.numel()

    return (loss_sum / count.clamp_min(1.0)) * weight_scale


def get_sparse_discrete_attribute_loss(
    loss_config,
    logits,
    target,
    centers,
    valid_mask,
    confidence=None,
    full_hw=None,
):
    loss_type = cfg_get(loss_config, 'type', 'cross_entropy')
    if loss_type != 'cross_entropy':
        raise ValueError(f"Discrete attribute baseline requires cross_entropy, got {loss_type!r}.")

    max_samples = cfg_get(loss_config, 'max_samples', 65536)
    coords = _sample_valid_orientation_coords(valid_mask, max_samples)
    if coords.numel() == 0:
        return logits.sum() * 0.0

    if centers.dim() > 1:
        centers = centers[0]
    centers = centers.to(device=target.device, dtype=target.dtype)
    target_h, target_w = target.shape[-2:]
    if full_hw is None:
        full_hw = logits.new_tensor([target_h, target_w]).view(1, 2).expand(logits.shape[0], -1)
    elif full_hw.dim() == 1:
        full_hw = full_hw.view(1, 2)
    full_hw = full_hw.to(device=logits.device, dtype=torch.long)
    sampled_logits = _gather_sparse_feature_map(logits, coords, full_hw)

    b_idx = coords[:, 0].long()
    y_idx = coords[:, 1].long()
    x_idx = coords[:, 2].long()
    target_scalar = target.squeeze(1) if target.dim() == 4 else target
    sampled_target = target_scalar[b_idx, y_idx, x_idx]
    class_target = torch.argmin(
        torch.abs(sampled_target.unsqueeze(1) - centers.view(1, -1)),
        dim=1,
    )

    loss = F.cross_entropy(sampled_logits.float(), class_target.long(), reduction='none')
    if bool(cfg_get(loss_config, 'normalize_by_log_classes', True)):
        loss = loss / math.log(sampled_logits.shape[1])

    final_weight = loss.new_ones(loss.shape)
    if confidence is not None:
        conf = confidence.squeeze(1) if confidence.dim() == 4 else confidence
        final_weight = final_weight * conf[b_idx, y_idx, x_idx].float()

    weighted_sum = (loss * final_weight).sum()
    weight_sum = final_weight.sum().clamp_min(1.0)
    return (weighted_sum / weight_sum) * float(cfg_get(loss_config, 'weight', 1.0))


@torch.no_grad()
def _sample_joint_orientation_angle_metrics(output, data, valid_mask, max_samples=16384):
    coords = _sample_valid_orientation_coords(valid_mask, max_samples)
    if coords.numel() == 0:
        zero = output['objectness'].new_tensor(0.0)
        return zero, zero

    _, _, target_h, target_w = data['approach_dir'].shape
    feat_samples = _gather_orientation_features(output, coords, (target_h, target_w))
    classifier_weight = output['_orientation_weight'][0].float()
    classifier_bias = output['_orientation_bias'][0].float()
    chunk_size = 4096
    pred_classes = []
    for start in range(0, feat_samples.shape[0], chunk_size):
        end = min(start + chunk_size, feat_samples.shape[0])
        logits = F.linear(feat_samples[start:end].float(), classifier_weight, classifier_bias)
        pred_classes.append(torch.argmax(logits, dim=1))
    pred_classes = torch.cat(pred_classes, dim=0)

    num_views, num_angles = _joint_orientation_meta(output)
    pred_app, pred_grip = decode_joint_orientation_classes(pred_classes, num_views, num_angles)
    pred_app = pred_app.to(device=feat_samples.device, dtype=feat_samples.dtype)
    pred_grip = pred_grip.to(device=feat_samples.device, dtype=feat_samples.dtype)

    b_idx = coords[:, 0].long()
    y_idx = coords[:, 1].long()
    x_idx = coords[:, 2].long()
    target_app = data['approach_dir'].permute(0, 2, 3, 1)[b_idx, y_idx, x_idx]
    target_grip = data['gripper_dir'].permute(0, 2, 3, 1)[b_idx, y_idx, x_idx]

    app_cos = F.cosine_similarity(pred_app, target_app, dim=1, eps=1e-6)
    grip_cos = F.cosine_similarity(pred_grip, target_grip, dim=1, eps=1e-6).abs()
    app_angle = torch.acos(torch.clamp(app_cos, -1, 1)) * 180 / np.pi
    grip_angle = torch.acos(torch.clamp(grip_cos, -1, 1)) * 180 / np.pi
    return app_angle.mean(), grip_angle.mean()


def select_gt_targets(
    output,
    gt_sparse_list,
    device,
    depth_img=None,
    objectness_mask=None,
    propagate=True,
    prop_radius=3,
    prop_depth_thresh=0.005,
    normalize_dirs=True,
    prediction_aware=True,
):
    """
    Target selection for the current v2 sparse GT format: (N, K, ...).

    Policy:
      - approach_dir -> global top-1 view (k=0)
      - gripper_dir/width/depth/score -> either predicted-approach-aligned best_k
        or fixed k=0 when prediction_aware=False.

    The joint discrete 300x12 orientation baseline sets prediction_aware=False
    so approach and gripper targets come from one coherent top-1 grasp.

    Returns:
      target_maps, valid_mask, confidence
    """
    B, _, H, W = output['objectness'].shape

    target_maps = {
        'graspness':   torch.zeros((B, 1, H, W), device=device),
        'approach_dir':torch.zeros((B, 3, H, W), device=device),
        'gripper_dir': torch.zeros((B, 3, H, W), device=device),
        'grasp_width': torch.zeros((B, 1, H, W), device=device),
        'grasp_depth': torch.zeros((B, 1, H, W), device=device),
        'grasp_score': torch.zeros((B, 1, H, W), device=device),
    }

    valid_mask = torch.zeros((B, 1, H, W), device=device)

    for b in range(B):
        gt_data = gt_sparse_list[b]
        indices = gt_data['valid_pixel_indices']  # (N,2) [v,u]
        if indices is None or len(indices) == 0:
            continue

        v, u = indices[:, 0], indices[:, 1]

        # --- GT unpack --- (v2 format: N, K, 3 — no P dimension)
        gt_app   = gt_data['approach_dir'].to(device)   # (N, K, 3)
        gt_grip  = gt_data['gripper_dir'].to(device)    # (N, K, 3)
        gt_width = gt_data['grasp_width'].to(device)    # (N, K)
        gt_depth = gt_data['grasp_depth'].to(device)    # (N, K)
        gt_score = gt_data['grasp_score'].to(device)    # (N, K)
        gt_graspness = gt_data['graspness'].to(device)  # (N,)

        N, K_views, _ = gt_app.shape

        # --- Predictions at sparse pixels ---
        pred_app, pred_grip = None, None
        if prediction_aware:
            if 'approach_dir' not in output or 'gripper_dir' not in output:
                raise ValueError(
                    "prediction_aware_assignment=True requires dense "
                    "approach_dir and gripper_dir outputs."
                )
            pred_app  = output['approach_dir'][b, :, v, u].permute(1, 0)  # (N,3)
            pred_grip = output['gripper_dir'][b, :, v, u].permute(1, 0)   # (N,3)

            if normalize_dirs:
                pred_app  = F.normalize(pred_app,  dim=-1, eps=1e-6)
                pred_grip = F.normalize(pred_grip, dim=-1, eps=1e-6)

        # (A) approach target: best view (k=0)
        target_app_top1 = gt_app[:, 0, :]  # (N,3)

        # (B) best_k: choose view by predicted approach (or fixed k=0 if not prediction_aware)
        valid_k_mask = (gt_score > 0)  # (N,K)
        has_valid_k = valid_k_mask.any(dim=1)  # (N,)
        if prediction_aware:
            gt_app_unique = F.normalize(gt_app, dim=-1, eps=1e-6) if normalize_dirs else gt_app  # (N,K,3)
            sim_app = torch.sum(pred_app.unsqueeze(1) * gt_app_unique, dim=-1)  # (N,K)
            sim_app = sim_app.masked_fill(~valid_k_mask, -1e9)
            best_k = torch.argmax(sim_app, dim=1)  # (N,)
        else:
            best_k = torch.zeros(N, dtype=torch.long, device=device)  # fixed top-1

        # gather best_k
        n_idx = torch.arange(N, device=device)
        score_best_k   = gt_score[n_idx, best_k]       # (N,)
        gt_grip_best_k = gt_grip[n_idx, best_k, :]     # (N,3)
        width_best_k   = gt_width[n_idx, best_k]       # (N,)
        depth_best_k   = gt_depth[n_idx, best_k]       # (N,)

        if normalize_dirs:
            gt_grip_best_k = F.normalize(gt_grip_best_k, dim=-1, eps=1e-6)

        # (C+D) gripper/score/width/depth: best_k view (v2 has no P dimension)
        target_grip_top1_under_bestk = gt_grip_best_k  # (N,3)
        has_valid_p = (score_best_k > 0)               # (N,)
        point_valid = has_valid_k & has_valid_p

        target_score_match = score_best_k   # (N,)
        target_width_match = width_best_k   # (N,)
        target_depth_match = depth_best_k   # (N,)

        # zero-out invalid points (mask will also remove them from loss, but keep tensors clean)
        if (~point_valid).any():
            target_grip_top1_under_bestk = target_grip_top1_under_bestk.clone()
            target_width_match = target_width_match.clone()
            target_depth_match = target_depth_match.clone()
            target_score_match = target_score_match.clone()

            target_grip_top1_under_bestk[~point_valid] = 0.0
            target_width_match[~point_valid] = 0.0
            target_depth_match[~point_valid] = 0.0
            target_score_match[~point_valid] = 0.0

        # ============================================================
        # (E) Scatter back
        # ============================================================
        target_maps['graspness'][b, 0, v, u] = gt_graspness

        # policy-aligned targets
        target_maps['approach_dir'][b, :, v, u] = target_app_top1.permute(1, 0)                 # (k=0,p=0)
        target_maps['gripper_dir'][b, :, v, u]  = target_grip_top1_under_bestk.permute(1, 0)    # (best_k,p_top1)
        target_maps['grasp_width'][b, 0, v, u]  = target_width_match
        target_maps['grasp_depth'][b, 0, v, u]  = target_depth_match
        target_maps['grasp_score'][b, 0, v, u]  = target_score_match

        valid_mask[b, 0, v, u] = point_valid.float()

    # --- Dense propagation ---
    confidence = None
    if propagate and depth_img is not None:
        target_maps, confidence, valid_mask, debug_info = dense_gt_propagation(
            target_maps, valid_mask, depth_img,
            objectness_mask=objectness_mask, radius=prop_radius, depth_thresh=prop_depth_thresh,
        )

    return target_maps, valid_mask, confidence

def get_loss_function(loss_config, pred, target, valid_mask=None, beta=0.1, kappa=None, confidence=None):
    loss_type = loss_config['type']
    weight = loss_config['weight']

    # --- 1. Base Loss Calculation ---
    if loss_type == 'bce_with_logits':
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    elif loss_type == 'bce':
        loss = F.binary_cross_entropy(pred, target, reduction='none')
    elif loss_type == 'smooth_l1':
        loss = F.smooth_l1_loss(pred, target, reduction='none', beta=beta)
    elif loss_type == 'l1':
        loss = F.l1_loss(pred, target, reduction='none')
    elif loss_type == 'l2':
        loss = F.mse_loss(pred, target, reduction='none')
    elif loss_type == 'symmetric_smooth_l1':
        diff = pred - target
        symm_diff = - pred - target
        loss = torch.where(torch.abs(diff) < torch.abs(symm_diff), diff, symm_diff)
        loss = F.smooth_l1_loss(loss, torch.zeros_like(loss), reduction='none', beta=beta)
    elif loss_type == 'cross_entropy':
        loss = F.cross_entropy(pred, target, reduction='none', ignore_index=-1)
    elif loss_type == 'cosine_similarity':
        # Output: (B, H, W) -> 3D Tensor
        loss = 1 - F.cosine_similarity(pred, target, dim=1, eps=1e-6)
    elif loss_type == 'symmetric_cosine_similarity':
        cos_sim = F.cosine_similarity(pred, target, dim=1, eps=1e-6)
        loss = 1 - torch.abs(cos_sim)
    elif loss_type == 'acosine_similarity':
        cos_sim = F.cosine_similarity(pred, target, dim=1, eps=1e-6)
        loss = torch.acos(torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6))
    elif loss_type == 'symmetric_acosine_similarity':
        cos_sim = F.cosine_similarity(pred, target, dim=1, eps=1e-6)
        loss = torch.acos(torch.clamp(torch.abs(cos_sim), -1 + 1e-6, 1 - 1e-6))
    
    # --- 2. NLL Losses (require kappa) ---
    elif loss_type in ['vonmf', 'symmetric_vonmf', 'angmf', 'symmetric_angmf']:
        if kappa is None:
            raise ValueError(f"Loss type '{loss_type}' requires kappa output from the model.")
        cos_sim = F.cosine_similarity(pred, target, dim=1, eps=1e-6)
        if 'symmetric' in loss_type:
            cos_sim = torch.abs(cos_sim)
        dot = torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6)
        kappa_squeezed = kappa.squeeze(1)  # (B,1,H,W) -> (B,H,W) to match dot
        if 'vonmf' in loss_type:
            loss = nll_vonmf(dot, kappa_squeezed)
        else:
            loss = nll_angmf(dot, kappa_squeezed)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    # --- 3. Apply Weighting (Mask * Confidence) Robustly ---
    
    final_weight = torch.ones_like(loss)
    
    # [Fix] Apply Valid Mask using align_and_expand
    if valid_mask is not None:
        valid_mask_aligned = align_and_expand(valid_mask.float(), loss)
        final_weight = final_weight * valid_mask_aligned

    # [Fix] Apply Confidence using align_and_expand
    if confidence is not None:
        confidence_aligned = align_and_expand(confidence.float(), loss)
        final_weight = final_weight * confidence_aligned

    # --- 4. Compute Weighted Average ---
    # Sum(Loss * Weight) / Sum(Weight)
    weighted_loss_sum = (loss * final_weight).sum()
    total_weight = final_weight.sum().clamp_min(1.0)
    return (weighted_loss_sum / total_weight) * weight


def compute_all_losses(output, data, loss_configs):
    """
    Compute all losses with Dynamic Label Assignment support.
    """
    losses = {}

    # [NEW] 1. Dynamic Label Assignment Logic
    # 만약 데이터에 Top-K GT 리스트('gt_sparse')가 있다면 동적 할당 수행

    # Get propagation config from loss_configs if available
    propagate = getattr(loss_configs, 'propagate_gt', False)
    prop_radius = getattr(loss_configs, 'prop_radius', 3)
    prop_depth_thresh = getattr(loss_configs, 'prop_depth_thresh', 0.005)
    confidence_weighting = getattr(loss_configs, 'confidence_weighting', False)
    prediction_aware = getattr(loss_configs, 'prediction_aware_assignment', True)

    # Get depth image for propagation
    depth_img = data.get('depth', None)

    # Get objectness mask: propagation only within object regions
    objectness_mask = data.get('segm', None)

    best_targets, valid_mask, confidence = select_gt_targets(
        output, data['gt_sparse'], output['objectness'].device,
        depth_img=depth_img,
        objectness_mask=objectness_mask,
        propagate=propagate,
        prop_radius=prop_radius,
        prop_depth_thresh=prop_depth_thresh,
        prediction_aware=prediction_aware,
    )
    # 매칭된 픽셀만 Valid Mask로 사용
    # put best_targets into data for compatibility
    data.update(best_targets)
    data['valid_mask'] = valid_mask

    if not confidence_weighting:
        confidence = None
    else:
        data['confidence'] = confidence

    quality_cfg = getattr(loss_configs, 'quality_aware', None)
    quality_enabled = getattr(quality_cfg, 'enabled', False)
    quality_min_weight = getattr(quality_cfg, 'min_weight', 0.2)
    quality_targets = getattr(quality_cfg, 'targets', ['approach', 'gripper', 'score', 'width', 'depth'])

    quality_confidence = None
    if quality_enabled:
        quality_confidence = best_targets['grasp_score'].detach().clamp(min=quality_min_weight, max=1.0)
        if confidence is not None:
            quality_confidence = confidence * quality_confidence

    # Confidence는 propagation에서 계산된 값 사용 (없으면 None)
    # confidence = best_targets['grasp_score']
    

    # =================================================================
    # 2. Loss Calculation
    # =================================================================

  # (A) Objectness
    if 'segm' in data:
        losses['segm'] = get_loss_function(
            loss_configs.objectness,
            output['objectness'],
            (data['segm'] > 0).float()
        )

    # (B) Graspness (Invariant)
    losses['graspness'] = get_loss_function(
        loss_configs.graspness,
        output['graspness'],
        data['graspness'],
        valid_mask=valid_mask,
        confidence=confidence,
        beta=loss_configs.graspness.get('beta', 0.0)
    )

    joint_discrete_orientation = (
        '_orientation_feat' in output
        and '_orientation_weight' in output
        and '_orientation_bias' in output
    )
    if joint_discrete_orientation:
        orientation_cfg = cfg_get(loss_configs, 'orientation', None)
        if orientation_cfg is None:
            orientation_weight = (
                float(cfg_get(loss_configs.approach, 'weight', 3.0))
                + float(cfg_get(loss_configs.gripper, 'weight', 3.0))
            )
            orientation_cfg = {
                'type': 'cross_entropy',
                'weight': orientation_weight,
                'max_samples': 65536,
                'chunk_size': 4096,
            }
        if cfg_get(orientation_cfg, 'type', 'cross_entropy') == 'mse':
            losses['orientation'] = get_sparse_joint_orientation_loss_mse(
                orientation_cfg,
                output,
                data['gt_sparse'],
                output['objectness'].device,
            )
        else:
            losses['orientation'] = get_sparse_joint_orientation_loss(
                orientation_cfg,
                output,
                best_targets['approach_dir'],
                best_targets['gripper_dir'],
                valid_mask,
                confidence=quality_confidence
                if ('approach' in quality_targets or 'gripper' in quality_targets)
                else confidence,
            )
    else:
        # (C) Approach Direction (Target: Top-1 Best)
        # 네트워크가 현재 엉뚱한 곳을 보더라도, "여기가 정답이야"라고 Top-1을 가리킴
        losses['approach'] = get_loss_function(
            loss_configs.approach,
            output['approach_dir'],
            best_targets['approach_dir'], # (Top-1 Value)
            valid_mask=valid_mask,
            confidence=quality_confidence if 'approach' in quality_targets else confidence,
            kappa=output.get('approach_kappa', None)
        )

        # (D) Gripper Direction (Target: Nearest to Pred Approach)
        # 네트워크가 보는 방향(Pred App) 기준으로 가장 말이 되는 Gripper 각도와 비교
        losses['gripper'] = get_loss_function(
            loss_configs.gripper,
            output['gripper_dir'],
            best_targets['gripper_dir'], # (Nearest Value)
            valid_mask=valid_mask,
            confidence=quality_confidence if 'gripper' in quality_targets else confidence,
            kappa=output.get('gripper_kappa', None)
        )

    # (E) Score, Width, Depth (Target: same selected grasp candidate)
    discrete_attributes = all(
        key in output
        for key in (
            '_attribute_score_logits',
            '_attribute_width_logits',
            '_attribute_depth_logits',
        )
    )
    if discrete_attributes:
        losses['score'] = get_sparse_discrete_attribute_loss(
            loss_configs.score,
            output['_attribute_score_logits'],
            best_targets['grasp_score'],
            output['_grasp_score_centers'],
            valid_mask,
            confidence=quality_confidence if 'score' in quality_targets else confidence,
            full_hw=output.get('_orientation_full_hw', None),
        )
        losses['width'] = get_sparse_discrete_attribute_loss(
            loss_configs.width,
            output['_attribute_width_logits'],
            best_targets['grasp_width'],
            output['_grasp_width_centers'],
            valid_mask,
            confidence=quality_confidence if 'width' in quality_targets else confidence,
            full_hw=output.get('_orientation_full_hw', None),
        )
        losses['depth'] = get_sparse_discrete_attribute_loss(
            loss_configs.depth,
            output['_attribute_depth_logits'],
            best_targets['grasp_depth'],
            output['_grasp_depth_centers'],
            valid_mask,
            confidence=quality_confidence if 'depth' in quality_targets else confidence,
            full_hw=output.get('_orientation_full_hw', None),
        )
    else:
        losses['score'] = get_loss_function(
            loss_configs.score,
            output['grasp_score'],
            best_targets['grasp_score'],
            valid_mask=valid_mask,
            confidence=quality_confidence if 'score' in quality_targets else confidence,
            beta=loss_configs.score.get('beta', 0.1)
        )

        gt_width_normalized = best_targets['grasp_width'] * 10.0
        losses['width'] = get_loss_function(
            loss_configs.width,
            output['grasp_width'],
            gt_width_normalized,
            valid_mask=valid_mask,
            confidence=quality_confidence if 'width' in quality_targets else confidence,
            beta=loss_configs.width.get('beta', 0.0)
        )

        gt_depth_normalized = best_targets['grasp_depth'] * 25.0
        losses['depth'] = get_loss_function(
            loss_configs.depth,
            output['grasp_depth'],
            gt_depth_normalized,
            valid_mask=valid_mask,
            confidence=quality_confidence if 'depth' in quality_targets else confidence,
            beta=loss_configs.depth.get('beta', 0.0)
        )

    # Ranking Loss (Nearest GT끼리 비교)
    if 'ranking_loss' in loss_configs and loss_configs.ranking_loss.weight > 0:
        rl_cfg = loss_configs.ranking_loss
        ranking_loss_fn = PairwiseRankingLoss(
            num_samples=rl_cfg.get('num_samples', 2000),
            min_gt_diff=rl_cfg.get('min_gt_diff', 0.1),
            margin=rl_cfg.get('margin', rl_cfg.get('min_gt_diff', 0.1)),
        )
        for target in rl_cfg.targets:
            tgt_key = target if target in best_targets else 'grasp_' + target
            if tgt_key in best_targets:
                losses['ranking_' + target] = ranking_loss_fn(
                    output[tgt_key],
                    best_targets[tgt_key],
                    valid_mask
                ) * rl_cfg.weight

    # Object-wise Ranking Loss (물체별 독립 스케일 GT에 사용. e.g. graspness)
    if 'object_wise_ranking_loss' in loss_configs and loss_configs.object_wise_ranking_loss.weight > 0:
        ow_cfg = loss_configs.object_wise_ranking_loss
        segm = data.get('segm', None)
        if segm is not None:
            ow_ranking_fn = ObjectWiseRankingLoss(
                num_samples_per_obj=ow_cfg.get('num_samples_per_obj', 400),
                min_gt_diff=ow_cfg.get('min_gt_diff', 0.05),
                margin=ow_cfg.get('margin', ow_cfg.get('min_gt_diff', 0.05)),
            )
            for target in ow_cfg.targets:
                tgt_key = target if target in best_targets else 'grasp_' + target
                if tgt_key in best_targets:
                    losses['object_wise_ranking_' + target] = ow_ranking_fn(
                        output[tgt_key],
                        best_targets[tgt_key],
                        valid_mask,
                        segm,
                    ) * ow_cfg.weight


    losses['total'] = sum(v for k, v in losses.items() if not k.startswith('_'))
    losses['_pred_width_scaled'] = output['grasp_width'] * 0.1
    losses['_pred_depth_scaled'] = output['grasp_depth'] * 0.04

    return losses, valid_mask


def compute_all_metrics(output, data, valid_mask, pred_width_scaled=None, pred_depth_scaled=None):
    """
    Compute all metrics for grasp prediction.

    Args:
        output: Dict of model predictions
        data: Dict of ground truth data
        valid_mask: Boolean mask for valid pixels (B, 1, H, W)
        pred_width_scaled: Pre-scaled width predictions (optional)
        pred_depth_scaled: Pre-scaled depth predictions (optional)

    Returns:
        Dict containing all metrics
    """
    metrics = {}
    valid_pixels = valid_mask.squeeze(1)  # (B, H, W)
    
    # for key in output:
    #     print(f"Output key: {key}, shape: {output[key].shape}")
    # for key in data:
    #     if not isinstance(data[key], torch.Tensor):
    #         continue
    #     print(f"Data key: {key}, shape: {data[key].shape}")
    
    # Graspness MAE (valid_mask: B, 1, H, W)
    # Note: graspness is already sigmoid-activated in the model output (HierarchicalGraspHeadv3/v4)
    graspness_mae = torch.abs(output['graspness'] - data['graspness'])  # (B, 1, H, W)
    metrics['graspness_mae'] = (graspness_mae * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    # Score MAE (valid_mask: B, 1, H, W)
    score_mae = torch.abs(output['grasp_score'] - data['grasp_score'])  # (B, 1, H, W)
    metrics['score_mae'] = (score_mae * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    if 'approach_dir' in output and 'gripper_dir' in output:
        # Approach angle error (degrees) (valid_pixels: B, H, W)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        approach_cos = cos(output['approach_dir'], data['approach_dir'])  # (B, H, W)
        approach_angle = torch.acos(torch.clamp(approach_cos, -1, 1)) * 180 / np.pi  # (B, H, W)
        metrics['approach_angle_error'] = (approach_angle * valid_pixels.float()).sum() / (valid_pixels.float().sum() + 1e-6)

        # Gripper angle error (degrees) (valid_pixels: B, H, W)
        # Use symmetric similarity to account for 180-degree symmetry
        gripper_cos = cos(output['gripper_dir'], data['gripper_dir'])  # (B, H, W)
        gripper_cos_abs = torch.abs(gripper_cos)  # Account for 180-degree symmetry
        gripper_angle = torch.acos(torch.clamp(gripper_cos_abs, -1, 1)) * 180 / np.pi  # (B, H, W)
        metrics['gripper_angle_error'] = (gripper_angle * valid_pixels.float()).sum() / (valid_pixels.float().sum() + 1e-6)
    elif '_orientation_feat' in output:
        app_err, grip_err = _sample_joint_orientation_angle_metrics(output, data, valid_mask)
        metrics['approach_angle_error'] = app_err
        metrics['gripper_angle_error'] = grip_err
    else:
        zero = output['objectness'].new_tensor(0.0)
        metrics['approach_angle_error'] = zero
        metrics['gripper_angle_error'] = zero

    # Width MAE (scaled prediction) (valid_mask: B, 1, H, W)
    if pred_width_scaled is None:
        pred_width_scaled = output['grasp_width'] * 0.1
    width_mae = torch.abs(pred_width_scaled - data['grasp_width'])  # (B, 1, H, W)
    metrics['width_mae'] = (width_mae * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    # Depth MAE (scaled prediction) (valid_mask: B, 1, H, W)
    if pred_depth_scaled is None:
        pred_depth_scaled = output['grasp_depth'] * 0.04
    depth_mae = torch.abs(pred_depth_scaled - data['grasp_depth'])  # (B, 1, H, W)
    metrics['depth_mae'] = (depth_mae * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return metrics
