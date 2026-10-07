"""
GraspNet-1Billion 데이터셋을 사용하여 픽셀 단위 grasp ground truth를 생성하는 스크립트 (v3)

v3 변경사항:
- 모든 view에 대해 모든 in-plane rotation (angle)의 gripper_dir를 저장
- 저장 형태: (Np, K, P, 3) where K=top views, P=angles (score 기준 정렬됨)
- width, depth, score도 (Np, K, P) 형태로 저장
- argsort 후 저장하여 score 기준 정렬
"""
import os
import argparse
import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from tqdm import tqdm
import cv2

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_utils import *
from graspnetAPI.utils.utils import generate_views
from graspnetAPI.grasp import GraspGroup

from graspnetAPI import GraspNet

import copy

# 상수 정의
NUM_VIEWS = 300      # V: view 개수
NUM_ANGLES = 12      # A: in-plane rotation 개수
NUM_DEPTHS = 3       # D: depth 개수
MAX_WIDTH = 0.10     # grasp 최대 너비
GRASP_HEIGHT = 0.02  # grasp 높이
TOP_K_VIEWS = 300    # 저장할 top-k view 개수
TOP_P_ANGLES = 12    # 저장할 top-p angle 개수 (전체 저장)

# 커맨드라인 인자 파싱
parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/data/Grasp/GraspNet-1Billion',
                    help='GraspNet 데이터셋 루트 경로')
parser.add_argument('--camera', default='kinect',
                    help='카메라 타입 [realsense/kinect]')
parser.add_argument('--scene_id', type=int, default=37,
                    help='처리할 씬 ID')
parser.add_argument('--vis', action='store_true',
                    help='시각화 여부')

cfgs = parser.parse_args()

root = cfgs.root


def load_collision_labels(root, scene_id):
    """씬의 collision label을 로드합니다."""
    collision_path = os.path.join(
        root, 'collision_label',
        f'scene_{str(scene_id).zfill(4)}',
        'collision_labels.npz'
    )
    labels = np.load(collision_path, mmap_mode='r')

    collision_labels = {}
    for i in range(len(labels)):
        collision_labels[i] = labels[f'arr_{i}']
    return collision_labels


def load_grasp_labels(root, obj_indices):
    """유효한 객체들의 grasp label을 로드합니다."""
    num_objects = 88
    valid_obj_idxs = []
    grasp_labels = {}

    for obj_id in tqdm(range(num_objects), desc='Grasp label 로딩 중...'):
        if obj_id not in obj_indices:
            continue

        aligned_obj_id = obj_id + 1
        valid_obj_idxs.append(aligned_obj_id)

        label_path = os.path.join(root, 'grasp_label', f'{str(obj_id).zfill(3)}_labels.npz')
        label = np.load(label_path, mmap_mode='r')

        grasp_labels[aligned_obj_id] = (
            label['points'].astype(np.float32),
            label['offsets'].astype(np.float32),
            label['scores'].astype(np.float32)
        )

    return valid_obj_idxs, grasp_labels


def project_3d_to_2d(points, intrinsic):
    """3D 포인트를 2D 픽셀 좌표로 투영합니다."""
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy

    pixels = np.stack([u, v], axis=1)
    return pixels


def load_scene_data(root, scene_id, ann_id, camera_type):
    """씬 데이터를 로드합니다."""
    scene_dir = os.path.join(root, f"scenes/scene_{str(scene_id).zfill(4)}/{camera_type}")

    rgb_path = os.path.join(scene_dir, f"rgb/{str(ann_id).zfill(4)}.png")
    depth_path = os.path.join(scene_dir, f"depth/{str(ann_id).zfill(4)}.png")
    segment_path = os.path.join(scene_dir, f"label/{str(ann_id).zfill(4)}.png")
    meta_path = os.path.join(scene_dir, f"meta/{str(ann_id).zfill(4)}.mat")

    rgb = np.array(Image.open(rgb_path)) / 255.0
    depth = np.array(Image.open(depth_path))
    seg = np.array(Image.open(segment_path))

    meta = scio.loadmat(meta_path)
    obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
    poses = meta['poses']
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    return rgb, depth, seg, obj_idxs, poses, intrinsic, factor_depth


def compute_valid_grasp_mask(fric_coefs, collision, grasp_offsets, max_width):
    """유효한 grasp을 선택합니다."""
    return (fric_coefs > 0) & (~collision) & (grasp_offsets[:, :, :, :, 2] <= max_width)


def compute_grasp_scores(fric_coefs, valid_mask):
    """friction coefficient를 기반으로 grasp score를 계산합니다."""
    scores = np.zeros_like(fric_coefs, dtype=np.float32)
    scores[valid_mask] = 1.1 - fric_coefs[valid_mask]
    return scores


def compute_graspness(scores):
    """각 grasp point의 graspness를 계산합니다."""
    scores = torch.from_numpy(scores).float().cuda()
    graspness = torch.sum(scores.reshape(scores.shape[0], -1) > 0, dim=-1).float()

    mask = graspness > 0
    if not mask.any():
        return graspness.cpu().numpy()

    nonzero = graspness[mask]
    scaled = (nonzero - torch.min(nonzero)) / (torch.max(nonzero) - torch.min(nonzero) + 1e-5)

    graspness = torch.zeros_like(graspness)
    graspness[mask] = scaled
    return graspness.cpu().numpy()


if __name__ == "__main__":
    scene_id = cfgs.scene_id
    camera_type = cfgs.camera

    print("Collision label 로딩 중...")
    collision_labels = load_collision_labels(root, scene_id)
    print("Grasp label 로딩 중...")
    g = GraspNet(root, camera_type)
    obj_indices = g.getObjIds(scene_id)
    valid_obj_idxs, grasp_labels = load_grasp_labels(root, obj_indices)

    for ann_id in tqdm(range(256)):
        
        # Save as npz
        output_path = os.path.join(
            root, 'pixelwise_grasp_gt_top300', camera_type,
            f'scene_{str(scene_id).zfill(4)}_ann_{str(ann_id).zfill(4)}.npz'
        )
        if os.path.exists(output_path):
            continue
        
        
        rgb, depth, seg, obj_idxs, poses, intrinsic, factor_depth = load_scene_data(
            root, scene_id, ann_id, camera_type
        )
        print(f"씬 {scene_id}, 어노테이션 {ann_id} 처리 중...")

        camera_info = CameraInfo(
            1280.0, 720.0,
            intrinsic[0][0], intrinsic[1][1],
            intrinsic[0][2], intrinsic[1][2],
            factor_depth
        )
        cloud_np = create_point_cloud_from_depth_image(depth, camera_info, organized=True).astype(np.float32)
        cloud = torch.from_numpy(cloud_np).cuda()

        template_views = generate_views(NUM_VIEWS)

        graspness_map = np.zeros(depth.shape, dtype=np.float32)

        # v3: 모든 angle을 저장하기 위한 데이터 구조
        all_top_k_data = {
            'trans': [],
            'approach_dirs': [],   # (Np, K, P, 3)
            'gripper_dirs': [],    # (Np, K, P, 3)
            'widths': [],          # (Np, K, P)
            'depths': [],          # (Np, K, P)
            'scores': []           # (Np, K, P)
        }

        for i, obj_idx in enumerate(obj_idxs):
            g_points, g_offsets, fric_coefs = copy.deepcopy(grasp_labels[obj_idx])
            collision = copy.deepcopy(collision_labels[i])
            pose = poses[:, :, i]

            obj_mask = (seg == obj_idx)
            obj_cloud = cloud[obj_mask]
            if obj_cloud.shape[0] == 0:
                continue
            visible_grasp, visible_cloud = remove_invisible_grasp_points(
                obj_cloud, g_points, pose, th=0.01
            )

            g_points = g_points[visible_grasp]      # (N, 3)
            g_offsets = g_offsets[visible_grasp]    # (N, V, A, D, 3)
            fric_coefs = fric_coefs[visible_grasp]  # (N, V, A, D)
            collision = collision[visible_grasp]    # (N, V, A, D)

            valid_grasp_mask = compute_valid_grasp_mask(
                fric_coefs, collision, g_offsets, MAX_WIDTH
            )
            if np.sum(valid_grasp_mask) == 0:
                continue
            scores = compute_grasp_scores(fric_coefs, valid_grasp_mask)  # (N, V, A, D)

            Np = scores.shape[0]

            # Select top-k views by mean score
            view_score = np.mean(scores.reshape(Np, NUM_VIEWS, -1), axis=-1)  # (Np, V)
            top_k_view_inds = np.argsort(view_score, axis=1)[:, ::-1][:, :TOP_K_VIEWS]  # (Np, K)

            # For each top-k view, get scores for all angles: (Np, K, A, D)
            top_k_view_score = scores[np.arange(Np)[:, None], top_k_view_inds, :, :]        # (Np, K, A, D)
            top_k_view_offsets = g_offsets[np.arange(Np)[:, None], top_k_view_inds, :, :, :] # (Np, K, A, D, 3)

            # For each angle, select best depth
            # angle_depth_score: (Np, K, A) - max over D
            angle_depth_score = np.max(top_k_view_score, axis=-1)  # (Np, K, A)
            best_depth_inds = np.argmax(top_k_view_score, axis=-1)  # (Np, K, A)

            # Sort angles by score for each view (descending)
            angle_sort_inds = np.argsort(angle_depth_score, axis=-1)[:, :, ::-1]  # (Np, K, A) sorted indices
            top_p_angle_inds = angle_sort_inds[:, :, :TOP_P_ANGLES]  # (Np, K, P)

            # Gather scores for top-p angles: (Np, K, P)
            top_kp_scores = np.take_along_axis(angle_depth_score, top_p_angle_inds, axis=-1)  # (Np, K, P)

            # Gather best depth indices for top-p angles
            top_p_depth_inds = np.take_along_axis(best_depth_inds, top_p_angle_inds, axis=-1)  # (Np, K, P)

            # Gather offsets for top-p angles with best depth: (Np, K, P, 3)
            Np_idx = np.arange(Np)[:, None, None]       # (Np, 1, 1)
            K_idx = np.arange(TOP_K_VIEWS)[None, :, None]  # (1, K, 1)

            top_kp_offsets = top_k_view_offsets[
                Np_idx,
                K_idx,
                top_p_angle_inds,
                top_p_depth_inds,
                :
            ]  # (Np, K, P, 3)

            top_kp_widths = top_kp_offsets[:, :, :, 2]  # (Np, K, P)
            top_kp_depths = top_kp_offsets[:, :, :, 1]  # (Np, K, P)
            top_kp_angles = top_kp_offsets[:, :, :, 0]  # (Np, K, P)

            # Compute approach and gripper directions for all (K, P) combinations
            all_views = np.tile(template_views, [Np, 1, 1])  # (Np, V, 3)
            top_k_views = all_views[np.arange(Np)[:, None], top_k_view_inds, :]  # (Np, K, 3)

            # For each (k, p) pair, compute rotation matrix
            top_kp_approach_dirs = np.zeros((Np, TOP_K_VIEWS, TOP_P_ANGLES, 3), dtype=np.float32)
            top_kp_gripper_dirs = np.zeros((Np, TOP_K_VIEWS, TOP_P_ANGLES, 3), dtype=np.float32)

            for k in range(TOP_K_VIEWS):
                for p in range(TOP_P_ANGLES):
                    views_kp = top_k_views[:, k, :]       # (Np, 3)
                    angles_kp = top_kp_angles[:, k, p]    # (Np,)
                    rotations_kp = batch_viewpoint_params_to_matrix(-views_kp, angles_kp)  # (Np, 3, 3)
                    rotations_kp = np.matmul(pose[:3, :3], rotations_kp)  # (Np, 3, 3)

                    top_kp_approach_dirs[:, k, p, :] = rotations_kp[:, :, 0]  # (Np, 3)
                    top_kp_gripper_dirs[:, k, p, :] = rotations_kp[:, :, 1]   # (Np, 3)

            trans = transform_point_cloud(g_points, pose)  # (Np, 3)

            graspness = compute_graspness(scores)
            projected_pixels = project_3d_to_2d(trans, intrinsic)
            projected_pixels = np.round(projected_pixels).astype(np.int32)
            for j in range(projected_pixels.shape[0]):
                u, v = projected_pixels[j]
                if u < 0 or u >= depth.shape[1] or v < 0 or v >= depth.shape[0]:
                    continue
                graspness_map[v, u] = max(graspness_map[v, u], graspness[j])

            # Store all data
            all_top_k_data['trans'].append(trans)                        # (Np, 3)
            all_top_k_data['approach_dirs'].append(top_kp_approach_dirs) # (Np, K, P, 3)
            all_top_k_data['gripper_dirs'].append(top_kp_gripper_dirs)   # (Np, K, P, 3)
            all_top_k_data['widths'].append(top_kp_widths)               # (Np, K, P)
            all_top_k_data['depths'].append(top_kp_depths)               # (Np, K, P)
            all_top_k_data['scores'].append(top_kp_scores)               # (Np, K, P)

        # Concatenate all objects
        print("\n=== Processing top-k grasp data ===")

        if len(all_top_k_data['trans']) == 0:
            print(f"No valid grasps for scene {scene_id}, ann {ann_id}")
            continue

        all_trans = np.concatenate(all_top_k_data['trans'], axis=0)                  # (N_total, 3)
        all_approach_dirs = np.concatenate(all_top_k_data['approach_dirs'], axis=0)  # (N_total, K, P, 3)
        all_gripper_dirs = np.concatenate(all_top_k_data['gripper_dirs'], axis=0)    # (N_total, K, P, 3)
        all_widths = np.concatenate(all_top_k_data['widths'], axis=0)                # (N_total, K, P)
        all_depths = np.concatenate(all_top_k_data['depths'], axis=0)                # (N_total, K, P)
        all_scores = np.concatenate(all_top_k_data['scores'], axis=0)                # (N_total, K, P)

        num_grasps = all_trans.shape[0]
        print(f"Total grasp points: {num_grasps}")

        # Filter valid grasps (score > 0 for best view, best angle)
        valid_grasp_mask = all_scores[:, 0, 0] > 0
        all_trans = all_trans[valid_grasp_mask]
        all_approach_dirs = all_approach_dirs[valid_grasp_mask]
        all_gripper_dirs = all_gripper_dirs[valid_grasp_mask]
        all_widths = all_widths[valid_grasp_mask]
        all_depths = all_depths[valid_grasp_mask]
        all_scores = all_scores[valid_grasp_mask]

        num_grasps = all_trans.shape[0]
        print(f"Valid grasp points (after filtering): {num_grasps}")

        # Filter out upward-facing grasps (check best view, best angle)
        best_approach_dirs = all_approach_dirs[:, 0, 0, :]  # (N, 3)
        downward_mask = best_approach_dirs[:, 2] >= 0

        all_trans = all_trans[downward_mask]
        all_approach_dirs = all_approach_dirs[downward_mask]
        all_gripper_dirs = all_gripper_dirs[downward_mask]
        all_widths = all_widths[downward_mask]
        all_depths = all_depths[downward_mask]
        all_scores = all_scores[downward_mask]

        num_grasps = all_trans.shape[0]
        print(f"Valid grasp points (after direction filtering): {num_grasps}")

        # Fix gripper direction ambiguity - make gripper point downward
        for k in range(TOP_K_VIEWS):
            for p in range(TOP_P_ANGLES):
                gripper_z = all_gripper_dirs[:, k, p, 2]
                flip_mask = gripper_z > 0
                all_gripper_dirs[flip_mask, k, p, :] *= -1

        # Convert to torch for nearest neighbor
        grasp_centers = torch.from_numpy(all_trans.astype(np.float32)).cuda()

        valid_mask = (graspness_map > 0) & (depth > 0)
        valid_pixels = np.argwhere(valid_mask)

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        if isinstance(factor_depth, np.ndarray):
            factor_depth = float(factor_depth.flatten()[0])

        v_coords, u_coords = valid_pixels[:, 0], valid_pixels[:, 1]
        z_vals = depth[v_coords, u_coords] / factor_depth
        x_vals = (u_coords - cx) * z_vals / fx
        y_vals = (v_coords - cy) * z_vals / fy

        pixel_3d_points = np.stack([x_vals, y_vals, z_vals], axis=1).astype(np.float32)
        pixel_3d_points_torch = torch.from_numpy(pixel_3d_points).cuda()

        # Nearest neighbor assignment
        chunk_size = 10000
        num_pixels = len(valid_pixels)
        nearest_indices = torch.zeros(num_pixels, dtype=torch.long, device='cuda')

        for i in tqdm(range(0, num_pixels, chunk_size), desc="NN assignment"):
            end_i = min(i + chunk_size, num_pixels)
            pixel_chunk = pixel_3d_points_torch[i:end_i]
            dists = torch.cdist(pixel_chunk, grasp_centers, p=2)
            nearest_indices[i:end_i] = torch.argmin(dists, dim=1)

        nearest_indices = nearest_indices.cpu().numpy()

        # Populate sparse data
        H, W = depth.shape
        num_valid = len(valid_pixels)

        # v3: (N_valid, K, P, ...) 형태로 저장
        sparse_approach_dir = np.zeros((num_valid, TOP_K_VIEWS, TOP_P_ANGLES, 3), dtype=np.float32)
        sparse_gripper_dir = np.zeros((num_valid, TOP_K_VIEWS, TOP_P_ANGLES, 3), dtype=np.float32)
        sparse_grasp_score = np.zeros((num_valid, TOP_K_VIEWS, TOP_P_ANGLES), dtype=np.float32)
        sparse_grasp_width = np.zeros((num_valid, TOP_K_VIEWS, TOP_P_ANGLES), dtype=np.float32)
        sparse_grasp_depth = np.zeros((num_valid, TOP_K_VIEWS, TOP_P_ANGLES), dtype=np.float32)
        sparse_graspness = np.zeros((num_valid,), dtype=np.float32)

        for idx, (v, u) in enumerate(valid_pixels):
            grasp_idx = nearest_indices[idx]
            sparse_approach_dir[idx] = all_approach_dirs[grasp_idx]  # (K, P, 3)
            sparse_gripper_dir[idx] = all_gripper_dirs[grasp_idx]    # (K, P, 3)
            sparse_grasp_width[idx] = all_widths[grasp_idx]          # (K, P)
            sparse_grasp_score[idx] = all_scores[grasp_idx]          # (K, P)
            sparse_grasp_depth[idx] = all_depths[grasp_idx]          # (K, P)
            sparse_graspness[idx] = graspness_map[v, u]

        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.savez(
            output_path,
            # Sparse data (only valid pixels)
            valid_pixel_indices=valid_pixels.astype(np.int16),       # (N_valid, 2) - [v, u]
            graspness=sparse_graspness.astype(np.float16),           # (N_valid,)
            grasp_score=sparse_grasp_score.astype(np.float16),       # (N_valid, K, P)
            grasp_width=sparse_grasp_width.astype(np.float16),       # (N_valid, K, P)
            grasp_depth=sparse_grasp_depth.astype(np.float16),       # (N_valid, K, P)
            approach_dir=sparse_approach_dir.astype(np.float16),     # (N_valid, K, P, 3)
            gripper_dir=sparse_gripper_dir.astype(np.float16),       # (N_valid, K, P, 3)
            # Metadata
            image_shape=np.array([H, W], dtype=np.int16),
            xyz=cloud_np.astype(np.float16),                         # (H, W, 3)
            top_k=TOP_K_VIEWS,
            top_p=TOP_P_ANGLES
        )

        print(f"Saved to {output_path}")
        print(f"  - approach_dir shape: {sparse_approach_dir.shape}")
        print(f"  - gripper_dir shape: {sparse_gripper_dir.shape}")
        print(f"  - grasp_score shape: {sparse_grasp_score.shape}")

        if cfgs.vis:
            print("\n=== Saving visualizations ===")
            vis_output_dir = os.path.join('pixelwise_grasp_gt_v3_vis', f'scene_{str(scene_id).zfill(4)}')
            os.makedirs(vis_output_dir, exist_ok=True)

            # Best view, best angle visualization (k=0, p=0)
            valid_mask_img = np.zeros((H, W), dtype=bool)
            valid_mask_img[valid_pixels[:, 0], valid_pixels[:, 1]] = True

            approach_dir_map = np.zeros((H, W, 3), dtype=np.float32)
            gripper_dir_map = np.zeros((H, W, 3), dtype=np.float32)
            grasp_score_map = np.zeros((H, W), dtype=np.float32)

            for idx, (v, u) in enumerate(valid_pixels):
                approach_dir_map[v, u] = sparse_approach_dir[idx, 0, 0]
                gripper_dir_map[v, u] = sparse_gripper_dir[idx, 0, 0]
                grasp_score_map[v, u] = sparse_grasp_score[idx, 0, 0]

            # RGB
            rgb_vis = (rgb * 255).astype(np.uint8)
            rgb_img = Image.fromarray(rgb_vis)
            rgb_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_rgb.png'))

            # Approach direction
            approach_dir_vis = (approach_dir_map + 1) / 2.0
            approach_dir_vis = (approach_dir_vis * 255).astype(np.uint8)
            approach_dir_vis[~valid_mask_img] = [0, 0, 0]
            approach_dir_img = Image.fromarray(approach_dir_vis)
            approach_dir_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_approach_dir.png'))

            # Gripper direction
            gripper_dir_vis = (gripper_dir_map + 1) / 2.0
            gripper_dir_vis = (gripper_dir_vis * 255).astype(np.uint8)
            gripper_dir_vis[~valid_mask_img] = [0, 0, 0]
            gripper_dir_img = Image.fromarray(gripper_dir_vis)
            gripper_dir_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_gripper_dir.png'))

            # Score map
            grasp_score_vis = (grasp_score_map * 255).astype(np.uint8)
            grasp_score_vis_colored = cv2.applyColorMap(grasp_score_vis, cv2.COLORMAP_JET)
            grasp_score_vis_colored[~valid_mask_img] = [0, 0, 0]
            grasp_score_img = Image.fromarray(cv2.cvtColor(grasp_score_vis_colored, cv2.COLOR_BGR2RGB))
            grasp_score_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_grasp_score.png'))

            print(f"Saved visualizations to: {vis_output_dir}")
