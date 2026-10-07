"""
GraspNet-1Billion 데이터셋을 사용하여 픽셀 단위 grasp ground truth를 생성하는 스크립트
"""
import os
import argparse
import numpy as np
import scipy.io as scio
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_utils import *
from graspnetAPI.utils.utils import generate_views
from graspnetAPI.grasp import GraspGroup, Grasp

from graspnetAPI import GraspNet

import copy

def get_gripper_control_points():
    """Get control points for the gripper.

    """
    # default gripper width is 0.14m
    control_points = np.array([[-0.06, 0, 0], [-0.02, 0, 0], 
                               [-0.02, -0.04, 0], [-0.02, 0.04, 0],
                                [0.02, -0.04, 0], [0.02, 0.04, 0]])
    return control_points


def create_arrow_mesh(origin, direction, length=0.05, color=[0, 1, 0]):
    """Create an arrow mesh in Open3D.

    Arrow starts at origin and points along direction with specified length.

    Args:
        origin: (3,) starting point of arrow
        direction: (3,) direction vector (will be normalized)
        length: length of the arrow (from origin to tip)
        color: RGB color [0-1]

    Returns:
        arrow mesh (cylinder + cone)
    """
    # Normalize direction
    direction = np.array(direction)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Arrow parameters
    cylinder_radius = length * 0.02
    cylinder_height = length * 0.7
    cone_radius = length * 0.04
    cone_height = length * 0.3

    # Create cylinder (arrow shaft) - centered at origin initially
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=cylinder_radius,
        height=cylinder_height
    )
    cylinder.paint_uniform_color(color)

    # Create cone (arrow head) - centered at origin initially
    cone = o3d.geometry.TriangleMesh.create_cone(
        radius=cone_radius,
        height=cone_height
    )
    cone.paint_uniform_color(color)

    # Position cone at the end of cylinder
    cone.translate([0, 0, cylinder_height / 2 + cone_height / 2])

    # Combine cylinder and cone
    arrow = cylinder + cone

    # Rotate arrow to align with direction
    # Default arrow points along +Z axis, rotate to desired direction
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm > 1e-6:  # Need to rotate
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))

        # Rodrigues rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        arrow.rotate(R, center=[0, 0, 0])
    elif np.dot(z_axis, direction) < 0:  # Points opposite direction
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        arrow.rotate(R, center=[0, 0, 0])

    # Shift arrow so it starts at origin (not centered at origin)
    # After rotation, the arrow spans from -cylinder_height/2 to cylinder_height/2 + cone_height
    # along the direction vector. We want it to start at 0, so shift along direction
    # shift_distance = cylinder_height / 2
    # arrow.translate(direction * shift_distance)

    # Now translate to the actual origin point
    arrow.translate(origin)

    return arrow

# 상수 정의
VALID_OBJ_INDICES = [0, 2, 5, 7, 8, 9, 11, 14, 15, 17, 18, 20, 21, 22, 26, 27, 29, 30,
                     34, 36, 37, 38, 40, 41, 43, 44, 46, 48, 51, 52, 56, 57, 58, 60,
                     61, 62, 63, 66, 69, 70]

# Grasp 파라미터
NUM_VIEWS = 300      # V: view 개수
NUM_ANGLES = 12      # A: in-plane rotation 개수
NUM_DEPTHS = 3       # D: depth 개수
MAX_WIDTH = 0.10     # grasp 최대 너비
GRASP_HEIGHT = 0.02  # grasp 높이
TOP_K_VIEWS = 300      # 저장할 top-k view 개수

# 커맨드라인 인자 파싱
# 커맨드라인 인자 파싱
parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/data/Grasp/GraspNet-1Billion',
                    help='GraspNet 데이터셋 루트 경로')
parser.add_argument('--camera', default='realsense',
                    help='카메라 타입 [realsense/kinect]')
parser.add_argument('--scene_id', type=int, default=37,
                    help='처리할 씬 ID')
parser.add_argument('--ann_id', type=int, default=0,
                    help='어노테이션 ID')
parser.add_argument('--vis', action='store_true',
                    help='시각화 여부')

cfgs = parser.parse_args()

# 데이터 경로 설정
root = cfgs.root
grasp_label_folder = os.path.join(root, "grasp_label")
scenes_folder = os.path.join(root, "scenes")
collision_label_folder = os.path.join(root, "collision_label")

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
        # 유효한 객체만 처리
        if obj_id not in obj_indices:
            continue

        # label png와 정렬하기 위해 +1
        aligned_obj_id = obj_id + 1
        valid_obj_idxs.append(aligned_obj_id)

        # grasp label 로드
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
    """씬 데이터를 로드합니다 (RGB, depth, segmentation, metadata)."""
    scene_dir = os.path.join(root, f"scenes/scene_{str(scene_id).zfill(4)}/{camera_type}")

    rgb_path = os.path.join(scene_dir, f"rgb/{str(ann_id).zfill(4)}.png")
    depth_path = os.path.join(scene_dir, f"depth/{str(ann_id).zfill(4)}.png")
    segment_path = os.path.join(scene_dir, f"label/{str(ann_id).zfill(4)}.png")
    meta_path = os.path.join(scene_dir, f"meta/{str(ann_id).zfill(4)}.mat")

    # 이미지 로드
    rgb = np.array(Image.open(rgb_path)) / 255.0
    depth = np.array(Image.open(depth_path))
    seg = np.array(Image.open(segment_path))

    # 메타데이터 로드
    meta = scio.loadmat(meta_path)
    obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
    poses = meta['poses']
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    return rgb, depth, seg, obj_idxs, poses, intrinsic, factor_depth


def compute_valid_grasp_mask(fric_coefs, collision, grasp_offsets, max_width):
    """유효한 grasp을 선택합니다 (high friction, no collision, within max width)."""
    return (fric_coefs > 0) & (~collision) & (grasp_offsets[:, :, :, :, 2] <= max_width)


def compute_grasp_scores(fric_coefs, valid_mask):
    """friction coefficient를 기반으로 grasp score를 계산합니다."""
    scores = np.zeros_like(fric_coefs, dtype=np.float32)
    scores[valid_mask] = 1.1 - fric_coefs[valid_mask]
    return scores


def compute_graspness(scores):
    """각 grasp point의 graspness를 계산합니다 (adaptive scaling)."""
    # 유효한 grasp 개수 기반으로 graspness 계산
    scores = torch.from_numpy(scores).float().cuda()
    graspness = torch.sum(scores.reshape(scores.shape[0], -1) > 0, dim=-1).float()

    mask = graspness > 0
    if not mask.any():
        return graspness

    # Min-max normalization 후 [0.1, 1.0] 범위로 스케일링
    nonzero = graspness[mask]
    scaled = (nonzero - torch.min(nonzero)) / (torch.max(nonzero) - torch.min(nonzero) + 1e-5)

    graspness = torch.zeros_like(graspness)
    graspness[mask] = scaled #* 0.9 + 0.1
    graspness = graspness.cpu().numpy()

    return graspness

def convert_to_grasp(approach_dir, gripper_dir, width, score, center, depth=0.02, height=0.02, object_id=0):
    """주어진 파라미터로부터 Grasp 객체를 생성합니다."""
    R = np.eye(3)
    R[:3, 0] = approach_dir / np.linalg.norm(approach_dir)
    R[:3, 1] = gripper_dir / np.linalg.norm(gripper_dir)
    grasp_y = np.cross(R[:3, 1], R[:3, 0])
    R[:3, 2] = grasp_y / np.linalg.norm(grasp_y)

    grasp_array = np.array([score, width, height, depth, *R.reshape(9), *center, object_id], dtype=np.float32)
    grasp = Grasp()
    grasp.grasp_array = grasp_array
    return grasp



if __name__ == "__main__":
    # 파라미터 설정
    scene_id = cfgs.scene_id
    camera_type = cfgs.camera
    # 레이블 로드 (씬 전체에 대해 한 번만 수행)
    print("Collision label 로딩 중...")
    collision_labels = load_collision_labels(root, scene_id)
    print("Grasp label 로딩 중...")
    g = GraspNet(root, camera_type)
    obj_indices = g.getObjIds(scene_id)
    valid_obj_idxs, grasp_labels = load_grasp_labels(root, obj_indices)

    for ann_id in tqdm(range(256)):
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


        # 카메라 정보 및 포인트 클라우드 생성
        camera_info = CameraInfo(
            1280.0, 720.0,
            intrinsic[0][0], intrinsic[1][1],
            intrinsic[0][2], intrinsic[1][2],
            factor_depth
        )
        cloud_np = create_point_cloud_from_depth_image(depth, camera_info, organized=True).astype(np.float32)
        cloud = torch.from_numpy(cloud_np).cuda()

        template_views = generate_views(NUM_VIEWS)

        # 각 객체별로 grasp 처리
        grasp_group = GraspGroup()
        graspness_map = np.zeros(depth.shape, dtype=np.float32)

        # Initialize top-k data storage
        all_top_k_data = {
            'trans': [],
            'approach_dirs': [],
            'gripper_dirs': [],
            'widths': [],
            'depths': [],
            'scores': []
        }

        for i, obj_idx in enumerate(obj_idxs):
            # Grasp label 로드
            g_points, g_offsets, fric_coefs = copy.deepcopy(grasp_labels[obj_idx])
            collision = copy.deepcopy(collision_labels[i])
            pose = poses[:, :, i]

            # 1) 객체 segmentation mask 생성
            obj_mask = (seg == obj_idx)
            obj_pixels = np.argwhere(obj_mask)  # (N_obj, 2) — [v, u]

            # 2) 보이지 않는 grasp point 제거
            obj_cloud = cloud[obj_mask]
            if obj_cloud.shape[0] == 0:
                continue
            visible_grasp, visible_cloud = remove_invisible_grasp_points(
                obj_cloud, g_points, pose, th=0.01
            )

            # visible grasp만 남기기
            g_points = g_points[visible_grasp]  # (N, 3)
            g_offsets = g_offsets[visible_grasp]  # (N, V, A, D, 3)
            fric_coefs = fric_coefs[visible_grasp]  # (N, V, A, D)
            collision = collision[visible_grasp]  # (N, V, A, D)

            # 5) 유효한 grasp 선택 및 score 계산
            valid_grasp_mask = compute_valid_grasp_mask(
                fric_coefs, collision, g_offsets, MAX_WIDTH
            )
            if np.sum(valid_grasp_mask) == 0:
                continue
            scores = compute_grasp_scores(fric_coefs, valid_grasp_mask) # (N, V, A, D)
        
            # select by view, angle, depth
            # select top-k views
            Np = scores.shape[0]
            view_score = np.mean(scores.reshape(Np, NUM_VIEWS, -1), axis=-1)  # (Np, V)
            top_k_view_inds = np.argsort(view_score, axis=1)[:, ::-1][:, :TOP_K_VIEWS]  # (Np, K)

            # For each top-k view, select best angle and depth
            # top_k_view_score: (Np, K, A, D)
            top_k_view_score = scores[np.arange(Np)[:, None], top_k_view_inds, :, :]  # (Np, K, A, D)

            # select by angle for each view
            angle_score = np.mean(top_k_view_score, axis=-1)  # (Np, K, A)
            top_k_angle_inds = np.argmax(angle_score, axis=-1)  # (Np, K)

            # Gather best angle scores: (Np, K, D)
            top_k_view_angle_score = top_k_view_score[
                np.arange(Np)[:, None],
                np.arange(TOP_K_VIEWS)[None, :],
                top_k_angle_inds,
                :
            ]  # (Np, K, D)

            # select by depth for each view
            top_k_depth_inds = np.argmax(top_k_view_angle_score, axis=-1)  # (Np, K)

            # Get final scores and offsets for each top-k view
            top_k_g_offsets = g_offsets[
                np.arange(Np)[:, None],
                top_k_view_inds,
                top_k_angle_inds,
                top_k_depth_inds,
                :
            ]  # (Np, K, 3)

            top_k_grasp_score = top_k_view_angle_score[
                np.arange(Np)[:, None],
                np.arange(TOP_K_VIEWS)[None, :],
                top_k_depth_inds
            ]  # (Np, K)

            # For grasp_group, use best view (k=0) for backward compatibility
            best_g_offsets = top_k_g_offsets[:, 0, :]  # (Np, 3)
            grasp_score = top_k_grasp_score[:, 0:1]  # (Np, 1)

            angles = best_g_offsets[:, 0:1] # (Np, 1)
            depths = best_g_offsets[:, 1:2] # (Np, 1)
            widths = best_g_offsets[:, 2:3] # (Np, 1)
            heights = GRASP_HEIGHT * np.ones_like(widths)

            views = np.tile(template_views, [Np, 1, 1])  # (Np, V, 3)
            best_view_inds = top_k_view_inds[:, 0]  # (Np,)
            views = views[np.arange(Np), best_view_inds, :]  # (Np, 3)
            rotations = batch_viewpoint_params_to_matrix(-views, angles[:,0])  # (Np, 3, 3)
            rotations = np.matmul(pose[:3, :3], rotations)
            rotations = rotations.reshape((-1,9))

            trans = transform_point_cloud(g_points, pose)  # (Np, 3)
            object_ids = obj_idx * np.ones((Np,1), dtype=np.int32)

            obj_grasp_array = np.hstack([grasp_score, widths, heights, depths, rotations, trans, object_ids]).astype(np.float32)
            grasp_group.grasp_group_array = np.concatenate((grasp_group.grasp_group_array, obj_grasp_array))


            graspness = compute_graspness(scores) # (Np,) Instance-wise Normalization
            # compute u, v for each grasp point, assign graspness to those pixels
            projected_pixels = project_3d_to_2d(trans, intrinsic)  # (Np, 2) [u, v]
            projected_pixels = np.round(projected_pixels).astype(np.int32)
            for j in range(projected_pixels.shape[0]):
                u, v = projected_pixels[j]
                if u < 0 or u >= depth.shape[1] or v < 0 or v >= depth.shape[0]:
                    continue
                graspness_map[v, u] = max(graspness_map[v, u], graspness[j])

            # Store top-k view information for this object's grasp points
            # top_k_angles, top_k_depths, top_k_widths: (Np, K)
            top_k_angles = top_k_g_offsets[:, :, 0]  # (Np, K)
            top_k_depths_offset = top_k_g_offsets[:, :, 1]  # (Np, K)
            top_k_widths = top_k_g_offsets[:, :, 2]  # (Np, K)

            # Compute rotations and directions for all top-k views
            all_views = np.tile(template_views, [Np, 1, 1])  # (Np, V, 3)
            top_k_views = all_views[np.arange(Np)[:, None], top_k_view_inds, :]  # (Np, K, 3)

            # For each top-k view, compute approach and gripper directions
            top_k_approach_dirs = []
            top_k_gripper_dirs = []
            for k in range(TOP_K_VIEWS):
                views_k = top_k_views[:, k, :]  # (Np, 3)
                angles_k = top_k_angles[:, k]  # (Np,)
                rotations_k = batch_viewpoint_params_to_matrix(-views_k, angles_k)  # (Np, 3, 3)
                rotations_k = np.matmul(pose[:3, :3], rotations_k)  # (Np, 3, 3)

                # Approach direction is the first column of rotation matrix
                approach_dir_k = rotations_k[:, :, 0]  # (Np, 3)
                # Gripper direction is the second column
                gripper_dir_k = rotations_k[:, :, 1]  # (Np, 3)

                top_k_approach_dirs.append(approach_dir_k)
                top_k_gripper_dirs.append(gripper_dir_k)

            top_k_approach_dirs = np.stack(top_k_approach_dirs, axis=1)  # (Np, K, 3)
            top_k_gripper_dirs = np.stack(top_k_gripper_dirs, axis=1)  # (Np, K, 3)

            # Store for later map population
            if 'all_top_k_data' not in dir():
                all_top_k_data = {
                    'trans': [],
                    'approach_dirs': [],
                    'gripper_dirs': [],
                    'widths': [],
                    'depths': [],
                    'scores': []
                }

            all_top_k_data['trans'].append(trans)  # (Np, 3)
            all_top_k_data['approach_dirs'].append(top_k_approach_dirs)  # (Np, K, 3)
            all_top_k_data['gripper_dirs'].append(top_k_gripper_dirs)  # (Np, K, 3)
            all_top_k_data['widths'].append(top_k_widths)  # (Np, K)
            all_top_k_data['depths'].append(top_k_depths_offset)  # (Np, K)
            all_top_k_data['scores'].append(top_k_grasp_score)  # (Np, K)

        # visualize
        # if cfgs.vis:
        #     print("\n=== 시각화 중 ===")
        #     scene_pcd = o3d.geometry.PointCloud()
        #     scene_pcd.points = o3d.utility.Vector3dVector(cloud.cpu().numpy().reshape(-1,3))

        # remove zero score grasps
        mask = grasp_group.grasp_group_array[:,0] > 0
        grasp_group.grasp_group_array = grasp_group.grasp_group_array[mask]
        print(f"처리 후 grasp 개수: {grasp_group.grasp_group_array.shape[0]}")

        # ========== Concatenate all top-k data ==========
        print("\n=== Processing top-k grasp data ===")

        if len(all_top_k_data['trans']) == 0:
            print(f"No valid grasps for scene {scene_id}, ann {ann_id}")
            continue

        # Concatenate data from all objects
        all_trans = np.concatenate(all_top_k_data['trans'], axis=0)  # (N_total, 3)
        all_approach_dirs = np.concatenate(all_top_k_data['approach_dirs'], axis=0)  # (N_total, K, 3)
        all_gripper_dirs = np.concatenate(all_top_k_data['gripper_dirs'], axis=0)  # (N_total, K, 3)
        all_widths = np.concatenate(all_top_k_data['widths'], axis=0)  # (N_total, K)
        all_depths = np.concatenate(all_top_k_data['depths'], axis=0)  # (N_total, K)
        all_scores = np.concatenate(all_top_k_data['scores'], axis=0)  # (N_total, K)

        num_grasps = all_trans.shape[0]
        print(f"Total grasp points: {num_grasps}")

        # Filter valid grasps (score > 0 for best view)
        valid_grasp_mask = all_scores[:, 0] > 0
        all_trans = all_trans[valid_grasp_mask]
        all_approach_dirs = all_approach_dirs[valid_grasp_mask]
        all_gripper_dirs = all_gripper_dirs[valid_grasp_mask]
        all_widths = all_widths[valid_grasp_mask]
        all_depths = all_depths[valid_grasp_mask]
        all_scores = all_scores[valid_grasp_mask]

        num_grasps = all_trans.shape[0]
        print(f"Valid grasp points (after filtering): {num_grasps}")

        # Filter out upward-facing grasps and fix gripper direction ambiguity
        # Check approach direction for best view (k=0)
        best_approach_dirs = all_approach_dirs[:, 0, :]  # (N, 3)
        downward_mask = best_approach_dirs[:, 2] >= 0  # approach_dir.z >= 0 (facing downward)

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
            gripper_z = all_gripper_dirs[:, k, 2]
            flip_mask = gripper_z > 0
            all_gripper_dirs[flip_mask, k, :] *= -1

        # Convert to torch tensors
        grasp_centers = torch.from_numpy(all_trans.astype(np.float32)).cuda()  # (N_grasps, 3)

        # Create valid mask - check if pixel has graspness AND valid depth
        valid_mask = (graspness_map > 0) & (depth > 0)
        valid_pixels = np.argwhere(valid_mask)  # (N_pixels, 2) [v, u]

        # Vectorized 3D reconstruction
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        # Extract scalar value from factor_depth (it might be [[1000.]] instead of 1000.0)
        if isinstance(factor_depth, np.ndarray):
            factor_depth = float(factor_depth.flatten()[0])

        v_coords, u_coords = valid_pixels[:, 0], valid_pixels[:, 1]
        z_vals = depth[v_coords, u_coords] / factor_depth
        x_vals = (u_coords - cx) * z_vals / fx
        y_vals = (v_coords - cy) * z_vals / fy

        pixel_3d_points = np.stack([x_vals, y_vals, z_vals], axis=1).astype(np.float32)  # (N_pixels, 3)
        pixel_3d_points_torch = torch.from_numpy(pixel_3d_points).cuda()

        # ========== Vectorized nearest neighbor assignment ==========
        # Compute pairwise distances: (N_pixels, N_grasps)
        # Using chunking to avoid OOM for large pixel counts
        chunk_size = 10000
        num_pixels = len(valid_pixels)

        nearest_indices = torch.zeros(num_pixels, dtype=torch.long, device='cuda')

        for i in tqdm(range(0, num_pixels, chunk_size)):
            end_i = min(i + chunk_size, num_pixels)
            pixel_chunk = pixel_3d_points_torch[i:end_i]  # (chunk_size, 3)

            # Compute distances: ||pixel - grasp_center||_2
            # Shape: (chunk_size, N_grasps)
            dists = torch.cdist(pixel_chunk, grasp_centers, p=2)

            # Find nearest grasp for each pixel in chunk
            nearest_indices[i:end_i] = torch.argmin(dists, dim=1)

        # Move results back to CPU
        nearest_indices = nearest_indices.cpu().numpy()

        # ========== Populate sparse data for top-k views ==========
        # Instead of dense maps, store only valid pixel data
        H, W = depth.shape
        num_valid = len(valid_pixels)

        # Sparse storage: (N_valid, K, ...) for each attribute
        sparse_approach_dir = np.zeros((num_valid, TOP_K_VIEWS, 3), dtype=np.float32)
        sparse_gripper_dir = np.zeros((num_valid, TOP_K_VIEWS, 3), dtype=np.float32)
        sparse_grasp_score = np.zeros((num_valid, TOP_K_VIEWS), dtype=np.float32)
        sparse_grasp_width = np.zeros((num_valid, TOP_K_VIEWS), dtype=np.float32)
        sparse_grasp_depth = np.zeros((num_valid, TOP_K_VIEWS), dtype=np.float32)
        sparse_graspness = np.zeros((num_valid,), dtype=np.float32)

        # Assign values for all valid pixels
        for idx, (v, u) in enumerate(valid_pixels):
            grasp_idx = nearest_indices[idx]
            sparse_approach_dir[idx] = all_approach_dirs[grasp_idx]  # (K, 3)
            sparse_gripper_dir[idx] = all_gripper_dirs[grasp_idx]    # (K, 3)
            sparse_grasp_width[idx] = all_widths[grasp_idx]          # (K,)
            sparse_grasp_score[idx] = all_scores[grasp_idx]          # (K,)
            sparse_grasp_depth[idx] = all_depths[grasp_idx]          # (K,)
            sparse_graspness[idx] = graspness_map[v, u]

        valid_mask = np.zeros((H, W), dtype=bool)
        valid_mask[valid_pixels[:, 0], valid_pixels[:, 1]] = True

        # Also create dense maps for visualization (only k=0, best view)
        approach_dir_map = np.zeros((H, W, 3), dtype=np.float32)
        gripper_dir_map = np.zeros((H, W, 3), dtype=np.float32)
        grasp_score_map = np.zeros((H, W), dtype=np.float32)
        grasp_width_map = np.zeros((H, W), dtype=np.float32)
        grasp_depth_map = np.zeros((H, W), dtype=np.float32)

        for idx, (v, u) in enumerate(valid_pixels):
            approach_dir_map[v, u] = sparse_approach_dir[idx, 0]
            gripper_dir_map[v, u] = sparse_gripper_dir[idx, 0]
            grasp_width_map[v, u] = sparse_grasp_width[idx, 0]
            grasp_score_map[v, u] = sparse_grasp_score[idx, 0]
            grasp_depth_map[v, u] = sparse_grasp_depth[idx, 0]

        # ========== Visualization meshes (optional) ==========
        # if cfgs.vis:
        #     meshes = []
        #     print("\n=== Creating visualization meshes ===")
        #     # Visualize best view (k=0) directions
        #     for i in range(min(num_grasps, 1000)):  # Limit for performance
        #         center = all_trans[i]
        #         approach_dir = all_approach_dirs[i, 0]  # k=0 (best view)
        #         gripper_dir = all_gripper_dirs[i, 0]

        #         approach_dir_mesh = create_arrow_mesh(center, approach_dir, length=0.04, color=[1,0,0])
        #         gripper_dir_mesh = create_arrow_mesh(center, gripper_dir, length=0.04, color=[0,1,0])
        #         meshes.extend([approach_dir_mesh, gripper_dir_mesh])
        #     o3d.visualization.draw_geometries([scene_pcd] + meshes)


        # save it as npz file with sparse storage for efficiency
        # Sparse format: store only valid pixel indices and their values
        # This is much more memory-efficient for training

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(
            output_path,
            # Sparse data (only valid pixels)
            valid_pixel_indices=valid_pixels.astype(np.int16),    # (N_valid, 2) - [v, u] coordinates
            graspness=sparse_graspness.astype(np.float16),        # (N_valid,)
            grasp_score=sparse_grasp_score.astype(np.float16),    # (N_valid, K)
            grasp_width=sparse_grasp_width.astype(np.float16),    # (N_valid, K)
            grasp_depth=sparse_grasp_depth.astype(np.float16),    # (N_valid, K)
            approach_dir=sparse_approach_dir.astype(np.float16),  # (N_valid, K, 3)
            gripper_dir=sparse_gripper_dir.astype(np.float16),    # (N_valid, K, 3)
            # Metadata
            image_shape=np.array([H, W], dtype=np.int16),  # for reconstruction
            xyz=cloud_np.astype(np.float16),  # (H, W, 3) - keep dense for point cloud
            top_k=TOP_K_VIEWS
        )
        # To reconstruct dense map when loading:
        # H, W = data['image_shape']
        # dense_map = np.zeros((H, W, ...))
        # dense_map[data['valid_pixel_indices'][:, 0], data['valid_pixel_indices'][:, 1]] = data['sparse_data']
            

        if cfgs.vis:
            print("\n=== Saving visualizations ===")
        # Create output directory for visualizations
            vis_output_dir = os.path.join('pixelwise_grasp_gt_vis', f'scene_{str(scene_id).zfill(4)}')
            os.makedirs(vis_output_dir, exist_ok=True)

            # 1. Visualize RGB image
            rgb_vis = (rgb * 255).astype(np.uint8)
            rgb_img = Image.fromarray(rgb_vis)
            rgb_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_rgb.png'))

            # 2. Visualize graspness map
            graspness_vis = (graspness_map * 255).astype(np.uint8)
            graspness_vis_colored = cv2.applyColorMap(graspness_vis, cv2.COLORMAP_JET)
            # Mask out invalid regions
            graspness_vis_colored[~valid_mask] = [0, 0, 0]
            graspness_img = Image.fromarray(cv2.cvtColor(graspness_vis_colored, cv2.COLOR_BGR2RGB))
            graspness_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_graspness.png'))

            # Use best view (k=0) for visualization - already created as dense maps above
            grasp_score_map_vis = grasp_score_map  # (H, W)
            grasp_width_map_vis = grasp_width_map  # (H, W)
            grasp_depth_map_vis = grasp_depth_map  # (H, W)
            approach_dir_map_vis = approach_dir_map  # (H, W, 3)
            gripper_dir_map_vis = gripper_dir_map  # (H, W, 3)

            grasp_score_vis = (grasp_score_map_vis * 255).astype(np.uint8)
            grasp_score_vis_colored = cv2.applyColorMap(grasp_score_vis, cv2.COLORMAP_JET)
            grasp_score_vis_colored[~valid_mask] = [0, 0, 0]
            grasp_score_img = Image.fromarray(cv2.cvtColor(grasp_score_vis_colored, cv2.COLOR_BGR2RGB))
            grasp_score_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_grasp_score.png'))

            # 3. Visualize grasp width map
            grasp_width_vis = np.clip(grasp_width_map_vis / MAX_WIDTH * 255, 0, 255).astype(np.uint8)
            grasp_width_vis_colored = cv2.applyColorMap(grasp_width_vis, cv2.COLORMAP_JET)
            grasp_width_vis_colored[~valid_mask] = [0, 0, 0]
            grasp_width_img = Image.fromarray(cv2.cvtColor(grasp_width_vis_colored, cv2.COLOR_BGR2RGB))
            grasp_width_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_grasp_width.png'))

            # 4. Visualize grasp depth map
            max_grasp_depth = 0.04  # assuming max depth is 4cm
            grasp_depth_vis = np.clip(grasp_depth_map_vis / max_grasp_depth * 255, 0, 255).astype(np.uint8)
            grasp_depth_vis_colored = cv2.applyColorMap(grasp_depth_vis, cv2.COLORMAP_JET)
            grasp_depth_vis_colored[~valid_mask] = [0, 0, 0]
            grasp_depth_img = Image.fromarray(cv2.cvtColor(grasp_depth_vis_colored, cv2.COLOR_BGR2RGB))
            grasp_depth_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_grasp_depth.png'))

            # 5. Visualize approach direction map as RGB
            approach_dir_vis = (approach_dir_map_vis + 1) / 2.0  # Normalize from [-1,1] to [0,1]
            approach_dir_vis = (approach_dir_vis * 255).astype(np.uint8)
            approach_dir_vis[~valid_mask] = [0, 0, 0]
            approach_dir_img = Image.fromarray(approach_dir_vis)
            approach_dir_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_approach_dir.png'))

            # 6. Visualize gripper direction map as RGB
            gripper_dir_vis = (gripper_dir_map_vis + 1) / 2.0
            gripper_dir_vis = (gripper_dir_vis * 255).astype(np.uint8)
            gripper_dir_vis[~valid_mask] = [0, 0, 0]
            gripper_dir_img = Image.fromarray(gripper_dir_vis)
            gripper_dir_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_gripper_dir.png'))

            # 7. Visualize valid mask
            valid_mask_vis = (valid_mask.astype(np.uint8) * 255)
            valid_mask_img = Image.fromarray(valid_mask_vis)
            valid_mask_img.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_valid_mask.png'))

            # 8. Create 2D arrow visualization on RGB image
            vis_img = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()

            # Sample pixels for visualization (too many arrows would clutter the image)
            arrow_pixels = np.argwhere(valid_mask)

            # Downsample for cleaner visualization
            step = max(1, len(arrow_pixels) // 500)  # Show ~500 arrows max
            arrow_pixels = arrow_pixels[::step]

            # Arrow visualization parameters
            arrow_length_3d = 0.03  # 3cm in 3D space
            arrow_color_approach = (0, 255, 0)  # green
            arrow_thickness = 1
            arrow_tip_length = 0.3

            for (v, u) in arrow_pixels:
                direction_3d = approach_dir_map_vis[v, u]

                if np.linalg.norm(direction_3d) < 1e-6:
                    continue

                # Get the 3D point from depth
                z = float(depth[v, u] / factor_depth)
                if z < 0.1:
                    continue

                x = float((u - cx) * z / fx)
                y = float((v - cy) * z / fy)
                point_3d = np.array([x, y, z], dtype=np.float32)

                # Calculate end point in 3D
                end_point_3d = point_3d + direction_3d.flatten() * arrow_length_3d

                # Project both start and end points to 2D
                start_2d = project_3d_to_2d(point_3d[np.newaxis, :], intrinsic)
                end_2d = project_3d_to_2d(end_point_3d[np.newaxis, :], intrinsic)

                # Convert to integer pixel coordinates
                start_u, start_v = float(start_2d[0, 0]), float(start_2d[0, 1])
                end_u, end_v = float(end_2d[0, 0]), float(end_2d[0, 1])

                start_point = (int(start_u), int(start_v))
                end_point = (int(end_u), int(end_v))

                # Check if endpoints are within image bounds
                if (0 <= end_point[0] < depth.shape[1] and
                    0 <= end_point[1] < depth.shape[0]):
                    cv2.arrowedLine(vis_img, start_point, end_point,
                                arrow_color_approach, arrow_thickness,
                                tipLength=arrow_tip_length)

            # Save 2D arrow visualization
            vis_img_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            vis_img_pil.save(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_arrows_2d.png'))

            # 8. Create a combined visualization grid
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # RGB
            axes[0, 0].imshow(rgb_vis)
            axes[0, 0].set_title('RGB Image', fontsize=14)
            axes[0, 0].axis('off')

            # Graspness
            axes[0, 1].imshow(graspness_img)
            axes[0, 1].set_title(f'Graspness Map (min={graspness_map.min():.3f}, max={graspness_map.max():.3f})', fontsize=14)
            axes[0, 1].axis('off')

            # Width
            axes[0, 2].imshow(grasp_width_img)
            axes[0, 2].set_title(f'Width Map (min={grasp_width_map_vis[valid_mask].min():.3f}, max={grasp_width_map_vis[valid_mask].max():.3f}m)', fontsize=14)
            axes[0, 2].axis('off')

            # Approach direction
            axes[1, 0].imshow(approach_dir_img)
            axes[1, 0].set_title('Approach Direction (RGB encoding)', fontsize=14)
            axes[1, 0].axis('off')

            # Gripper direction
            axes[1, 1].imshow(gripper_dir_img)
            axes[1, 1].set_title('Gripper Direction (RGB encoding)', fontsize=14)
            axes[1, 1].axis('off')

            # Arrows on RGB
            axes[1, 2].imshow(vis_img_pil)
            axes[1, 2].set_title(f'Approach Arrows ({len(arrow_pixels)} shown)', fontsize=14)
            axes[1, 2].axis('off')

            plt.suptitle(f'Scene {scene_id}, Annotation {ann_id} - Pixel-wise Grasp Ground Truth\nValid pixels: {np.sum(valid_mask)}',
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_output_dir, f'ann_{str(ann_id).zfill(4)}_combined.png'), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\n✓ Saved visualizations to: {vis_output_dir}")
            print(f"  - RGB image")
            print(f"  - Graspness map (JET colormap)")
            print(f"  - Width map (JET colormap)")
            print(f"  - Approach direction map (RGB encoding)")
            print(f"  - Gripper direction map (RGB encoding)")
            print(f"  - Valid mask")
            print(f"  - 2D arrow visualization")
            print(f"  - Combined grid visualization")

