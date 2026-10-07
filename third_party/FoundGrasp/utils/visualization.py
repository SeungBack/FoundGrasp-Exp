import cv2
import numpy as np
import os
import torch


def _ensure_float32_tensors(obj):
    """float16/bfloat16 텐서를 float32로 변환 (NumPy 호환용)."""
    if isinstance(obj, dict):
        return {k: _ensure_float32_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        if t.dtype in (torch.float16, torch.bfloat16):
            t = t.float()
        return t
    return obj


def apply_colormap(data, cmap=cv2.COLORMAP_TURBO):
    """
    Normalize valid data to 0-255 and apply colormap.
    NaN values are ignored (rendered as Black).
    """
    # Convert to float numpy array if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    data = np.asarray(data, dtype=np.float32)
    
    valid_mask = ~np.isnan(data)

    if not np.any(valid_mask):
        h, w = data.shape
        return np.zeros((h, w, 3), dtype=np.uint8)

    v_min = np.nanmin(data)
    v_max = np.nanmax(data)

    data_filled = data.copy()
    data_filled[~valid_mask] = v_min 

    if v_max - v_min > 1e-6:
        norm = (data_filled - v_min) / (v_max - v_min)
    else:
        norm = np.zeros_like(data_filled)
    
    img_u8 = (norm * 255).astype(np.uint8)
    vis_color = cv2.applyColorMap(img_u8, cmap)

    vis_color[~valid_mask] = 0
    return vis_color

def add_title(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, img.shape[0] / 288)  # ~2.0 for 576px, scales down for H/4
    thickness = max(1, int(font_scale))
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (0, 0), (text_w + 10, text_h + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (5, text_h + 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

def thicken_points(img, size=3):
    if size <= 1: return img
    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def _prepare_scalar_map(key, vis):
    vis_f = vis.astype(np.float32)

    # Kappa often has a long positive tail, so compress the dynamic range
    # for visualization while preserving ordering.
    if 'kappa' in key:
        vis_f = np.log1p(np.clip(vis_f - 1.0, 0.0, None))

    vis_f[vis_f == 0] = np.nan
    return vis_f


def _to_numpy(data):
    if hasattr(data, 'numpy'):
        return data.numpy()
    return np.array(data)


def _make_color_bgr(data):
    if data.shape[0] == 3:
        data = data.transpose(1, 2, 0)
    if data.dtype != np.uint8:
        vis_img = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    else:
        vis_img = data.copy()
    return cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)


def _make_depth_bgr(data):
    vis = data[0] if data.ndim == 3 else data
    vis = vis.copy()
    vis[vis < 0.3] = np.nan
    return apply_colormap(vis)



def _append_visualization(vis_results, vis_images, key, vis_img, save_dir=None):
    vis_img_with_title = add_title(vis_img.copy(), key)
    vis_images.append(vis_img_with_title)
    vis_results[key] = vis_img_with_title
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)



def visualize_and_save_opencv(sample, save_dir=None, prefix='', point_size=3):
    """
    Visualize sample data and optionally save to disk.

    Args:
        sample: dict containing data to visualize
        save_dir: directory to save visualizations (if None, only returns dict)
        point_size: size for sparse point visualization

    Returns:
        dict: visualization results with keys as data names and values as BGR images
    """
    sample = _ensure_float32_tensors(sample)

    vis_results = {}
    vis_images = []

    for key in sample:
        if key in ['pixel_values', 'intrinsic', 'scene_id', 'im_id']:
            continue
            
        data = sample[key]
        if hasattr(data, 'numpy'):
            data = data.numpy()
        else:
            data = np.array(data)

        vis_img = None
        is_sparse = False

        # 1. RGB Image
        if key == 'og_color':
            vis_img = _make_color_bgr(data)

        # 2. XYZ Coordinate Map (수정됨)
        elif key == 'xyz':
            # Data Shape: (3, H, W) -> (H, W, 3)
            vis = data.transpose(1, 2, 0).copy()
            h, w, c = vis.shape
            
            # Z값이 0인 곳은 데이터가 없는 곳으로 간주 (Mask 생성)
            # vis[:, :, 2] 가 Z축
            valid_mask = (vis[:, :, 2] != 0)
            
            # 결과 이미지 (BGR) 초기화
            vis_img = np.zeros((h, w, 3), dtype=np.uint8)

            if np.any(valid_mask):
                # X, Y, Z 각 채널별로 Normalize (0~255)
                # X: data[0], Y: data[1], Z: data[2]
                
                normalized_channels = []
                for i in range(3):
                    c_data = vis[:, :, i]
                    # 유효한 픽셀들만 뽑아서 min/max 계산
                    valid_pixels = c_data[valid_mask]
                    
                    if len(valid_pixels) > 0:
                        v_min, v_max = valid_pixels.min(), valid_pixels.max()
                        v_min = min(0, v_min)
                        if v_max - v_min > 1e-6:
                            norm = (c_data - v_min) / (v_max - v_min)
                        else:
                            norm = np.zeros_like(c_data)
                    else:
                        norm = np.zeros_like(c_data)
                        
                    normalized_channels.append((norm * 255).astype(np.uint8))
                
                # 병합: 통상적으로 X=Red, Y=Green, Z=Blue 로 시각화함
                # OpenCV는 BGR 순서이므로 [Z, Y, X] 순서로 넣어야 함
                # normalized_channels[0] -> X (Red)
                # normalized_channels[1] -> Y (Green)
                # normalized_channels[2] -> Z (Blue)
                
                vis_img = cv2.merge([
                    normalized_channels[2], # B (Z)
                    normalized_channels[1], # G (Y)
                    normalized_channels[0]  # R (X)
                ])
                
                # Z=0 (Invalid)인 영역 검정색 처리
                vis_img[~valid_mask] = 0

        # 3. Depth Map
        elif key == 'depth':
            vis_img = _make_depth_bgr(data)

        # 4. Direction Vectors
        elif 'dir' in key:
            is_sparse = True
            vis = data.transpose(1, 2, 0).copy()

            # Handle gripper direction symmetry
            # Gripper direction is symmetric: d and -d represent the same grasp
            # Normalize to always point in a consistent direction (e.g., where z > 0 or x > 0)
            if 'gripper' in key:
                mask = np.any(vis != 0, axis=-1)
                # Flip vectors where the first non-zero component is negative
                # This ensures consistent visualization
                for i in range(vis.shape[0]):
                    for j in range(vis.shape[1]):
                        if mask[i, j]:
                            vec = vis[i, j]
                            # Find first non-zero component
                            # Prioritize z-axis for consistency
                            if abs(vec[2]) > 1e-6:
                                if vec[2] < 0:
                                    vis[i, j] = -vec
                            elif abs(vec[0]) > 1e-6:
                                if vec[0] < 0:
                                    vis[i, j] = -vec
                            elif abs(vec[1]) > 1e-6:
                                if vec[1] < 0:
                                    vis[i, j] = -vec

            vis_norm = np.zeros_like(vis)
            vis_norm[..., 0] = (vis[..., 0] + 1.0) / 2.0
            vis_norm[..., 1] = (vis[..., 1] + 1.0) / 2.0
            vis_norm[..., 2] = (vis[..., 2] + 1.0) / 2.0

            vis_norm = np.clip(vis_norm, 0, 1)
            mask = np.any(vis != 0, axis=-1)

            vis_bgr = (vis_norm * 255).astype(np.uint8)

            vis_img = np.zeros_like(vis_bgr)
            vis_img[mask] = vis_bgr[mask]

        # 5. Grasp / Sparse Maps
        elif key in ['grasp_score', 'grasp_width', 'grasp_depth', 'graspness', 'objectness', 'valid_mask', 'segm', 'confidence',
                     'depth_mask', 'depth_corruption_mask', 'depth_removed', 'depth_delta', 'xyz_delta', 'approach_kappa', 'gripper_kappa',
                     'pred_grasp_score', 'pred_grasp_width', 'pred_grasp_depth', 'pred_graspness', 'pred_objectness',
                     'pred_valid_mask', 'pred_segm', 'pred_approach_kappa', 'pred_gripper_kappa',]:
            is_sparse = True if 'pred' not in key else False
            vis = data[0] if data.ndim == 3 else data

            vis_f = _prepare_scalar_map(key, vis)
            vis_img = apply_colormap(vis_f)

        # 6. Default
        else:
            # print(f"Visualization for key '{key}' not implemented. Skipping.")
            continue
            # vis = data[0] if data.ndim > 2 else data
            # vis_img = apply_colormap(vis)

        if 'pred' in key:
            # masking with segm > 0
            if 'segm' in sample:
                segm = sample['segm']
                segm = segm.numpy() if hasattr(segm, 'numpy') else np.array(segm)
                segm_mask = segm[0] > 0
                if segm_mask.shape != vis_img.shape[:2]:
                    segm_mask = cv2.resize(
                        segm_mask.astype(np.uint8),
                        (vis_img.shape[1], vis_img.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                vis_img[~segm_mask] = 0

        if vis_img is not None:
            if is_sparse:
                vis_img = thicken_points(vis_img, size=point_size)

            _append_visualization(vis_results, vis_images, key, vis_img, save_dir=save_dir)

    # Create grid visualization
    if not vis_images:
        return vis_results

    sorted_keys = []
    processed_keys = set()

    # 1. 기본 입력 데이터 먼저 배치 (맨 윗줄)
    base_keys = ['og_color', 'depth', 'xyz', 'depth_mask']
    for key in base_keys:
        if key in vis_results:
            sorted_keys.append(key)
            processed_keys.add(key)

    # 2. {Key}와 {pred_Key} 짝짓기
    all_keys = list(vis_results.keys())
    # 'pred_'로 시작하지 않고, 이미 처리되지 않은 키들을 찾음 (예: graspness, score 등)
    target_keys = [k for k in all_keys if k not in processed_keys and not k.startswith('pred_')]
    target_keys.sort()  # 이름순 정렬 (보기 좋게)

    for key in target_keys:
        pred_key = f"pred_{key}"
        
        # Ground Truth (또는 Base) 추가
        sorted_keys.append(key)
        processed_keys.add(key)
        
        # 대응되는 Prediction이 있으면 바로 옆에 추가
        if pred_key in vis_results:
            sorted_keys.append(pred_key)
            processed_keys.add(pred_key)

    # 3. 나머지 키들 추가 (짝 없는 pred 값이나 기타 데이터)
    remaining_keys = [k for k in all_keys if k not in processed_keys]
    remaining_keys.sort()
    # depth_mask는 시각화 x
    # if 'depth_mask' in remaining_keys:
    #     remaining_keys.remove('depth_mask')
    sorted_keys.extend(remaining_keys)

    # 4. 정렬된 순서대로 이미지 리스트 다시 생성
    vis_images_sorted = [vis_results[k] for k in sorted_keys]
    # 맨 첫번째 

    # ==========================================================================
    # Grid 이미지 생성
    # ==========================================================================
    h, w = vis_images_sorted[0].shape[:2]
    resized_images = [cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img for img in vis_images_sorted]

    n_cols = 4  # 4칸씩 그리면 (GT, Pred) (GT, Pred) 형태로 보기 좋게 나옵니다.
    n_rows = (len(resized_images) + n_cols - 1) // n_cols
    pad_count = (n_rows * n_cols) - len(resized_images)
    
    for _ in range(pad_count):
        resized_images.append(np.zeros((h, w, 3), dtype=np.uint8))

    rows = [cv2.hconcat(resized_images[i*n_cols : (i+1)*n_cols]) for i in range(n_rows)]
    final_grid = cv2.vconcat(rows)

    # Store grid in results
    vis_results['all_data_grid'] = final_grid

    # Save grid if save_dir is provided
    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, prefix+"all_data_grid.jpg"), final_grid)
        print("Saved grid visualization.")

    return vis_results


def log_to_rerun(data, vis_result_2d, points_3d, gg, batch_idx):
    """
    Log visualization data to rerun for interactive 3D viewing.

    Args:
        data: dict containing batch data
        output: dict containing model predictions
        points_3d: (N, 3) numpy array of 3D points
        gg: GraspGroup object
        batch_idx: current batch index for timeline
    """
    try:
        import rerun as rr
    except ImportError:
        print("Warning: rerun not installed. Skipping rerun visualization.")
        return

    # Set timeline
    rr.set_time("frame_idx", sequence=batch_idx)

    # Free 3D rotation: disable up-axis constraint
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    rr.log("world/camera/vis_result_2d", rr.Image(vis_result_2d['all_data_grid']))

    # Log point cloud
    log_point_cloud_to_rerun(data, points_3d)

    # Log grasps
    log_grasps_to_rerun(gg)


def log_point_cloud_to_rerun(data, points_3d):
    """
    Log point cloud to rerun.

    Args:
        data: dict containing og_color and xyz (already indexed, no batch dimension)
        points_3d: (N, 3) numpy array of 3D points
    """
    try:
        import rerun as rr
    except ImportError:
        return

    valid_mask = points_3d[:, 2] > 0
    valid_points = points_3d[valid_mask]

    if len(valid_points) > 0:
        if 'og_color' in data:
            rgb = _ensure_float32_tensors(data['og_color'])
            rgb = rgb.numpy() if hasattr(rgb, 'numpy') else np.array(rgb)  # (3, H, W)
            xyz_map = _ensure_float32_tensors(data['xyz'])
            xyz_map = xyz_map.numpy() if hasattr(xyz_map, 'numpy') else np.array(xyz_map)  # (3, H, W)

            rgb_flat = rgb.reshape(3, -1).T  # (N, 3)
            xyz_flat = xyz_map.reshape(3, -1).T  # (N, 3)

            valid_mask_flat = xyz_flat[:, 2] > 0
            valid_points_viz = xyz_flat[valid_mask_flat]
            colors_viz = rgb_flat[valid_mask_flat]

            rr.log("world/point_cloud", rr.Points3D(valid_points_viz, colors=colors_viz, radii=0.001))
        else:
            rr.log("world/point_cloud", rr.Points3D(valid_points, radii=0.001))


def log_grasps_to_rerun(gg):
    """
    Log grasps as mesh to rerun.

    Args:
        gg: GraspGroup object
    """
    try:
        import rerun as rr
    except ImportError:
        return

    def log_empty_grasps():
        rr.log(
            "world/grasps_mesh",
            rr.Mesh3D(
                vertex_positions=np.empty((0, 3), dtype=np.float32),
                triangle_indices=np.empty((0, 3), dtype=np.uint32),
                vertex_colors=np.empty((0, 3), dtype=np.float32),
            ),
        )

    if len(gg) == 0:
        log_empty_grasps()
        return

    # Convert grasps to open3d meshes
    gg_o3d_list = gg.to_open3d_geometry_list()

    # Merge all meshes into one
    all_vertices = []
    all_triangles = []
    all_colors = []
    vertex_offset = 0

    for grasp_mesh in gg_o3d_list:
        # Extract vertices
        verts = np.asarray(grasp_mesh.vertices)
        if len(verts) == 0:
            continue

        # Extract triangles with offset
        tris = np.asarray(grasp_mesh.triangles)

        # Extract colors
        if grasp_mesh.has_vertex_colors():
            cols = np.asarray(grasp_mesh.vertex_colors)
        else:
            # Default red color
            cols = np.tile([1.0, 0.0, 0.0], (len(verts), 1))

        all_vertices.append(verts)
        all_triangles.append(tris + vertex_offset)
        all_colors.append(cols)

        vertex_offset += len(verts)

    # Log merged mesh
    if not all_vertices:
        log_empty_grasps()
        return

    combined_vertices = np.vstack(all_vertices)
    combined_triangles = np.vstack(all_triangles)
    combined_colors = np.vstack(all_colors)

    rr.log(
        "world/grasps_mesh",
        rr.Mesh3D(
            vertex_positions=combined_vertices,
            triangle_indices=combined_triangles,
            vertex_colors=combined_colors
        )
    )
