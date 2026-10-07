"""
Generate per-pixel CAD surface distance (cad_surface_dist) for training frames.

For each training frame (scene 0-99), computes the distance from each valid
sensor xyz point to the nearest point on the corresponding CAD surface
(pose-transformed). Saves sparse (N_valid,) float16 to supplementary npz files.

Existing pixelwise_grasp_gt_top300 files are NOT modified.

Output:
    {root}/pixelwise_cad_surface_dist/realsense/scene_XXXX_ann_XXXX.npz
        cad_surface_dist: (N_valid,) float16   # meters

Usage:
    python scripts/gen_cad_surface_dist.py --root /data/Grasp/GraspNet-1Billion
    python scripts/gen_cad_surface_dist.py --root /data/Grasp/GraspNet-1Billion --workers 8
"""
import argparse
import os
import numpy as np
import scipy.io as scio
import scipy.spatial
import cv2
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/data/Grasp/GraspNet-1Billion')
parser.add_argument('--camera', default='realsense')
parser.add_argument('--scenes', default='0,99', help='scene range inclusive, e.g. 0,99')
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--boundary_erosion', type=int, default=2, help='px to erode object mask boundary')
args = parser.parse_args()


def load_cad_surface(root, obj_id):
    """Load 5mm-downsampled CAD surface points (model coords, float32).

    GraspNet cls_indexes are 1-indexed (1..88); model dirs are 0-indexed (000..087).
    """
    model_dir_idx = obj_id - 1  # convert to 0-indexed
    path = os.path.join(root, 'models', f'{model_dir_idx:03d}', 'surface_5mm.npy')
    if not os.path.exists(path):
        return None
    return np.load(path)  # (M, 3)


def transform_points(pts, pose):
    """pts: (N,3) model coords -> camera coords using (3,4) pose."""
    R, t = pose[:3, :3], pose[:3, 3:]
    return (R @ pts.T + t).T  # (N, 3)


def min_dist_kdtree(obs_pts, cad_pts):
    """
    obs_pts: (N, 3) numpy  -- sensor xyz for object pixels
    cad_pts: (M, 3) numpy  -- transformed CAD surface
    returns: (N,) numpy    -- distance to nearest CAD point
    """
    tree = scipy.spatial.cKDTree(cad_pts)
    dists, _ = tree.query(obs_pts, workers=1)
    return dists.astype(np.float32)


def process_frame(args_tuple):
    root, camera, scene_id, ann_id, boundary_erosion = args_tuple

    scene_str = f'scene_{scene_id:04d}'
    ann_str = f'{ann_id:04d}'

    # Output path
    out_dir = os.path.join(root, 'pixelwise_cad_surface_dist', camera)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{scene_str}_ann_{ann_str}.npz')
    if os.path.exists(out_path):
        return True

    # Load existing GT (xyz + valid_pixel_indices)
    gt_path = os.path.join(
        root, 'pixelwise_grasp_gt_top300', camera,
        f'{scene_str}_ann_{ann_str}.npz'
    )
    if not os.path.exists(gt_path):
        return False

    with np.load(gt_path) as gt:
        xyz_dense = gt['xyz'].astype(np.float32)          # (H, W, 3)
        valid_pix = gt['valid_pixel_indices'].astype(np.int32)  # (N, 2) [v, u]

    # Load segmentation and meta
    scene_dir = os.path.join(root, 'scenes', scene_str, camera)
    segm = cv2.imread(
        os.path.join(scene_dir, 'label', f'{ann_str}.png'),
        cv2.IMREAD_UNCHANGED
    )
    meta = scio.loadmat(os.path.join(scene_dir, 'meta', f'{ann_str}.mat'))
    obj_ids = meta['cls_indexes'].flatten().astype(np.int32)
    poses = meta['poses']  # (3, 4, N_obj)

    H, W = xyz_dense.shape[:2]
    d_surf_dense = np.full((H, W), np.nan, dtype=np.float32)

    erode_kernel = np.ones((boundary_erosion * 2 + 1,) * 2, dtype=np.uint8)

    for i, obj_id in enumerate(obj_ids):
        cad_pts = load_cad_surface(root, obj_id)
        if cad_pts is None or len(cad_pts) == 0:
            continue

        pose = poses[:, :, i]  # (3, 4)
        cad_cam = transform_points(cad_pts, pose)  # (M, 3) camera frame

        # Object mask with boundary erosion
        obj_mask = (segm == obj_id).astype(np.uint8)
        if obj_mask.sum() == 0:
            continue
        if boundary_erosion > 0:
            obj_mask = cv2.erode(obj_mask, erode_kernel, iterations=1)

        # Valid pixels within this object (must have valid xyz)
        obj_valid = (obj_mask > 0) & (xyz_dense[:, :, 2] > 0.01)
        if not np.any(obj_valid):
            continue

        obj_pix = np.argwhere(obj_valid)        # (K, 2)
        obs_pts = xyz_dense[obj_pix[:, 0], obj_pix[:, 1]]  # (K, 3)

        d = min_dist_kdtree(obs_pts, cad_cam)
        d_surf_dense[obj_pix[:, 0], obj_pix[:, 1]] = d

    # Sparse: only at valid_pixel_indices (same indexing as GT)
    v, u = valid_pix[:, 0], valid_pix[:, 1]
    cad_surface_dist = d_surf_dense[v, u]  # (N_valid,), NaN where no object

    # Replace NaN (background / no CAD supervision) with -1.0
    # → exp(-d/τ) ≈ 0, handled in dataloader by masking
    cad_surface_dist = np.where(np.isnan(cad_surface_dist), -1.0, cad_surface_dist)

    np.savez_compressed(out_path, cad_surface_dist=cad_surface_dist.astype(np.float16))
    return True


def main():
    lo, hi = [int(x) for x in args.scenes.split(',')]
    scene_ids = list(range(lo, hi + 1))

    tasks = [
        (args.root, args.camera, s, a, args.boundary_erosion)
        for s in scene_ids
        for a in range(256)
    ]

    print(f"Generating cad_surface_dist for {len(tasks)} frames "
          f"(scenes {lo}-{hi}, workers={args.workers}) ...")

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_frame, tasks), total=len(tasks)))
    else:
        results = [process_frame(t) for t in tqdm(tasks)]

    ok = sum(results)
    print(f"Done: {ok}/{len(tasks)} frames processed.")

    # Sanity check
    sample_path = os.path.join(
        args.root, 'pixelwise_cad_surface_dist', args.camera,
        f'scene_{lo:04d}_ann_0000.npz'
    )
    if os.path.exists(sample_path):
        d = np.load(sample_path)['cad_surface_dist'].astype(np.float32)
        valid = d[d >= 0]
        print(f"\nSanity check scene_{lo:04d}_ann_0000:")
        print(f"  N_valid with CAD: {len(valid)} / {len(d)}")
        print(f"  d_surf [mm]: mean={valid.mean()*1000:.1f}  "
              f"p50={np.median(valid)*1000:.1f}  p95={np.percentile(valid,95)*1000:.1f}")


if __name__ == '__main__':
    main()
