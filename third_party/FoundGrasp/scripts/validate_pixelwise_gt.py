"""
Validate converted v2 compressed pixelwise_grasp_gt_top300 npz files.
Checks: loadable, correct keys, v2 shape contract, no NaN/Inf, value ranges.
"""
import os
import argparse
import random
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

REQUIRED_KEYS = {
    'valid_pixel_indices', 'graspness', 'grasp_score', 'grasp_width',
    'grasp_depth', 'approach_dir', 'gripper_dir', 'image_shape', 'xyz', 'top_k',
}


def validate_file(path):
    errors = []
    try:
        with np.load(path, allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
    except Exception as e:
        return (path, [f'load error: {e}'])

    # Keys
    missing = REQUIRED_KEYS - set(d.keys())
    if missing:
        errors.append(f'missing keys: {missing}')

    # v2 shape contract
    app = d.get('approach_dir')
    if app is None:
        return (path, errors + ['no approach_dir'])

    if app.ndim != 3:
        errors.append(f'approach_dir ndim={app.ndim} (expected 3 for v2)')
        return (path, errors)  # rest of checks assume v2

    N, K, _ = app.shape
    top_k = int(d['top_k']) if 'top_k' in d else None

    if top_k is not None and K != top_k:
        errors.append(f'K={K} != top_k={top_k}')

    # Shape consistency across arrays
    for key in ('gripper_dir',):
        arr = d.get(key)
        if arr is not None and arr.shape != (N, K, 3):
            errors.append(f'{key} shape {arr.shape} != ({N},{K},3)')
    for key in ('grasp_score', 'grasp_width', 'grasp_depth'):
        arr = d.get(key)
        if arr is not None and arr.shape != (N, K):
            errors.append(f'{key} shape {arr.shape} != ({N},{K})')
    if 'graspness' in d and d['graspness'].shape != (N,):
        errors.append(f"graspness shape {d['graspness'].shape} != ({N},)")
    if 'valid_pixel_indices' in d and d['valid_pixel_indices'].shape[0] != N:
        errors.append(f"valid_pixel_indices N={d['valid_pixel_indices'].shape[0]} != {N}")
    if 'image_shape' in d and d['image_shape'].shape != (2,):
        errors.append(f"image_shape {d['image_shape'].shape} != (2,)")

    # NaN / Inf
    for key in ('approach_dir', 'gripper_dir', 'grasp_score', 'grasp_width', 'grasp_depth', 'graspness', 'xyz'):
        arr = d.get(key)
        if arr is not None and arr.dtype.kind == 'f':
            arr_f = arr.astype(np.float32)
            if np.any(np.isnan(arr_f)):
                errors.append(f'{key} contains NaN')
            if np.any(np.isinf(arr_f)):
                errors.append(f'{key} contains Inf')

    # Value ranges
    for key in ('graspness', 'grasp_score'):
        arr = d.get(key)
        if arr is not None:
            arr_f = arr.astype(np.float32)
            lo, hi = float(arr_f.min()), float(arr_f.max())
            if lo < -0.01 or hi > 1.01:
                errors.append(f'{key} out of [0,1]: min={lo:.4f} max={hi:.4f}')

    # Unit vector check (sample to keep fast)
    for key in ('approach_dir', 'gripper_dir'):
        arr = d.get(key)
        if arr is not None and arr.size > 0:
            flat = arr.astype(np.float32).reshape(-1, 3)
            idx = np.random.choice(len(flat), min(1000, len(flat)), replace=False)
            norms = np.linalg.norm(flat[idx], axis=-1)
            valid_mask = norms > 0.01  # skip zero vectors
            if valid_mask.any():
                bad = np.abs(norms[valid_mask] - 1.0) > 0.1
                if bad.any():
                    errors.append(f'{key} not unit vectors (sample norm range [{norms[valid_mask].min():.3f}, {norms[valid_mask].max():.3f}])')

    return (path, errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/Grasp/GraspNet-1Billion')
    parser.add_argument('--camera', default='realsense', choices=['realsense', 'kinect', 'both'])
    parser.add_argument('--sample', type=int, default=0,
                        help='Random sample N files per camera (0 = all)')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--delete_bad', action='store_true',
                        help='Delete files that fail validation (so they can be regenerated)')
    args = parser.parse_args()

    random.seed(args.seed)
    cameras = ['realsense', 'kinect'] if args.camera == 'both' else [args.camera]
    gt_root = os.path.join(args.root, 'pixelwise_grasp_gt_top300')

    all_files = []
    for cam in cameras:
        cam_dir = os.path.join(gt_root, cam)
        if not os.path.exists(cam_dir):
            print(f'{cam}: directory not found, skipping')
            continue
        files = sorted(Path(cam_dir).glob('*.npz'))
        if args.sample > 0:
            files = random.sample(files, min(args.sample, len(files)))
        all_files.extend(str(f) for f in files)
        print(f'{cam}: {len(files)} files to validate')

    print(f'Total: {len(all_files)} files\n')

    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(validate_file, all_files, chunksize=8),
            total=len(all_files)
        ))

    ok, fail = 0, 0
    deleted = 0
    for path, errors in results:
        if errors:
            fail += 1
            print(f'FAIL {path}')
            for e in errors:
                print(f'     {e}')
            if args.delete_bad:
                os.remove(path)
                deleted += 1
                print(f'     → deleted')
        else:
            ok += 1

    print(f'\nResults: {ok} OK, {fail} FAILED out of {len(all_files)} files')
    if args.delete_bad and deleted:
        print(f'Deleted {deleted} bad files — re-run gen_pixelwise_gt_mp.py to regenerate them')


if __name__ == '__main__':
    main()
