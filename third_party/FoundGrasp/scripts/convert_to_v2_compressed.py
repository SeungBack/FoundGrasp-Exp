"""
Convert existing pixelwise_grasp_gt_top300 npz files to v2 compressed format.
  - v3 (N, K, P, ...) → v2 (N, K, ...) by taking P=0 (best angle, score-sorted)
  - v2 uncompressed    → v2 compressed
Skips files already in v2 compressed format.
"""
import os
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


def _is_v2_compressed(path, d):
    if d['approach_dir'].ndim != 3:
        return False
    uncompressed_bytes = sum(v.nbytes for v in d.values() if hasattr(v, 'nbytes'))
    return os.path.getsize(path) < uncompressed_bytes * 0.5


def convert_file(path):
    try:
        with np.load(path, allow_pickle=True) as f:
            d = {k: f[k] for k in f.files}

        if _is_v2_compressed(path, d):
            return (path, 'skipped')

        is_v3 = (d['approach_dir'].ndim == 4)
        if is_v3:
            d['approach_dir'] = d['approach_dir'][:, :, 0, :]
            d['gripper_dir']  = d['gripper_dir'][:, :, 0, :]
            d['grasp_score']  = d['grasp_score'][:, :, 0]
            d['grasp_width']  = d['grasp_width'][:, :, 0]
            d['grasp_depth']  = d['grasp_depth'][:, :, 0]
            d.pop('top_p', None)

        tmp_path = path[:-4] + '.tmp.npz'
        try:
            np.savez_compressed(tmp_path, **d)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        return (path, 'converted' if is_v3 else 'recompressed')
    except Exception as e:
        return (path, f'error: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/Grasp/GraspNet-1Billion')
    parser.add_argument('--camera', default='both', choices=['realsense', 'kinect', 'both'])
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    gt_root = os.path.join(args.root, 'pixelwise_grasp_gt_top300')
    cameras = ['realsense', 'kinect'] if args.camera == 'both' else [args.camera]

    all_files = []
    for cam in cameras:
        cam_dir = os.path.join(gt_root, cam)
        if os.path.exists(cam_dir):
            files = sorted(Path(cam_dir).glob('*.npz'))
            all_files.extend(str(f) for f in files)
            print(f"{cam}: {len(files)} files")

    print(f"Total: {len(all_files)} files")

    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(convert_file, all_files, chunksize=4),
            total=len(all_files)
        ))

    counts = {'skipped': 0, 'converted': 0, 'recompressed': 0, 'error': 0}
    errors = []
    for path, status in results:
        key = status if not status.startswith('error') else 'error'
        counts[key] += 1
        if status.startswith('error'):
            errors.append(f"  {path}: {status}")

    print(f"\nResults: {counts}")
    if errors:
        print("Errors:")
        print('\n'.join(errors))


if __name__ == '__main__':
    main()
