# convert_h5_to_npz_mp.py
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

DTYPE_MAP = {
    'graspness': np.float16,
    'grasp_score': np.float16,
    'grasp_width': np.float16,
    'grasp_depth': np.float16,
    'approach_dir': np.float16,
    'gripper_dir': np.float16,
    'valid_mask': np.uint8,
    'xyz': np.float16
}

def process_one(args):
    h5_path, output_dir = args
    out = {}

    try:
        with h5py.File(h5_path, 'r') as f:
            for key, dtype in DTYPE_MAP.items():
                if key not in f:
                    continue
                arr = f[key][()]
                if arr.dtype == bool:
                    arr = arr.astype(np.uint8)
                out[key] = arr.astype(dtype)

        npz_path = output_dir / h5_path.with_suffix('.npz').name
        np.savez(npz_path, **out)

    except Exception as e:
        return f"[ERROR] {h5_path}: {e}"

    return None


def convert_h5_to_npz_mp(h5_dir, output_dir, num_workers=None):
    h5_dir = Path(h5_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = list(h5_dir.glob('*.h5'))
    tasks = [(p, output_dir) for p in h5_files]

    if num_workers is None:
        num_workers = min(cpu_count(), num_workers)

    print(f"Using {num_workers} workers for {len(tasks)} files")

    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_one, tasks),
                           total=len(tasks)):
            if result is not None:
                print(result)


if __name__ == "__main__":
    convert_h5_to_npz_mp(
        '/data/Grasp/GraspNet-1Billion/pixelwise_grasp_gt_hdf5/realsense',
        '/data/Grasp/GraspNet-1Billion/pixelwise_grasp_gt/realsense',
        num_workers=64
    )
