"""
Preprocess CAD models for depth reliability estimation.

Loads nontextured.ply for each of the 92 GraspNet-1B objects,
voxel-downsamples at 5 mm, and saves as surface_5mm.npy.

Run once before gen_cad_surface_dist.py.

Usage:
    python scripts/preprocess_cad_models.py --root /data/Grasp/GraspNet-1Billion
"""
import argparse
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/data/Grasp/GraspNet-1Billion')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel size in meters (default 5mm)')
args = parser.parse_args()

models_dir = os.path.join(args.root, 'models')
model_ids = sorted(os.listdir(models_dir))

print(f"Processing {len(model_ids)} models at voxel_size={args.voxel_size*1000:.0f}mm ...")

for model_id in tqdm(model_ids):
    ply_path = os.path.join(models_dir, model_id, 'nontextured.ply')
    out_path = os.path.join(models_dir, model_id, 'surface_5mm.npy')

    if not os.path.exists(ply_path):
        print(f"  [SKIP] {model_id}: nontextured.ply not found")
        continue

    if os.path.exists(out_path):
        continue  # already done

    pcd = o3d.io.read_point_cloud(ply_path)
    pcd_down = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    pts = np.asarray(pcd_down.points, dtype=np.float32)
    np.save(out_path, pts)

print(f"\nDone. Example sizes:")
for model_id in model_ids[:3]:
    out_path = os.path.join(models_dir, model_id, 'surface_5mm.npy')
    if os.path.exists(out_path):
        pts = np.load(out_path)
        print(f"  {model_id}: {len(pts)} points")
