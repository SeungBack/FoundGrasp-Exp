"""AnyGrasp SDK worker: RGB-D -> SDK opening-width grasps in camera metres.

The SDK documents width as required gripper opening. Preserve it exactly;
never infer a hidden x1.2 margin from a different model's implementation.
SDK binaries and license remain in their original workspace; a private runtime
folder provides the layout required by the SDK's loader.
"""
import argparse
import importlib.machinery
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from efg.worker_protocol import serve


def points_from_depth(depth, K, depth_max_m=1.5):
    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)
    if depth.ndim != 2 or K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Invalid depth/K shape or intrinsics")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.isfinite(depth_max_m) or depth_max_m <= 0:
        raise ValueError("Focal lengths and depth limit must be positive")
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0) & (depth < depth_max_m))
    z = depth[v, u]
    return np.ascontiguousarray(np.column_stack((
        (u - K[0, 2]) / K[0, 0] * z,
        (v - K[1, 2]) / K[1, 1] * z, z)), dtype=np.float32)


def predict(detector, depth, K, *, depth_max_m=1.5, nms=True):
    points = points_from_depth(depth, K, depth_max_m)
    if not len(points):
        return np.empty((0, 17), dtype=np.float64)
    gg = detector.get_grasp(points, {"dense_grasp": False, "collision_detection": True})
    if gg is None or not len(gg):
        return np.empty((0, 17), dtype=np.float64)
    if nms:
        gg = gg.nms()
    gg = gg.sort_by_score()
    # Width is SDK required opening, not a reconstructed object width.
    return np.asarray(gg.grasp_group_array, dtype=np.float64).copy()


def sdk_runtime(sdk_dir, work_dir):
    sdk = Path(sdk_dir).expanduser().resolve()
    license_dir = sdk / "license_registration" / "license"
    if not (license_dir / "licenseCfg.json").is_file():
        raise RuntimeError(f"AnyGrasp license directory missing: {license_dir}")
    libraries = [sdk / "grasp_detection" / "gsnet_versions" / ("gsnet" + suffix)
                 for suffix in importlib.machinery.EXTENSION_SUFFIXES]
    library = next((p for p in libraries if p.is_file()), None)
    if library is None:
        raise RuntimeError(f"AnyGrasp SDK binary for Python {sys.version_info[:2]} missing")
    runtime = Path(work_dir).resolve() / "sdk_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for name, source in (("gsnet.so", library), ("license", license_dir)):
        target = runtime / name
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        target.symlink_to(source, target_is_directory=source.is_dir())
    return runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--sdk_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max_gripper_width", type=float, default=0.1)
    ap.add_argument("--gripper_height", type=float, default=0.03)
    ap.add_argument("--depth_max_m", type=float, default=1.5)
    ap.add_argument("--no-nms", action="store_true", help="Common benchmark: retain SDK candidates")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"AnyGrasp checkpoint missing: {checkpoint}")
    if not 0 < args.max_gripper_width <= .1 or not 0 < args.gripper_height < .2:
        raise ValueError("Invalid gripper dimensions")
    work_dir = str(Path(args.work_dir).resolve())
    runtime = sdk_runtime(args.sdk_dir, work_dir)
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    import torch
    from gsnet import create_detector

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    detector = create_detector(argparse.Namespace(
        checkpoint_path=str(checkpoint), max_gripper_width=args.max_gripper_width,
        gripper_height=args.gripper_height))
    if detector is None:
        raise RuntimeError("AnyGrasp license validation failed; see worker log")
    runtime_info = {'python': sys.version.split()[0], 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                    'seed': args.seed, 'nms': not args.no_nms,
                    'depth_max_m': args.depth_max_m, 'collision_detection': True,
                    'width_semantics': 'required_opening_m'}
    (Path(work_dir) / 'runtime.json').write_text(json.dumps(runtime_info, indent=2))
    print(f"-> AnyGrasp loaded {checkpoint}; width=required_opening_m; nms={not args.no_nms}", flush=True)

    def infer(rgb, depth, K):
        return predict(detector, depth, K, depth_max_m=args.depth_max_m, nms=not args.no_nms)

    serve(work_dir, infer, poll_interval=.005, synchronize=torch.cuda.synchronize)


if __name__ == "__main__":
    main()
