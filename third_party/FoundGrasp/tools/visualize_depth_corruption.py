import argparse
import os
import random
import sys
import time

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.graspnet1b import GraspNet1BDataset
from utils.config import load_config_with_base, resolve_config_path
from utils.visualization import apply_colormap, visualize_and_save_opencv


def _load_optional_config(config_arg):
    if not config_arg:
        return None
    config_path = resolve_config_path(config_arg)
    return load_config_with_base(config_path)


def _seed_all(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def _add_panel_label(img, text):
    banner_h = 44
    labeled = cv2.copyMakeBorder(img, banner_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def _hconcat_resized(left, right):
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.hconcat([left, right])


def _resolve_pair(clean_vis, corrupt_vis, key):
    left_key = key
    left_label = "before"
    right_label = "after"

    if key == "depth_delta":
        left_key = "depth"
        left_label = "clean depth"
        right_label = "depth delta"
    elif key == "depth_delta_overlay":
        left_key = "depth_delta_overlay"
        left_label = "clean depth"
        right_label = "depth delta overlay"
    elif key == "xyz_delta":
        left_key = "xyz"
        left_label = "clean xyz"
        right_label = "xyz delta"
    elif key == "xyz_delta_overlay":
        left_key = "xyz_delta_overlay"
        left_label = "clean color"
        right_label = "xyz delta overlay"
    elif key == "depth_corruption_mask":
        left_key = "depth"
        left_label = "clean depth"
        right_label = "corruption mask"
    elif key == "depth_removed":
        left_key = "depth"
        left_label = "clean depth"
        right_label = "depth removed"

    left = clean_vis.get(left_key)
    right = corrupt_vis.get(key)
    return left, right, left_label, right_label


def _make_depth_corruption_cfg(args, cfg):
    if cfg is not None:
        depth_cfg = cfg.training.get("depth_corruption", {})
        if args.force_prob is not None:
            depth_cfg.prob = args.force_prob
        return {
            "type": depth_cfg.get("type", "none"),
            "prob": depth_cfg.get("prob", 0.0),
            "num_patches": list(depth_cfg.get("num_patches", depth_cfg.get("num_blobs", [2, 5]))),
            "patch_size_range": list(depth_cfg.get("patch_size_range", depth_cfg.get("radius_range", [12, 40]))),
            "edge_dropout": depth_cfg.get("edge_dropout", 0.0),
            "edge_kernel": depth_cfg.get("edge_kernel", 5),
            "edge_grad_thresh": depth_cfg.get("edge_grad_thresh", 0.02),
            "far_bias": depth_cfg.get("far_bias", 0.0),
            "sigma_range": list(depth_cfg.get("sigma_range", [0.001, 0.002])),
            "sigma_depth_scale": depth_cfg.get("sigma_depth_scale", 0.5),
        }

    return {
        "type": "sensor_style",
        "prob": 1.0 if args.force_prob is None else args.force_prob,
        "num_patches": args.num_patches,
        "patch_size_range": args.patch_size_range,
        "edge_dropout": args.edge_dropout,
        "edge_kernel": args.edge_kernel,
        "edge_grad_thresh": args.edge_grad_thresh,
        "far_bias": args.far_bias,
        "sigma_range": args.sigma_range,
        "sigma_depth_scale": args.sigma_depth_scale,
    }


def _normalize_aug_tokens(augmentation, keep_geom_aug=False):
    if not augmentation:
        return None
    aug_tokens = [token.strip() for token in str(augmentation).split(",") if token.strip()]
    aug_tokens = [token for token in aug_tokens if token != "color"]
    if not keep_geom_aug:
        aug_tokens = [token for token in aug_tokens if token not in {"crop", "hflip", "vflip"}]
    return ",".join(aug_tokens) if aug_tokens else None


def _build_dataset_kwargs(args, cfg):
    if cfg is not None:
        augmentation = _normalize_aug_tokens(cfg.training.get("augmentation", None), keep_geom_aug=args.keep_geom_aug)
        return {
            "root": cfg.dataset.root,
            "camera": cfg.dataset.camera,
            "split": args.split or cfg.dataset.get("split", "train"),
            "image_size": tuple(cfg.dataset.image_size),
            "backbone_name": cfg.model.backbone.name,
            "augmentation": augmentation,
            "input_type": cfg.model.get("input_type", "rgbd"),
        }

    augmentation = _normalize_aug_tokens(args.augmentation, keep_geom_aug=args.keep_geom_aug)
    return {
        "root": args.root,
        "camera": args.camera,
        "split": args.split or "train",
        "image_size": tuple(args.image_size),
        "backbone_name": args.backbone_name,
        "augmentation": augmentation,
        "input_type": args.input_type,
    }


def _extract_vis_sample(sample, corruption_mask=None, depth_removed=None, depth_delta=None, xyz_delta=None):
    vis_sample = {
        "og_color": sample["og_color"],
        "depth": sample["depth"],
        "depth_mask": sample["depth_mask"],
        "xyz": sample["xyz"],
        "scene_id": sample.get("scene_id", "unknown"),
        "im_id": sample.get("im_id", 0),
    }
    if corruption_mask is not None:
        vis_sample["depth_corruption_mask"] = corruption_mask
    if depth_removed is not None:
        vis_sample["depth_removed"] = depth_removed
    if depth_delta is not None:
        vis_sample["depth_delta"] = depth_delta
    if xyz_delta is not None:
        vis_sample["xyz_delta"] = xyz_delta
    return vis_sample


def _build_corruption_visuals(clean_sample, corrupt_sample, depth_corruption_cfg):
    clean_depth = clean_sample["depth"]
    corrupt_depth = corrupt_sample["depth"]
    clean_xyz = clean_sample["xyz"]
    corrupt_xyz = corrupt_sample["xyz"]
    clean_valid = clean_depth > 0.1
    corrupt_valid = corrupt_depth > 0.1
    stable_valid = clean_valid & corrupt_valid

    depth_delta = (clean_depth - corrupt_depth).abs()
    xyz_delta = torch.linalg.norm(clean_xyz - corrupt_xyz, dim=0, keepdim=True)
    depth_delta = torch.where(stable_valid, depth_delta, torch.full_like(depth_delta, float("nan")))
    xyz_delta = torch.where(stable_valid, xyz_delta, torch.full_like(xyz_delta, float("nan")))

    corruption_type = str(depth_corruption_cfg.get("type", "none")).lower()
    extra_maps = {
        "depth_delta": depth_delta,
        "xyz_delta": xyz_delta,
    }
    keys = ["og_color", "depth", "depth_mask", "xyz", "depth_delta_overlay", "xyz_delta_overlay", "depth_delta", "xyz_delta"]

    if corruption_type == "sensor_style":
        corruption_mask = ((clean_depth > 0.1) & (corrupt_depth <= 0.1)).float()
        depth_removed = (clean_depth - corrupt_depth).clamp_min(0.0)
        extra_maps["depth_corruption_mask"] = corruption_mask
        extra_maps["depth_removed"] = depth_removed
        keys.extend(["depth_corruption_mask", "depth_removed"])

    return extra_maps, keys


def _make_clean_color_bgr(sample):
    color = sample["og_color"].detach().cpu().numpy()
    color = np.clip(color.transpose(1, 2, 0), 0.0, 1.0)
    color = (color * 255.0).astype(np.uint8)
    return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)


def _make_clean_depth_bgr(sample):
    depth = sample["depth"][0].detach().cpu().numpy().astype(np.float32)
    depth = depth.copy()
    depth[depth <= 0.1] = np.nan
    return apply_colormap(depth)


def _make_delta_overlay(base_bgr, delta_map, alpha=0.65):
    delta = delta_map[0].detach().cpu().numpy().astype(np.float32)
    valid = np.isfinite(delta)
    if not np.any(valid):
        return base_bgr.copy()

    delta_vis = delta.copy()
    hi = np.nanpercentile(delta_vis[valid], 99.0)
    if hi <= 1e-8:
        return base_bgr.copy()
    delta_vis = np.clip(delta_vis, 0.0, hi)
    heat = apply_colormap(delta_vis)
    mask = valid & (delta_vis > 1e-8)

    out = base_bgr.astype(np.float32).copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * heat[mask].astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _measure_fetch_time(dataset, index, seed, warmup, repeats):
    for i in range(warmup):
        _seed_all(seed + i)
        _ = dataset[index]

    times_ms = []
    for i in range(repeats):
        _seed_all(seed + 1000 + i)
        start = time.perf_counter()
        _ = dataset[index]
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def main():
    parser = argparse.ArgumentParser(description="Visualize clean vs sensor-style depth corruption.")
    parser.add_argument("--config", type=str, default=None, help="Optional config path/name.")
    parser.add_argument("--root", type=str, default="/data/Grasp/GraspNet-1Billion")
    parser.add_argument("--camera", type=str, default="realsense")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=[576, 1024], metavar=("H", "W"))
    parser.add_argument("--backbone-name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--augmentation", type=str, default=None)
    parser.add_argument("--input-type", type=str, default="xyz")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="figures/depth_corruption")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--force-prob", type=float, default=None, help="Override config depth corruption prob.")
    parser.add_argument("--num-patches", "--num-blobs", dest="num_patches", type=int, nargs=2, default=[2, 5])
    parser.add_argument("--patch-size-range", "--radius-range", dest="patch_size_range", type=int, nargs=2, default=[12, 40])
    parser.add_argument("--edge-dropout", type=float, default=0.1)
    parser.add_argument("--edge-kernel", type=int, default=5)
    parser.add_argument("--edge-grad-thresh", type=float, default=0.02)
    parser.add_argument("--far-bias", type=float, default=0.0)
    parser.add_argument("--sigma-range", type=float, nargs=2, default=[0.001, 0.002])
    parser.add_argument("--sigma-depth-scale", type=float, default=0.5)
    parser.add_argument("--keep-geom-aug", action="store_true", help="Keep crop/hflip/vflip for visualization.")
    args = parser.parse_args()

    cfg = _load_optional_config(args.config)
    dataset_kwargs = _build_dataset_kwargs(args, cfg)
    depth_corruption_cfg = _make_depth_corruption_cfg(args, cfg)

    try:
        clean_dataset = GraspNet1BDataset(**dataset_kwargs, depth_corruption=None)
        corrupt_dataset = GraspNet1BDataset(**dataset_kwargs, depth_corruption=depth_corruption_cfg)
    except OSError as exc:
        raise SystemExit(
            "Failed to load image processor. Pass a valid `--config` or set "
            "`--backbone-name facebook/dinov2-base`.\n"
            f"Original error: {exc}"
        ) from exc

    _seed_all(args.seed)
    clean_sample = clean_dataset[args.index]
    _seed_all(args.seed)
    corrupt_sample = corrupt_dataset[args.index]

    extra_maps, keys = _build_corruption_visuals(clean_sample, corrupt_sample, depth_corruption_cfg)

    clean_vis = visualize_and_save_opencv(_extract_vis_sample(clean_sample), save_dir=None)
    corrupt_vis = visualize_and_save_opencv(
        _extract_vis_sample(
            corrupt_sample,
            corruption_mask=extra_maps.get("depth_corruption_mask"),
            depth_removed=extra_maps.get("depth_removed"),
            depth_delta=extra_maps.get("depth_delta"),
            xyz_delta=extra_maps.get("xyz_delta"),
        ),
        save_dir=None,
    )
    clean_depth_bgr = _make_clean_depth_bgr(clean_sample)
    clean_color_bgr = _make_clean_color_bgr(clean_sample)
    clean_vis["depth_delta_overlay"] = clean_depth_bgr
    corrupt_vis["depth_delta_overlay"] = _make_delta_overlay(clean_depth_bgr, extra_maps["depth_delta"])
    clean_vis["xyz_delta_overlay"] = clean_color_bgr
    corrupt_vis["xyz_delta_overlay"] = _make_delta_overlay(clean_color_bgr, extra_maps["xyz_delta"])

    clean_times = _measure_fetch_time(clean_dataset, args.index, args.seed, args.warmup, args.repeats)
    corrupt_times = _measure_fetch_time(corrupt_dataset, args.index, args.seed, args.warmup, args.repeats)
    clean_mean = float(np.mean(clean_times))
    corrupt_mean = float(np.mean(corrupt_times))
    overhead_ms = corrupt_mean - clean_mean
    overhead_ratio = (corrupt_mean / max(clean_mean, 1e-6)) - 1.0

    os.makedirs(args.save_dir, exist_ok=True)
    rows = []
    for key in keys:
        left, right, left_label, right_label = _resolve_pair(clean_vis, corrupt_vis, key)
        if right is None:
            continue
        if left is None:
            left = np.zeros_like(right)
        pair = _hconcat_resized(_add_panel_label(left, left_label), _add_panel_label(right, right_label))
        rows.append(pair)
        cv2.imwrite(os.path.join(args.save_dir, f"{key}_comparison.png"), pair)

    if rows:
        max_w = max(img.shape[1] for img in rows)
        padded_rows = []
        for img in rows:
            if img.shape[1] < max_w:
                pad_w = max_w - img.shape[1]
                img = cv2.copyMakeBorder(img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded_rows.append(img)
        grid = cv2.vconcat(padded_rows)
        cv2.imwrite(os.path.join(args.save_dir, "comparison_grid.png"), grid)

    timing_lines = [
        f"clean_fetch_ms_mean: {clean_mean:.3f}",
        f"clean_fetch_ms_std: {float(np.std(clean_times)):.3f}",
        f"corrupt_fetch_ms_mean: {corrupt_mean:.3f}",
        f"corrupt_fetch_ms_std: {float(np.std(corrupt_times)):.3f}",
        f"overhead_ms_mean: {overhead_ms:.3f}",
        f"overhead_ratio: {overhead_ratio * 100.0:.2f}%",
        f"warmup: {args.warmup}",
        f"repeats: {args.repeats}",
    ]
    with open(os.path.join(args.save_dir, "timing.txt"), "w", encoding="ascii") as f:
        f.write("\n".join(timing_lines) + "\n")

    print(f"Saved comparison to {args.save_dir}")
    print(f"Scene: {clean_sample['scene_id']}, image: {clean_sample['im_id']}, index: {args.index}")
    print(f"Depth corruption config: {depth_corruption_cfg}")
    print(
        "Fetch time: "
        f"clean {clean_mean:.2f} ms, corrupt {corrupt_mean:.2f} ms, "
        f"overhead {overhead_ms:.2f} ms ({overhead_ratio * 100.0:.2f}%)"
    )


if __name__ == "__main__":
    main()
