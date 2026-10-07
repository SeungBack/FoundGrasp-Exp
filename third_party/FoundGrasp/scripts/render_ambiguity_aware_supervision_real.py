#!/usr/bin/env python3
"""Render a real-data Ambiguity-Aware Supervision figure for FoundGrasp.

The script intentionally uses real GraspNet-style point cloud geometry and
pixelwise grasp labels. It saves two drafts:

  - Option A: one compact local 3D panel.
  - Option B: three micro-panels; this is copied to the final requested path.

The renderer prefers pixelwise_grasp_gt_v3 when available. On this workspace the
available GT is pixelwise_grasp_gt_top300, whose per-pixel top-300 candidates are
used as valid orientation targets.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_ROOT = "/data/Grasp/GraspNet-1Billion"
DEFAULT_CAMERA = "realsense"
DEFAULT_SCENE = 0
DEFAULT_ANN = 19
DEFAULT_PIXEL_INDEX = 1034

SURFACE_GRAY = "#9a9a9a"
SURFACE_DARK = "#5f5f5f"
SUPPORT_GREEN = "#2ca25f"
SEED_GREEN = "#0b6b3a"
APPROACH_RED = "#d62728"
GRIPPER_ORANGE = "#e68613"
DARK = "#222222"


@dataclass
class Candidate:
    gt_path: str
    gt_dir_name: str
    scene_id: int
    ann_id: int
    pixel_index: int
    v: int
    u: int
    chosen: list[int]
    best_index: int
    best_score: float
    diversity: float


@dataclass
class FigureData:
    candidate: Candidate
    center: np.ndarray
    surface_points: np.ndarray
    support_points: np.ndarray
    sparse_points: np.ndarray
    approach_dir: np.ndarray
    gripper_dirs: np.ndarray
    rgb_crop: np.ndarray | None
    support_radius_px: int
    support_depth_thresh: float
    object_id: int


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _parse_scene_ann(path: str) -> tuple[int, int]:
    name = os.path.basename(path)
    m = re.search(r"scene_(\d+)_ann_(\d+)\.npz", name)
    if not m:
        raise ValueError(f"Cannot parse scene/ann from {path}")
    return int(m.group(1)), int(m.group(2))


def _find_gt_root(root: str, camera: str) -> tuple[str, str]:
    for name in ("pixelwise_grasp_gt_v3", "pixelwise_grasp_gt_top300"):
        path = os.path.join(root, name, camera)
        if os.path.isdir(path) and glob.glob(os.path.join(path, "*.npz")):
            return path, name
    raise FileNotFoundError(
        f"No pixelwise GT found under {root}. Expected pixelwise_grasp_gt_v3 "
        f"or pixelwise_grasp_gt_top300 with camera folder '{camera}'."
    )


def _load_gt(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        out = {key: data[key] for key in data.files}
    return out


def _flatten_candidate_axis(gt: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten v3 (N,K,P,...) or top300 (N,K,...) labels to (N,C,...)."""
    score = gt["grasp_score"].astype(np.float32)
    approach = gt["approach_dir"].astype(np.float32)
    gripper = gt["gripper_dir"].astype(np.float32)
    width = gt["grasp_width"].astype(np.float32)
    depth = gt["grasp_depth"].astype(np.float32)

    if score.ndim == 3:
        n = score.shape[0]
        score = score.reshape(n, -1)
        width = width.reshape(n, -1)
        depth = depth.reshape(n, -1)
        approach = approach.reshape(n, -1, 3)
        gripper = gripper.reshape(n, -1, 3)
    elif score.ndim != 2:
        raise ValueError(f"Unsupported score shape: {score.shape}")

    return {
        "score": score,
        "width": width,
        "depth": depth,
        "approach": _norm(approach),
        "gripper": _norm(gripper),
    }


def _greedy_axis_diverse(
    scores: np.ndarray,
    grippers: np.ndarray,
    candidates: list[int],
    max_axes: int = 6,
    axis_cos_thresh: float = 0.94,
) -> list[int]:
    chosen: list[int] = []
    ordered = sorted(candidates, key=lambda j: -float(scores[j]))
    for idx in ordered:
        g = grippers[idx]
        if not chosen:
            chosen.append(idx)
            continue
        # Gripper axis is symmetric, so abs(dot) measures axis similarity.
        if all(abs(float(np.dot(g, grippers[c]))) < axis_cos_thresh for c in chosen):
            chosen.append(idx)
        if len(chosen) >= max_axes:
            break
    return chosen


def _candidate_diversity(grippers: np.ndarray, chosen: list[int]) -> float:
    if len(chosen) < 2:
        return 0.0
    axes = grippers[chosen]
    dots = np.abs(axes @ axes.T)
    return float((1.0 - dots[np.triu_indices(len(chosen), 1)]).mean())


def _candidate_from_file(
    gt_path: str,
    gt_dir_name: str,
    pixel_index: int | None,
    approach_cos_thresh: float,
    score_thresh: float,
) -> Candidate | None:
    gt = _load_gt(gt_path)
    flat = _flatten_candidate_axis(gt)
    pixels = gt["valid_pixel_indices"].astype(np.int32)
    scores = flat["score"]
    approach = flat["approach"]
    gripper = flat["gripper"]
    scene_id, ann_id = _parse_scene_ann(gt_path)

    if pixel_index is not None:
        order = [pixel_index]
    else:
        max_score = scores.max(axis=1)
        order = np.argsort(-max_score)[: min(800, len(max_score))]

    for pi in order:
        if pi < 0 or pi >= len(pixels):
            continue
        best_index = int(np.argmax(scores[pi]))
        best_score = float(scores[pi, best_index])
        if best_score <= 0:
            continue

        app0 = approach[pi, best_index]
        valid = np.where(scores[pi] > score_thresh)[0]
        valid = [
            int(j)
            for j in valid
            if float(np.dot(app0, approach[pi, j])) > approach_cos_thresh
        ]
        if len(valid) < 3:
            continue

        chosen = _greedy_axis_diverse(scores[pi], gripper[pi], valid)
        if len(chosen) < 3:
            continue

        v, u = map(int, pixels[pi])
        return Candidate(
            gt_path=gt_path,
            gt_dir_name=gt_dir_name,
            scene_id=scene_id,
            ann_id=ann_id,
            pixel_index=int(pi),
            v=v,
            u=u,
            chosen=chosen,
            best_index=best_index,
            best_score=best_score,
            diversity=_candidate_diversity(gripper[pi], chosen),
        )
    return None


def _auto_select_candidate(
    gt_root: str,
    gt_dir_name: str,
    max_files: int = 250,
    approach_cos_thresh: float = 0.90,
    score_thresh: float = 0.15,
) -> Candidate:
    files = sorted(glob.glob(os.path.join(gt_root, "scene_*_ann_*.npz")))[:max_files]
    ranked: list[tuple[float, Candidate]] = []

    for path in files:
        candidate = _candidate_from_file(
            path,
            gt_dir_name,
            pixel_index=None,
            approach_cos_thresh=approach_cos_thresh,
            score_thresh=score_thresh,
        )
        if candidate is None:
            continue
        # Prefer high score, several distinct axes, and approach-consistent ambiguity.
        rank = candidate.best_score + 0.35 * len(candidate.chosen) + 0.8 * candidate.diversity
        ranked.append((rank, candidate))

    if not ranked:
        raise RuntimeError("Could not find a suitable ambiguous supervision sample.")
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def _load_scene_rgb(root: str, camera: str, scene_id: int, ann_id: int) -> np.ndarray | None:
    path = os.path.join(
        root,
        "scenes",
        f"scene_{scene_id:04d}",
        camera,
        "rgb",
        f"{ann_id:04d}.png",
    )
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("RGB"))


def _load_scene_label(root: str, camera: str, scene_id: int, ann_id: int) -> np.ndarray | None:
    path = os.path.join(
        root,
        "scenes",
        f"scene_{scene_id:04d}",
        camera,
        "label",
        f"{ann_id:04d}.png",
    )
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path))


def _build_figure_data(
    root: str,
    camera: str,
    candidate: Candidate,
    surface_window_px: int = 90,
    surface_stride: int = 2,
    surface_radius_m: float = 0.085,
    support_radius_px: int = 28,
    support_depth_thresh: float = 0.008,
) -> FigureData:
    gt = _load_gt(candidate.gt_path)
    flat = _flatten_candidate_axis(gt)
    xyz = gt["xyz"].astype(np.float32)
    pixels = gt["valid_pixel_indices"].astype(np.int32)
    h, w = xyz.shape[:2]
    v, u = candidate.v, candidate.u
    center = xyz[v, u].astype(np.float32)

    label = _load_scene_label(root, camera, candidate.scene_id, candidate.ann_id)
    object_id = 0
    if label is not None and 0 <= v < label.shape[0] and 0 <= u < label.shape[1]:
        object_id = int(label[v, u])

    y0, y1 = max(0, v - surface_window_px), min(h, v + surface_window_px + 1)
    x0, x1 = max(0, u - surface_window_px), min(w, u + surface_window_px + 1)

    local_xyz = xyz[y0:y1:surface_stride, x0:x1:surface_stride]
    local_label = None
    if label is not None:
        local_label = label[y0:y1:surface_stride, x0:x1:surface_stride]

    surface = local_xyz.reshape(-1, 3)
    surface_mask = np.isfinite(surface).all(axis=1)
    surface_mask &= surface[:, 2] > 0.1
    surface_mask &= np.linalg.norm(surface - center, axis=1) < surface_radius_m

    if local_label is not None and object_id > 0:
        obj_mask = (local_label.reshape(-1) == object_id)
        if int((surface_mask & obj_mask).sum()) > 150:
            surface_mask &= obj_mask
    surface_points = surface[surface_mask]

    sy0, sy1 = max(0, v - support_radius_px), min(h, v + support_radius_px + 1)
    sx0, sx1 = max(0, u - support_radius_px), min(w, u + support_radius_px + 1)
    yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
    support_xyz = xyz[sy0:sy1, sx0:sx1]
    support_mask = np.isfinite(support_xyz).all(axis=-1)
    support_mask &= support_xyz[..., 2] > 0.1
    support_mask &= (np.square(yy - v) + np.square(xx - u)) <= support_radius_px**2
    support_mask &= np.abs(support_xyz[..., 2] - center[2]) <= support_depth_thresh
    if label is not None and object_id > 0:
        support_label = label[sy0:sy1, sx0:sx1]
        support_mask &= support_label == object_id
    support_points = support_xyz[support_mask]

    sparse_mask = (
        (pixels[:, 0] >= y0)
        & (pixels[:, 0] < y1)
        & (pixels[:, 1] >= x0)
        & (pixels[:, 1] < x1)
    )
    sparse_pix = pixels[sparse_mask]
    sparse_points = xyz[sparse_pix[:, 0], sparse_pix[:, 1]] if len(sparse_pix) else center[None]
    sparse_valid = np.isfinite(sparse_points).all(axis=1) & (sparse_points[:, 2] > 0.1)
    sparse_points = sparse_points[sparse_valid]
    if len(sparse_points) > 180:
        rng = np.random.default_rng(7)
        sparse_points = sparse_points[rng.choice(len(sparse_points), 180, replace=False)]

    rgb = _load_scene_rgb(root, camera, candidate.scene_id, candidate.ann_id)
    rgb_crop = rgb[y0:y1, x0:x1] if rgb is not None else None

    approach_dir = flat["approach"][candidate.pixel_index, candidate.best_index]
    gripper_dirs = flat["gripper"][candidate.pixel_index, candidate.chosen]

    return FigureData(
        candidate=candidate,
        center=center,
        surface_points=surface_points,
        support_points=support_points,
        sparse_points=sparse_points,
        approach_dir=_norm(approach_dir),
        gripper_dirs=_norm(gripper_dirs),
        rgb_crop=rgb_crop,
        support_radius_px=support_radius_px,
        support_depth_thresh=support_depth_thresh,
        object_id=object_id,
    )


def _centered(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    return points - center[None, :]


def _set_equal_3d(ax, points: np.ndarray, pad: float = 0.012) -> None:
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    mid = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins) / 2.0 + pad)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span * 0.75, mid[2] + span * 0.75)
    ax.set_box_aspect((1, 1, 0.72))


def _style_3d(ax, title: str | None = None) -> None:
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=26, azim=-58)
    if title:
        ax.set_title(title, fontsize=9.2, pad=0, color=DARK)


def _draw_surface(ax, data: FigureData, support: bool = False, sparse: bool = False) -> None:
    surface = _centered(data.surface_points, data.center)
    if len(surface) > 2200:
        rng = np.random.default_rng(3)
        surface = surface[rng.choice(len(surface), 2200, replace=False)]

    z = surface[:, 2]
    z_norm = (z - z.min()) / (z.ptp() + 1e-9)
    gray = 0.56 + 0.22 * z_norm
    colors = np.stack([gray, gray, gray], axis=1)
    ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], s=4.5, c=colors, alpha=0.72, linewidths=0)

    if support and len(data.support_points):
        support_pts = _centered(data.support_points, data.center)
        if len(support_pts) > 900:
            rng = np.random.default_rng(11)
            support_pts = support_pts[rng.choice(len(support_pts), 900, replace=False)]
        ax.scatter(
            support_pts[:, 0],
            support_pts[:, 1],
            support_pts[:, 2],
            s=13,
            c=SUPPORT_GREEN,
            alpha=0.86,
            linewidths=0,
            depthshade=False,
        )

    if sparse and len(data.sparse_points):
        sparse_pts = _centered(data.sparse_points, data.center)
        ax.scatter(
            sparse_pts[:, 0],
            sparse_pts[:, 1],
            sparse_pts[:, 2],
            s=12,
            facecolors="none",
            edgecolors=SEED_GREEN,
            linewidths=0.55,
            alpha=0.42,
            depthshade=False,
        )

    ax.scatter([0], [0], [0], s=48, c=DARK, edgecolors="white", linewidths=0.7, depthshade=False)
    _set_equal_3d(ax, surface)


def _draw_orientation(ax, data: FigureData, show_many: int = 6) -> None:
    _draw_surface(ax, data, support=False, sparse=False)
    arrow_len = 0.060
    app = data.approach_dir
    ax.quiver(
        0,
        0,
        0,
        app[0],
        app[1],
        app[2],
        length=arrow_len,
        color=APPROACH_RED,
        linewidth=2.2,
        arrow_length_ratio=0.22,
        normalize=True,
    )

    bar_len = 0.076
    axes = data.gripper_dirs[:show_many]
    for i, g in enumerate(axes):
        half = 0.5 * bar_len * g
        alpha = 0.96 if i == 0 else 0.68
        lw = 3.0 if i == 0 else 2.0
        ax.plot(
            [-half[0], half[0]],
            [-half[1], half[1]],
            [-half[2], half[2]],
            color=GRIPPER_ORANGE,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
        )
        # Tiny end caps make the bidirectional-bar encoding read as an axis.
        ax.scatter(
            [-half[0], half[0]],
            [-half[1], half[1]],
            [-half[2], half[2]],
            s=8 if i == 0 else 5,
            c=GRIPPER_ORANGE,
            alpha=alpha,
            depthshade=False,
        )


def _draw_kappa_panel(ax) -> None:
    theta = np.linspace(-np.pi / 2, np.pi / 2, 240)
    broad = np.exp(1.4 * np.cos(2 * theta))
    sharp = np.exp(7.0 * np.cos(2 * theta))
    broad = broad / broad.max()
    sharp = sharp / sharp.max()

    ax.plot(np.degrees(theta), broad, color="#76a9d8", linewidth=2.0)
    ax.fill_between(np.degrees(theta), 0, broad, color="#76a9d8", alpha=0.18, linewidth=0)
    ax.plot(np.degrees(theta), sharp, color=GRIPPER_ORANGE, linewidth=2.2)
    ax.fill_between(np.degrees(theta), 0, sharp, color=GRIPPER_ORANGE, alpha=0.22, linewidth=0)
    ax.axvline(0, color=DARK, linewidth=0.8, alpha=0.5)

    ax.annotate("", xy=(48, 1.15), xytext=(-48, 1.15), arrowprops=dict(arrowstyle="<->", color=GRIPPER_ORANGE, lw=1.7))
    ax.text(0, 1.24, "gripper axis", ha="center", va="bottom", fontsize=8.0, color=DARK)
    ax.text(0, 1.06, r"$g \equiv -g$", ha="center", va="top", fontsize=8.0, color=DARK)
    ax.text(-72, 0.55, "low kappa", color="#377eb8", fontsize=8.0, ha="left")
    ax.text(24, 0.77, "high kappa", color=GRIPPER_ORANGE, fontsize=8.0, ha="left")

    ax.set_title("(c) Symmetric target sharpness", fontsize=9.2, color=DARK, pad=4)
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, 1.34)
    ax.set_xlabel("orientation angle", fontsize=7.5, labelpad=1)
    ax.set_yticks([])
    ax.set_xticks([-90, 0, 90])
    ax.tick_params(axis="x", labelsize=7, length=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["bottom"].set_linewidth(0.7)


def _add_legend(fig, anchor=(0.5, 0.01)) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE_GRAY, markeredgecolor="none", markersize=5, label="local surface"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=SUPPORT_GREEN, markeredgecolor="none", markersize=5, label="propagated support"),
        plt.Line2D([0], [0], color=APPROACH_RED, linewidth=2.2, label="approach"),
        plt.Line2D([0], [0], color=GRIPPER_ORANGE, linewidth=2.2, label="valid gripper axes"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=anchor,
        ncol=4,
        frameon=False,
        fontsize=7.4,
        handlelength=1.6,
        columnspacing=1.2,
    )


def render_option_a(data: FigureData, out_base: str) -> None:
    fig = plt.figure(figsize=(5.25, 4.25))
    ax = fig.add_axes([0.02, 0.08, 0.78, 0.86], projection="3d")
    _draw_surface(ax, data, support=True, sparse=True)
    _draw_orientation(ax, data, show_many=6)
    _style_3d(ax, "Sparse label, propagated support, and valid axes")

    ax.text2D(0.03, 0.04, "real local GraspNet surface", transform=ax.transAxes, fontsize=7.4, color=SURFACE_DARK)
    ax.text2D(0.03, 0.96, "Option A", transform=ax.transAxes, fontsize=8.2, color=DARK, weight="bold")

    inset = fig.add_axes([0.73, 0.16, 0.23, 0.25])
    _draw_kappa_panel(inset)
    inset.set_title("kappa cue", fontsize=8.2, pad=2)
    inset.set_xlabel("")
    _add_legend(fig, anchor=(0.44, 0.02))

    _save(fig, out_base)


def render_option_b(data: FigureData, out_base: str) -> None:
    fig = plt.figure(figsize=(9.4, 3.25))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.15, 1.15, 0.86],
        left=0.015,
        right=0.985,
        bottom=0.17,
        top=0.91,
        wspace=0.03,
    )

    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    _draw_surface(ax0, data, support=True, sparse=True)
    _style_3d(ax0, "(a) Depth-aware propagation")
    ax0.text2D(0.05, 0.04, f"radius {data.support_radius_px}px, depth {data.support_depth_thresh*1000:.0f} mm", transform=ax0.transAxes, fontsize=7.1, color=SURFACE_DARK)

    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    _draw_orientation(ax1, data, show_many=6)
    _style_3d(ax1, "(b) Multiple valid orientations")
    ax1.text2D(0.07, 0.04, "bidirectional bars encode symmetry", transform=ax1.transAxes, fontsize=7.1, color=SURFACE_DARK)

    ax2 = fig.add_subplot(gs[0, 2])
    _draw_kappa_panel(ax2)

    fig.text(
        0.02,
        0.975,
        "Ambiguity-aware supervision from real GraspNet labels",
        ha="left",
        va="top",
        fontsize=9.8,
        color=DARK,
        weight="bold",
    )
    _add_legend(fig, anchor=(0.48, 0.035))
    _save(fig, out_base)


def _save(fig, out_base: str) -> None:
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{out_base}.{ext}"
        fig.savefig(path, dpi=420, bbox_inches="tight", pad_inches=0.025)
        print(f"saved {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--scene-id", type=int, default=DEFAULT_SCENE)
    parser.add_argument("--ann-id", type=int, default=DEFAULT_ANN)
    parser.add_argument("--pixel-index", type=int, default=DEFAULT_PIXEL_INDEX)
    parser.add_argument("--auto", action="store_true", help="Search for a sample instead of using the default selected one")
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args()

    gt_root, gt_dir_name = _find_gt_root(args.root, args.camera)
    if args.auto:
        candidate = _auto_select_candidate(gt_root, gt_dir_name)
    else:
        gt_path = os.path.join(
            gt_root,
            f"scene_{args.scene_id:04d}_ann_{args.ann_id:04d}.npz",
        )
        candidate = _candidate_from_file(
            gt_path,
            gt_dir_name,
            pixel_index=args.pixel_index,
            approach_cos_thresh=0.90,
            score_thresh=0.15,
        )
        if candidate is None:
            print("Default sample was not suitable; falling back to automatic search.")
            candidate = _auto_select_candidate(gt_root, gt_dir_name)

    data = _build_figure_data(args.root, args.camera, candidate)

    print("selected sample:")
    print(f"  gt: {candidate.gt_path}")
    print(f"  scene: scene_{candidate.scene_id:04d}, ann: {candidate.ann_id:04d}, camera: {args.camera}")
    print(f"  pixel_index: {candidate.pixel_index}, pixel: (v={candidate.v}, u={candidate.u})")
    print(f"  gt_dir: {candidate.gt_dir_name}, object_id: {data.object_id}")
    print(f"  chosen candidate indices: {candidate.chosen}")
    print(f"  best_score: {candidate.best_score:.3f}, axis_diversity: {candidate.diversity:.3f}")
    print(f"  surface points: {len(data.surface_points)}, support points: {len(data.support_points)}")

    out_dir = args.out_dir
    draft_a = os.path.join(out_dir, "ambiguity_aware_supervision_real_draft_a")
    draft_b = os.path.join(out_dir, "ambiguity_aware_supervision_real_draft_b")
    final = os.path.join(out_dir, "ambiguity_aware_supervision_real")

    render_option_a(data, draft_a)
    render_option_b(data, draft_b)

    for ext in ("png", "pdf"):
        shutil.copyfile(f"{draft_b}.{ext}", f"{final}.{ext}")
        print(f"selected Option B -> {final}.{ext}")


if __name__ == "__main__":
    main()
