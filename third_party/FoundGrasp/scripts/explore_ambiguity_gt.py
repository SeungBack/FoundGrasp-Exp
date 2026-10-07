"""
Ambiguity-Aware Supervision 그림용 데이터 탐색 스크립트.

어떤 장면, 어떤 픽셀이 그림에 적합한지 찾기 위한 탐색 도구.

Usage:
    python scripts/explore_ambiguity_gt.py --scene_ids 0 1 2 37 --ann_id 0 --out_dir /tmp/ambiguity_explore
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

GT_ROOT = '/data/Grasp/GraspNet-1Billion/pixelwise_grasp_gt_top300/realsense'
SCENE_ROOT = '/data/Grasp/GraspNet-1Billion/scenes'


def load_scene(scene_id, ann_id):
    scene_str = f'scene_{str(scene_id).zfill(4)}'
    rgb = np.array(Image.open(f'{SCENE_ROOT}/{scene_str}/realsense/rgb/{str(ann_id).zfill(4)}.png'))
    depth = np.array(Image.open(f'{SCENE_ROOT}/{scene_str}/realsense/depth/{str(ann_id).zfill(4)}.png'))
    gt = np.load(f'{GT_ROOT}/{scene_str}_ann_{str(ann_id).zfill(4)}.npz', allow_pickle=True)
    return rgb, depth, gt


def compute_angular_spread(approach_dirs, scores):
    """각 픽셀의 valid approach direction angular spread (std in degrees)."""
    N = scores.shape[0]
    spreads = np.zeros(N)
    valid_counts = (scores > 0).sum(axis=1)

    for i in range(N):
        valid = approach_dirs[i][scores[i] > 0]
        if len(valid) < 2:
            spreads[i] = 0.0
            continue
        mean_dir = valid.mean(axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm < 1e-6:
            spreads[i] = 90.0
            continue
        mean_dir /= norm
        dots = np.clip((valid * mean_dir).sum(axis=1), -1, 1)
        spreads[i] = np.arccos(dots).std() * 180 / np.pi

    return spreads, valid_counts


def pick_candidate_pixels(pixels, spreads, valid_counts, n=4):
    """
    대비가 잘 되는 픽셀 쌍을 고른다:
    - high_spread: valid grasps >= 20, spread가 큰 것 (ambiguous)
    - low_spread:  valid grasps >= 20, spread가 작은 것 (consistent)
    """
    mask = valid_counts >= 20
    if mask.sum() < 4:
        mask = valid_counts >= 5

    idx = np.where(mask)[0]
    sub_spreads = spreads[idx]

    sorted_by_spread = idx[np.argsort(sub_spreads)]
    # low spread: bottom n, high spread: top n
    low_idx = sorted_by_spread[:n]
    high_idx = sorted_by_spread[-n:]

    return low_idx, high_idx


def plot_hemisphere(ax, approach_dirs, scores, title, color_by_score=True):
    """
    Valid approach directions를 반구 투영으로 시각화.
    x-y plane에서 z > 0인 방향들을 polar-like으로 표시.
    """
    valid = approach_dirs[scores > 0]
    valid_scores = scores[scores > 0]

    if len(valid) == 0:
        ax.set_title(title, fontsize=7)
        return

    # Ensure unit vectors
    norms = np.linalg.norm(valid, axis=1, keepdims=True)
    valid = valid / np.clip(norms, 1e-6, None)

    # Stereographic-like: project onto xy plane, dot with z gives elevation
    # x_proj = x / (1 - z), y_proj = y / (1 - z)  -- but simpler: just use (x, y)
    # Use azimuth + elevation on unit sphere:
    # elevation from camera forward (assume z ~ camera optical axis)
    elev = np.arcsin(np.clip(valid[:, 2], -1, 1)) * 180 / np.pi
    azim = np.arctan2(valid[:, 1], valid[:, 0]) * 180 / np.pi

    sc = ax.scatter(azim, elev, c=valid_scores if color_by_score else 'steelblue',
                    cmap='plasma', s=8, alpha=0.7,
                    vmin=0, vmax=1 if color_by_score else None)

    spread_std = elev.std() + azim.std()
    ax.set_title(f'{title}\n{len(valid)} valid, spread≈{spread_std:.0f}°', fontsize=7)
    ax.set_xlabel('Azimuth (°)', fontsize=6)
    ax.set_ylabel('Elevation (°)', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    return sc


def plot_gripper_symmetry(ax, gripper_dirs, scores, title):
    """
    Gripper direction의 g ≡ -g 대칭성 시각화.
    gripper_dirs: (K, 3) — K views에 대한 gripper direction
    """
    valid = gripper_dirs[scores > 0]
    if len(valid) == 0:
        ax.set_title(title, fontsize=7)
        return

    norms = np.linalg.norm(valid, axis=1, keepdims=True)
    valid = valid / np.clip(norms, 1e-6, None)

    # Map all to upper hemisphere (z >= 0) by flipping if z < 0
    flipped = valid.copy()
    flip_mask = flipped[:, 2] < 0
    flipped[flip_mask] *= -1

    # Project to 2D: use x, y of upper hemisphere directions
    azim = np.arctan2(flipped[:, 1], flipped[:, 0]) * 180 / np.pi
    elev = np.arcsin(np.clip(flipped[:, 2], -1, 1)) * 180 / np.pi

    ax.scatter(azim, elev, c='coral', s=8, alpha=0.7, label='g')

    # Also show raw (some negative z)
    azim_raw = np.arctan2(valid[:, 1], valid[:, 0]) * 180 / np.pi
    elev_raw = np.arcsin(np.clip(valid[:, 2], -1, 1)) * 180 / np.pi
    ax.scatter(azim_raw, elev_raw, c='steelblue', s=4, alpha=0.4, label='raw')

    ax.set_title(f'{title}\n(coral=symmetric)', fontsize=7)
    ax.set_xlabel('Azimuth (°)', fontsize=6)
    ax.set_ylabel('Elevation (°)', fontsize=6)
    ax.tick_params(labelsize=5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)


def make_scene_figure(scene_id, ann_id, out_dir):
    try:
        rgb, depth, gt = load_scene(scene_id, ann_id)
    except FileNotFoundError as e:
        print(f'Skip scene {scene_id}: {e}')
        return

    pixels = gt['valid_pixel_indices'].astype(np.int32)   # (N, 2) [v, u]
    scores = gt['grasp_score'].astype(np.float32)          # (N, K)
    app_dirs = gt['approach_dir'].astype(np.float32)       # (N, K, 3)
    grip_dirs = gt['gripper_dir'].astype(np.float32)       # (N, K, 3)
    graspness = gt['graspness'].astype(np.float32)         # (N,)

    spreads, valid_counts = compute_angular_spread(app_dirs, scores)
    low_idx, high_idx = pick_candidate_pixels(pixels, spreads, valid_counts, n=4)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Scene {scene_id:04d}, ann {ann_id:04d} — Approach direction spread', fontsize=11)

    # --- Row 1: Scene overview ---
    ax_rgb = fig.add_axes([0.01, 0.55, 0.28, 0.40])
    ax_depth = fig.add_axes([0.30, 0.55, 0.28, 0.40])
    ax_spread = fig.add_axes([0.59, 0.55, 0.28, 0.40])

    ax_rgb.imshow(rgb)
    ax_rgb.set_title(f'RGB (scene {scene_id:04d})', fontsize=8)
    ax_rgb.axis('off')

    depth_vis = depth.astype(np.float32)
    depth_vis[depth_vis == 0] = np.nan
    ax_depth.imshow(depth_vis, cmap='viridis')
    ax_depth.set_title('Depth', fontsize=8)
    ax_depth.axis('off')

    # Scatter valid pixels colored by spread
    ax_spread.imshow(rgb)
    sc = ax_spread.scatter(pixels[:, 1], pixels[:, 0], c=spreads,
                           cmap='RdYlGn_r', s=2, alpha=0.8, vmin=0, vmax=40)
    # Mark selected pixels
    for idx in low_idx:
        ax_spread.plot(pixels[idx, 1], pixels[idx, 0], 'bs', ms=6, mfc='none', mew=1.5)
    for idx in high_idx:
        ax_spread.plot(pixels[idx, 1], pixels[idx, 0], 'r^', ms=6, mfc='none', mew=1.5)
    plt.colorbar(sc, ax=ax_spread, fraction=0.03, label='Spread (°)')
    ax_spread.set_title('Approach spread (blue□=low / red△=high)', fontsize=8)
    ax_spread.axis('off')

    # --- Row 2: Hemisphere plots for selected pixels ---
    n_show = 4
    for col, idx in enumerate(list(low_idx[:2]) + list(high_idx[:2])):
        label_type = 'LOW spread' if col < 2 else 'HIGH spread'
        v, u = pixels[idx]
        vc = valid_counts[idx]
        sp = spreads[idx]

        ax = fig.add_axes([0.01 + col * 0.22, 0.05, 0.20, 0.40])
        plot_hemisphere(ax, app_dirs[idx], scores[idx],
                        f'({u},{v}) {label_type}\n{vc} valid, std={sp:.1f}°')
        # Mark on spread image
        marker = 's' if col < 2 else '^'
        color = 'blue' if col < 2 else 'red'
        ax_spread.plot(u, v, marker=marker, ms=8, color=color, mew=2)
        ax_rgb.plot(u, v, marker=marker, ms=6, color=color, mew=2)

    # Add gripper symmetry plot for one interesting pixel
    best_high = high_idx[-1]
    v, u = pixels[best_high]
    ax_sym = fig.add_axes([0.89, 0.05, 0.10, 0.40])
    plot_gripper_symmetry(ax_sym, grip_dirs[best_high], scores[best_high],
                          f'Gripper dirs\n({u},{v})')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'scene{scene_id:04d}_ann{ann_id:04d}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

    # Also save a summary CSV
    summary_path = os.path.join(out_dir, f'scene{scene_id:04d}_ann{ann_id:04d}_pixels.txt')
    with open(summary_path, 'w') as f:
        f.write('type,pixel_idx,v,u,valid_count,spread_std\n')
        for idx in low_idx:
            v, u = pixels[idx]
            f.write(f'low,{idx},{v},{u},{valid_counts[idx]:.0f},{spreads[idx]:.2f}\n')
        for idx in high_idx:
            v, u = pixels[idx]
            f.write(f'high,{idx},{v},{u},{valid_counts[idx]:.0f},{spreads[idx]:.2f}\n')
    print(f'Pixel candidates: {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_ids', type=int, nargs='+', default=[0, 1, 5, 10, 37, 50])
    parser.add_argument('--ann_id', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='/tmp/ambiguity_explore')
    args = parser.parse_args()

    for sid in args.scene_ids:
        make_scene_figure(sid, args.ann_id, args.out_dir)

    print(f'\nDone. Results in {args.out_dir}/')
