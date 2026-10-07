"""
Ambiguity-Aware Supervision 논문 그림용 3D 시각화.

물체 point cloud 위에 valid approach direction 화살표를 3D로 그린다.
두 픽셀 비교: high-count (many valid grasps) vs. 같은 물체의 다른 픽셀.
g ≡ -g 대칭성 패널도 생성.

Usage:
    python scripts/render_ambiguity_3d.py --scene_id 37 --ann_id 0 --out_dir /tmp/ambiguity_3d
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import open3d as o3d

GT_ROOT = '/data/Grasp/GraspNet-1Billion/pixelwise_grasp_gt_top300/realsense'
SCENE_ROOT = '/data/Grasp/GraspNet-1Billion/scenes'


def load_data(scene_id, ann_id):
    scene_str = f'scene_{str(scene_id).zfill(4)}'
    rgb = np.array(Image.open(f'{SCENE_ROOT}/{scene_str}/realsense/rgb/{str(ann_id).zfill(4)}.png'))
    seg = np.array(Image.open(f'{SCENE_ROOT}/{scene_str}/realsense/label/{str(ann_id).zfill(4)}.png'))
    gt = np.load(f'{GT_ROOT}/{scene_str}_ann_{str(ann_id).zfill(4)}.npz', allow_pickle=True)
    return rgb, seg, gt


def make_arrow_mesh(origin, direction, length=0.02, radius=0.001, color=[1, 0.4, 0.1]):
    """Open3D arrow mesh from origin in direction."""
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius,
        cone_radius=radius * 2.5,
        cylinder_height=length * 0.75,
        cone_height=length * 0.25,
    )
    arrow.paint_uniform_color(color)
    # Default arrow points in +Z; rotate to direction
    direction = np.array(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    z = np.array([0, 0, 1.0])
    axis = np.cross(z, direction)
    axis_len = np.linalg.norm(axis)
    if axis_len > 1e-6:
        axis /= axis_len
        angle = np.arccos(np.clip(np.dot(z, direction), -1, 1))
        R = arrow.get_rotation_matrix_from_axis_angle(axis * angle)
        arrow.rotate(R, center=[0, 0, 0])
    elif np.dot(z, direction) < 0:
        R = arrow.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
        arrow.rotate(R, center=[0, 0, 0])
    arrow.translate(origin)
    return arrow


def make_gripper_mesh(origin, approach_dir, gripper_dir, width=0.06, depth=0.02, color=[0.2, 0.5, 1.0]):
    """Simple parallel-jaw gripper as line set."""
    approach_dir = np.array(approach_dir, dtype=np.float64)
    approach_dir /= np.linalg.norm(approach_dir)
    gripper_dir = np.array(gripper_dir, dtype=np.float64)
    gripper_dir -= np.dot(gripper_dir, approach_dir) * approach_dir
    gripper_dir /= np.linalg.norm(gripper_dir)

    # Gripper geometry: two finger tips + connecting bar
    # Fingers at ±width/2 along gripper_dir, extending depth along -approach_dir
    tip1 = origin + gripper_dir * width / 2
    tip2 = origin - gripper_dir * width / 2
    base1 = tip1 - approach_dir * depth
    base2 = tip2 - approach_dir * depth
    palm = origin - approach_dir * depth

    points = [tip1, tip2, base1, base2, palm]
    lines = [[0, 2], [1, 3], [2, 4], [3, 4]]  # fingers + palm bar

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(points))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines))
    ls.paint_uniform_color(color)
    return ls


def render_scene_o3d(geoms, camera_pos, camera_target, up, width=640, height=480, bg_color=(1, 1, 1, 1)):
    """Offscreen render a list of Open3D geometries."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.array(bg_color))
    renderer.scene.scene.set_sun_light([0.707, 0, -0.707], [1.0, 1.0, 1.0], 75000)
    renderer.scene.scene.enable_sun_light(True)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultLit'

    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = 'unlitLine'
    mat_line.line_width = 3.0

    for i, geom in enumerate(geoms):
        name = f'geom_{i}'
        if isinstance(geom, o3d.geometry.LineSet):
            renderer.scene.add_geometry(name, geom, mat_line)
        else:
            renderer.scene.add_geometry(name, geom, mat)

    renderer.setup_camera(60.0,
                          np.array(camera_target),
                          np.array(camera_pos),
                          np.array(up))
    img = np.asarray(renderer.render_to_image())
    del renderer
    return img


def find_best_pixel(pixels, scores, app_dirs, seg, label_id, min_count=30, mode='max_count'):
    """Find a pixel on a given object label."""
    valid_counts = (scores > 0).sum(axis=1)
    obj_mask = np.array([seg[v, u] == label_id for v, u in pixels])
    obj_idx = np.where(obj_mask)[0]
    if len(obj_idx) == 0:
        return None

    obj_counts = valid_counts[obj_idx]
    ok = obj_counts >= min_count
    if ok.sum() == 0:
        ok = obj_counts >= obj_counts.max() // 2

    cand = obj_idx[ok]
    cand_counts = valid_counts[cand]

    if mode == 'max_count':
        return cand[np.argmax(cand_counts)]
    elif mode == 'max_spread':
        spreads = []
        for i in cand:
            valid = app_dirs[i][scores[i] > 0]
            if len(valid) < 3:
                spreads.append(0.0)
                continue
            m = valid.mean(0); m /= max(np.linalg.norm(m), 1e-6)
            dots = np.clip((valid * m).sum(1), -1, 1)
            spreads.append(np.degrees(np.arccos(dots)).std())
        return cand[np.argmax(spreads)]
    return cand[0]


def render_approach_arrows(xyz, rgb_img, seg, gt, pixel_idx, label_id,
                           arrow_length=0.025, max_arrows=30, cam_offset=np.array([0, -0.15, 0.1])):
    """Render point cloud of one object + approach arrows at one pixel."""
    pixels = gt['valid_pixel_indices'].astype(np.int32)
    scores = gt['grasp_score'].astype(np.float32)
    app_dirs = gt['approach_dir'].astype(np.float32)

    v, u = pixels[pixel_idx]
    origin = xyz[v, u]

    valid = scores[pixel_idx] > 0
    dirs = app_dirs[pixel_idx][valid]
    sc = scores[pixel_idx][valid]

    # Subsample
    if len(dirs) > max_arrows:
        idx = np.linspace(0, len(dirs) - 1, max_arrows, dtype=int)
        dirs = dirs[idx]
        sc = sc[idx]

    # Normalize scores for color
    sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-6)

    # Object point cloud
    obj_mask = seg == label_id
    ys, xs = np.where(obj_mask)
    pts = xyz[ys, xs]
    colors = rgb_img[ys, xs] / 255.0
    # Filter valid depth
    valid_z = pts[:, 2] > 0
    pts = pts[valid_z]; colors = colors[valid_z]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geoms = [pcd]

    # Approach arrows — color by score (blue=low, red=high)
    cmap = plt.cm.plasma
    for d, s in zip(dirs, sc_norm):
        c = np.array(cmap(s)[:3])
        geoms.append(make_arrow_mesh(origin, d, length=arrow_length, radius=0.0008, color=c))

    # Highlight the selected pixel
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
    sphere.translate(origin)
    sphere.paint_uniform_color([1.0, 0.1, 0.1])
    geoms.append(sphere)

    # Camera: look at origin from slightly above/behind
    cam_pos = origin + cam_offset
    img = render_scene_o3d(geoms, cam_pos, origin, up=[0, 0, 1],
                           width=640, height=480)
    return img, len(dirs)


def render_gripper_symmetry(xyz, rgb_img, seg, gt, pixel_idx, label_id,
                             cam_offset=np.array([0, -0.15, 0.08])):
    """Render g ≡ -g: one grasp + its 180° flip around approach axis."""
    pixels = gt['valid_pixel_indices'].astype(np.int32)
    scores = gt['grasp_score'].astype(np.float32)
    app_dirs = gt['approach_dir'].astype(np.float32)
    grip_dirs = gt['gripper_dir'].astype(np.float32)

    v, u = pixels[pixel_idx]
    origin = xyz[v, u]

    # Pick the highest-score valid grasp
    valid_idx = np.where(scores[pixel_idx] > 0)[0]
    best = valid_idx[np.argmax(scores[pixel_idx][valid_idx])]
    app = app_dirs[pixel_idx][best]
    grip = grip_dirs[pixel_idx][best]

    # Object point cloud
    obj_mask = seg == label_id
    ys, xs = np.where(obj_mask)
    pts = xyz[ys, xs]; colors = rgb_img[ys, xs] / 255.0
    valid_z = pts[:, 2] > 0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[valid_z])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid_z])

    # Approach arrow
    app_arrow = make_arrow_mesh(origin, app, length=0.03, radius=0.001, color=[0.4, 0.4, 0.4])

    # Gripper g (blue)
    g1 = make_gripper_mesh(origin, app, grip, color=[0.2, 0.4, 1.0])

    # Gripper -g (red, flip gripper_dir by 180° around approach_dir axis)
    grip_neg = -grip
    g2 = make_gripper_mesh(origin, app, grip_neg, color=[1.0, 0.3, 0.3])

    # Highlight origin
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
    sphere.translate(origin)
    sphere.paint_uniform_color([0.0, 0.8, 0.2])
    geoms = [pcd, app_arrow, g1, g2, sphere]

    cam_pos = origin + cam_offset
    img = render_scene_o3d(geoms, cam_pos, origin, up=[0, 0, 1], width=640, height=480)
    return img


def main(args):
    print(f'Loading scene {args.scene_id}, ann {args.ann_id}...')
    rgb, seg, gt = load_data(args.scene_id, args.ann_id)
    xyz = gt['xyz'].astype(np.float32)
    pixels = gt['valid_pixel_indices'].astype(np.int32)
    scores = gt['grasp_score'].astype(np.float32)
    app_dirs = gt['approach_dir'].astype(np.float32)

    label_ids = [int(x) for x in args.labels]

    os.makedirs(args.out_dir, exist_ok=True)

    panels = []
    panel_titles = []

    for lid in label_ids:
        # Find max-count pixel (representative, many valid grasps)
        px_count = find_best_pixel(pixels, scores, app_dirs, seg, lid,
                                   min_count=args.min_count, mode='max_count')
        # Find max-spread pixel (most ambiguous)
        px_spread = find_best_pixel(pixels, scores, app_dirs, seg, lid,
                                    min_count=args.min_count // 2, mode='max_spread')

        if px_count is None:
            print(f'  label {lid}: no valid pixel found, skip')
            continue

        v, u = pixels[px_count]
        vc = (scores[px_count] > 0).sum()
        print(f'  label {lid}: max-count pixel ({v},{u}), {vc} valid grasps')

        # Adjust camera based on object centroid
        obj_mask = seg == lid
        ys, xs = np.where(obj_mask)
        pts = xyz[ys, xs]
        valid_z = pts[:, 2] > 0
        centroid = pts[valid_z].mean(axis=0) if valid_z.sum() > 0 else xyz[v, u]

        # Camera offset: try from above-side
        offset = np.array([0.0, -0.18, 0.10])

        img, n_arrows = render_approach_arrows(
            xyz, rgb, seg, gt, px_count, lid,
            arrow_length=args.arrow_length,
            max_arrows=args.max_arrows,
            cam_offset=offset
        )
        out_path = os.path.join(args.out_dir, f'label{lid}_count_arrows.png')
        Image.fromarray(img).save(out_path)
        panels.append(img)
        panel_titles.append(f'label {lid}\n({v},{u}) — {n_arrows} valid approaches')

        # Gripper symmetry for the same pixel
        img_sym = render_gripper_symmetry(
            xyz, rgb, seg, gt, px_count, lid, cam_offset=offset
        )
        out_path_sym = os.path.join(args.out_dir, f'label{lid}_gripper_sym.png')
        Image.fromarray(img_sym).save(out_path_sym)

        print(f'    saved: {out_path}')
        print(f'    saved: {out_path_sym}')

    if len(panels) == 0:
        print('No panels generated.')
        return

    # Compose figure
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, panels, panel_titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    fig.suptitle(f'Scene {args.scene_id:04d} — Valid approach directions per pixel', fontsize=11)
    out_fig = os.path.join(args.out_dir, f'scene{args.scene_id:04d}_3d_overview.png')
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nComposite figure: {out_fig}')

    # Also: gripper symmetry panel
    sym_imgs = []
    for lid in label_ids:
        p = os.path.join(args.out_dir, f'label{lid}_gripper_sym.png')
        if os.path.exists(p):
            sym_imgs.append((np.array(Image.open(p)), f'label {lid} — g vs −g'))

    if sym_imgs:
        fig2, axes2 = plt.subplots(1, len(sym_imgs), figsize=(6 * len(sym_imgs), 5))
        if len(sym_imgs) == 1:
            axes2 = [axes2]
        for ax, (img, title) in zip(axes2, sym_imgs):
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        fig2.suptitle('Gripper symmetry: blue=g, red=−g (both valid)', fontsize=11)
        out_sym = os.path.join(args.out_dir, f'scene{args.scene_id:04d}_gripper_sym.png')
        fig2.savefig(out_sym, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'Gripper symmetry figure: {out_sym}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=int, default=37)
    parser.add_argument('--ann_id', type=int, default=0)
    parser.add_argument('--labels', type=int, nargs='+', default=[10, 52],
                        help='Object label IDs to visualize')
    parser.add_argument('--min_count', type=int, default=30,
                        help='Min valid grasps for pixel selection')
    parser.add_argument('--arrow_length', type=float, default=0.025)
    parser.add_argument('--max_arrows', type=int, default=40)
    parser.add_argument('--out_dir', type=str, default='/tmp/ambiguity_3d')
    args = parser.parse_args()
    main(args)
