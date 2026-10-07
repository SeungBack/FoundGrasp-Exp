#!/usr/bin/env python3
"""Render a draft figure for ambiguity-aware supervision using only Pillow."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


Vec3 = tuple[float, float, float]

WHITE = (255, 255, 255, 255)
TEXT = (34, 34, 34, 255)
SURFACE_LINE = (228, 228, 228, 255)
SURFACE_POINT = (217, 217, 217, 255)
SEED_GREEN = (27, 158, 119, 255)
SUPPORT_GREEN = (153, 216, 201, 175)
APPROACH_RED = (194, 59, 34, 255)
GRIPPER_ORANGE = (240, 140, 0, 220)
GRIPPER_ORANGE_FAINT = (240, 140, 0, 135)
BLACK = (0, 0, 0, 255)


def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def vec_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_norm(v: Vec3) -> float:
    return math.sqrt(vec_dot(v, v))


def normalize(v: Vec3) -> Vec3:
    n = vec_norm(v)
    if n < 1e-8:
        return v
    return vec_scale(v, 1.0 / n)


def rotate_z(v: Vec3, angle_deg: float) -> Vec3:
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


def surface_height(x: float, y: float) -> float:
    bump = 0.16 * math.exp(-2.8 * ((x + 0.25) ** 2 + 0.7 * (y - 0.1) ** 2))
    tilt = -0.06 * y + 0.03 * x
    step = 0.10 if x > 0.22 else 0.0
    ripple = 0.015 * math.sin(3.5 * y) * math.cos(2.2 * x)
    return bump + tilt + step + ripple


def make_surface(num: int = 19) -> list[list[Vec3]]:
    xs = [(-1.0 + 2.0 * i / (num - 1)) for i in range(num)]
    ys = [(-0.9 + 1.8 * j / (num - 1)) for j in range(num)]
    rows: list[list[Vec3]] = []
    for y in ys:
        row = []
        for x in xs:
            row.append((x, y, surface_height(x, y)))
        rows.append(row)
    return rows


def tangent_frame() -> tuple[Vec3, Vec3, Vec3]:
    approach = normalize((0.20, -0.28, 0.94))
    tangent_a = normalize(vec_cross(approach, (0.0, 0.0, 1.0)))
    tangent_b = normalize(vec_cross(approach, tangent_a))
    return approach, tangent_a, tangent_b


def rotation_matrix(elev_deg: float = 24.0, azim_deg: float = -58.0) -> tuple[tuple[float, float, float], ...]:
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    rot_z = (
        (math.cos(azim), -math.sin(azim), 0.0),
        (math.sin(azim), math.cos(azim), 0.0),
        (0.0, 0.0, 1.0),
    )
    rot_x = (
        (1.0, 0.0, 0.0),
        (0.0, math.cos(elev), -math.sin(elev)),
        (0.0, math.sin(elev), math.cos(elev)),
    )
    return mat_mul(rot_x, rot_z)


def mat_mul(a, b):
    out = []
    for r in range(3):
        row = []
        for c in range(3):
            row.append(
                a[r][0] * b[0][c] +
                a[r][1] * b[1][c] +
                a[r][2] * b[2][c]
            )
        out.append(tuple(row))
    return tuple(out)


ROT = rotation_matrix()


def apply_rot(v: Vec3) -> Vec3:
    return (
        ROT[0][0] * v[0] + ROT[0][1] * v[1] + ROT[0][2] * v[2],
        ROT[1][0] * v[0] + ROT[1][1] * v[1] + ROT[1][2] * v[2],
        ROT[2][0] * v[0] + ROT[2][1] * v[1] + ROT[2][2] * v[2],
    )


class PanelProjector:
    def __init__(self, box: tuple[int, int, int, int], scale: float = 176.0):
        self.x0, self.y0, self.x1, self.y1 = box
        self.cx = 0.5 * (self.x0 + self.x1)
        self.cy = 0.58 * self.y0 + 0.42 * self.y1
        self.scale = scale

    def project(self, point: Vec3) -> tuple[float, float, float]:
        rot = apply_rot(point)
        px = self.cx + self.scale * rot[0]
        py = self.cy - self.scale * (rot[1] + 0.2 * rot[2])
        return px, py, rot[2]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(26, bold=True)
FONT_LABEL = load_font(21, bold=True)
FONT_TEXT = load_font(18)
FONT_SMALL = load_font(16)


def draw_panel_header(draw: ImageDraw.ImageDraw, box, title: str, label: str) -> None:
    x0, y0, _, _ = box
    draw.text((x0 + 6, y0 + 6), label, fill=TEXT, font=FONT_LABEL)
    draw.text((x0 + 34, y0 + 6), title, fill=TEXT, font=FONT_TITLE)


def line(draw: ImageDraw.ImageDraw, p0, p1, fill, width: int = 2) -> None:
    draw.line((p0[0], p0[1], p1[0], p1[1]), fill=fill, width=width)


def draw_arrow(draw: ImageDraw.ImageDraw, proj: PanelProjector, origin: Vec3,
               vec: Vec3, color, length: float, width: int = 4) -> None:
    start = proj.project(origin)
    end = proj.project(vec_add(origin, vec_scale(normalize(vec), length)))
    line(draw, start, end, color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ang = math.atan2(dy, dx)
    head_len = 13.0
    spread = 0.56
    left = (end[0] - head_len * math.cos(ang - spread), end[1] - head_len * math.sin(ang - spread))
    right = (end[0] - head_len * math.cos(ang + spread), end[1] - head_len * math.sin(ang + spread))
    draw.polygon([end[:2], left, right], fill=color)


def draw_disc(draw: ImageDraw.ImageDraw, proj: PanelProjector, point: Vec3,
              radius: int, fill, outline=None) -> None:
    x, y, _ = proj.project(point)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline)


def draw_bar(draw: ImageDraw.ImageDraw, proj: PanelProjector, center: Vec3,
             direction: Vec3, length: float, color, width: int = 4,
             lift: float = 0.012) -> None:
    lifted = vec_add(center, (0.0, 0.0, lift))
    offset = vec_scale(normalize(direction), length)
    p0 = proj.project(vec_sub(lifted, offset))
    p1 = proj.project(vec_add(lifted, offset))
    line(draw, p0, p1, color, width=width)


def draw_surface(draw: ImageDraw.ImageDraw, proj: PanelProjector, surface: list[list[Vec3]]) -> None:
    rows = len(surface)
    cols = len(surface[0])
    projected = [[proj.project(surface[r][c]) for c in range(cols)] for r in range(rows)]
    for r in range(rows):
        for c in range(cols - 1):
            line(draw, projected[r][c], projected[r][c + 1], SURFACE_LINE, width=1)
    for c in range(cols):
        for r in range(rows - 1):
            line(draw, projected[r][c], projected[r + 1][c], SURFACE_LINE, width=1)
    for r in range(rows):
        for c in range(cols):
            px, py, _ = projected[r][c]
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=SURFACE_POINT)


def draw_support_points(draw: ImageDraw.ImageDraw, proj: PanelProjector, points: Iterable[Vec3]) -> None:
    projected = [proj.project(point) for point in points]
    projected.sort(key=lambda item: item[2])
    for x, y, _ in projected:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=SUPPORT_GREEN)


def panel_sparse_propagation(draw, box, surface) -> None:
    draw_panel_header(draw, box, "Sparse labels and depth-aware propagation", "a")
    proj = PanelProjector(box)
    draw_surface(draw, proj, surface)

    seed_xy = [(-0.55, 0.30), (-0.10, -0.05), (0.05, 0.22)]
    seeds = [(x, y, surface_height(x, y)) for x, y in seed_xy]
    center = seeds[1]

    support_points = []
    for row in surface:
        for point in row:
            dist_xy = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
            depth_delta = abs(point[2] - center[2])
            if dist_xy < 0.33 and depth_delta < 0.045:
                support_points.append((point[0], point[1], point[2] + 0.003))
    draw_support_points(draw, proj, support_points)

    for seed in seeds:
        draw_disc(draw, proj, vec_add(seed, (0.0, 0.0, 0.01)), 7, SEED_GREEN, outline=WHITE)

    anchor = proj.project(vec_add(seeds[0], (0.0, 0.0, 0.02)))
    draw.text((anchor[0] - 88, anchor[1] - 70), "sparse labels", fill=SEED_GREEN, font=FONT_TEXT)
    anchor = proj.project(vec_add(center, (0.0, 0.0, 0.02)))
    draw.text(
        (anchor[0] - 156, anchor[1] + 80),
        "propagated support\n(stays on the same local surface)",
        fill=TEXT,
        font=FONT_SMALL,
    )


def panel_multi_orientation(draw, box, surface) -> None:
    draw_panel_header(draw, box, "Multiple valid orientations at one grasp point", "b")
    proj = PanelProjector(box)
    draw_surface(draw, proj, surface)
    approach, tangent_a, tangent_b = tangent_frame()
    center = (-0.05, 0.00, surface_height(-0.05, 0.00))

    draw_disc(draw, proj, vec_add(center, (0.0, 0.0, 0.012)), 7, BLACK, outline=WHITE)
    draw_arrow(draw, proj, center, approach, APPROACH_RED, length=0.26)

    for idx, angle_deg in enumerate([-58, -28, 0, 28, 58]):
        angle = math.radians(angle_deg)
        direction = vec_add(vec_scale(tangent_a, math.cos(angle)), vec_scale(tangent_b, math.sin(angle)))
        color = GRIPPER_ORANGE if idx == 2 else GRIPPER_ORANGE_FAINT
        width = 5 if idx == 2 else 4
        draw_bar(draw, proj, center, direction, length=0.16, color=color, width=width)

    tip = proj.project(vec_add(center, vec_scale(normalize(approach), 0.28)))
    draw.text((tip[0] - 34, tip[1] - 42), "fixed approach", fill=APPROACH_RED, font=FONT_TEXT)
    anchor = proj.project(vec_add(center, (0.0, 0.0, 0.04)))
    draw.text(
        (anchor[0] - 160, anchor[1] + 92),
        "valid gripper orientations\nat the same grasp point",
        fill=TEXT,
        font=FONT_SMALL,
    )


def panel_kappa(draw, box, surface) -> None:
    draw_panel_header(draw, box, "Broader vs. sharper orientation targets", "c")
    proj = PanelProjector(box)
    draw_surface(draw, proj, surface)
    approach, tangent_a, tangent_b = tangent_frame()
    low_center = (-0.38, 0.03, surface_height(-0.38, 0.03))
    high_center = (0.38, -0.02, surface_height(0.38, -0.02))

    panels = [
        (low_center, [-68, -44, -22, 0, 22, 44, 68], "low kappa", -115),
        (high_center, [-18, -9, 0, 9, 18], "high kappa", -70),
    ]
    for center, angles, label, dx in panels:
        draw_disc(draw, proj, vec_add(center, (0.0, 0.0, 0.012)), 7, BLACK, outline=WHITE)
        draw_arrow(draw, proj, center, approach, APPROACH_RED, length=0.22)
        mid = len(angles) // 2
        for idx, angle_deg in enumerate(angles):
            angle = math.radians(angle_deg)
            direction = vec_add(vec_scale(tangent_a, math.cos(angle)), vec_scale(tangent_b, math.sin(angle)))
            color = GRIPPER_ORANGE if idx == mid else GRIPPER_ORANGE_FAINT
            draw_bar(draw, proj, center, direction, length=0.14, color=color, width=4)
        anchor = proj.project(vec_add(center, (0.0, 0.0, 0.06)))
        draw.text((anchor[0] + dx, anchor[1] - 82), label, fill=TEXT, font=FONT_TEXT)

    low_anchor = proj.project(vec_add(low_center, (0.0, 0.0, 0.02)))
    high_anchor = proj.project(vec_add(high_center, (0.0, 0.0, 0.02)))
    draw.text((low_anchor[0] - 112, low_anchor[1] + 98), "broader angular target", fill=TEXT, font=FONT_SMALL)
    draw.text((high_anchor[0] - 98, high_anchor[1] + 98), "sharper angular target", fill=TEXT, font=FONT_SMALL)


def draw_legend(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    y = height - 42
    x = 126
    gap = 328

    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=SEED_GREEN, outline=WHITE)
    draw.text((x + 18, y - 12), "sparse supervision seed", fill=TEXT, font=FONT_SMALL)

    x += gap
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=SUPPORT_GREEN)
    draw.text((x + 18, y - 12), "depth-aware propagated support", fill=TEXT, font=FONT_SMALL)

    x += gap
    draw.line((x - 18, y, x + 18, y), fill=APPROACH_RED, width=4)
    draw.polygon([(x + 18, y), (x + 6, y - 6), (x + 6, y + 6)], fill=APPROACH_RED)
    draw.text((x + 28, y - 12), "approach direction", fill=TEXT, font=FONT_SMALL)

    x += 260
    draw.line((x - 18, y, x + 18, y), fill=GRIPPER_ORANGE, width=5)
    draw.text((x + 28, y - 12), "valid gripper orientation set", fill=TEXT, font=FONT_SMALL)


def build_image() -> Image.Image:
    width, height = 1740, 640
    image = Image.new("RGBA", (width, height), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    surface = make_surface()
    boxes = [
        (38, 26, 560, 560),
        (608, 26, 1130, 560),
        (1178, 26, 1700, 560),
    ]

    panel_sparse_propagation(draw, boxes[0], surface)
    panel_multi_orientation(draw, boxes[1], surface)
    panel_kappa(draw, boxes[2], surface)
    draw_legend(draw, width, height)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/ambiguity_aware_supervision_draft.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = build_image()
    image.save(args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
