"""시각화 전용 그리퍼 근사 메쉬. TCP z=접근, y=개폐; 단위 m.

제조사 CAD가 아닌 치수 기반 근사다. 핑거 접촉면은 y=±width/2, 끝은 TCP z=0.
"""
import itertools

import numpy as np
import trimesh


def bevel_box(size, center, radius=0.003):
    half = np.asarray(size) / 2
    r = min(radius, float(half.min()) * .8)
    points = []
    # 모서리와 꼭짓점을 깎은 26면 하우징.
    for signs in itertools.product((-1, 1), repeat=3):
        for axis in range(3):
            v = half - r
            v[axis] = half[axis]
            points.append(v * signs)
    mesh = trimesh.convex.convex_hull(points)
    mesh.apply_translation(center)
    return mesh


def gripper_parts(tcp_z):
    """(경로, 메쉬, RGB). left/right는 개폐 시 이동할 자식 그룹."""
    dark, metal, rubber = (45, 52, 62), (178, 188, 198), (38, 42, 48)
    mount = trimesh.creation.cylinder(radius=.033, height=.012, sections=48)
    mount.apply_translation([0, 0, -tcp_z + .006])
    parts = [('mount', mount, metal)]
    length = max(.04, tcp_z - .095 - .012)
    parts += [('housing', bevel_box([.068, .106, length], [0, 0, -tcp_z + .012 + length/2], .007), dark),
              ('face', bevel_box([.058, .094, .009], [0, 0, -.0995], .003), metal),
              ('inset', bevel_box([.002, .063, length*.52], [.034, 0, -tcp_z + .012 + length*.52], .0005), (72, 82, 94))]
    for side, sign in [('left', 1), ('right', -1)]:
        parts += [(f'{side}/carriage', bevel_box([.040, .024, .018], [0, sign*.012, -.087], .003), metal),
                  (f'{side}/finger', bevel_box([.024, .014, .058], [0, sign*.011, -.050], .0025), metal),
                  (f'{side}/tip', bevel_box([.027, .010, .028], [0, sign*.005, -.014], .003), rubber)]
        for i in range(4):
            parts.append((f'{side}/tip_rib{i}', bevel_box([.023, .0015, .0018],
                          [0, sign*.00075, -.006-i*.005], .0005), (65, 70, 78)))
    return parts
