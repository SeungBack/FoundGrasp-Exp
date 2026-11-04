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


def gripper_parts(tcp_z, tip_length_m=.080):
    """(경로, 메쉬, RGB). left/right는 개폐 시 이동할 자식 그룹.

    tip_length_m is the whole replaceable finger, from its root to TCP z=0,
    including the distal contact pad. It is not added to the metal finger.
    The remaining flange-to-root length belongs to the housing/carriage model.
    """
    tcp_z, tip_length_m = float(tcp_z), float(tip_length_m)
    if (not np.isfinite([tcp_z, tip_length_m]).all() or tip_length_m <= 0
            or tcp_z <= tip_length_m + .027):
        raise ValueError("그리퍼 그림 치수는 양수이고 TCP 길이에 팁·장착부가 들어가야 합니다")
    dark, metal, rubber = (45, 52, 62), (178, 188, 198), (38, 42, 48)
    mount = trimesh.creation.cylinder(radius=.033, height=.012, sections=48)
    mount.apply_translation([0, 0, -tcp_z + .006])
    parts = [('mount', mount, metal)]
    length = tcp_z - tip_length_m - .015 - .012
    parts += [('housing', bevel_box([.068, .106, length], [0, 0, -tcp_z + .012 + length/2], .007), dark),
              ('face', bevel_box([.058, .094, .009], [0, 0, -tip_length_m - .0195], .003), metal),
              ('inset', bevel_box([.002, .063, length*.52], [.034, 0, -tcp_z + .012 + length*.52], .0005), (72, 82, 94))]
    pad = min(.028, tip_length_m * .35)
    overlap = min(.007, pad * .25)
    finger_end = -(pad - overlap)
    finger_length = tip_length_m + finger_end
    for side, sign in [('left', 1), ('right', -1)]:
        parts += [(f'{side}/carriage', bevel_box([.040, .024, .018], [0, sign*.012, -tip_length_m - .007], .003), metal),
                  (f'{side}/finger', bevel_box([.024, .014, finger_length],
                                             [0, sign*.011, (-tip_length_m + finger_end)/2], .0025), metal),
                  (f'{side}/tip', bevel_box([.027, .010, pad], [0, sign*.005, -pad/2], .003), rubber)]
        for i in range(4):
            parts.append((f'{side}/tip_rib{i}', bevel_box([.023, .0015, min(.0018, pad*.07)],
                          [0, sign*.00075, -pad*(.2+i*.18)], .0005), (65, 70, 78)))
    return parts
