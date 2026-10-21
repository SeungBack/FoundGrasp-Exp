"""xArm7 시각화용 기구학 — MuJoCo Menagerie MJCF(third_party/xarm7_mjcf)의 링크 체인.

외형은 xarm7_visual의 제조사 GLB(재질 포함)를 우선 사용한다.
시각화 전용이다 (제어·IK는 xArm SDK가 한다). 루트 바디(link_base)의 받침대 높이
(pos z 0.12)는 버리고 base 좌표계 원점에 둔다 — xArm SDK의 base 좌표계와 같다.
영점 자세에서 link7(플랜지) 위치 = (0.206, 0, 0.1205) m = xArm7 DH 값.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import THIRD_PARTY

MJCF_PATH = THIRD_PARTY / "xarm7_mjcf" / "xarm7_nohand.xml"
VISUAL_DIR = THIRD_PARTY / "xarm7_visual"


def _quat_wxyz_to_R(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _axis_angle_R(axis, angle):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


@dataclass
class Link:
    name: str
    T_parent: np.ndarray        # 부모 링크 → 이 링크 (관절 0일 때)
    axis: "np.ndarray | None"   # 관절 축 (링크 좌표계), 없으면 고정
    mesh: "Path | None"


class XArm7Model:
    def __init__(self, mjcf_path=MJCF_PATH):
        root = ET.parse(mjcf_path).getroot()
        mesh_dir = Path(mjcf_path).parent / root.find("compiler").get("meshdir", ".")
        self.links = []

        def walk(body, is_root):
            T = np.eye(4)
            if not is_root:  # 루트 받침대 오프셋은 버린다 (base 원점 = SDK base)
                T[:3, 3] = [float(v) for v in body.get("pos", "0 0 0").split()]
                T[:3, :3] = _quat_wxyz_to_R(
                    [float(v) for v in body.get("quat", "1 0 0 0").split()])
            joint = body.find("joint")
            axis = None
            if joint is not None:
                axis = np.array([float(v) for v in joint.get("axis", "0 0 1").split()])
            geom = body.find("geom")
            mesh = mesh_dir / f"{geom.get('mesh')}.stl" if geom is not None else None
            if Path(mjcf_path).resolve() == MJCF_PATH.resolve():
                visual = VISUAL_DIR / f"{body.get('name')}.glb"
                if visual.is_file():
                    mesh = visual
            self.links.append(Link(body.get("name"), T, axis, mesh))
            for child in body.findall("body"):
                walk(child, False)

        walk(root.find("worldbody").find("body"), True)
        self.n_joints = sum(link.axis is not None for link in self.links)

    def link_poses(self, q_rad):
        """관절각(rad, 7개) → [(링크 이름, base→링크 4x4)] (체인 순서)."""
        q = list(q_rad)
        T = np.eye(4)
        out, j = [], 0
        for link in self.links:  # 단일 체인 (분기 없음)
            T = T @ link.T_parent
            if link.axis is not None:
                R = np.eye(4)
                R[:3, :3] = _axis_angle_R(link.axis, q[j])
                T = T @ R
                j += 1
            out.append((link.name, T.copy()))
        return out

    def flange_pose(self, q_rad):
        return self.link_poses(q_rad)[-1][1]
