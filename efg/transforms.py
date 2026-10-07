"""좌표 변환: 카메라 좌표계 그립(gg) → base 좌표계 → 툴(TCP) 목표 자세.

- hand-eye 결과(T_flange_cam)는 configs/calib/hand_eye.yaml (calib.hand_eye) 사용
- graspnetAPI 그립 좌표계: x = 접근 방향, y = 핑거 개폐 방향
- 툴 좌표계(xArm TCP): +z = 플랜지→툴끝(접근 방향), 핑거 개폐는 툴 y축으로 가정
"""
import numpy as np
import yaml


def load_hand_eye(path):
    """hand_eye_latest.yaml → T_flange_cam (4x4)."""
    with open(path) as f:
        d = yaml.safe_load(f)
    return np.asarray(d["T_flange_cam"]["matrix"], dtype=np.float64)


def grasp_to_base(T_base_cam, g):
    """단일 grasp(graspnetAPI Grasp)를 base 좌표계로 변환.

    반환: (R_base_grasp 3x3, p_base 3,) — x=접근, y=개폐 방향 유지.
    """
    R = T_base_cam[:3, :3] @ g.rotation_matrix
    p = T_base_cam[:3, :3] @ g.translation + T_base_cam[:3, 3]
    return R, p


def tool_pose_from_grasp(R_base_grasp, p_base, closing_sign=1,
                         insertion_m=0.0):
    """그립 자세 → 툴(TCP) 목표 SE(3).

    툴 z축 = 접근 방향(grasp x), 툴 y축 = 개폐 방향(closing_sign * grasp y).
    평행 그리퍼는 y/-y가 등가이므로 closing_sign(±1)으로 두 해를 만든다.
    insertion_m: 그립 중심점에서 접근 방향으로 추가 삽입 깊이
    (graspnet per-grasp depth + 여유. TCP=툴끝 기준 접근 부족 보정).
    """
    z = R_base_grasp[:, 0]
    z = z / np.linalg.norm(z)
    y = closing_sign * R_base_grasp[:, 1]
    y = y - np.dot(y, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    T[:3, 3] = p_base + insertion_m * z
    return T


def offset_along_tool_z(T_tool, dist_m):
    """툴 z축(접근 방향) 기준 -dist만큼 물러난 자세 (pre-grasp 용)."""
    T = T_tool.copy()
    T[:3, 3] = T[:3, 3] - dist_m * T[:3, 2]
    return T


def lifted(T_tool, height_m):
    """base +z로 height만큼 올린 자세 (자세 유지, 위치만 상승)."""
    T = T_tool.copy()
    T[:3, 3] = T[:3, 3] + np.array([0.0, 0.0, height_m])
    return T


def approach_tilt_deg(R_base_grasp):
    """접근 벡터와 base -z(수직 아래) 사이 각도 (deg)."""
    a = R_base_grasp[:, 0]
    a = a / np.linalg.norm(a)
    cos = np.clip(np.dot(a, np.array([0.0, 0.0, -1.0])), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))
