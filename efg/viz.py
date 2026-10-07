"""그립 자세 2D 시각화 — 2지 그리퍼 와이어프레임을 이미지에 투영.

grasp 프레임(graspnetAPI): x=접근 방향, y=핑거 개폐 방향.
손가락 2개(접근 방향, 폭=width) + 베이스 바 + 접근 꼬리를 선분으로 그린다.
"""
import numpy as np

from .roi import project_px

BASE_BACK_M = 0.02   # 그립 중심에서 베이스 바까지 (접근 반대 방향)
TAIL_M = 0.03        # 베이스 바 뒤 꼬리 길이
MIN_FINGER_M = 0.02  # depth가 작아도 손가락이 보이게 하는 최소 길이


def gripper_segments_px(g, K):
    """단일 grasp → 픽셀 선분 [((u1,v1),(u2,v2)), ...]. 투영 불가 시 []."""
    R, t = g.rotation_matrix, g.translation
    x, y = R[:, 0], R[:, 1]
    hw = float(g.width) / 2.0
    d = max(float(g.depth), MIN_FINGER_M)
    b = t - BASE_BACK_M * x  # 베이스 바 중심
    pts = np.array([
        b + hw * y, t + hw * y + d * x,   # 손가락 1
        b - hw * y, t - hw * y + d * x,   # 손가락 2
        b + hw * y, b - hw * y,           # 베이스 바
        b, b - TAIL_M * x,                # 꼬리
    ])
    uv = project_px(pts, K)
    if not np.isfinite(uv).all():
        return []
    uv = uv.astype(int)
    return [(tuple(uv[i]), tuple(uv[i + 1])) for i in range(0, len(uv), 2)]


def draw_grippers(bgr, gg, K, color=(0, 165, 255), thickness=1, max_n=None):
    """GraspGroup 전체를 그리퍼 와이어프레임으로 그린다 (in-place, BGR)."""
    import cv2

    n = len(gg) if max_n is None else min(max_n, len(gg))
    for i in range(n - 1, -1, -1):  # 상위 score가 위에 그려지게 역순
        for a, b in gripper_segments_px(gg[i], K):
            cv2.line(bgr, a, b, color, thickness, cv2.LINE_AA)
    return bgr
