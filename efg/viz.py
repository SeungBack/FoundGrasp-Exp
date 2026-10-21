"""그립 자세 시각화 — 2지 그리퍼 와이어프레임 (3D 선분, 이미지 투영).

grasp 프레임(graspnetAPI): x=접근 방향, y=핑거 개폐 방향.
손가락 2개(접근 방향, 폭=width) + 베이스 바 + 접근 꼬리를 선분으로 그린다.
"""
import numpy as np

from .roi import project_px

BASE_BACK_M = 0.02   # 그립 중심에서 베이스 바까지 (접근 반대 방향)
TAIL_M = 0.03        # 베이스 바 뒤 꼬리 길이
MIN_FINGER_M = 0.02  # depth가 작아도 손가락이 보이게 하는 최소 길이


def score_colors(scores):
    """표시 점수 범위를 Matplotlib RdYlGn으로 매핑 (성공 확률 아님)."""
    from matplotlib import colormaps
    s = np.asarray(scores, dtype=float)
    if not len(s):
        return np.empty((0, 3), dtype=np.uint8)
    lo, hi = float(np.min(s)), float(np.max(s))
    t = (s-lo)/(hi-lo) if hi > lo else np.full_like(s, .5)
    return np.rint(colormaps["RdYlGn"](t)[:, :3] * 255).astype(np.uint8)


def display_candidate_indices(gg, max_n=25):
    """점수가 높은 순서로 12mm 이내 중복 지점을 덜어낸 표시용 인덱스."""
    selected = []
    positions = np.asarray(gg.translations)
    for i in np.argsort(-np.asarray(gg.scores), kind="stable"):
        if selected and np.any(np.linalg.norm(positions[selected]-positions[i], axis=1) < .012):
            continue
        selected.append(int(i))
        if len(selected) >= max(1, max_n):
            break
    return np.asarray(selected, dtype=int)


def draw_score_grippers(bgr, gg, K, max_n=25):
    import cv2
    order = display_candidate_indices(gg, max_n)
    colors = score_colors(np.asarray(gg.scores)[order])
    for rank in range(len(order)-1, -1, -1):
        i = int(order[rank])
        thickness = 7 if rank == 0 else 2
        color = tuple(int(c) for c in colors[rank][::-1])
        for a, b in gripper_segments_px(gg[i], K):
            if rank == 0:
                cv2.line(bgr, a, b, (250, 250, 250), thickness+6, cv2.LINE_AA)
            cv2.line(bgr, a, b, (20, 24, 28), thickness+2, cv2.LINE_AA)
            cv2.line(bgr, a, b, color, thickness, cv2.LINE_AA)
        if rank == 0:
            segs = gripper_segments_px(gg[i], K)
            if segs:
                x, y = segs[-1][-1]
                anchor = (max(5, min(bgr.shape[1]-190, x+10)), max(25, min(bgr.shape[0]-5, y-12)))
                text = f"TOP 1  {float(gg.scores[i]):.3f}"
                cv2.putText(bgr, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, .65, (15,15,15), 5, cv2.LINE_AA)
                cv2.putText(bgr, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2, cv2.LINE_AA)


def gripper_segments_3d(g, width=None):
    """단일 grasp → 카메라 좌표계 선분 끝점 (8,3): [a0,b0,a1,b1,...] 4개 선분.

    width: 그릴 폭 (기본 g.width — 실행 개방 폭을 보이려면 지정)."""
    R, t = g.rotation_matrix, g.translation
    x, y = R[:, 0], R[:, 1]
    hw = float(g.width if width is None else width) / 2.0
    d = max(float(g.depth), MIN_FINGER_M)
    b = t - BASE_BACK_M * x  # 베이스 바 중심
    return np.array([
        b + hw * y, t + hw * y + d * x,   # 손가락 1
        b - hw * y, t - hw * y + d * x,   # 손가락 2
        b + hw * y, b - hw * y,           # 베이스 바
        b, b - TAIL_M * x,                # 꼬리
    ])


def gripper_segments_px(g, K):
    """단일 grasp → 픽셀 선분 [((u1,v1),(u2,v2)), ...]. 투영 불가 시 []."""
    uv = project_px(gripper_segments_3d(g), K)
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
