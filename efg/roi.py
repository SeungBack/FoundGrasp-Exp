"""스캔 영역(ROI) — 이미지 2D 사각형으로 그립 후보 필터·시각화.

FoundGrasp는 전체 프레임으로 추론하고, 후보 그립 중심을 K로 이미지에
투영해 ROI 내부만 남긴다 (입력 마스킹은 depth 경계 왜곡을 만들 수 있어
후처리 필터를 택함). 박스 테두리 등 영역 밖 그립이 제거된다.
"""
import numpy as np

from .paths import import_grasp_group


def project_px(points_cam, K):
    """카메라 좌표계 점들(N,3) → 픽셀 (N,2). z<=0은 NaN."""
    p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * p[:, 0] / z + K[0, 2]
        v = K[1, 1] * p[:, 1] / z + K[1, 2]
    uv = np.stack([u, v], axis=1)
    uv[z <= 1e-6] = np.nan
    return uv


def filter_roi(gg, K, roi_px):
    """GraspGroup에서 그립 중심 투영이 ROI 내부인 후보만 반환 (score 순 유지)."""
    GraspGroup = import_grasp_group()

    if len(gg) == 0:
        return gg
    x1, y1, x2, y2 = roi_px
    uv = project_px(gg.translations, K)
    mask = ((uv[:, 0] >= x1) & (uv[:, 0] <= x2)
            & (uv[:, 1] >= y1) & (uv[:, 1] <= y2))
    mask &= np.isfinite(uv).all(axis=1)
    sub = GraspGroup()
    sub.grasp_group_array = gg.grasp_group_array[mask].copy()
    return sub


def exclude_near(gg, T_base_cam, points_base, radius_m):
    """base 좌표계 지점들(직전 실패 그립) 반경 내 후보 제외."""
    GraspGroup = import_grasp_group()

    if len(gg) == 0 or not points_base:
        return gg
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    centers = gg.translations @ R.T + t  # (N,3) base
    keep = np.ones(len(gg), dtype=bool)
    for p in points_base:
        keep &= np.linalg.norm(centers - np.asarray(p), axis=1) > radius_m
    sub = GraspGroup()
    sub.grasp_group_array = gg.grasp_group_array[keep].copy()
    return sub


def draw_overlay(bgr, roi_px, status="", marks=None):
    """ROI 사각형 + 상태 텍스트 + 그립 마크 오버레이 (in-place, BGR).

    marks 항목 2형식 지원:
      (u, v, color_bgr, label)                       — 점 마크
      {"segments_px": [((u1,v1),(u2,v2)),...],
       "color": bgr, "label": str, "anchor": (u,v)}  — 그리퍼 와이어프레임
    한글은 OpenCV 기본 폰트가 지원하지 않으므로 status는 영문 사용.
    """
    import cv2

    if roi_px is not None:
        x1, y1, x2, y2 = [int(v) for v in roi_px]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(bgr, "SCAN ROI", (x1 + 4, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    for m in (marks or []):
        if isinstance(m, dict):
            color = m.get("color", (0, 0, 255))
            for a, b in m.get("segments_px", []):
                cv2.line(bgr, tuple(int(c) for c in a),
                         tuple(int(c) for c in b), color, 2, cv2.LINE_AA)
            if m.get("label") and m.get("anchor") is not None:
                u, v = (int(c) for c in m["anchor"])
                cv2.putText(bgr, m["label"], (u + 8, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            continue
        u, v, color, label = m
        u, v = int(u), int(v)
        cv2.circle(bgr, (u, v), 6, color, 2)
        if label:
            cv2.putText(bgr, label, (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if status:
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 36), (30, 30, 30), -1)
        cv2.putText(bgr, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    return bgr
