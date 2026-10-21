"""빈 박스 높이맵 기반 하강 제한 — 박스 바닥·벽과 그리퍼 충돌 방지.

scripts/scan_box_floor.py 가 빈 박스를 스캔해 base 좌표 2.5D 높이맵(셀별
최고 표면 z)을 저장하면, 실행 시 선정 그립의 TCP 끝 + 양 핑거 끝이
`표면 z + clearance` 아래로 내려가지 않도록 삽입량을 줄인다.
줄여도 부족하면(삽입 하한 미달) 후보를 탈락시켜 재선정을 유도한다.
"""
import copy

import numpy as np


class FloorMap:
    """base 좌표 xy 격자 → 표면 최고 z (m). 격자 밖은 최근접 셀 값 사용."""

    def __init__(self, z, x0, y0, cell):
        self.z = np.asarray(z, dtype=np.float64)
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)

    @classmethod
    def load(cls, path):
        try:
            d = np.load(path, allow_pickle=True)
        except FileNotFoundError:
            raise SystemExit(
                f"높이맵 없음: {path} — scripts/scan_box_floor.py 로 빈 박스를 "
                "스캔하거나 floor_limit.enabled: false 로 끄세요")
        return cls(d["z"], d["x0"], d["y0"], d["cell"])

    def surface_z(self, xy):
        """(N,2) base xy → (N,) 표면 z. 격자 밖은 가장자리 셀로 클램프."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        i = np.clip(np.round((xy[:, 1] - self.y0) / self.cell).astype(int),
                    0, self.z.shape[0] - 1)
        j = np.clip(np.round((xy[:, 0] - self.x0) / self.cell).astype(int),
                    0, self.z.shape[1] - 1)
        return self.z[i, j]


def clamp_to_floor(sel, g, fmap, cfg):
    """선정 그립(sel)의 삽입량을 높이맵에 맞춰 축소.

    검사점: TCP 끝 + 양 핑거 끝(그립 폭/2 + 5mm 여유, 개폐 방향).
    반환: (조정된 SelectedGrasp 또는 None, 줄인 거리 m).
    None = 삽입 하한(min_insertion_m)까지 줄여도 여유 부족 — 후보 탈락 권고.
    """
    fl = cfg["floor_limit"]
    s_cfg = cfg["selection"]
    clearance = float(fl.get("clearance_m", 0.008))
    min_ins = float(fl.get("min_insertion_m", 0.010))
    insertion = float(s_cfg.get("approach_extra_m", 0.0))
    if s_cfg.get("use_grasp_depth", True):
        insertion += float(g.depth)
    budget = insertion - min_ins  # 뒤로 물릴 수 있는 최대 거리

    sel = copy.deepcopy(sel)
    half = float(getattr(sel, "open_width_m", g.width)) / 2.0 + 0.005  # 실행 개방 폭 기준 핑거 위치
    total_back = 0.0
    for _ in range(5):  # 후퇴 시 xy도 변해 표면 z가 달라질 수 있어 반복 수렴
        T = sel.T_tcp_grasp
        a, y, tip = T[:3, 2], T[:3, 1], T[:3, 3]
        pts = np.stack([tip, tip + half * y, tip - half * y])
        v = float(np.max(fmap.surface_z(pts[:, :2]) + clearance - pts[:, 2]))
        if v <= 1e-4:
            return sel, total_back
        d_back = v / max(-a[2], 0.2)  # 접근 하향 성분으로 환산 (tilt 제한 내 ≥0.9)
        if total_back + d_back > budget:
            return None, total_back + d_back
        total_back += d_back
        shift = -d_back * a
        for name in ("T_tcp_grasp", "T_tcp_pre", "T_tcp_lift"):
            M = getattr(sel, name).copy()
            M[:3, 3] = M[:3, 3] + shift
            setattr(sel, name, M)
    return None, total_back  # 수렴 실패 — 보수적으로 탈락
