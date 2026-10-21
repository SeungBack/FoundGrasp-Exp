"""floor_limit — 높이맵 하강 제한 로직 (하드웨어 불필요)."""
import numpy as np

from efg.floor import FloorMap, clamp_to_floor
from efg.selection import SelectedGrasp
from efg.transforms import lifted, offset_along_tool_z

CFG = {"floor_limit": {"clearance_m": 0.008, "min_insertion_m": 0.010},
       "selection": {"approach_extra_m": 0.04, "use_grasp_depth": True}}


class G:  # graspnetAPI Grasp 대역
    width, depth = 0.04, 0.02


def flat_map(z=0.0, size=0.4, cell=0.005):
    n = int(size / cell)
    return FloorMap(np.full((n, n), z), x0=0.0, y0=-size / 2, cell=cell)


def vertical_sel(tip_z):
    """접근 = base -z (툴 z축 아래), 개폐 = base y."""
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = [-1, 0, 0], [0, 1, 0], [0, 0, -1]
    T[:3, 3] = [0.2, 0.0, tip_z]
    return SelectedGrasp(0, 0.9, 0.04, 0.048, 1, T, offset_along_tool_z(T, 0.1),
                         lifted(T, 0.15), T[:3, 3].copy())


def test_above_surface_untouched():
    sel, back = clamp_to_floor(vertical_sel(0.05), G(), flat_map(0.0), CFG)
    assert sel is not None and back == 0.0
    assert np.isclose(sel.T_tcp_grasp[2, 3], 0.05)


def test_below_surface_backs_off_to_clearance():
    sel, back = clamp_to_floor(vertical_sel(0.0), G(), flat_map(0.0), CFG)
    assert sel is not None and np.isclose(back, 0.008, atol=1e-4)
    assert np.isclose(sel.T_tcp_grasp[2, 3], 0.008, atol=1e-4)
    # pre/lift 자세도 같은 양만큼 함께 이동
    assert np.isclose(sel.T_tcp_pre[2, 3], 0.1 + 0.008, atol=1e-4)
    assert np.isclose(sel.T_tcp_lift[2, 3], 0.15 + 0.008, atol=1e-4)


def test_exceeding_budget_rejects():
    # 삽입 0.06, 하한 0.01 → 예산 0.05. 표면이 0.06 위면 후퇴 0.068 필요 → 탈락
    sel, back = clamp_to_floor(vertical_sel(0.0), G(), flat_map(0.06), CFG)
    assert sel is None and back > 0.05


def test_finger_tip_checked_not_only_tcp():
    """개폐 방향(base y) 한쪽 핑거 아래만 높은 벽 → 핑거 끝 기준으로 후퇴."""
    fm = flat_map(0.0)
    half = 0.048 / 2 + 0.005  # 실행 개방 폭(open_width_m) 기준 핑거 끝
    i = np.round((0.0 + half - fm.y0) / fm.cell).astype(int)  # +y 핑거 끝 셀
    fm.z[i, :] = 0.03
    sel, back = clamp_to_floor(vertical_sel(0.0), G(), fm, CFG)
    assert sel is not None and np.isclose(back, 0.038, atol=1e-3)
