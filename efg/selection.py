"""그립 후보 선정 — score 상위 순으로 필터 + 로봇 IK 적합성 체크.

pre-grasp / grasp / lift 세 자세가 모두 IK 가능해야 채택한다.
평행 그리퍼의 y/-y 등가성 때문에 자세당 두 방향(closing_sign ±1)을 시도한다.
"""
from dataclasses import dataclass, field

import numpy as np

from .transforms import (approach_tilt_deg, grasp_to_base, lifted,
                         offset_along_tool_z, tool_pose_from_grasp)


@dataclass
class SelectedGrasp:
    index: int
    score: float
    width_m: float
    closing_sign: int
    T_tcp_grasp: np.ndarray
    T_tcp_pre: np.ndarray
    T_tcp_lift: np.ndarray
    p_base: np.ndarray
    tilt_deg: float = 0.0  # 접근 방향의 수직(base -z) 대비 기울임


@dataclass
class SelectionReport:
    selected: "SelectedGrasp | None" = None
    rejected: list = field(default_factory=list)  # (index, 사유)

    def summary(self, max_lines=15):
        lines = [f"  #{i:02d} {reason}" for i, reason in
                 self.rejected[:max_lines]]
        if len(self.rejected) > max_lines:
            lines.append(f"  ... 외 {len(self.rejected) - max_lines}건")
        return "\n".join(lines)


def _tcp_to_flange(robot, T_base_tcp):
    from .se3 import inv_se3
    return T_base_tcp @ inv_se3(robot.T_flange_tcp)


def _ik_ok(robot, T_base_tcp):
    return robot.check_flange_ik(_tcp_to_flange(robot, T_base_tcp))


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _j7_deg(robot, T_base_tcp):
    """목표 TCP 자세 IK 해의 마지막 축(J7) 각도 (deg). 미지원/실패 시 None."""
    fn = getattr(robot, "flange_ik_joints_deg", None)
    if fn is None:
        return None
    j = fn(_tcp_to_flange(robot, T_base_tcp))
    if j is None or len(j) < 7:
        return None
    return float(j[6])


def _camera_lateral_dir_tcp(cfg, robot):
    """손목 카메라 돌출 방향의 툴 z 수직 성분 (TCP 프레임 단위 벡터).

    hand-eye 병진(플랜지→카메라)을 TCP 프레임으로 돌린 뒤 툴 축(z) 성분을
    제거 — 그립 자세 회전을 곱하면 base에서 카메라가 향하는 쪽이 된다.
    robot이 None이면 플랜지≈TCP 회전으로 근사한다.
    """
    from .transforms import load_hand_eye
    u = load_hand_eye(cfg["calib"]["hand_eye"])[:3, 3]
    if robot is not None:
        u = robot.T_flange_tcp[:3, :3].T @ u
    u[2] = 0.0
    n = np.linalg.norm(u)
    return u / n if n > 1e-6 else None


def select_grasp(gg, T_base_cam, cfg, robot=None):
    """gg(카메라 좌표계, score 내림차순)에서 실행 가능 후보 선정.

    기본은 score 상위 순이지만, 최고 실행가능 score와의 차이가
    score_tie_tolerance 이내인 후보들 중에서는 접근이 가장 수직에 가까운
    것을 선택한다 (수직 파지의 성공률이 높음 — 2026-07-21 현장 관찰).
    robot이 None이면 IK 체크를 건너뛰고 기하 필터만 적용한다.
    반환: SelectionReport (selected가 None이면 실패).
    """
    sel = cfg["selection"]
    xs, ys, zs = (sel["workspace_x_m"], sel["workspace_y_m"],
                  sel["workspace_z_m"])
    tol = float(sel.get("score_tie_tolerance", 0.0))
    # camera_up (옵션, 기본 off): 손목 카메라가 아래(바닥·박스)를 향하는 자세를
    # 배제 — sign 두 해 중 카메라 상향 쪽 우선, 하향 한도 초과 해는 탈락
    cam_cfg = sel.get("camera_up") or {}
    cam_u = (_camera_lateral_dir_tcp(cfg, robot)
             if cam_cfg.get("prefer", False) else None)
    cam_min_updot = -np.sin(np.radians(float(cam_cfg.get("max_down_deg",
                                                         20.0))))
    report = SelectionReport()
    pool = []  # score band 안의 실행 가능 후보들

    for i in range(len(gg)):
        g = gg[i]
        # 이미 실행 가능 후보가 있고 score band를 벗어나면 탐색 종료
        if pool and g.score < pool[0].score - tol:
            break
        if g.score < sel["min_score"]:
            report.rejected.append((i, f"score {g.score:.3f} < min"))
            continue
        if g.width > sel["max_grasp_width_m"]:
            report.rejected.append((i, f"width {g.width * 1000:.0f}mm > max"))
            continue

        R_b, p_b = grasp_to_base(T_base_cam, g)
        if not (xs[0] <= p_b[0] <= xs[1] and ys[0] <= p_b[1] <= ys[1]
                and zs[0] <= p_b[2] <= zs[1]):
            report.rejected.append(
                (i, f"작업공간 밖 {np.round(p_b, 3).tolist()}"))
            continue
        tilt = approach_tilt_deg(R_b)
        if tilt > sel["approach_max_tilt_deg"]:
            report.rejected.append((i, f"접근 기울임 {tilt:.0f}° > max"))
            continue

        insertion = float(sel.get("approach_extra_m", 0.0))
        if sel.get("use_grasp_depth", True):
            insertion += float(g.depth)
        # 두 closing_sign 모두 평가: J7 허용 범위를 벗어나는 방향은 탈락시키고,
        # 둘 다 가능하면 손목 회전(J7)이 작은 쪽을 선택 (파지 형태는 동일 —
        # 불필요한 180° 손목 회전·케이블 꼬임 방지)
        j7_lo, j7_hi = sel.get("j7_limit_deg", [-89.0, 90.0])
        home_j7 = float(cfg["robot"]["home_joints_deg"][6])
        feasible = []
        for sign in (1, -1):
            T_grasp = tool_pose_from_grasp(R_b, p_b, closing_sign=sign,
                                           insertion_m=insertion)
            updot = None
            if cam_u is not None:
                updot = float((T_grasp[:3, :3] @ cam_u)[2])
                if updot < cam_min_updot:
                    report.rejected.append(
                        (i, f"카메라 하향 {np.degrees(np.arcsin(-updot)):.0f}° "
                            f"(sign={sign:+d})"))
                    continue
            T_pre = offset_along_tool_z(T_grasp, sel["pre_grasp_offset_m"])
            T_lift = lifted(T_grasp, sel["lift_height_m"])
            wrist = None
            if robot is not None and sel.get("ik_check_per_candidate", True):
                bad = next((name for name, T in
                            [("pre", T_pre), ("grasp", T_grasp),
                             ("lift", T_lift)] if not _ik_ok(robot, T)), None)
                if bad is not None:
                    report.rejected.append(
                        (i, f"IK 불가 ({bad}, sign={sign:+d})"))
                    continue
                # 세 자세 모두 J7 해가 허용 범위 안이어야 함
                j7_out = None
                for name, T in [("pre", T_pre), ("grasp", T_grasp),
                                ("lift", T_lift)]:
                    j7 = _j7_deg(robot, T)
                    if j7 is not None and not (j7_lo <= j7 <= j7_hi):
                        j7_out = (name, j7)
                        break
                    if name == "grasp" and j7 is not None:
                        wrist = abs(_wrap_deg(j7 - home_j7))
                if j7_out is not None:
                    report.rejected.append(
                        (i, f"J7 {j7_out[1]:.0f}° 범위 밖 "
                            f"[{j7_lo:.0f},{j7_hi:.0f}] "
                            f"({j7_out[0]}, sign={sign:+d})"))
                    continue
            feasible.append((sign, T_grasp, T_pre, T_lift, wrist, updot))
        if not feasible:
            continue
        if len(feasible) == 2:
            # 카메라 상향 우선 (updot 차이가 유의미할 때) → 아니면 J7 회전 최소
            if cam_u is not None and abs(feasible[0][5] - feasible[1][5]) > 0.05:
                feasible.sort(key=lambda f: -f[5])
                print(f"   sign 선택: {feasible[0][0]:+d} (카메라 상향, "
                      f"updot {feasible[0][5]:+.2f} vs {feasible[1][5]:+.2f})")
            elif all(f[4] is not None for f in feasible):
                feasible.sort(key=lambda f: f[4])
                print(f"   sign 선택: {feasible[0][0]:+d} "
                      f"(J7 회전 {feasible[0][4]:.0f}° vs {feasible[1][4]:.0f}°)")
        sign, T_grasp, T_pre, T_lift, _, _ = feasible[0]
        pool.append(SelectedGrasp(
            index=i, score=float(g.score), width_m=float(g.width),
            closing_sign=sign, T_tcp_grasp=T_grasp, T_tcp_pre=T_pre,
            T_tcp_lift=T_lift, p_base=p_b, tilt_deg=float(tilt)))

    if pool:
        top = pool[0]  # 최고 score 실행 가능 후보
        pool.sort(key=lambda c: (c.tilt_deg, -c.score))
        chosen = pool[0]
        if chosen.index != top.index:
            print(f"   수직 우선 선택: #{chosen.index} "
                  f"(tilt {chosen.tilt_deg:.0f}°, score {chosen.score:.3f}) "
                  f"← 최고점 #{top.index} (tilt {top.tilt_deg:.0f}°, "
                  f"Δscore {top.score - chosen.score:.3f}, band {len(pool)}개)")
        elif len(pool) > 1:
            print(f"   수직 우선 확인: 최고점 #{top.index}가 band {len(pool)}개 중 "
                  f"가장 수직 (tilt {top.tilt_deg:.0f}°)")
        report.selected = chosen
    return report
