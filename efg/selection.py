"""그립 후보 선정 — score 순위대로 기하·로봇 IK 적합성을 검사해 첫 통과 후보 채택.

pre-grasp / grasp / transfer 경유 자세가 모두 IK 가능해야 채택한다.
파지·상승 각각의 180° 등가 방향을 조합해 관절 이동 비용이 작은 경로를 고른다.

공정성: score는 순위로만 쓴다. 방법마다 score 분포가 달라 절대값 기준(최소 score,
score 차 이내 수직 우선)은 방법 간 비교를 왜곡하므로 두지 않는다. 폭 한계·충돌
필터는 앞 단계(efg/postprocess.py)에서 모든 방법에 똑같이 적용된다.
"""
from dataclasses import dataclass, field

import numpy as np

from .motion import transfer_waypoint
from .console import detail
from .transforms import (approach_tilt_deg, grasp_to_base, lifted,
                         offset_along_tool_z, tool_pose_from_grasp)


@dataclass
class SelectedGrasp:
    index: int
    score: float
    width_m: float          # 검출기 raw 예측 폭
    open_width_m: float     # 실행 개방 폭 (postprocess.open_width_m)
    closing_sign: int
    T_tcp_grasp: np.ndarray
    T_tcp_pre: np.ndarray
    T_tcp_lift: np.ndarray
    p_base: np.ndarray
    tilt_deg: float = 0.0  # 접근 방향의 수직(base -z) 대비 기울임
    transfer_sign: int = 1  # 상승 목표: 실측 홈 방향(+1) 또는 툴 z축 180°(-1)


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


def _ik_joints(robot, T_base_tcp):
    """Finite seven-joint endpoint solution; unavailable IK is not feasible."""
    fn = getattr(robot, "flange_ik_joints_deg", None)
    if fn is None:
        return None
    j = fn(_tcp_to_flange(robot, T_base_tcp))
    if j is None:
        return None
    j = np.asarray(j, dtype=float)
    return j if j.shape == (7,) and np.isfinite(j).all() else None


def _joint_route_cost(*waypoints):
    """Sum of largest joint changes per leg, then total travel as a tie-break.

    Endpoint heuristic, not a Cartesian path duration prediction. Do not wrap
    angles: -170 -> +170 is 340 degrees with bounded joints/cables, not 20.
    """
    changes = np.abs(np.diff(np.asarray(waypoints, dtype=float), axis=0))
    return float(changes.max(axis=1).sum()), float(changes.sum())


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
    """gg(카메라 좌표계, score 내림차순)에서 순위대로 첫 실행 가능 후보 선정.

    robot이 None이면 IK 체크를 건너뛰고 기하 필터만 적용한다.
    반환: SelectionReport (selected가 None이면 실패).
    """
    from .postprocess import open_width_m

    sel = cfg["selection"]
    xs, ys, zs = (sel["workspace_x_m"], sel["workspace_y_m"],
                  sel["workspace_z_m"])
    # Keep the camera-down hard limit; prefer camera-up only for equal route costs.
    cam_cfg = sel.get("camera_up") or {}
    cam_u = (_camera_lateral_dir_tcp(cfg, robot)
             if cam_cfg.get("prefer", False) else None)
    cam_min_updot = -np.sin(np.radians(float(cam_cfg.get("max_down_deg",
                                                         20.0))))
    report = SelectionReport()
    use_ik = robot is not None and sel.get("ik_check_per_candidate", True)
    current = robot.get_joints_deg() if use_ik else None
    drop_joints = np.asarray(cfg["drop"]["joints_deg"], dtype=float) if use_ik else None
    if use_ik and (np.shape(current) != (7,) or not np.isfinite(current).all()
                   or drop_joints.shape != (7,) or not np.isfinite(drop_joints).all()):
        raise RuntimeError("대칭 경로 비교용 현재/릴리즈 관절 값이 잘못되었습니다")

    for i in range(len(gg)):
        g = gg[i]
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
        # Compare complete routes within this grasp score rank, not across grasps.
        j7_lo, j7_hi = sel.get("j7_limit_deg", [-89.0, 90.0])
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
            if not use_ik:
                feasible.append((sign, T_grasp, T_pre, T_lift, (0., 0.), updot, 1))
                continue

            def checked_joints(name, T):
                joints = _ik_joints(robot, T)
                if joints is None:
                    report.rejected.append((i, f"IK 불가 ({name}, sign={sign:+d})"))
                elif not j7_lo <= joints[6] <= j7_hi:
                    report.rejected.append((i, f"J7 {joints[6]:.0f}° 범위 밖 "
                                            f"[{j7_lo:.0f},{j7_hi:.0f}] "
                                            f"({name}, sign={sign:+d})"))
                    joints = None
                return joints

            q_pre = checked_joints("pre", T_pre)
            if q_pre is None:
                continue
            q_grasp = checked_joints("grasp", T_grasp)
            if q_grasp is None:
                continue
            for transfer_sign in (1, -1):
                T_transfer, _ = transfer_waypoint(
                    T_pre, T_lift, cfg["drop"], robot.R_home_tcp,
                    closing_sign=transfer_sign)
                if cam_u is not None and float((T_transfer[:3, :3] @ cam_u)[2]) < cam_min_updot:
                    report.rejected.append((i, f"카메라 하향 (transfer={transfer_sign:+d})"))
                    continue
                q_transfer = checked_joints(f"transfer={transfer_sign:+d}", T_transfer)
                if q_transfer is None:
                    continue
                cost = _joint_route_cost(current, q_pre, q_grasp, q_pre, q_transfer, drop_joints)
                feasible.append((sign, T_grasp, T_pre, T_lift, cost, updot, transfer_sign))
        if not feasible:
            continue
        feasible.sort(key=lambda f: (*f[4], -(f[5] if f[5] is not None else 0.)))
        sign, T_grasp, T_pre, T_lift, cost, _, transfer_sign = feasible[0]
        if use_ik:
            detail(f"대칭 선택: 파지 {sign:+d} · 상승 {transfer_sign:+d} · "
                   f"경로 관절비용 {cost[0]:.1f}° / 총 이동 {cost[1]:.1f}° "
                   f"(가능 조합 {len(feasible)}/4)")
        report.selected = SelectedGrasp(
            index=i, score=float(g.score), width_m=float(g.width),
            open_width_m=open_width_m(g.width, cfg),
            closing_sign=sign, T_tcp_grasp=T_grasp, T_tcp_pre=T_pre,
            T_tcp_lift=T_lift, p_base=p_b, tilt_deg=float(tilt),
            transfer_sign=transfer_sign)
        break
    return report
