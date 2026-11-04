"""그립 후보 선정 — score 순위대로 기하·로봇 IK 적합성을 검사해 첫 통과 후보 채택.

pre-grasp / grasp / transfer 경유 자세와 상승 중간 표본이 IK 가능해야 채택한다.
파지의 180° 등가 방향과 설정에서 허용한 상승 방향 중 관절 이동 비용이 작은 경로를 고른다.

공정성: score는 순위로만 쓴다. 방법마다 score 분포가 달라 절대값 기준(최소 score,
score 차 이내 수직 우선)은 방법 간 비교를 왜곡하므로 두지 않는다. 폭 한계·충돌
필터는 앞 단계(efg/postprocess.py)에서 모든 방법에 똑같이 적용된다.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .transfer_plan import (transfer_waypoint, transfer_signs, rise_path_error,
                            tcp_ik_joints, _PathIKError)
from .console import detail
from .motion_command import PlanRejected
from .transforms import (approach_tilt_deg, grasp_to_base, lifted,
                         offset_along_tool_z, tool_pose_from_grasp)
from .transforms import camera_lateral_dir_tcp as _camera_lateral_dir_tcp

if TYPE_CHECKING:
    from .pick_plan import PickPlan


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
    transfer_sign: int = 1  # 설정된 상승 기본 방향(+1) 또는 그 툴 z축 180°(-1)
    j7_range_deg: "tuple[float, float] | None" = None  # 선정 시 IK 표본 범위; 실측 경로 아님
    pick_plan: "PickPlan | None" = None  # 접근 전 전체 경로 검사에서 통과한 불변 명령


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


def _ik_joints(robot, T_base_tcp):
    """Reject geometric IK failures; never turn broken reads into an empty scene."""
    try:
        return tcp_ik_joints(robot, T_base_tcp, '후보 IK')
    except _PathIKError:
        return None


def _joint_route_cost(*waypoints):
    """Sum of largest joint changes per leg, then total travel as a tie-break.

    Endpoint heuristic, not a Cartesian path duration prediction. Do not wrap
    angles: -170 -> +170 is 340 degrees with bounded joints/cables, not 20.
    """
    changes = np.abs(np.diff(np.asarray(waypoints, dtype=float), axis=0))
    return float(changes.max(axis=1).sum()), float(changes.sum())


def candidate_j7_limits(selection_cfg):
    """Reserve wrist travel during selection; execution keeps the hard limits.

    Unseeded controller IK may return a different branch after approaching.
    This reserve rejects boundary-adjacent candidates, not a certified bound
    on that discrepancy. Actual-state planning and live monitoring still apply.
    """
    limits = np.asarray(selection_cfg.get("j7_limit_deg", [-89., 90.]), dtype=float)
    margin = float(selection_cfg.get("j7_selection_margin_deg", 0.))
    if (limits.shape != (2,) or not np.isfinite(limits).all()
            or not np.isfinite(margin) or margin < 0
            or limits[0] + margin >= limits[1] - margin):
        raise ValueError("selection.j7_selection_margin_deg는 유한한 0 이상의 값이며 "
                         "j7_limit_deg 안에 선정 범위를 남겨야 합니다")
    return float(limits[0] + margin), float(limits[1] - margin)


def select_grasp(gg, T_base_cam, cfg, robot=None, *, candidate_prepare=None):
    """gg(카메라 좌표계, score 내림차순)에서 순위대로 첫 실행 가능 후보 선정.

    robot이 None이면 IK 체크를 건너뛰고 기하 필터만 적용한다.
    candidate_prepare가 있으면 각 대칭의 전체 경로를 계획한다. PlanRejected는
    다른 대칭/후보를 검토하게 하고, 통신·장비 오류는 호출자에게 전달한다.
    반환: SelectionReport (selected가 None이면 실패).
    """
    from .postprocess import open_width_m

    sel = cfg["selection"]
    j7_lo, j7_hi = candidate_j7_limits(sel)
    xs, ys, zs = (sel["workspace_x_m"], sel["workspace_y_m"],
                  sel["workspace_z_m"])
    # Keep the camera-down hard limit; prefer camera-up only for equal route costs.
    cam_cfg = sel.get("camera_up") or {}
    cam_u = (_camera_lateral_dir_tcp(cfg, robot)
             if cam_cfg.get("prefer", False) else None)
    cam_min_updot = -np.sin(np.radians(float(cam_cfg.get("max_down_deg",
                                                         20.0))))
    report = SelectionReport()
    upper_signs = transfer_signs(cfg['drop'])
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
        feasible = []
        joint_samples = {}
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
                    report.rejected.append((i, f"J7 {joints[6]:.2f}° 선정 범위 밖 "
                                            f"[{j7_lo:g},{j7_hi:g}] "
                                            f"({name}, sign={sign:+d})"))
                    joints = None
                return joints

            q_pre = checked_joints("pre", T_pre)
            if q_pre is None:
                continue
            q_grasp = checked_joints("grasp", T_grasp)
            if q_grasp is None:
                continue
            if candidate_prepare is not None:
                # The route planner considers alternative lifts. A single
                # configured upper endpoint must not reject all those routes.
                cost = _joint_route_cost(current, q_pre, q_grasp, q_pre, drop_joints)
                feasible.append((sign, T_grasp, T_pre, T_lift, cost, updot, 1))
                joint_samples[sign, 1] = [q_pre, q_grasp]
                continue
            for transfer_sign in upper_signs:
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
                joint_samples[sign, transfer_sign] = [q_pre, q_grasp, q_transfer]
        if not feasible:
            continue
        feasible.sort(key=lambda f: (*f[4], -(f[5] if f[5] is not None else 0.)))
        chosen = None
        for option in feasible:
            sign, T_grasp, T_pre, T_lift, cost, _, transfer_sign = option
            if use_ik and candidate_prepare is None:
                target, _ = transfer_waypoint(T_pre, T_lift, cfg['drop'], robot.R_home_tcp,
                                              closing_sign=transfer_sign)
                reason = rise_path_error(
                    robot, T_pre, target, cfg['robot'], (j7_lo, j7_hi),
                    joint_samples=joint_samples[sign, transfer_sign])
                if reason:
                    report.rejected.append((i, f"{reason} (sign={sign:+d}, transfer={transfer_sign:+d})"))
                    continue
            j7_range = None
            if use_ik:
                wrist = np.asarray(joint_samples[sign, transfer_sign])[:, 6]
                j7_range = float(wrist.min()), float(wrist.max())
            candidate = SelectedGrasp(
                index=i, score=float(g.score), width_m=float(g.width),
                open_width_m=open_width_m(g.width, cfg),
                closing_sign=sign, T_tcp_grasp=T_grasp, T_tcp_pre=T_pre,
                T_tcp_lift=T_lift, p_base=p_b, tilt_deg=float(tilt),
                transfer_sign=transfer_sign, j7_range_deg=j7_range)
            if candidate_prepare is not None:
                try:
                    candidate = candidate_prepare(candidate)
                except PlanRejected as exc:
                    report.rejected.append((i, f"전체 경로 기각 (sign={sign:+d}): {exc}"))
                    continue
            chosen = option, candidate
            break
        if chosen is None:
            continue
        option, candidate = chosen
        sign, T_grasp, T_pre, T_lift, cost, _, transfer_sign = option
        if use_ik:
            detail(f"대칭 선택: 파지 {sign:+d} · 상승 {transfer_sign:+d} · "
                   f"경로 관절비용 {cost[0]:.1f}° / 총 이동 {cost[1]:.1f}° "
                   f"(가능 조합 {len(feasible)}/{2 * len(upper_signs)})")
        report.selected = candidate
        break
    return report
