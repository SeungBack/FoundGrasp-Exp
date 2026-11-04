"""Transfer geometry and preparation, separated from motion submission.

IK sampling is a feasibility screen, not certification of the controller path.
A prepared plan is tied to the measured robot state and never changed in flight.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .se3 import inv_se3, pose_delta


def _require_stopped(robot):
    """A matching pose alone is insufficient when another command is queued."""
    state_code, state = robot.arm.get_state()
    queue_code, pending = robot.arm.get_cmdnum()
    error_code, errors = robot.arm.get_err_warn_code()
    if state_code or queue_code or error_code or state != 2 or pending != 0 or errors[0]:
        raise RuntimeError(f'상승 계획 시작 정지 확인 실패 — 이동 미전송 '
                           f'(state={state}, error/warn={errors}, pending={pending}, '
                           f'read_codes={[state_code, queue_code, error_code]})')


def tcp_ik_joints(robot, T_tcp, label, j7_limits=None):
    q = robot.flange_ik_joints_deg(T_tcp @ inv_se3(robot.T_flange_tcp))
    if q is None or np.shape(q) != (7,) or not np.isfinite(q).all():
        raise RuntimeError(f"IK 불가 ({label}, 반환={q})")
    q = np.asarray(q, dtype=float)
    if j7_limits is not None and not j7_limits[0] <= q[6] <= j7_limits[1]:
        raise RuntimeError(f"J7 {q[6]:.2f}° 제한 [{j7_limits[0]:g}, {j7_limits[1]:g}] 초과 "
                           f"({label}, IK 관절={np.round(q, 2).tolist()})")
    return q


@dataclass(frozen=True)
class TransferPlan:
    start_tcp: np.ndarray
    start_joints: np.ndarray
    pre: np.ndarray
    upper: np.ndarray
    transfer_sign: int

    def __post_init__(self):
        for name in ('start_tcp', 'start_joints', 'pre', 'upper'):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def check_start(self, robot):
        _require_stopped(robot)
        actual, _ = robot.get_tcp_pose()
        joints = np.asarray(robot.get_joints_deg(), dtype=float)
        if np.shape(actual) != (4, 4) or not np.isfinite(actual).all():
            raise RuntimeError('상승 계획 시작 TCP 판독 오류 — 이동 미전송')
        pos, angle = pose_delta(actual, self.start_tcp)
        if (joints.shape != (7,) or not np.isfinite(joints).all()
                or pos > .002 or angle > 1.
                or np.max(np.abs(joints - self.start_joints)) > .5):
            raise RuntimeError('상승 계획 이후 로봇 시작 자세가 변경됨 — 이동 미전송')


def prepare_transfer(robot, T_pre, T_lift, robot_cfg, drop_cfg, *, transfer_sign=1,
                     j7_limits=None, camera_up=None, allow_symmetry=False, check=None):
    """Choose a feasible upper symmetry BEFORE closing/enqueueing.

    Never changes the grasp/pre-grasp pose. A common retreat failure cannot be
    solved by switching the upper orientation, so it aborts immediately.
    camera_up is (tool-frame lateral direction, minimum world-up dot product).
    """
    if check:
        check()
    _require_stopped(robot)
    current, _ = robot.get_tcp_pose()
    joints = np.asarray(robot.get_joints_deg(), dtype=float)
    if np.shape(current) != (4, 4) or not np.isfinite(current).all():
        raise RuntimeError('상승 계획용 실제 TCP 판독 오류')
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise RuntimeError('상승 계획용 실제 관절 판독 오류')
    if j7_limits is not None and not j7_limits[0] <= joints[6] <= j7_limits[1]:
        raise RuntimeError(f'실측 J7 {joints[6]:.2f}° 제한 초과 — 상승 계획 중단')
    if T_pre[2, 3] <= current[2, 3]:
        raise RuntimeError('after-grasp 복귀 목표가 현재 위치보다 높지 않습니다')
    try:
        tcp_ik_joints(robot, T_pre, '복귀 pre-grasp', j7_limits)
    except RuntimeError as exc:
        raise RuntimeError(f'{exc}; 실측 관절={np.round(joints, 2).tolist()}') from exc
    failures = []
    signs = (transfer_sign, -transfer_sign) if allow_symmetry else (transfer_sign,)
    for sign in signs:
        if check:
            check()
        target, _ = transfer_waypoint(T_pre, T_lift, drop_cfg, robot.R_home_tcp, sign)
        if camera_up is not None:
            direction, min_updot = camera_up
            if float((target[:3, :3] @ direction)[2]) < min_updot:
                failures.append(f'상승 {sign:+d}: 카메라 하향 제한')
                continue
        try:
            tcp_ik_joints(robot, target, f'상승 {sign:+d} 경유점', j7_limits)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        reason = rise_path_error(robot, T_pre, target, robot_cfg, j7_limits, check)
        if reason:
            failures.append(f'상승 {sign:+d}: {reason}')
            continue
        plan = TransferPlan(current, joints, T_pre, target, sign)
        plan.check_start(robot)
        return plan
    raise RuntimeError('상승 경로 준비 실패: ' + '; '.join(failures)
                       + f'; 실측 관절={np.round(joints, 2).tolist()}')


def transfer_waypoint(T_pre, T_lift, drop_cfg, R_home_tcp, closing_sign=1):
    """Rise at fixed XY toward home or its tool-z half-turn equivalent."""
    if closing_sign not in (1, -1):
        raise RuntimeError("잘못된 상승 대칭 방향")
    radius = float(drop_cfg.get("transfer_clearance_margin_mm",
                                drop_cfg.get("lift_blend_radius_mm", 10)))
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("상승 높이 여유는 양수여야 합니다")
    rotation = np.asarray(R_home_tcp, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise RuntimeError("홈 TCP 자세 기록이 없습니다")
    target = T_pre.copy()
    clearance = max(float(T_lift[2, 3]), float(drop_cfg.get("via_height_m", .35)),
                    float(T_pre[2, 3]) + .05)
    radius = min(radius, (clearance - T_pre[2, 3]) * 1000 * .25)
    target[2, 3] = clearance + radius / 1000
    target[:3, :3] = rotation @ np.diag([closing_sign, closing_sign, 1.])
    return target, radius


def reorientation_speed(T_start, T_end, cfg):
    """Conservative translation-speed cap from required TCP rotation time.

    This is a path-average planning heuristic, not a joint velocity guarantee.
    Controller speed/singularity protection remains enabled.
    """
    distance, angle = pose_delta(T_start, T_end)
    speed = min(float(cfg["lift_speed_mm_s"]),
                float(cfg.get("reorient_speed_mm_s", 100)))
    angular_rate = float(cfg.get("reorient_angular_speed_deg_s", 30))
    if not np.isfinite([speed, angular_rate]).all() or min(speed, angular_rate) <= 0:
        raise RuntimeError("자세 전환 속도 설정은 유한한 양수여야 합니다")
    if angle > 1e-3:
        speed = min(speed, distance * 1000 * angular_rate / angle)
    if speed < 1:
        raise RuntimeError("자세 전환 거리가 회전량에 비해 너무 짧습니다")
    return speed, angle


def rise_path_error(robot, start, target, cfg, j7_limits=None, check=None):
    """Sample nominal straight rise + shortest rotation; return rejection reason.

    Feasibility screening only: SDK endpoint IK cannot certify the controller's
    blended trajectory, branch continuity, singularities or swept collisions.
    In particular, firmware 2.7.1 ignores the SDK's ref_angles parameter.
    """
    step_mm = float(cfg.get("rise_ik_step_mm", 20))
    step_deg = float(cfg.get("rise_ik_step_deg", 5))
    if not np.isfinite([step_mm, step_deg]).all() or min(step_mm, step_deg) <= 0:
        raise ValueError("상승 IK 검사 간격은 유한한 양수여야 합니다")
    distance, angle = pose_delta(start, target)
    count = max(2, int(np.ceil(max(distance * 1000 / step_mm, angle / step_deg))))
    rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack(
        [start[:3, :3], target[:3, :3]])))
    for i in range(1, count):
        if check:
            check()
        fraction = i / count
        T = np.eye(4)
        T[:3, 3] = start[:3, 3] * (1 - fraction) + target[:3, 3] * fraction
        T[:3, :3] = rotations(fraction).as_matrix()
        location = f"상승 경로 {fraction:.0%}, z={T[2, 3] * 1000:.1f}mm"
        try:
            tcp_ik_joints(robot, T, location, j7_limits)
        except RuntimeError as exc:
            return str(exc)
    return None


