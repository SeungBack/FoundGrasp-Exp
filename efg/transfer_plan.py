"""Transfer geometry and preparation, separated from motion submission.

IK sampling is a feasibility screen, not certification of the controller path.
A prepared plan is tied to the measured robot state and never changed in flight.
"""
from dataclasses import dataclass, replace
import re

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .se3 import inv_se3, pose_delta, se3_to_xarm_pose, xarm_pose_to_se3
from .motion_command import MotionCommand, MotionCheckError, PlanRejected, preflight_with_retiming


class _PathIKError(RuntimeError):
    """A geometric IK rejection, never a transport or malformed-data failure."""


def _require_stopped(robot):
    """A matching pose alone is insufficient when another command is queued."""
    state_code, state = robot.arm.get_state()
    queue_code, pending = robot.arm.get_cmdnum()
    error_code, errors = robot.arm.get_err_warn_code()
    if (state_code or queue_code or error_code or state != 2 or pending != 0
            or errors[0] or robot.arm.mode != 0):
        raise RuntimeError(f'상승 계획 시작 정지 확인 실패 — 이동 미전송 '
                           f'(mode={robot.arm.mode}, state={state}, error/warn={errors}, pending={pending}, '
                           f'read_codes={[state_code, queue_code, error_code]})')


def tcp_ik_joints(robot, T_tcp, label, j7_limits=None):
    q = robot.flange_ik_joints_deg(T_tcp @ inv_se3(robot.T_flange_tcp))
    if q is None:
        failure = getattr(robot, 'last_ik_failure', None)
        reason = failure if isinstance(failure, str) else f'반환={q}'
        if isinstance(failure, str):
            code = re.search(r'SDK code=(-?\d+)', failure)
            # get_inverse_kinematics returns 10 (RET_IS_INVALID) when it has
            # no valid result. This API status is distinct from a native motion
            # preflight's code=1 + only_check_result=21 combination. Code 1 on
            # an IK query means HAS_ERROR, not a geometric rejection.
            if code is None or int(code.group(1)) != 10:
                raise RuntimeError(f"IK 조회 실패 ({label}, {reason})")
            # INVALID can take precedence over fault flags in the SDK reply.
            # Never hide an actual controller fault behind an unavailable IK.
            _require_stopped(robot)
        raise _PathIKError(f"IK 불가 ({label}, {reason})")
    if np.shape(q) != (7,) or not np.isfinite(q).all():
        raise RuntimeError(f"IK 관절 판독 오류 ({label}, 반환={q})")
    q = np.asarray(q, dtype=float)
    if j7_limits is not None and not j7_limits[0] <= q[6] <= j7_limits[1]:
        raise _PathIKError(f"J7 {q[6]:.2f}° 제한 [{j7_limits[0]:g}, {j7_limits[1]:g}] 초과 "
                           f"({label}, IK 관절={np.round(q, 2).tolist()})")
    return q


@dataclass(frozen=True)
class TransferRoute:
    """Geometric route and immutable commands, before tying it to live state."""
    name: str
    pre: np.ndarray
    upper: np.ndarray
    transfer_sign: int
    commands: tuple
    start: np.ndarray | None = None

    def __post_init__(self):
        for name in ('pre', 'upper', 'start'):
            original = getattr(self, name)
            if original is None and name == 'start':
                continue
            value = np.array(original, dtype=float, copy=True)
            if value.shape != (4, 4) or not np.isfinite(value).all():
                raise ValueError(f'{name}는 유한한 4×4 자세여야 합니다')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        commands = tuple(self.commands)
        if (len(commands) < 3 or any(c.kind != 'cartesian' for c in commands[:-1])
                or commands[-1].kind != 'joint'):
            raise ValueError('복귀 경로는 Cartesian 경유점들과 마지막 조인트 이송으로 구성해야 합니다')
        for command, target in ((commands[0], self.pre), (commands[-2], self.upper)):
            if not np.allclose(xarm_pose_to_se3(command.target), target, atol=1e-8, rtol=0):
                raise ValueError('복귀 경로 자세와 명령 목표가 일치하지 않습니다')
        object.__setattr__(self, 'commands', commands)


@dataclass(frozen=True)
class TransferPlan:
    start_tcp: np.ndarray
    start_joints: np.ndarray
    pre: np.ndarray
    upper: np.ndarray
    transfer_sign: int
    commands: tuple = ()
    j7_limits: tuple | None = None
    speed_scale: float = 1.
    route_name: str = 'configured'

    def __post_init__(self):
        for name in ('start_tcp', 'start_joints', 'pre', 'upper'):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'commands', tuple(self.commands))
        if self.j7_limits is not None:
            object.__setattr__(self, 'j7_limits', tuple(self.j7_limits))

    @property
    def continuous(self):
        return bool(self.commands and self.commands[-2].radius > 0)

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
                     j7_limits=None, camera_up=None, allow_symmetry=False, check=None,
                     preferred_route=None):
    """Search return routes from measured state before closing or enqueueing.

    A check-only rejection may try another geometric route while the real arm
    remains unchanged and healthy. Actual faults and transport errors propagate.
    The grasp/pre-grasp geometry and clearance height never change in this search.
    """
    transfer_signs(drop_cfg, transfer_sign, allow_symmetry=allow_symmetry)
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
    guard = TransferPlan(current, joints, T_pre, T_pre, transfer_sign, j7_limits=j7_limits)
    try:
        tcp_ik_joints(robot, T_pre, '복귀 pre-grasp', j7_limits)
    except _PathIKError as exc:
        if check:
            check()
        guard.check_start(robot)
        raise PlanRejected(f'{exc}; 실측 관절={np.round(joints, 2).tolist()}') from exc
    failures = []
    rejected_retreats = set()
    routes = transfer_route_candidates(
        current, T_pre, T_lift, robot_cfg, drop_cfg, robot.R_home_tcp,
        transfer_sign=transfer_sign, allow_symmetry=allow_symmetry,
        preferred_route=preferred_route)
    for route in routes:
        if route.commands[0] in rejected_retreats:
            continue  # The same first command has no knowledge of later waypoints.
        if check:
            check()
        reason = route_path_error(robot, route, robot_cfg, j7_limits, camera_up, check)
        if reason:
            if check:
                check()
            guard.check_start(robot)
            failures.append(f'상승 {route.transfer_sign:+d}: {reason} [경로={route.name}]')
            continue
        commands = route.commands
        if j7_limits is not None and not j7_limits[0] <= commands[-1].target[6] <= j7_limits[1]:
            raise RuntimeError('릴리즈 목표 J7 제한 초과')
        plan = TransferPlan(current, joints, T_pre, route.upper, route.transfer_sign,
                            commands, j7_limits, route_name=route.name)
        try:
            commands, scale = preflight_with_retiming(
                robot.arm, commands, check_start=lambda: plan.check_start(robot), check=check)
        except MotionCheckError as exc:
            if check:
                check()
            plan.check_start(robot)
            if not exc.is_path_rejection:
                raise
            failures.append(f'상승 {route.transfer_sign:+d}: {exc} [경로={route.name}]')
            if exc.command_index == 0:
                rejected_retreats.add(route.commands[0])
            continue
        plan = replace(plan, commands=commands, speed_scale=scale)
        plan.check_start(robot)
        return plan
    if check:
        check()
    guard.check_start(robot)
    raise PlanRejected('상승 경로 준비 실패: ' + '; '.join(failures)
                       + f'; 실측 관절={np.round(joints, 2).tolist()}')


def _proper_rotation(value, label):
    rotation = np.asarray(value, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise RuntimeError(f'{label}는 유효한 3×3 회전행렬이어야 합니다')
    return rotation


def closest_upright_rotation(R_pre, R_home):
    """Align the tool axis to home's axis with the least orientation change.

    Home's yaw is not a transport constraint. Parallel-transport the current
    heading by the shortest rotation of the tool z axis instead of prescribing
    a full home orientation. Antiparallel axes admit many equal minima; choose
    the pre-grasp x axis deterministically for the half turn.
    """
    pre = _proper_rotation(R_pre, 'pre-grasp 자세')
    home = _proper_rotation(R_home, '홈 TCP 자세')
    cross = np.cross(pre[:, 2], home[:, 2])
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(pre[:, 2], home[:, 2]), -1., 1.))
    if sine > 1e-12:
        axis_angle = cross / sine * np.arctan2(sine, cosine)
        alignment = Rotation.from_rotvec(axis_angle).as_matrix()
    elif cosine < 0:
        alignment = Rotation.from_rotvec(pre[:, 0] * np.pi).as_matrix()
    else:
        alignment = np.eye(3)
    return alignment @ pre


def transfer_signs(drop_cfg, preferred=1, *, allow_symmetry=True):
    """Allowed upper orientations; grasp retention forbids a wrist half turn."""
    if preferred not in (1, -1):
        raise RuntimeError("잘못된 상승 대칭 방향")
    orientation = drop_cfg.get('lift_orientation', 'upright')
    if orientation not in ('grasp', 'upright', 'home'):
        raise RuntimeError("drop.lift_orientation은 'grasp', 'upright', 'home' 중 하나여야 합니다")
    if orientation == 'grasp':
        if preferred != 1:
            raise RuntimeError("파지 자세 유지(grasp) 상승에는 180° 대칭 방향을 적용할 수 없습니다")
        return (1,)
    return (preferred, -preferred) if allow_symmetry else (preferred,)


def transfer_waypoint(T_pre, T_lift, drop_cfg, R_home_tcp, closing_sign=1):
    """Return upper waypoint and bounded outgoing blend radius (0 = stop).

    Grasp transport retains the pre-grasp rotation through the Cartesian lift.
    Upright preserves the nearest heading; 'home' retains the legacy full home
    orientation. Only the latter two allow a tool-z half turn for jaw symmetry.
    The nominal corner gets at least one radius above the clearance height.
    This is not a swept-volume certificate for the subsequent joint trajectory.
    """
    transfer_signs(drop_cfg, closing_sign, allow_symmetry=False)
    orientation = drop_cfg.get('lift_orientation', 'upright')
    if orientation == 'grasp':
        rotation = _proper_rotation(T_pre[:3, :3], 'pre-grasp 자세')
    elif orientation == 'upright':
        rotation = closest_upright_rotation(T_pre[:3, :3], R_home_tcp)
    elif orientation == 'home':
        rotation = _proper_rotation(R_home_tcp, '홈 TCP 자세')
    minimum = _minimum_transfer_height(drop_cfg)
    radius = float(drop_cfg.get("transfer_clearance_margin_mm",
                                drop_cfg.get("lift_blend_radius_mm", 10)))
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("상승 높이 여유는 양수여야 합니다")
    target = T_pre.copy()
    clearance = max(float(T_lift[2, 3]), float(drop_cfg.get("via_height_m", .35)),
                    float(T_pre[2, 3]) + .05, minimum or 0.)
    radius = min(radius, (clearance - T_pre[2, 3]) * 1000 * .25)
    blend = float(drop_cfg.get('transfer_blend_radius_mm', 0))
    if not np.isfinite(blend) or blend < 0:
        raise RuntimeError('drop.transfer_blend_radius_mm는 유한한 0 이상의 값이어야 합니다')
    blend = min(blend, (clearance - T_pre[2, 3]) * 1000 * .25)
    target[2, 3] = clearance + max(radius, blend) / 1000
    target[:3, :3] = rotation @ np.diag([closing_sign, closing_sign, 1.])
    return target, blend


def transfer_commands(current, pre, upper, upper_radius, robot_cfg, drop_cfg):
    """Compile complete absolute commands once for checking and execution."""
    retreat_radius = float(robot_cfg.get('retreat_blend_radius_mm', 5))
    if not np.isfinite(retreat_radius) or retreat_radius < 0:
        raise RuntimeError('잘못된 복귀 블렌딩 반경')
    retreat_radius = min(retreat_radius, np.linalg.norm(pre[:3, 3] - current[:3, 3]) * 250,
                         (upper[2, 3] - pre[2, 3]) * 250)
    retreat_speed = min(float(robot_cfg['lift_speed_mm_s']),
                        float(robot_cfg.get('retreat_speed_mm_s', 150)))
    rise_speed, _ = reorientation_speed(pre, upper, robot_cfg)
    return (
        MotionCommand('cartesian', tuple(se3_to_xarm_pose(pre)), retreat_speed,
                      float(robot_cfg.get('lift_acc_mm_s2', 500)),
                      retreat_radius if retreat_radius > .01 else -1, '복귀'),
        MotionCommand('cartesian', tuple(se3_to_xarm_pose(upper)), rise_speed,
                      float(robot_cfg.get('reorient_acc_mm_s2', 150)),
                      upper_radius if upper_radius > 0 else -1, '상승'),
        # radius=0 selects MoveJointB for an incoming blend, with no outgoing
        # rounded corner. No successor is queued; final arrival is still checked.
        MotionCommand('joint', tuple(drop_cfg['joints_deg']),
                      float(drop_cfg['joint_speed_deg_s']),
                      float(drop_cfg.get('joint_acc_deg_s2', 50)),
                      0 if upper_radius > 0 else -1, '이송'),
    )


def transfer_route_candidates(current, T_pre, T_lift, robot_cfg, drop_cfg, R_home_tcp, *,
                              transfer_sign=1, allow_symmetry=False, preferred_route=None):
    """Search height and orientation alternatives within explicit clearance bounds.

    Without a minimum-height setting, retain the configured upper height. With
    an explicit minimum, sample upward at height_search_step_m intervals and
    include the exact nominal endpoint. The original object-escape lift and
    pre-grasp + 50mm remain hard lower bounds. At each height try the configured
    orientation, half tilt correction, then nearest upright. Legacy dual-sign
    search uses the two configured signs and nearest upright instead, retaining
    three routes per height. Rotation alternatives retain a short vertical guard
    so rounded rotation cannot start below pre.
    """
    signs = transfer_signs(drop_cfg, transfer_sign, allow_symmetry=allow_symmetry)
    routes = []

    def append(route):
        # Keep a preferred, already retimed command sequence rather than replace
        # it with a faster reconstruction having the same path geometry.
        for old in routes:
            if len(old.commands) == len(route.commands) and all(
                    a.kind == b.kind and abs(a.radius - b.radius) < 1e-7
                    and np.allclose(a.target, b.target, atol=1e-8, rtol=0)
                    for a, b in zip(old.commands, route.commands)):
                return
        routes.append(route)

    if preferred_route is not None:
        if not isinstance(preferred_route, TransferRoute):
            raise TypeError('preferred_route는 TransferRoute여야 합니다')
        if not np.allclose(preferred_route.pre, T_pre, atol=1e-8, rtol=0):
            raise RuntimeError('사전 선정한 복귀 경로와 pre-grasp 목표가 달라졌습니다')
        if drop_cfg.get('lift_orientation', 'upright') == 'grasp' and preferred_route.transfer_sign != 1:
            raise RuntimeError('파지 방향 유지 모드의 복귀 경로에 180° 대칭을 사용할 수 없습니다')
        append(replace(preferred_route, start=current))
    search = bool(drop_cfg.get('route_search', False))
    minimum = _minimum_transfer_height(drop_cfg)
    step = float(drop_cfg.get('height_search_step_m', .02))
    if not np.isfinite(step) or step <= 0:
        raise ValueError('drop.height_search_step_m는 유한한 양수여야 합니다')
    nominal_height = max(float(T_lift[2, 3]), float(T_pre[2, 3]) + .05,
                         float(drop_cfg.get('via_height_m', .35)), minimum or 0.)
    heights = [('nominal', nominal_height)]
    if search and minimum is not None:
        lower_height = max(minimum, float(T_lift[2, 3]), float(T_pre[2, 3]) + .05)
        if lower_height < nominal_height - 1e-6:
            # Do not silently truncate a dense search: the nominal endpoint may
            # be the only feasible route. Reject unreasonable settings instead.
            interval_count = (nominal_height - lower_height) / step
            if not np.isfinite(interval_count) or interval_count > 63 + 1e-10:
                raise ValueError('상승 높이 후보가 64개를 초과합니다 — '
                                 'drop.height_search_step_m를 늘리거나 높이 범위를 줄이세요')
            intervals = max(1, int(np.ceil(interval_count - 1e-10)))
            heights = [('lower' if index == 0 else f'level_{index}',
                        lower_height + index * step) for index in range(intervals)]
            heights.append(('nominal', nominal_height))
    upright = (closest_upright_rotation(T_pre[:3, :3], R_home_tcp)
               if search else T_pre[:3, :3])
    has_tilt = not np.allclose(upright, T_pre[:3, :3], atol=1e-8, rtol=0)
    rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack([T_pre[:3, :3], upright])))
    for height_name, height in heights:
        height_cfg = dict(drop_cfg, via_height_m=height)
        prefix = '' if height_name == 'nominal' else height_name + '_'
        for sign in signs:
            upper, radius = transfer_waypoint(T_pre, T_lift, height_cfg, R_home_tcp, sign)
            name = 'configured' if sign == transfer_sign else 'configured_symmetry'
            append(TransferRoute(prefix + name, T_pre, upper, sign,
                                 transfer_commands(current, T_pre, upper, radius, robot_cfg, drop_cfg),
                                 current))
        if not search or not has_tilt:
            continue
        upper, radius = transfer_waypoint(T_pre, T_lift, height_cfg, R_home_tcp, transfer_sign)
        guard = T_pre.copy()
        guard[2, 3] += .2 * (upper[2, 3] - T_pre[2, 3])
        corrections = ((1., 'upright'),) if len(signs) > 1 else ((.5, 'half_tilt'), (1., 'upright'))
        for fraction, name in corrections:
            target = upper.copy()
            target[:3, :3] = rotations(fraction).as_matrix()
            incoming_radius = min(radius, (target[2, 3] - guard[2, 3]) * 250)
            retreat, rise, transfer = transfer_commands(
                current, T_pre, target, incoming_radius, robot_cfg, drop_cfg)
            retreat = replace(retreat, radius=(
                min(retreat.radius, (guard[2, 3] - T_pre[2, 3]) * 250)
                if retreat.radius > 0 else retreat.radius))
            guard_radius = min(float(robot_cfg.get('retreat_blend_radius_mm', 5)),
                               (guard[2, 3] - T_pre[2, 3]) * 250,
                               (target[2, 3] - guard[2, 3]) * 250)
            guard_speed, _ = reorientation_speed(T_pre, guard, robot_cfg)
            rise_speed, _ = reorientation_speed(guard, target, robot_cfg)
            guard_command = MotionCommand(
                'cartesian', tuple(se3_to_xarm_pose(guard)), guard_speed,
                float(robot_cfg.get('reorient_acc_mm_s2', 150)),
                guard_radius if guard_radius > .01 else -1, '자세 유지 상승')
            append(TransferRoute(prefix + name, T_pre, target, 1,
                                 (retreat, guard_command, replace(rise, speed=rise_speed), transfer),
                                 current))
    return routes


def _minimum_transfer_height(drop_cfg):
    """The operator supplies lateral clearance; do not infer it from the scene."""
    value = drop_cfg.get('min_transfer_height_m')
    if value is None:
        return None
    minimum = float(value)
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError('drop.min_transfer_height_m는 null 또는 유한한 양수여야 합니다')
    return minimum


def _path_samples(start, target, cfg):
    step_mm = float(cfg.get('rise_ik_step_mm', 20))
    step_deg = float(cfg.get('rise_ik_step_deg', 5))
    if not np.isfinite([step_mm, step_deg]).all() or min(step_mm, step_deg) <= 0:
        raise ValueError('상승 IK 검사 간격은 유한한 양수여야 합니다')
    distance, angle = pose_delta(start, target)
    count = max(2, int(np.ceil(max(distance * 1000 / step_mm, angle / step_deg))))
    rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack(
        [start[:3, :3], target[:3, :3]])))
    for index in range(1, count):
        fraction = index / count
        T = np.eye(4)
        T[:3, 3] = start[:3, 3] * (1 - fraction) + target[:3, 3] * fraction
        T[:3, :3] = rotations(fraction).as_matrix()
        yield fraction, T


def route_path_error(robot, route, robot_cfg, j7_limits=None, camera_up=None,
                     check=None, joint_samples=None):
    """Screen every Cartesian endpoint and interior, including the retreat.

    Returns geometric rejection text only. Malformed IK, transport failures and
    external check exceptions propagate; callers must also verify real state.
    """
    poses = [xarm_pose_to_se3(command.target) for command in route.commands[:-1]]
    labels = ['복귀 pre-grasp'] + ['상승 중간 경유점'] * (len(poses) - 2)
    labels += [f'상승 {route.transfer_sign:+d} 경유점']
    if route.start is not None:
        poses.insert(0, route.start)
        labels.insert(0, '파지점')

    def evaluate(T, label):
        if check:
            check()
        if camera_up is not None:
            direction, min_updot = camera_up
            if float((T[:3, :3] @ direction)[2]) < min_updot:
                return f'카메라 하향 제한 ({label})'
        try:
            q = tcp_ik_joints(robot, T, label, j7_limits)
        except _PathIKError as exc:
            return str(exc)
        if joint_samples is not None:
            joint_samples.append(q.copy())
        return None

    # Test the final orientation first to discard grossly infeasible routes
    # before doing all of their interior queries.
    for index in [len(poses) - 1, *range(len(poses) - 1)]:
        reason = evaluate(poses[index], labels[index])
        if reason:
            return reason
    for index, (start, target) in enumerate(zip(poses, poses[1:])):
        leg = '복귀' if route.start is not None and index == 0 else '상승'
        for fraction, T in _path_samples(start, target, robot_cfg):
            reason = evaluate(T, f'{leg} 경로 {fraction:.0%}, z={T[2, 3] * 1000:.1f}mm')
            if reason:
                return reason
    return None


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


def rise_path_error(robot, start, target, cfg, j7_limits=None, check=None, *, joint_samples=None):
    """Sample nominal straight rise + shortest rotation; return rejection reason.

    Feasibility screening only: SDK endpoint IK cannot certify the controller's
    blended trajectory, branch continuity, singularities or swept collisions.
    In particular, firmware 2.7.1 ignores the SDK's ref_angles parameter.
    """
    for fraction, T in _path_samples(start, target, cfg):
        if check:
            check()
        location = f"상승 경로 {fraction:.0%}, z={T[2, 3] * 1000:.1f}mm"
        try:
            joints = tcp_ik_joints(robot, T, location, j7_limits)
            if joint_samples is not None:
                joint_samples.append(joints)
        except _PathIKError as exc:
            return str(exc)
    return None


