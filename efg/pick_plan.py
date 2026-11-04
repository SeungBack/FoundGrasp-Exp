"""Prepare a complete pick route before moving away from the scan pose.

The native controller checks approach, retreat, rise and transfer as one shadow
chain. Only the exact accepted commands are returned. This is a feasibility
screen; the measured grasp state must still be checked before closing.
"""
from dataclasses import dataclass, replace

import numpy as np

from .motion_command import (MotionCheckError, PlanRejected, approach_commands,
                             preflight_with_retiming)
from .se3 import pose_delta
from .transfer_plan import (_require_stopped, TransferRoute,
                            transfer_route_candidates, route_path_error)
from .transforms import camera_lateral_dir_tcp


def _check_start(robot, start_tcp, start_joints):
    _require_stopped(robot)
    actual, _ = robot.get_tcp_pose()
    joints = np.asarray(robot.get_joints_deg(), dtype=float)
    if (np.shape(actual) != (4, 4) or not np.isfinite(actual).all()
            or joints.shape != (7,) or not np.isfinite(joints).all()):
        raise RuntimeError('파지 계획 시작 자세 판독 오류 — 이동 미전송')
    pos, angle = pose_delta(actual, start_tcp)
    if pos > .002 or angle > 1. or np.max(np.abs(joints - start_joints)) > .5:
        raise RuntimeError('파지 계획 이후 로봇 시작 자세가 변경됨 — 이동 미전송')


@dataclass(frozen=True)
class PickPlan:
    start_tcp: np.ndarray
    start_joints: np.ndarray
    grasp: np.ndarray
    pre: np.ndarray
    approach_commands: tuple
    transfer_route: TransferRoute
    speed_scale: float = 1.
    j7_range_deg: tuple | None = None

    def __post_init__(self):
        for name in ('start_tcp', 'start_joints', 'grasp', 'pre'):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            shape = (7,) if name == 'start_joints' else (4, 4)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'파지 계획 {name} 값이 잘못되었습니다')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'approach_commands', tuple(self.approach_commands))
        if self.j7_range_deg is not None:
            object.__setattr__(self, 'j7_range_deg', tuple(self.j7_range_deg))

    def check_start(self, robot):
        _check_start(robot, self.start_tcp, self.start_joints)


def prepare_pick(robot, sel, cfg, *, check=None, j7_limits=None):
    """Choose a whole route at the actual scan state without requesting motion.

    j7_limits may be narrowed by the selection margin. The caller owns the arm;
    expected path rejection is returned as PlanRejected only after confirming
    the same actual healthy, stopped state. All other failures propagate.
    """
    if check:
        check()
    _require_stopped(robot)
    start_tcp, _ = robot.get_tcp_pose()
    start_joints = np.asarray(robot.get_joints_deg(), dtype=float)
    if (np.shape(start_tcp) != (4, 4) or not np.isfinite(start_tcp).all()
            or start_joints.shape != (7,) or not np.isfinite(start_joints).all()):
        raise RuntimeError('파지 계획용 실제 시작 자세 판독 오류')
    start_tcp = np.asarray(start_tcp, dtype=float).copy()
    start_joints = start_joints.copy()
    hard_limits = cfg.get('selection', {}).get('j7_limit_deg', [-89., 90.])
    if not hard_limits[0] <= start_joints[6] <= hard_limits[1]:
        raise RuntimeError('실측 J7 제한 초과 — 파지 계획 중단')
    if j7_limits is None:
        j7_limits = hard_limits
    pre, grasp = sel.T_tcp_pre, sel.T_tcp_grasp
    if pre[2, 3] <= grasp[2, 3]:
        raise PlanRejected('pre-grasp 복귀 목표가 파지점보다 높지 않습니다')
    approach = approach_commands(start_tcp, pre, grasp, cfg['robot'])
    cam_cfg = cfg.get('selection', {}).get('camera_up') or {}
    direction = camera_lateral_dir_tcp(cfg, robot) if cam_cfg.get('prefer', False) else None
    camera_up = None if direction is None else (
        direction, -np.sin(np.radians(float(cam_cfg.get('max_down_deg', 20.)))))
    failures = []

    def guard():
        if check:
            check()
        _check_start(robot, start_tcp, start_joints)

    for route in transfer_route_candidates(
            grasp, pre, sel.T_tcp_lift, cfg['robot'], cfg['drop'], robot.R_home_tcp,
            transfer_sign=getattr(sel, 'transfer_sign', 1), allow_symmetry=True):
        guard()
        samples = []
        reason = route_path_error(robot, route, cfg['robot'], j7_limits,
                                  camera_up=camera_up, check=check, joint_samples=samples)
        if reason:
            guard()
            failures.append(f'{route.name}: {reason}')
            continue
        if not hard_limits[0] <= route.commands[-1].target[6] <= hard_limits[1]:
            raise RuntimeError('릴리즈 목표 J7 제한 초과 — 파지 계획 중단')
        try:
            accepted, scale = preflight_with_retiming(
                robot.arm, approach + route.commands, check_start=guard, check=check)
        except MotionCheckError as exc:
            guard()
            if not exc.is_path_rejection:
                raise
            failures.append(f'{route.name}: {exc}')
            continue
        guard()
        wrist = np.asarray(samples)[:, 6] if samples else None
        j7_range = None if wrist is None else (float(wrist.min()), float(wrist.max()))
        return PickPlan(start_tcp, start_joints, grasp, pre, accepted[:2],
                        replace(route, commands=accepted[2:]), scale, j7_range)
    guard()
    raise PlanRejected('파지 경로 준비 실패: ' + '; '.join(failures))
