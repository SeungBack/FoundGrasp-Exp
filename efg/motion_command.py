"""Immutable native xArm commands shared by preflight and execution.

Controller preflight screens self-collision, limits and overspeed. It does not
certify external obstacle clearance or the complete blended swept trajectory.
"""
from dataclasses import dataclass, replace
import math
from numbers import Integral


class PlanRejected(RuntimeError):
    """A path is unavailable while the measured robot remains healthy/stopped.

    Callers may select another candidate. This must never wrap transport errors,
    physical controller faults, user stops or camera failures.
    """


class MotionCheckError(RuntimeError):
    """Native preflight rejection; no actual movement was requested."""

    def __init__(self, message, *, command_index=None, code=None, result=None):
        super().__init__(message)
        self.command_index = command_index
        self.code = code
        self.result = result
        self.speed_scale = 1.

    @property
    def is_overspeed(self):
        # SDK 1.18 maps a nonzero native check reply to HAS_ERROR (1).
        # Transport/API errors and actual robot faults are not retry signals.
        return self.command_index is not None and self.code == 1 and self.result == 24

    @property
    def is_path_rejection(self):
        """Known native planning results; caller must still verify actual state."""
        return (self.command_index is not None and self.code == 1
                and self.result in (21, 22, 23, 24))

    def __str__(self):
        message = super().__str__()
        if self.speed_scale < 1:
            message += (f'; 검사 기준 명령 대비 속도 {self.speed_scale:.0%}, '
                        f'가속도 {self.speed_scale ** 2:.0%}에서도 기각')
        return message


@dataclass(frozen=True)
class MotionCommand:
    """Cartesian target: mm/radians; joint target: degrees.

    Cartesian speed/acceleration use mm/s and mm/s²; joint commands use deg/s
    and deg/s². The native blending radius is expressed in mm for both kinds.
    """
    kind: str
    target: tuple[float, ...]
    speed: float
    acceleration: float
    radius: float
    label: str

    def __post_init__(self):
        if self.kind not in ('cartesian', 'joint'):
            raise ValueError(f'알 수 없는 이동 종류: {self.kind}')
        target = tuple(float(value) for value in self.target)
        expected = 6 if self.kind == 'cartesian' else 7
        if len(target) != expected or not all(math.isfinite(value) for value in target):
            raise ValueError(f'{self.kind} 목표는 유한한 {expected}개 값이어야 합니다')
        object.__setattr__(self, 'target', target)
        for name in ('speed', 'acceleration', 'radius'):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f'{name}는 유한한 값이어야 합니다')
            if name == 'radius':
                if value < 0 and value != -1:
                    raise ValueError('블렌딩 반경은 -1 또는 0 이상이어야 합니다')
            elif value <= 0:
                raise ValueError(f'{name}는 양수여야 합니다')
            object.__setattr__(self, name, value)
        if not isinstance(self.label, str):
            raise ValueError('이동 설명은 문자열이어야 합니다')

    def submit(self, arm, *, only_check_type=0):
        """Submit exactly this command; check types 1/2/3 never request motion."""
        if only_check_type not in (0, 1, 2, 3):
            raise ValueError('only_check_type은 0, 1, 2, 3 중 하나여야 합니다')
        # Do not inherit the SDK's shared timing cache from a previous command.
        kwargs = dict(speed=self.speed, mvacc=self.acceleration, mvtime=0, radius=self.radius,
                      wait=False, only_check_type=only_check_type)
        if self.kind == 'cartesian':
            return arm.set_position(*self.target, is_radian=True, **kwargs)
        return arm.set_servo_angle(angle=list(self.target), is_radian=False, **kwargs)


def approach_commands(start, pre, grasp, cfg):
    """Compile the same rounded approach for whole-cycle checks and execution."""
    import numpy as np
    from .se3 import se3_to_xarm_pose

    radius = float(cfg.get('pregrasp_blend_radius_mm', 5))
    if not math.isfinite(radius) or radius < 0:
        raise ValueError('잘못된 pre-grasp 블렌딩 반경')
    lengths = (np.linalg.norm(pre[:3, 3] - start[:3, 3]),
               np.linalg.norm(grasp[:3, 3] - pre[:3, 3]))
    radius = min(radius, min(lengths) * 250)
    radius = radius if radius > .01 else -1
    acceleration = float(cfg.get('approach_acc_mm_s2', 500))
    travel_acceleration = float(cfg.get('travel_acc_mm_s2', acceleration))
    return (
        MotionCommand('cartesian', tuple(se3_to_xarm_pose(pre)),
                      float(cfg['travel_speed_mm_s']), travel_acceleration, radius, 'pre-grasp'),
        MotionCommand('cartesian', tuple(se3_to_xarm_pose(grasp)),
                      float(cfg['approach_speed_mm_s']), acceleration, -1, '접근'),
    )


def preflight_commands(arm, commands, check=None):
    """Check a native command chain from the actual stationary robot state.

    The caller owns the robot and must verify it remains stationary. Each chain
    starts with type 1 to overwrite any previous controller shadow state. Middle
    commands use type 3; the last uses type 2, restoring shadow state to actual.
    A single command uses type 1. After rejection/interruption no cleanup motion
    is sent: a later check must start with type 1 again. Per-call options avoid
    changing the SDK-global check mode, including on exceptions.
    """
    version = getattr(arm, 'version_number', None)
    if (not isinstance(version, (tuple, list)) or len(version) != 3
            or not all(isinstance(value, Integral) and not isinstance(value, bool)
                       and value >= 0 for value in version)
            or tuple(version) < (1, 11, 100)):
        raise MotionCheckError(f'컨트롤러 경로 사전 검사는 FW 1.11.100 이상이 필요합니다 (현재 {version})')
    commands = tuple(commands)
    if not commands or not all(isinstance(command, MotionCommand) for command in commands):
        raise ValueError('사전 검사에는 하나 이상의 MotionCommand가 필요합니다')
    for index, command in enumerate(commands):
        if check:
            check()
        check_type = 1 if index == 0 else (2 if index == len(commands) - 1 else 3)
        code = command.submit(arm, only_check_type=check_type)
        # Each native call resets this value, so capture it before any next call.
        result = getattr(arm, 'only_check_result', None)
        if code != 0 or result != 0:
            raise MotionCheckError(f'컨트롤러 경로 사전 검사 실패 ({command.label}, '
                                   f'code={code}, only_check_result={result})',
                                   command_index=index, code=code, result=result)


def preflight_with_retiming(arm, commands, *, check_start, check=None):
    """Bounded command-limit scaling for native overspeed rejections only.

    Positive outgoing radii connect commands into a phase; zero/negative radii
    end a phase at an exact stop. Retiming scales only the rejected phase, using
    its original limits (v*s, a*s²), with geometry and blend radii unchanged.
    Every reduction still checks the ENTIRE chain: the native shadow trajectory
    must begin at the actual state, not at an unexecuted intermediate waypoint.
    The returned scale is the minimum phase scale, not a common chain scale.

    Each phase permits three reductions, bounding attempts at 1 + 3 * phases.
    The controller still owns jerk/trajectory generation; this is not a proof
    of exact time-scaled or collision-free motion.
    check_start must reject changed, moving or faulted actual robot states.
    Every attempt restarts the native shadow chain at the actual stopped state.
    """
    original = tuple(commands)
    if not original or not all(isinstance(command, MotionCommand) for command in original):
        raise ValueError('사전 검사에는 하나 이상의 MotionCommand가 필요합니다')
    phase_for_command = [0]
    for previous in original[:-1]:
        phase_for_command.append(phase_for_command[-1] + (previous.radius <= 0))
    levels = (1., .8, .6, .5)
    phase_levels = [0] * (phase_for_command[-1] + 1)
    while True:
        if check:
            check()
        check_start()
        phase_scales = [levels[level] for level in phase_levels]
        candidate = tuple(
            command if phase_scales[phase] == 1. else replace(
                command, speed=command.speed * phase_scales[phase],
                acceleration=command.acceleration * phase_scales[phase] ** 2)
            for command, phase in zip(original, phase_for_command))
        try:
            preflight_commands(arm, candidate, check=check)
        except MotionCheckError as exc:
            index = exc.command_index
            valid_index = (isinstance(index, Integral) and not isinstance(index, bool)
                           and 0 <= index < len(original))
            phase = phase_for_command[index] if valid_index else None
            exc.speed_scale = phase_scales[phase] if valid_index else min(phase_scales)
            if (not exc.is_overspeed or not valid_index
                    or phase_levels[phase] == len(levels) - 1):
                raise
            phase_levels[phase] += 1
        else:
            if check:
                check()
            check_start()
            return candidate, min(phase_scales)
