"""로봇 연결·기본 이동 헬퍼 (robot.XArm7 위의 얇은 계층)."""
import time

import numpy as np

from .robot import XArm7
from .console import detail, row
from .motion_command import MotionCheckError, approach_commands, preflight_with_retiming
from .se3 import inv_se3, pose_delta, se3_to_xarm_pose, xarm_pose_to_se3


def _confirm_time(cfg, key):
    """Require multiple stopped samples even when shortening endpoint dwell."""
    value = float(cfg.get(key, .2))
    if not np.isfinite(value) or not .04 <= value <= 1.0:
        raise RuntimeError(f"{key}는 0.04~1초 범위여야 합니다")
    return value


def approach_blended(robot, T_pre, T_grasp, cfg, check=None, prepare=None, metrics=None,
                     *, commands=None, j7_limits=None):
    """Queue a rounded pre-grasp and exact-stop grasp; monitor until arrival.

    Camera exceptions propagate after stopping the controller (including its
    queued approach). Controller failures return False, without clearing errors.
    """
    started = time.perf_counter()
    width_s = 0.
    if metrics is not None:
        metrics.clear()
    try:
        confirm_s = _confirm_time(cfg, "approach_confirm_s")
        if check:
            check()
        if commands is None:
            start, _ = robot.get_tcp_pose()
            commands = approach_commands(start, T_pre, T_grasp, cfg)
        commands = tuple(commands)
        if (len(commands) != 2 or any(command.kind != 'cartesian' for command in commands)
                or commands[-1].radius != -1):
            raise RuntimeError("접근은 두 Cartesian 명령과 최종 정확 정지가 필요합니다")
        for command, target in zip(commands, (T_pre, T_grasp)):
            distance, angle = pose_delta(xarm_pose_to_se3(command.target), target)
            if distance > 1e-8 or angle > 1e-5:
                raise RuntimeError("준비된 접근 명령과 선정 목표가 다릅니다")
        for index, command in enumerate(commands):
            if check:
                check()
            code = command.submit(robot.arm, only_check_type=0)
            if code != 0:
                raise RuntimeError(f"접근 명령 거부 (code={code})")
            if index == 0 and prepare is not None:
                # Controller travels to pre-grasp while the host adjusts width.
                # Never enqueue the final approach before width verification.
                width_started = time.perf_counter()
                if not prepare():
                    raise RuntimeError("pre-grasp 이동 중 그리퍼 폭 조절 실패")
                width_s = time.perf_counter() - width_started

        queued_at = time.perf_counter()
        deadline = time.monotonic() + float(cfg.get("approach_timeout_s", 60))
        settled_since = None
        while time.monotonic() < deadline:
            if check:
                check()
            code, state = robot.arm.get_state()
            err_code, errors = robot.arm.get_err_warn_code()
            queue_code, pending = robot.arm.get_cmdnum()
            if code or err_code or queue_code or errors[0] or state >= 3:
                raise RuntimeError(f"접근 중 로봇 오류/정지 (state={state}, error={errors})")
            if j7_limits is not None:
                joint_code, raw_joints = robot.arm.get_servo_angle(is_radian=False)
                try:
                    joints = np.asarray(raw_joints, dtype=float)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("접근 중 실제 관절 판독 오류") from exc
                if joint_code or joints.shape != (7,) or not np.isfinite(joints).all():
                    raise RuntimeError("접근 중 실제 관절 판독 오류")
                if not j7_limits[0] <= joints[6] <= j7_limits[1]:
                    raise RuntimeError(f"접근 중 실측 J7 {joints[6]:.2f}° 제한 초과")
            actual, _ = robot.get_tcp_pose()
            pos, rot = pose_delta(actual, T_grasp)
            # State 2 is stopped. State 0 can precede motion startup, so it
            # must never be interpreted as completion of queued commands.
            if state == 2 and pending == 0 and pos <= 0.002 and rot <= 1.0:
                if settled_since is None:
                    settled_since = time.monotonic()
                confirmed_s = time.monotonic() - settled_since
                if confirmed_s >= confirm_s:
                    if metrics is not None:
                        metrics.update(width_s=width_s,
                                       confirm_s=confirmed_s,
                                       before_queue_s=queued_at - started,
                                       after_queue_s=time.perf_counter() - queued_at)
                    return True
            else:
                settled_since = None
            time.sleep(0.02)
        raise RuntimeError("접근 완료 대기 시간 초과/목표 미도달")
    except RuntimeError as exc:
        robot.stop_motion()
        # CameraError is a RuntimeError subclass: preserve camera handling.
        from .camera import CameraError
        if isinstance(exc, CameraError):
            raise
        print(f"[robot] {exc}")
        return False
    except BaseException:
        robot.stop_motion()
        raise


def _rise_diagnostics(robot, target):
    """Best-effort read-only snapshot before our stop request obscures state."""
    for label, read in (
        ("실측 TCP mm/deg", lambda: robot.arm.get_position(is_radian=False)),
        ("실측 관절 deg", lambda: robot.arm.get_servo_angle(is_radian=False)),
    ):
        try:
            print(f"  상승 중단 {label}: {read()}")
        except Exception as exc:
            print(f"  상승 중단 {label} 조회 실패: {exc}")
    pose = np.asarray(se3_to_xarm_pose(target))
    pose[3:] = np.degrees(pose[3:])
    print(f"  상승 목표 TCP mm/deg: {np.round(pose, 3).tolist()}")


def log_controller_fault(robot, errors):
    """Best-effort diagnostics for a latched fault, before stopping/resetting.

    Check-only rejections must not call this: the native detail may describe an
    older physical fault. Keep native servo IDs and unspecified tau units as-is.
    """
    if not errors or errors[0] not in (24, 31):
        return
    fault = errors[0]
    if fault == 31:
        print("  C31: 로봇 관절 전류 기반 충돌 감지 · 보호 정지")
    try:
        if fault == 24:
            code, info = robot.arm.get_c24_error_info(is_radian=False)
        else:
            code, info = robot.arm.get_c31_error_info()
        size = 2 if fault == 24 else 3
        if (code == 0 and len(info) >= size
                and np.isfinite(np.asarray(info[:size], dtype=float)).all()):
            # Preserve the native ID; do not guess whether it is zero-based.
            if fault == 24:
                print(f"  C24 상세: controller servo_id={info[0]}, "
                      f"속도={float(info[1]):.3f}°/s")
            else:
                print(f"  C31 상세: controller servo_id={info[0]}, "
                      f"theoretical_tau={float(info[1]):.3f}, "
                      f"actual_tau={float(info[2]):.3f} (SDK 원시값)")
        else:
            print(f"  C{fault} 상세 조회 실패: code={code}, info={info}")
    except Exception as exc:
        print(f"  C{fault} 상세 조회 실패: {exc}")


def _stopped_transfer_joints(robot, j7_limits, *, label="조인트 이송"):
    """Require an idle, healthy measured start for the separate joint move."""
    state_code, state = robot.arm.get_state()
    queue_code, pending = robot.arm.get_cmdnum()
    error_code, errors = robot.arm.get_err_warn_code()
    if (state_code or queue_code or error_code or state != 2 or pending != 0
            or errors[0] or robot.arm.mode != 0):
        log_controller_fault(robot, errors)
        raise RuntimeError(f"{label} 시작 상태 변경 — 이송 미전송 "
                           f"(mode={robot.arm.mode}, state={state}, error={errors}, pending={pending})")
    joints = np.asarray(robot.get_joints_deg(), dtype=float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise RuntimeError(f"{label} 시작 관절 판독 오류 — 이송 미전송")
    if j7_limits is not None and not j7_limits[0] <= joints[6] <= j7_limits[1]:
        raise RuntimeError(f"{label} 시작 J7 {joints[6]:.2f}° 제한 초과 — 이송 미전송")
    return joints


def _prepare_joint_transfer(robot, command, j7_limits, check=None, metrics=None):
    """Finalize the unsent joint phase at its measured stationary start.

    The input carries the original configured ceilings for this unsent phase.
    Bounded native overspeed retiming may reduce it to 50% speed / 25% acceleration.
    The target and exact-stop radius stay fixed; no joint motion is sent here.
    """
    started = time.perf_counter()
    if metrics is not None:
        metrics['failure_stage'] = '조인트 이송 사전 검사'
    try:
        before = _stopped_transfer_joints(robot, j7_limits).copy()

        def check_start():
            actual = _stopped_transfer_joints(robot, j7_limits)
            if np.max(np.abs(actual - before)) > .5:
                raise RuntimeError("조인트 이송 사전 검사 중 시작 자세 변경 — 이송 미전송")

        try:
            checked, scale = preflight_with_retiming(
                robot.arm, (command,), check_start=check_start, check=check)
        except MotionCheckError as exc:
            raise RuntimeError(
                f"조인트 이송 사전 검사 실패: {exc}; "
                f"시작 관절={np.round(before, 2).tolist()}, "
                f"목표 관절={np.round(command.target, 2).tolist()}, "
                f"검사 기준 속도={command.speed:g}°/s, "
                f"가속도={command.acceleration:g}°/s²") from exc
        accepted = checked[0]
        if metrics is not None:
            metrics.pop('failure_stage', None)
            metrics.update(joint_speed_scale=scale,
                           joint_speed_deg_s=accepted.speed,
                           joint_acc_deg_s2=accepted.acceleration)
        if scale < 1:
            row('이송 계획', f"{accepted.speed:g}°/s · {accepted.acceleration:g}°/s² · "
                f"상승 후 실측 자세에서 재검사 통과 (설정 속도의 {scale:.0%})")
        return accepted, check_start
    finally:
        if metrics is not None:
            metrics['joint_preflight_s'] = time.perf_counter() - started


def wait_tcp_stopped(robot, target, timeout_s, check=None, confirm_s=.2, j7_limits=None):
    """Barrier between Cartesian and joint motion: no later command is queued."""
    confirm_s = _confirm_time({"confirm_s": confirm_s}, "confirm_s")
    deadline = time.monotonic() + timeout_s
    settled_since = None
    while time.monotonic() < deadline:
        if check:
            check()
        code, state = robot.arm.get_state()
        err_code, errors = robot.arm.get_err_warn_code()
        queue_code, pending = robot.arm.get_cmdnum()
        if code or err_code or queue_code or errors[0] or state >= 3:
            log_controller_fault(robot, errors)
            _rise_diagnostics(robot, target)
            raise RuntimeError(f"상승 정지 확인 실패 — 조인트 이송 미전송 "
                               f"(state={state}, error={errors[0]}, warn={errors[1]}, pending={pending})")
        joint_code, raw_joints = robot.arm.get_servo_angle(is_radian=False)
        try:
            joints = np.asarray(raw_joints, dtype=float)
        except (TypeError, ValueError) as exc:
            _rise_diagnostics(robot, target)
            raise RuntimeError("상승 중 실제 관절 판독 오류 — 조인트 이송 미전송") from exc
        if joint_code or joints.shape != (7,) or not np.isfinite(joints).all():
            _rise_diagnostics(robot, target)
            raise RuntimeError(f"상승 중 실제 관절 판독 오류 — 조인트 이송 미전송 "
                               f"(code={joint_code}, joints={raw_joints})")
        if j7_limits is not None and not j7_limits[0] <= joints[6] <= j7_limits[1]:
            _rise_diagnostics(robot, target)
            raise RuntimeError(f"상승 중 실측 J7 {joints[6]:.2f}° 제한 "
                               f"[{j7_limits[0]:g}, {j7_limits[1]:g}]° 초과 — 조인트 이송 미전송")
        actual, _ = robot.get_tcp_pose()
        distance, angle = pose_delta(actual, target)
        if state == 2 and pending == 0 and distance <= .002 and angle <= 1:
            if settled_since is None:
                settled_since = time.monotonic()
            confirmed_s = time.monotonic() - settled_since
            if confirmed_s >= confirm_s:
                return confirmed_s
        else:
            settled_since = None
        time.sleep(.02)
    _rise_diagnostics(robot, target)
    raise RuntimeError("상승 정지/목표 도착 확인 시간 초과 — 조인트 이송 미전송")


def lift_and_transfer_blended(robot, plan, robot_cfg, drop_cfg, check=None,
                              on_stage=None, metrics=None):
    """Execute exactly the prepared commands, with optional upper blending.

    Continuous plans queue retreat/rise/transfer together and stop only at the
    release goal. Stopped plans retain the upper arrival barrier, then finalize
    the unsent joint phase from the measured upper configuration. Only that
    phase is checked afresh against its original ceilings before its first command.
    Neither mode replans or retries a phase after its physical motion starts.
    """
    if metrics is not None:
        metrics.clear()
    try:
        release_confirm_s = _confirm_time(drop_cfg, "arrival_confirm_s")
        upper_confirm_s = (None if plan.continuous else
                           _confirm_time(drop_cfg, "transfer_confirm_s"))
        if check:
            check()
        plan.check_start(robot)
        if (len(plan.commands) < 3 or plan.commands[-1].kind != 'joint'
                or any(command.kind != 'cartesian' for command in plan.commands[:-1])):
            raise RuntimeError("이송 명령 목록이 준비되지 않았습니다")
        transfer_index = len(plan.commands) - 1
        T_pre, target = plan.pre, plan.upper
        joints = np.asarray(plan.commands[-1].target, dtype=float)
        if plan.continuous:
            if on_stage:
                on_stage("Retreat / rise / transfer")
            detail(f"복귀 → 상승 통과 (반경 {plan.commands[-2].radius:g}mm) → "
                   f"이송 {plan.commands[-1].speed:g}°/s · 릴리즈에서 정지")
        rise_started = time.perf_counter()
        for index, command in enumerate(plan.commands):
            joint_guard = None
            if index == transfer_index and not plan.continuous:
                upper_confirmed = wait_tcp_stopped(
                    robot, target, float(drop_cfg.get("transfer_timeout_s", 60)), check,
                    confirm_s=upper_confirm_s, j7_limits=plan.j7_limits)
                if metrics is not None:
                    metrics.update(upper_confirm_s=upper_confirmed,
                                   retreat_rise_s=time.perf_counter() - rise_started)
                if on_stage:
                    on_stage("Preparing joint transfer")
                # The Cartesian motion may finish in a different redundant-arm
                # configuration than the earlier shadow plan. Replan ONLY this
                # unsent phase from the measured stop and original ceilings.
                # Continuous plans never reset limits or replan in flight.
                command, joint_guard = _prepare_joint_transfer(
                    robot, plan.nominal_commands[-1], plan.j7_limits,
                    check=check, metrics=metrics)
            if index == transfer_index and not plan.continuous and on_stage:
                on_stage("Joint transfer")
            if check:
                check()
            if joint_guard is not None:
                joint_guard()
            code = command.submit(robot.arm, only_check_type=0)
            if code:
                raise RuntimeError(f"{command.label} 명령 거부 (code={code})")
        deadline = time.monotonic() + float(drop_cfg.get("transfer_timeout_s", 60))
        settled_since = None
        while time.monotonic() < deadline:
            if check:
                check()
            code, state = robot.arm.get_state()
            err_code, errors = robot.arm.get_err_warn_code()
            queue_code, pending = robot.arm.get_cmdnum()
            pose_code, actual = robot.arm.get_servo_angle(is_radian=False)
            if code or err_code or queue_code or pose_code or errors[0] or state >= 3:
                # Read diagnostics before our STOP request; preserve the original failure.
                log_controller_fault(robot, errors)
                try:
                    tcp_code, tcp_pose = robot.arm.get_position(is_radian=False)
                    print(f"  이송 중단 위치: TCP(code={tcp_code})={tcp_pose}, "
                          f"joints(code={pose_code})={actual}, pending={pending}; "
                          f"pre={np.round(T_pre[:3, 3] * 1000, 1).tolist()}, "
                          f"upper={np.round(target[:3, 3] * 1000, 1).tolist()}")
                except Exception as diagnostic_error:
                    print(f"  중단 위치 조회 실패: {diagnostic_error}")
                raise RuntimeError(f"이송 중 로봇 오류/정지 (state={state}, error={errors})")
            actual = np.asarray(actual, dtype=float)
            if actual.shape != (7,) or not np.isfinite(actual).all():
                raise RuntimeError("이송 중 실제 관절 판독 오류")
            if plan.j7_limits is not None and not plan.j7_limits[0] <= actual[6] <= plan.j7_limits[1]:
                raise RuntimeError(f"이송 중 실측 J7 {actual[6]:.2f}° 제한 초과")
            arrived = np.max(np.abs(actual - joints)) <= .5
            if state == 2 and pending == 0 and arrived:
                if settled_since is None:
                    settled_since = time.monotonic()
                confirmed_s = time.monotonic() - settled_since
                if confirmed_s >= release_confirm_s:
                    if metrics is not None:
                        metrics["arrival_confirm_s"] = confirmed_s
                        if plan.continuous:
                            metrics["continuous_transfer_s"] = time.perf_counter() - rise_started
                    return True
            else:
                settled_since = None
            time.sleep(.02)
        raise RuntimeError("이송 완료 대기 시간 초과/목표 미도달")
    except RuntimeError as exc:
        robot.stop_motion()
        from .camera import CameraError
        if isinstance(exc, CameraError):
            raise
        print(f"[robot] {exc}")
        return False
    except BaseException:
        robot.stop_motion()
        raise


def recover_unclosed_pick(robot, pick_plan, robot_cfg, check=None,
                          j7_limits=None, metrics=None):
    """Return an unclosed grasp along its nominal approach, after native checks.

    The caller may use this only for a clean planning rejection before closing
    the gripper. A robot/camera fault, changed start or rejected return route
    sends no recovery motion. The inverse Cartesian path is checked anew; it is
    not assumed to reproduce the forward motion's redundant joint branch.
    """
    started = time.perf_counter()
    if metrics is not None:
        metrics.clear()
    try:
        if check:
            check()
        initial_joints = _stopped_transfer_joints(
            robot, j7_limits, label="재선정 복귀").copy()
        current, _ = robot.get_tcp_pose()
        current = np.asarray(current, dtype=float).copy()
        if current.shape != (4, 4) or not np.isfinite(current).all():
            raise RuntimeError("재선정 복귀 시작 TCP 판독 오류")
        distance, angle = pose_delta(current, pick_plan.grasp)
        if distance > .002 or angle > 1.:
            raise RuntimeError("파지 위치가 변경되어 역방향 복귀를 시작할 수 없습니다")
        if pick_plan.pre[2, 3] <= current[2, 3]:
            raise RuntimeError("재선정 복귀 pre-grasp가 현재 위치보다 높지 않습니다")

        def check_start():
            joints = _stopped_transfer_joints(robot, j7_limits, label="재선정 복귀")
            pose, _ = robot.get_tcp_pose()
            if np.shape(pose) != (4, 4) or not np.isfinite(pose).all():
                raise RuntimeError("재선정 복귀 시작 TCP 판독 오류")
            position_error, angle_error = pose_delta(current, pose)
            if (position_error > .002 or angle_error > 1.
                    or np.max(np.abs(joints - initial_joints)) > .5):
                raise RuntimeError("재선정 복귀 검사 중 시작 자세 변경 — 복귀 미전송")

        # Respect both the configured retreat limits and the accepted forward
        # segment limits, including any scaling chosen by whole-pick preflight.
        forward_pre, forward_grasp = pick_plan.approach_commands
        speed_cap = min(float(robot_cfg['travel_speed_mm_s']),
                        float(robot_cfg.get('retreat_speed_mm_s', 150.)))
        acceleration = min(float(robot_cfg.get('approach_acc_mm_s2', 500.)),
                           float(robot_cfg.get('lift_acc_mm_s2', 500.)),
                           forward_pre.acceleration, forward_grasp.acceleration)
        reverse_cfg = dict(robot_cfg,
                           travel_speed_mm_s=min(speed_cap, forward_grasp.speed),
                           approach_speed_mm_s=min(speed_cap, forward_pre.speed),
                           travel_acc_mm_s2=acceleration,
                           approach_acc_mm_s2=acceleration,
                           pregrasp_blend_radius_mm=max(0., forward_pre.radius))
        commands = approach_commands(current, pick_plan.pre, pick_plan.start_tcp, reverse_cfg)
        commands, scale = preflight_with_retiming(
            robot.arm, commands, check_start=check_start, check=check)
        check_start()
        checked_at = time.perf_counter()
        ok = approach_blended(
            robot, pick_plan.pre, pick_plan.start_tcp, reverse_cfg,
            check=check, commands=commands, j7_limits=j7_limits, metrics=metrics)
        if metrics is not None:
            metrics.update(recovery_preflight_s=checked_at - started,
                           recovery_speed_scale=scale)
        return ok
    except RuntimeError as exc:
        from .camera import CameraError
        if isinstance(exc, CameraError):
            raise
        print(f"[robot] 재선정 복귀 준비 실패: {exc}")
        return False


def connect_robot(cfg):
    robot = XArm7(cfg["robot"]["ip"])
    try:
        offset = cfg["robot"].get("tcp_offset_mm_deg")
        if offset is not None:
            robot.configure_tcp_offset(offset)
        payload = cfg.get("robot", {}).get("tcp_payload") or {}
        if payload.get("weight_kg"):
            robot.set_payload(payload["weight_kg"],
                              payload.get("com_mm", (0.0, 0.0, 0.0)))
        return robot
    except BaseException:
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"[robot] 초기화 실패 후 연결 종료 오류: {exc}")
        raise


def go_home(robot, cfg):
    r = cfg["robot"]
    acc = float(r.get("home_joint_acc_deg_s2", 50))
    if not np.isfinite(acc) or acc <= 0:
        raise ValueError("robot.home_joint_acc_deg_s2는 유한한 양수여야 합니다")
    robot.enable_motion()
    robot.move_joints(r["home_joints_deg"],
                      speed_deg_s=r["home_joint_speed_deg_s"], acc_deg_s2=acc)


def move_tcp(robot, T_base_tcp, speed_mm_s, arrive_tol_mm=10.0, acc_mm_s2=None):
    """툴(TCP) 목표 SE(3)로 직선 이동 + 도착 검증.

    이동 호출이 성공을 반환해도 실제 플랜지 자세를 읽어 목표와 비교한다
    (컨트롤러 상태 이상으로 이동 없이 반환되는 경우 차단).
    """
    T_flange = T_base_tcp @ inv_se3(robot.T_flange_tcp)
    kwargs = {"acc_mm_s2": acc_mm_s2} if acc_mm_s2 is not None else {}
    if not robot.move_flange(T_flange, speed_mm_s=speed_mm_s, **kwargs):
        return False
    T_actual, _ = robot.get_flange_pose()
    err_mm = float(np.linalg.norm(
        (T_actual[:3, 3] - T_flange[:3, 3]) * 1000.0))
    if err_mm > arrive_tol_mm:
        print(f"경고: 목표 미도달 (오차 {err_mm:.1f}mm > {arrive_tol_mm}mm) — "
              "이동 실패로 처리")
        return False
    return True
