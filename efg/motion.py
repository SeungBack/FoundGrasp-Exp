"""로봇 연결·기본 이동 헬퍼 (robot.XArm7 위의 얇은 계층)."""
import time

import numpy as np

from .robot import XArm7
from .console import detail
from .se3 import inv_se3, pose_delta, se3_to_xarm_pose


def _confirm_time(cfg, key):
    """Require multiple stopped samples even when shortening endpoint dwell."""
    value = float(cfg.get(key, .2))
    if not np.isfinite(value) or not .04 <= value <= 1.0:
        raise RuntimeError(f"{key}는 0.04~1초 범위여야 합니다")
    return value


def approach_blended(robot, T_pre, T_grasp, cfg, check=None, prepare=None, metrics=None):
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
        start, _ = robot.get_tcp_pose()
        lengths = [np.linalg.norm(T_pre[:3, 3] - start[:3, 3]),
                   np.linalg.norm(T_grasp[:3, 3] - T_pre[:3, 3])]
        radius = min(float(cfg.get("pregrasp_blend_radius_mm", 5)),
                     min(lengths) * 1000 * 0.25)
        if not np.isfinite(radius) or radius < 0:
            raise RuntimeError("잘못된 pre-grasp 블렌딩 반경")
        # A zero-length leg cannot be blended; negative radius means stop.
        radius = radius if radius > 0.01 else -1
        for index, (target, speed, blend) in enumerate((
            (T_pre, cfg["travel_speed_mm_s"], radius),
            (T_grasp, cfg["approach_speed_mm_s"], -1),
        )):
            if check:
                check()
            pose = se3_to_xarm_pose(target)
            code = robot.arm.set_position(
                *pose, speed=float(speed),
                mvacc=float(cfg.get("approach_acc_mm_s2", 500)),
                radius=blend, is_radian=True, wait=False)
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


def wait_tcp_stopped(robot, target, timeout_s, check=None, confirm_s=.2):
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
            raise RuntimeError(f"상승 정지 확인 실패 — 조인트 이송 미전송 "
                               f"(state={state}, error={errors}, pending={pending})")
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
    raise RuntimeError("상승 정지/목표 도착 확인 시간 초과 — 조인트 이송 미전송")


def lift_and_transfer_blended(robot, T_pre, T_lift, robot_cfg, drop_cfg, check=None,
                              on_stage=None, metrics=None, transfer_sign=1,
                              j7_limits=None):
    """Retreat to pre-grasp, rise while adopting home orientation, then transfer.

    Plan before enqueueing. Never retry via another route after partial enqueue.
    Cartesian retreat/rise can blend; joint transfer starts only after a verified stop.
    """
    if metrics is not None:
        metrics.clear()
    try:
        upper_confirm_s = _confirm_time(drop_cfg, "transfer_confirm_s")
        release_confirm_s = _confirm_time(drop_cfg, "arrival_confirm_s")
        if check:
            check()
        current, _ = robot.get_tcp_pose()
        target, radius = transfer_waypoint(
            T_pre, T_lift, drop_cfg, robot.R_home_tcp, closing_sign=transfer_sign)
        if T_pre[2, 3] <= current[2, 3]:
            raise RuntimeError("after-grasp 복귀 목표가 현재 위치보다 높지 않습니다")
        retreat_radius = float(robot_cfg.get("retreat_blend_radius_mm", 5))
        if not np.isfinite(retreat_radius) or retreat_radius < 0:
            raise RuntimeError("잘못된 복귀 블렌딩 반경")
        retreat_radius = min(retreat_radius, np.linalg.norm(T_pre[:3, 3] - current[:3, 3]) * 250,
                             (target[2, 3] - T_pre[2, 3]) * 250)
        retreat_radius = retreat_radius if retreat_radius > .01 else -1
        # Validate the raised waypoint before moving a held object.
        if not all(robot.check_flange_ik(T @ inv_se3(robot.T_flange_tcp))
                   for T in (T_pre, target)):
            raise RuntimeError("상승 경유점 IK 실패")
        if j7_limits is not None:
            for T in (T_pre, target):
                angles = robot.flange_ik_joints_deg(T @ inv_se3(robot.T_flange_tcp))
                if (angles is None or len(angles) != 7 or not np.isfinite(angles).all()
                        or not j7_limits[0] <= angles[6] <= j7_limits[1]):
                    raise RuntimeError("상승 경유점 J7 제한/IK 재검사 실패")
        joints = np.asarray(drop_cfg["joints_deg"], dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise RuntimeError("잘못된 릴리즈 관절 목표")
        joint_acc = float(drop_cfg.get("joint_acc_deg_s2", 50))
        if not np.isfinite(joint_acc) or joint_acc <= 0:
            raise RuntimeError("drop.joint_acc_deg_s2는 유한한 양수여야 합니다")
        rise_speed, angle = reorientation_speed(T_pre, target, robot_cfg)
        retreat_speed = min(float(robot_cfg["lift_speed_mm_s"]),
                            float(robot_cfg.get("retreat_speed_mm_s", 150)))
        joint_speed = float(drop_cfg["joint_speed_deg_s"])
        detail(f"복귀 {retreat_speed:.0f}mm/s → 자세 전환 {rise_speed:.1f}mm/s "
              f"(TCP 회전 {angle:.1f}°, 상승 대칭 {transfer_sign:+d}) "
              f"→ 정지 확인 → 이송 {joint_speed:.0f}°/s")
        rise_started = time.perf_counter()
        for waypoint, blend, speed, acc in (
            (T_pre, retreat_radius, retreat_speed, robot_cfg.get("lift_acc_mm_s2", 500)),
            (target, -1, rise_speed, robot_cfg.get("reorient_acc_mm_s2", 150)),
        ):
            if check:
                check()
            code = robot.arm.set_position(
                *se3_to_xarm_pose(waypoint), speed=speed, mvacc=float(acc),
                radius=blend, is_radian=True, wait=False)
            if code:
                raise RuntimeError(f"상승 명령 거부 (code={code})")
        upper_confirmed = wait_tcp_stopped(
            robot, target, float(drop_cfg.get("transfer_timeout_s", 60)), check,
            confirm_s=upper_confirm_s)
        if metrics is not None:
            metrics.update(upper_confirm_s=upper_confirmed,
                           retreat_rise_s=time.perf_counter() - rise_started)
        detail("상승 도착·정지 확인 완료 — 조인트 이송 시작 (혼합 블렌딩 없음)")
        if check:
            check()
        if on_stage:
            on_stage("Joint transfer")
        code = robot.arm.set_servo_angle(
            angle=joints.tolist(), speed=float(drop_cfg["joint_speed_deg_s"]),
            mvacc=joint_acc, radius=-1, is_radian=False, wait=False)
        if code:
            raise RuntimeError(f"이송 명령 거부 (code={code})")
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
                try:
                    tcp_code, tcp_pose = robot.arm.get_position(is_radian=False)
                    print(f"  이송 중단 위치: TCP(code={tcp_code})={tcp_pose}, "
                          f"joints(code={pose_code})={actual}, pending={pending}; "
                          f"pre={np.round(T_pre[:3, 3] * 1000, 1).tolist()}, "
                          f"upper={np.round(target[:3, 3] * 1000, 1).tolist()}")
                except Exception as diagnostic_error:
                    print(f"  중단 위치 조회 실패: {diagnostic_error}")
                raise RuntimeError(f"이송 중 로봇 오류/정지 (state={state}, error={errors})")
            arrived = np.max(np.abs(np.asarray(actual) - joints)) <= .5
            if state == 2 and pending == 0 and arrived:
                if settled_since is None:
                    settled_since = time.monotonic()
                confirmed_s = time.monotonic() - settled_since
                if confirmed_s >= release_confirm_s:
                    if metrics is not None:
                        metrics["arrival_confirm_s"] = confirmed_s
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


def connect_robot(cfg):
    robot = XArm7(cfg["robot"]["ip"])
    payload = cfg.get("robot", {}).get("tcp_payload") or {}
    if payload.get("weight_kg"):
        robot.set_payload(payload["weight_kg"],
                          payload.get("com_mm", (0.0, 0.0, 0.0)))
    return robot


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
