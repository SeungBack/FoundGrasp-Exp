"""파지-이적재 데모 루프 — run.py 가 호출한다.

  홈(스캔 자세) → 캡처 → 검출 → ROI 필터 → 선정(IK·J7·카메라 상향·하강 제한)
  → pre-grasp(그리퍼 개방 병행) → 저속 접근 → 파지
  → 성공: 상승 → 상방 경유(블렌딩) → 릴리즈 자세 → (페이로드 확인) → 놓기 → 반복
  → 실패: pre-grasp 복귀 → 홈 → 재스캔 (직전 실패 지점 반경 제외)
  ROI 내 후보 0개 → 대기, 사용자 입력(Enter 재개 / q 종료).
  제어된 종료(q·취소) 시 홈 복귀. 이동 실패 abort는 그 자리 정지.
"""
import threading
import time

import cv2
import numpy as np

from .camera import CameraStream, resolve_intrinsics
from .config import get, require
from .detectors import build_detector
from .floor import FloorMap, clamp_to_floor
from .motion import connect_robot, go_home, move_tcp
from .payload import load_payload_check
from .roi import draw_overlay, exclude_near, filter_roi, project_px
from .se3 import inv_se3
from .selection import select_grasp
from .transforms import load_hand_eye
from .viz import draw_grippers, gripper_segments_px

GRIPPER_OPEN_TIMEOUT_S = 15.0  # pre-open 병행 실행 대기 한도


def confirm(msg, auto):
    if auto:
        return True
    return input(f"{msg} [y/N] ").strip().lower() == "y"


def wait_user(stream, msg):
    """대기 후 사용자 입력. 반환: True=재개, False=종료."""
    stream.set_status("WAITING FOR USER (terminal)")
    ans = input(f"{msg} [Enter=재개 / q=종료] ").strip().lower()
    return ans != "q"


def _via_joints(robot, sel, d):
    """이송 경유점(상승 지점 위 via_height_m 높이)의 조인트 해.

    자세는 상승 자세 그대로 유지하고 base z만 올린다 — 박스 테두리보다 높은
    지점을 지나게 해 잡은 물체가 박스에 걸리지 않게 한다.
    반환: (조인트 리스트 | None, 경유 불필요 여부). None + False = IK 실패.
    """
    T = sel.T_tcp_lift.copy()
    z_via = float(d.get("via_height_m", 0.35))
    if T[2, 3] >= z_via:
        return None, True  # 이미 충분히 높음 — 경유 없이 릴리즈로 직행
    T[2, 3] = z_via
    j = robot.flange_ik_joints_deg(T @ inv_se3(robot.T_flange_tcp))
    return (None, False) if j is None else ([float(a) for a in j], False)


def _payload_verdict(robot, payload):
    """릴리즈 직전 페이로드 확인. 반환 (물체 있음 여부, 출력용 문구).

    판정 불가(자세 불일치·읽기 실패)는 통과 처리한다 — 실행 중인 데모를
    확인 수단의 문제로 중단시키지 않는다 (경고는 남긴다).
    """
    try:
        has, val = payload.measure(robot)
    except RuntimeError as e:
        print(f"  경고: 페이로드 확인 실패 ({e}) — 성공으로 간주")
        return True, None
    if has is None:
        print(f"  경고: 릴리즈 자세가 기준값 측정 자세와 다름 "
              f"(최대 {val:.2f}° > 허용) — 페이로드 확인 생략. "
              "릴리즈 자세를 다시 티칭했다면 scripts/calib_payload.py 재실행")
        return True, None
    return has, (f"Δ{val:+.3f} Nm (임계 {payload.min_delta_nm:.3f})")


def pick_and_place(robot, gripper, sel, cfg, auto, payload=None,
                   grip_check=True):
    """파지 → 릴리즈. 반환: ("success"|"fail"|"abort", 사유).

    fail: 파지 실패 (pre-grasp 복귀 완료 또는 릴리즈 자세, 재시도 가능).
    abort: 이동 실패/사용자 취소 — 충돌 가능성이 있으므로 그 자리에 정지.
    grip_check=False: 판정 결과를 출력만 하고 항상 이송을 진행한다.
    """
    r = cfg["robot"]
    d = cfg["drop"]

    # 그리퍼 개방과 pre-grasp 이동을 동시에 — 정지 상태에서 개방을 기다리지
    # 않고 이동 중에 끝나게 한다 (그리퍼는 Modbus, 로봇은 별도 연결로 독립).
    # enable_motion은 내부 sleep(0.5s)+컨트롤러 왕복이 있어 반드시 먼저 끝내둔다.
    robot.enable_motion()
    res = {}
    t0 = time.monotonic()

    def _open():
        res["ok"] = gripper.pre_open(sel.width_m)
        res["t"] = time.monotonic() - t0

    opener = threading.Thread(target=_open, daemon=True)
    opener.start()
    moved = move_tcp(robot, sel.T_tcp_pre, r["travel_speed_mm_s"])
    t_move = time.monotonic() - t0
    opener.join(timeout=GRIPPER_OPEN_TIMEOUT_S)
    print(f"  병행: 이동 {t_move:.1f}s / 개방 {res.get('t', float('nan')):.1f}s")
    if opener.is_alive() or not res.get("ok"):
        return "abort", "그리퍼 개방"
    if not moved:
        return "abort", "pre-grasp 이동"
    if not confirm("  그립 지점으로 접근할까요?", auto):
        return "abort", "사용자 취소"

    # 접근은 저속·저가속 — 물체 접촉 충격과 흔들림 최소화
    acc = r.get("approach_acc_mm_s2")
    if acc:
        robot.arm.set_tcp_maxacc(float(acc))
    try:
        ok_move = move_tcp(robot, sel.T_tcp_grasp, r["approach_speed_mm_s"])
    finally:
        if acc:  # 이후 구간(복귀·상승)은 기본 가속도로 복원
            robot.arm.set_tcp_maxacc(float(r.get("default_acc_mm_s2", 5000)))
    if not ok_move:
        return "abort", "접근 이동"

    ok, pos = gripper.grip()
    print(f"  파지: {'성공' if ok else '실패'} (finger {pos}%)"
          + ("" if grip_check else " — 판정 미적용, 이송 진행"))
    if not ok and grip_check:
        gripper.pre_open(sel.width_m)  # 완전 개방은 주변 물체와 충돌 위험
        if not move_tcp(robot, sel.T_tcp_pre, r["lift_speed_mm_s"]):
            return "abort", "파지 실패 후 복귀 이동"
        return "fail", "파지"

    if not move_tcp(robot, sel.T_tcp_lift, r["lift_speed_mm_s"]):
        return "abort", "상승 이동"  # 물체 파지 상태 — 자동 복귀가 2차 충돌 위험
    try:
        # pre-grasp 상방을 정지 없이 경유(MoveArcJoint 블렌딩)해 릴리즈로 직행.
        # 가속도 50 deg/s^2 = move_joints 기본과 동일 (페이로드 토크 스파이크로
        # 인한 충돌 감지(C31) 오작동 방지)
        v = float(d["joint_speed_deg_s"])
        via_joints, skip_via = _via_joints(robot, sel, d)
        code = 0 if skip_via else 1
        if via_joints is not None:
            code = robot.arm.set_servo_angle(
                angle=via_joints, speed=v, mvacc=50.0, is_radian=False,
                wait=False, radius=float(d.get("via_blend_radius", 50.0)))
        if code == 0:
            code = robot.arm.set_servo_angle(
                angle=[float(a) for a in d["joints_deg"]], speed=v,
                mvacc=50.0, is_radian=False, wait=True)
            if code != 0:
                print(f"  릴리즈 이동 오류 (code={code}, "
                      f"controller error={robot.arm.error_code})")
                return "abort", "릴리즈 이동"
        else:  # 경유점 IK 실패/블렌딩 거부 — 홈 경유로 폴백 (박스 회피 유지)
            print(f"  경유 블렌딩 불가 (code={code}) — 홈 경유로 폴백")
            robot.move_joints(r["home_joints_deg"], speed_deg_s=v)
            robot.move_joints(d["joints_deg"], speed_deg_s=v)
    except RuntimeError as e:  # move_joints는 실패 시 예외
        print(f"  릴리즈 이동 오류: {e}")
        return "abort", "릴리즈 이동"

    # 놓기 직전 페이로드 확인 (옵션) — 고정 자세의 관절 중력 토크를 빈 손 기준과
    # 비교. grip_check=False에서도 측정·출력은 유지 (임계값 실측 자료).
    if payload is not None:
        has_obj, detail = _payload_verdict(robot, payload)
        if detail:
            print(f"  페이로드: {detail} → "
                  f"{'물체 있음' if has_obj else '빈 손'}"
                  + ("" if grip_check else " (판정 미적용)"))
        if not has_obj and grip_check:
            gripper.open()  # 다음 파지를 위해 개방 (릴리즈 상방이라 충돌 없음)
            return "fail", "빈 손(페이로드)"
    gripper.open()
    return "success", None


def run(cfg, auto=False, dry_run=False):
    """데모 루프 실행. 반환: stats dict."""
    from pathlib import Path

    roi = require(cfg, "scan_roi_px", "scripts/set_roi.py 로 스캔 영역을 지정하세요")
    if not dry_run:
        require(cfg, "drop.joints_deg", "scripts/teach_drop.py 로 릴리즈 자세를 티칭하세요")
    out_cfg = cfg.get("output") or {}
    save_cycles = bool(out_cfg.get("save_cycles", False))
    outdir = Path(out_cfg.get("dir", "outputs"))
    if save_cycles:
        outdir.mkdir(parents=True, exist_ok=True)

    T_flange_cam = load_hand_eye(cfg["calib"]["hand_eye"])
    live = bool(get(cfg, "live.enabled", True))
    grip_check = bool(get(cfg, "grip_check.enabled", True))
    if not grip_check:
        print("파지 성공 판정 미적용 — 판정 결과를 출력만 하고 항상 이송을 "
              "진행합니다.\n  주의: 실패해도 재시도/실패 지점 제외가 동작하지 "
              "않아, 잡히지 않는 물체에서 같은 시도를 반복할 수 있습니다 "
              "(--auto 사용 시 특히).")

    # 빈 박스 하강 제한 (옵션) — dry-run은 실제 로봇 자세가 없어 미적용
    floor_map = None
    fl = cfg.get("floor_limit") or {}
    if fl.get("enabled", False) and not dry_run:
        floor_map = FloorMap.load(fl["map_path"])
        print(f"하강 제한 활성: 높이맵 {floor_map.z.shape} "
              f"(여유 {fl.get('clearance_m', 0.008) * 1000:.0f}mm)")

    # 페이로드 확인 (옵션) — 판정 미적용 모드에선 기준값 미측정이 실행을 막지 않음
    payload = None
    if not dry_run:
        payload = load_payload_check(cfg, required=grip_check)
        if payload is not None:
            print(f"페이로드 확인 활성: 임계 {payload.min_delta_nm:.3f} Nm "
                  f"({'방향 투영' if payload.ref_delta is not None else '크기'} 기준)"
                  + ("" if grip_check else " — 출력만"))

    robot = gripper = stream = None
    stats = {"cycles": 0, "success": 0, "fail": 0, "empty": 0}
    home_on_exit = False  # 제어된 종료(q·취소 등)에서만 True — 예외 시 모션 금지
    try:
        print("== 초기화: 카메라/검출기" + ("" if dry_run else "/로봇/그리퍼") + " ==")
        stream = CameraStream(cfg, live=live)
        stream.start()
        detector = build_detector(cfg)
        print(f"검출기: {detector.name}")
        if not dry_run:
            from .gripper import GripperCtl
            robot = connect_robot(cfg)
            gripper = GripperCtl(cfg).connect()

        failed_pts = []  # 직전 실패 그립 지점 (base) — 성공/대기 시 초기화
        fail_streak = 0
        unsafe_stop = False  # 이동 실패 abort — 자동 홈 복귀 금지(2차 충돌 위험)
        while not stream.quit_requested:
            stats["cycles"] += 1
            n = stats["cycles"]
            print(f"\n== 사이클 {n}: 스캔 자세(홈) + 캡처 ==")
            stream.set_status(f"CYCLE {n}: HOMING")
            if robot is not None:
                go_home(robot, cfg)
                T_base_flange, _ = robot.get_flange_pose()
            else:
                T_base_flange = np.eye(4)
            T_base_cam = T_base_flange @ T_flange_cam

            stream.set_status(f"CYCLE {n}: SCANNING")
            stream.set_marks([])
            time.sleep(0.3)  # 정지 후 temporal filter 안정화
            rgb, depth_m, K = stream.snapshot()
            K = resolve_intrinsics(K, cfg)

            # ROI 밖 depth 무효화(0) — 전체 프레임 추론 시 top-k 후보가 ROI 밖
            # (예: 릴리즈 박스의 물체 더미)에 몰려 ROI 내 후보가 0개가 되는 것을
            # 방지. 무효 depth는 센서에서 늘 생기는 정상 입력이라 인공 표면을
            # 만들지 않는다. 투영 필터는 경계 걸침 후보 제거용 안전망으로 유지.
            x1, y1, x2, y2 = [int(v) for v in roi]
            depth_roi = np.zeros_like(depth_m)
            depth_roi[y1:y2, x1:x2] = depth_m[y1:y2, x1:x2]

            gg = detector.detect(rgb, depth_roi, K)
            gg_roi = filter_roi(gg, K, roi)
            print(f"  후보 {len(gg)}개 → ROI 내 {len(gg_roi)}개")

            if len(gg_roi) == 0:
                print("  ROI 내 후보 없음 — 박스가 비었거나 검출 실패.")
                stream.set_status("ROI EMPTY - waiting user")
                failed_pts, fail_streak = [], 0
                if not wait_user(stream, "물체를 보충/확인 후 재개하세요."):
                    break
                continue

            # 사이클 기록(output.save_cycles): ROI + 후보 그리퍼(주황),
            # 선정 그립(빨강)은 아래에서 덧그려 갱신 저장
            vis = jpg = None
            if save_cycles:
                vis = np.ascontiguousarray(rgb[:, :, ::-1]).copy()
                draw_overlay(vis, roi)
                draw_grippers(vis, gg_roi, K, color=(0, 165, 255))
                jpg = outdir / f"cycle{n:03d}_grasps.jpg"
                cv2.imwrite(str(jpg), vis)

            pool = exclude_near(gg_roi, T_base_cam,
                                failed_pts, cfg["retry"]["exclude_radius_m"])
            # 선정 + (옵션) 하강 제한: 삽입 축소로 해결되면 채택, 하한까지
            # 줄여도 박스 여유가 부족하면 후보를 제외하고 재선정
            s = g_sel = None
            for _ in range(10):
                report = select_grasp(pool, T_base_cam, cfg, robot=robot)
                if report.selected is None:
                    break
                cand = report.selected
                if floor_map is None:
                    s, g_sel = cand, pool[cand.index]
                    break
                adj, d_back = clamp_to_floor(cand, pool[cand.index],
                                             floor_map, cfg)
                if adj is not None:
                    if d_back > 0:
                        z0 = cand.T_tcp_grasp[2, 3] * 1000
                        z1 = adj.T_tcp_grasp[2, 3] * 1000
                        print(f"  [floor_limit] 삽입 {d_back * 1000:.1f}mm 축소: "
                              f"TCP 목표 z {z0:.1f} → {z1:.1f}mm "
                              f"(표면 여유 {fl.get('clearance_m', 0.008) * 1000:.0f}mm 확보)")
                    s, g_sel = adj, pool[cand.index]
                    break
                print(f"  하강 제한: #{cand.index} 탈락 — 필요 후퇴 "
                      f"{d_back * 1000:.0f}mm가 삽입 여유 초과, 재선정")
                pool = exclude_near(pool, T_base_cam, [cand.p_base], 0.005)
            if s is None:
                print(f"  실행 가능한 그립 없음 (탈락 {len(report.rejected)}건)")
                print(report.summary())
                fail_streak += 1
                if fail_streak >= cfg["retry"]["max_fail_streak"]:
                    if not wait_user(stream, "연속 선정 실패 — 장면 정리 후 재개하세요."):
                        break
                    failed_pts, fail_streak = [], 0
                continue

            segs = gripper_segments_px(g_sel, K)
            uv = project_px([np.asarray(g_sel.translation)], K)[0]
            stream.set_marks([{
                "segments_px": segs, "color": (0, 0, 255),
                "label": f"#{s.index} {s.score:.2f}",
                "anchor": uv if np.isfinite(uv).all() else None,
            }])
            if vis is not None:  # 선정 그립(빨강)을 기록 이미지에도 갱신
                for a, b in segs:
                    cv2.line(vis, a, b, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(jpg), vis)
            stream.set_status(f"CYCLE {n}: GRASP score={s.score:.2f} "
                              f"w={s.width_m * 1000:.0f}mm")
            print(f"  선정: #{s.index} score {s.score:.3f}, "
                  f"width {s.width_m * 1000:.0f}mm, tilt {s.tilt_deg:.0f}°, "
                  f"지점 {np.round(s.p_base * 1000, 1).tolist()} mm")

            if dry_run:
                if not wait_user(stream, "[dry-run] 선정 완료."):
                    break
                continue

            outcome, reason = pick_and_place(robot, gripper, s, cfg, auto,
                                             payload=payload,
                                             grip_check=grip_check)
            if outcome == "abort":
                if reason in ("사용자 취소", "그리퍼 개방"):  # 로봇 정상 상태
                    print(f"[종료] {reason}")
                else:
                    unsafe_stop = True
                    print(f"[중단] {reason} — 로봇을 현재 위치에 정지합니다. "
                          "(비상: xArm Studio)")
                break
            if outcome == "success":
                stats["success"] += 1
                failed_pts, fail_streak = [], 0
                print(f"  이적재 성공 (누적 {stats['success']})")
            else:
                stats["fail"] += 1
                fail_streak += 1
                failed_pts.append(s.p_base)
                if reason == "빈 손(페이로드)":
                    # 그리퍼는 닫혔지만 손에 질량이 없음 — 애초에 못 잡았거나
                    # 이송 중 놓친 경우. 후자면 물체 위치가 바뀌었을 수 있다.
                    stats["empty"] += 1
                    print("  이송 후 빈 손 판정 — 파지 실패로 집계. "
                          "이송 중 낙하 여부를 확인하세요.")
                print(f"  파지 실패({reason}) — 홈 복귀 후 재스캔 "
                      f"(연속 {fail_streak}회)")
                if fail_streak >= cfg["retry"]["max_fail_streak"]:
                    if not wait_user(stream, "연속 파지 실패 — 장면 확인 후 재개하세요."):
                        break
                    failed_pts, fail_streak = [], 0
        home_on_exit = not unsafe_stop  # 루프의 정상 종료(break 포함)에서만 도달
    finally:
        if robot is not None and home_on_exit:
            try:
                print("\n종료 — 홈 위치로 복귀합니다.")
                go_home(robot, cfg)
            except Exception as e:
                print(f"홈 복귀 실패: {e} — 로봇 상태를 수동 확인하세요.")
        print(f"\n== 종료: 사이클 {stats['cycles']}, "
              f"성공 {stats['success']}, 실패 {stats['fail']}"
              + (f" (그중 이송 후 빈 손 {stats['empty']})"
                 if stats["empty"] else "") + " ==")
        if stream is not None:
            stream.stop()
        if gripper is not None:
            gripper.disconnect()
        if robot is not None:
            robot.disconnect()
    return stats
