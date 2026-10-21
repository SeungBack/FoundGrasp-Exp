"""파지-이적재 데모 루프 — run.py 가 호출한다.

  홈(스캔 자세) → 캡처 → 검출(detector.name) → 공유 후처리(폭 한계·충돌필터·정렬)
  → ROI 필터 → 선정(순위 순, IK·J7·카메라 상향·하강 제한)
  → pre-grasp 이동 중 그리퍼 폭 조절 → 완료 확인 → 접근 → 파지
  → 성공: 상승 → 상방 경유(블렌딩) → 릴리즈 자세 → (페이로드 확인) → 놓기 → 반복
  → 실패: pre-grasp 복귀 → 홈 → 재스캔 (직전 실패 지점 반경 제외)
  ROI 내 후보 0개 → 대기, 사용자 입력(Enter 재개 / q 종료).
  제어된 종료(Ctrl+C 1회·q·취소) 시 홈 복귀. 이동 실패 abort는 그 자리 정지.

FoundGrasp는 독립 detect() 경로를 사용한다. 입력 ROI·선정·모션은 공유하되
추가 공유 충돌 검사와 폭 마진은 비교군에만 적용한다 (efg/postprocess.py).
"""
import select
import signal
import sys
import threading
import time

import cv2
import numpy as np

from .camera import CameraError, resolve_intrinsics
from .camera_process import ProcessCameraStream as CameraStream
from .config import get, require
from .detectors import build_detector
from .floor import FloorMap, clamp_to_floor
from .motion import connect_robot, go_home, move_tcp, approach_blended, lift_and_transfer_blended, transfer_waypoint
from .payload import load_payload_check
from .postprocess import mask_depth_to_roi, shared_filter
from .roi import draw_overlay, exclude_near, filter_roi, project_px
from .se3 import inv_se3
from .selection import select_grasp
from .transforms import load_hand_eye
from .viz import draw_score_grippers, gripper_segments_px
from .viz_rerun import RerunViz

GRIPPER_OPEN_TIMEOUT_S = 15.0  # pre-open 병행 실행 대기 한도


def _read_input(prompt, stream=None):
    """터미널 입력 대기 중에도 카메라 실패를 확인한다 (Linux 실행 환경)."""
    if stream is None:
        return input(prompt)
    print(prompt, end="", flush=True)
    while True:
        stream.raise_if_failed()
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            answer = sys.stdin.readline()
            stream.raise_if_failed()
            if not answer:
                raise EOFError("터미널 입력 종료")
            return answer.rstrip("\n")


def confirm(msg, auto, stream=None):
    if stream is not None:
        stream.raise_if_failed()
    if auto:
        return True
    return _read_input(f"{msg} [y/N] ", stream).strip().lower() == "y"


def wait_user(stream, msg):
    """대기 후 사용자 입력. 반환: True=재개, False=종료."""
    stream.set_status("WAITING FOR USER (terminal)")
    ans = _read_input(f"{msg} [Enter=재개 / q=종료] ", stream).strip().lower()
    return ans != "q"


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
                   grip_check=True, stream=None, viz=None, stop=None):
    """파지 → 릴리즈. 반환: ("success"|"fail"|"abort", 사유).

    fail: 파지 실패 (pre-grasp 복귀 완료 또는 릴리즈 자세, 재시도 가능).
    abort: 이동 실패/사용자 취소 — 충돌 가능성이 있으므로 그 자리에 정지.
    grip_check=False: 판정 결과를 출력만 하고 항상 이송을 진행한다.
    """
    def phase(name):
        if viz is not None:
            viz.log_stage(name)

    r = cfg["robot"]
    d = cfg["drop"]
    if stream is not None:
        stream.raise_if_failed()

    # Also guard direct calls / disabled per-candidate IK before moving or closing.
    T_transfer, _ = transfer_waypoint(sel.T_tcp_pre, sel.T_tcp_lift, d, robot.R_home_tcp)
    if not robot.check_flange_ik(T_transfer @ inv_se3(robot.T_flange_tcp)):
        print(f"  상승 경유점 사전 IK 실패: TCP {np.round(T_transfer[:3, 3] * 1000, 1).tolist()} mm")
        return "abort", "상승 경유점 사전 IK"

    # 기본 모드에서는 pre-grasp 이동 중 폭 조절, 완료 확인 후 최종 접근한다.
    # enable_motion은 내부 sleep(0.5s)+컨트롤러 왕복이 있어 반드시 먼저 끝내둔다.
    robot.enable_motion()
    if stream is not None:
        stream.raise_if_failed()
    if auto:
        def check_approach():
            if stream is not None:
                stream.raise_if_failed()
            if stop is not None and stop.requested:
                raise RuntimeError("사용자 정지 요청")

        phase("Pre-grasp / adjusting width / approach")
        if not approach_blended(robot, sel.T_tcp_pre, sel.T_tcp_grasp,
                                r, check=check_approach,
                                prepare=lambda: gripper.pre_open(sel.open_width_m, check=check_approach)):
            return "abort", "연속 접근 이동"
    else:
        res = {}
        t0 = time.monotonic()

        def _open():
            res["ok"] = gripper.pre_open(sel.open_width_m)
            res["t"] = time.monotonic() - t0

        phase("Pre-grasp / opening")
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
        phase("Waiting for approach confirmation")
        if not confirm("  그립 지점으로 접근할까요?", auto, stream=stream):
            return "abort", "사용자 취소"
        if stream is not None:
            stream.raise_if_failed()

        phase("Approaching")
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

    if auto:
        check_approach()
    phase("Closing gripper")
    ok, pos = gripper.grip()
    print(f"  파지: {'성공' if ok else '실패'} (finger {pos}%)"
          + ("" if grip_check else " — 판정 미적용, 이송 진행"))
    if not ok and grip_check:
        phase("Reopening / retracting")
        gripper.pre_open(sel.open_width_m)  # 완전 개방은 주변 물체와 충돌 위험
        if not move_tcp(robot, sel.T_tcp_pre, r["lift_speed_mm_s"]):
            return "abort", "파지 실패 후 복귀 이동"
        return "fail", "파지"

    phase("Retreat / rise / stop / transfer")

    def check_transfer():
        if stream is not None:
            stream.raise_if_failed()
        if stop is not None and stop.requested:
            raise RuntimeError("사용자 정지 요청")

    if not lift_and_transfer_blended(robot, sel.T_tcp_pre, sel.T_tcp_lift, r, d,
                                     check=check_transfer):
        return "abort", "상승/이송 이동"

    # 놓기 직전 페이로드 확인 (옵션) — 고정 자세의 관절 중력 토크를 빈 손 기준과
    # 비교. grip_check=False에서도 측정·출력은 유지 (임계값 실측 자료).
    phase("At release pose")
    if payload is not None:
        has_obj, detail = _payload_verdict(robot, payload)
        if detail:
            print(f"  페이로드: {detail} → "
                  f"{'물체 있음' if has_obj else '빈 손'}"
                  + ("" if grip_check else " (판정 미적용)"))
        if not has_obj and grip_check:
            gripper.open()  # 다음 파지를 위해 개방 (릴리즈 상방이라 충돌 없음)
            return "fail", "빈 손(페이로드)"
    phase("Releasing")
    if not gripper.open():
        return "abort", "릴리즈 개방 실패"
    return "success", None


def method_label(cfg):
    """기록용 방법 이름 (foundgrasp는 refiner 여부 구분)."""
    name = cfg["detector"].get("name", "foundgrasp")
    if name == "foundgrasp" and not get(cfg, "detector.foundgrasp.refiner.enabled", True):
        return "foundgrasp_detonly"
    return name


class StopRequest:
    """Ctrl+C 1회 = 현재 사이클 후 정지(홈 복귀), 2회 = 즉시 KeyboardInterrupt."""

    def __init__(self):
        self.requested = False
        self._prev = signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        if self.requested:
            signal.signal(signal.SIGINT, self._prev)
            raise KeyboardInterrupt
        self.requested = True
        print("\n[정지 요청] 현재 사이클을 마치고 홈 복귀 후 종료합니다 "
              "(즉시 중단: Ctrl+C 한 번 더)")

    def restore(self):
        signal.signal(signal.SIGINT, self._prev)


def _home_flange_pose(cfg):
    """dry-run용 스캔 자세 플랜지 pose — 홈 관절각의 xArm7 모델 FK (시각화·선정 근사)."""
    from .robot_model import XArm7Model

    q = np.radians(np.asarray(cfg["robot"]["home_joints_deg"], dtype=np.float64))
    return XArm7Model().flange_pose(q), q


def run(cfg, auto=True, dry_run=False):
    """데모 루프 실행. 반환: stats dict."""
    from pathlib import Path

    roi = require(cfg, "scan_roi_px", "scripts/set_roi.py 로 스캔 영역을 지정하세요")
    if not dry_run:
        require(cfg, "drop.joints_deg", "scripts/teach_drop.py 로 릴리즈 자세를 티칭하세요")
    out_cfg = cfg.get("output") or {}
    save_cycles = bool(out_cfg.get("save_cycles", False))
    method = method_label(cfg)
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{method}" + ("_dryrun" if dry_run else "")
    outdir = Path(out_cfg.get("dir", "outputs")) / run_name
    if save_cycles:
        outdir.mkdir(parents=True, exist_ok=True)

    T_flange_cam = load_hand_eye(cfg["calib"]["hand_eye"])
    live = bool(get(cfg, "live.enabled", False))
    mask_roi = bool(get(cfg, "postprocess.roi_depth_mask", True))
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

    robot = gripper = stream = detector = viz = None
    stop = StopRequest()
    stats = {"cycles": 0, "success": 0, "fail": 0, "empty": 0}
    home_on_exit = False  # 제어된 종료(q·취소 등)에서만 True — 예외 시 모션 금지
    try:
        viz = RerunViz(cfg, run_name, out_dir=Path(out_cfg.get("dir", "outputs")))
        print("== 초기화: 카메라/검출기" + ("" if dry_run else "/로봇/그리퍼") + " ==")
        stream = CameraStream(cfg, live=live)
        stream.start()
        detector = build_detector(cfg)
        print(f"검출기: {detector.describe()}")
        viz.log_text(f"방법: {detector.describe()}  설정: {cfg['_config_path']}")
        warmup_runs = int(get(cfg, "detector.warmup_runs", 1))
        warm_rgb, warm_depth, warm_K = stream.snapshot()
        if warmup_runs > 0:
            print(f"검출기 워밍업 {warmup_runs}회 (로봇 연결 전, 결과 미사용)")
            warm_K = resolve_intrinsics(warm_K, cfg)
            if mask_roi:
                warm_depth = mask_depth_to_roi(warm_depth, roi)
            for _ in range(warmup_runs):
                detector.detect(warm_rgb, warm_depth, warm_K)
                stream.raise_if_failed()
        del warm_rgb, warm_depth, warm_K
        stream.raise_if_failed()
        if not dry_run:
            from .gripper import GripperCtl
            robot = connect_robot(cfg)
            gripper = GripperCtl(cfg).connect()
            viz.start_robot_stream(robot, gripper=gripper)

        inference_retry = 0
        max_inference_retries = int(cfg["retry"].get("max_inference_retries", 3))
        if max_inference_retries < 0:
            raise ValueError("retry.max_inference_retries는 0 이상이어야 합니다")
        failed_pts = []  # 직전 실패 그립 지점 (base) — 성공/대기 시 초기화
        fail_streak = 0
        unsafe_stop = False  # 이동 실패 abort — 자동 홈 복귀 금지(2차 충돌 위험)
        while not (stream.quit_requested or stop.requested):
            stream.raise_if_failed()
            if inference_retry == 0:
                stats["cycles"] += 1
                n = stats["cycles"]
                viz.set_cycle(n)
                viz.log_stage("Homing" if not dry_run else "Home pose preview")
                scan_label = "홈 자세 가정 (로봇 이동 없음)" if dry_run else "스캔 자세(홈)"
                print(f"\n== 사이클 {n}: {scan_label} + 캡처 ==")
                stream.set_status(f"CYCLE {n}: HOMING")
                if robot is not None:
                    go_home(robot, cfg)
                    T_base_flange, _ = robot.get_flange_pose()
                    robot.R_home_tcp = (T_base_flange @ robot.T_flange_tcp)[:3, :3].copy()
                else:
                    T_base_flange, q_home = _home_flange_pose(cfg)
                    viz.log_robot(q_home)
                T_base_cam = T_base_flange @ T_flange_cam

            viz.log_stage("Scanning")
            stream.set_status(f"CYCLE {n}: SCANNING")
            stream.set_marks([])
            if inference_retry == 0:
                time.sleep(0.3)  # 정지 후 temporal filter 안정화
            rgb, depth_m, K = (stream.snapshot(fresh=True) if inference_retry
                               else stream.snapshot())
            K = resolve_intrinsics(K, cfg)

            # 입력: ROI 밖 depth 무효화(0) — 전 방법 동일. 전체 프레임 추론 시 후보가
            # ROI 밖(릴리즈 박스 더미)에 몰려 ROI 내 후보가 0개가 되는 것을 방지.
            # 무효 depth는 센서에서 늘 생기는 정상 입력이라 인공 표면을 만들지 않는다.
            depth_in = mask_depth_to_roi(depth_m, roi) if mask_roi else depth_m
            viz.log_stage("Detecting / selecting")
            t0 = time.perf_counter()
            gg = detector.detect(rgb, depth_in, K)
            detect_ms = (time.perf_counter() - t0) * 1000.0
            # 실행 후처리: 폭 한계 → 비교군만 공유 충돌필터 → 정렬
            post_t0 = time.perf_counter()
            gg, counts = shared_filter(gg, depth_m, K, cfg)
            gg_roi = filter_roi(gg, K, roi)
            post_ms = (time.perf_counter() - post_t0) * 1000.0
            counts["roi"] = len(gg_roi)
            extra = getattr(detector, "last_refiner_ms", None)
            print(f"  검출 {detect_ms:.0f}ms" + (f" (refiner {extra:.0f}ms)" if extra else "")
                  + f" + 후처리 {post_ms:.0f}ms = {detect_ms + post_ms:.0f}ms"
                  + f" | 후보 {counts['detected']} → 폭 {counts['width_ok']} → "
                  f"충돌 {counts['collision_free']} → ROI {counts['roi']}")
            viz.log_scan(rgb, depth_m, K, T_base_cam, roi)
            viz.log_candidates(gg_roi, K, counts, detect_ms)
            viz.log_text(f"사이클 {n}: 검출 {detect_ms:.0f}ms + 후처리 {post_ms:.0f}ms, 후보 {counts}")

            if len(gg_roi) == 0:
                if inference_retry < max_inference_retries:
                    inference_retry += 1
                    print(f"  ROI 후보 없음 — 새 프레임 재추론 {inference_retry}/{max_inference_retries}")
                    continue
                print(f"  {max_inference_retries + 1}회 추론에서 실행 후보를 확보하지 못했습니다.")
                inference_retry = 0
                stream.set_status("NO CANDIDATE - waiting user")
                if not wait_user(stream, "재추론 한도 도달 — 장면 확인 후 재개하세요."):
                    break
                continue

            # 사이클 기록: Rerun과 같은 점수 색상/표시 후보, 선정은 청록색.
            vis = jpg = None
            if save_cycles:
                vis = np.ascontiguousarray(rgb[:, :, ::-1]).copy()
                draw_overlay(vis, roi)
                draw_score_grippers(vis, gg_roi, K, max_n=int(get(cfg, "rerun.max_candidates", 25)))
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
                viz.log_text(f"실행 가능한 그립 없음 (탈락 {len(report.rejected)}건)", level="WARN")
                if inference_retry < max_inference_retries:
                    inference_retry += 1
                    print(f"  선정 실패 — 새 프레임 재추론 {inference_retry}/{max_inference_retries}")
                    continue
                inference_retry = 0
                if not wait_user(stream, "재추론 한도 도달 — 실행 가능한 자세가 없습니다."):
                    break
                continue

            inference_retry = 0
            stream.raise_if_failed()
            g_one = pool[s.index:s.index + 1]
            segs = gripper_segments_px(g_sel, K)
            uv = project_px([np.asarray(g_sel.translation)], K)[0]
            stream.set_marks([{
                "segments_px": segs, "color": (255, 215, 65),
                "label": f"SELECTED {s.score:.2f}",
                "anchor": uv if np.isfinite(uv).all() else None,
            }])
            if vis is not None:  # 선정 그립을 기록 이미지에도 갱신
                for a, b in segs:
                    cv2.line(vis, a, b, (20, 24, 28), 6, cv2.LINE_AA)
                    cv2.line(vis, a, b, (255, 215, 65), 3, cv2.LINE_AA)
                cv2.imwrite(str(jpg), vis)
            stream.set_status(f"CYCLE {n}: GRASP score={s.score:.2f} "
                              f"w={s.width_m * 1000:.0f}mm")
            msg = (f"선정: 순위 {s.index + 1}/{len(pool)}, score {s.score:.6g}, "
                   f"폭 {s.width_m * 1000:.0f}mm → 개방 {s.open_width_m * 1000:.0f}mm, "
                   f"tilt {s.tilt_deg:.0f}°, 지점 {np.round(s.p_base * 1000, 1).tolist()} mm")
            print("  " + msg)
            viz.log_selected(g_one, s, K)
            viz.log_text(msg)

            if dry_run:
                viz.log_stage("Preview / waiting")
                if not wait_user(stream, "[dry-run] 선정 완료."):
                    break
                continue

            outcome, reason = pick_and_place(robot, gripper, s, cfg, auto,
                                             payload=payload,
                                             grip_check=grip_check, stream=stream, viz=viz, stop=stop)
            viz.log_outcome(g_one, s, outcome, reason)
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
        stream.raise_if_failed()
        home_on_exit = not unsafe_stop  # 루프의 정상 종료(break 포함)에서만 도달
    except CameraError as exc:
        print(f"\n[중단] {exc} — 추가 접근과 자동 홈 복귀를 수행하지 않습니다.")
        if viz is not None:
            viz.log_text(f"카메라 오류로 중단: {exc}", level="ERROR")
    finally:
        stop.restore()
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
        if viz is not None:
            viz.log_stage("Stopped")
            viz.log_text(f"실행 종료: 사이클 {stats['cycles']}")
            viz.close()
        if detector is not None:
            detector.close()
        if stream is not None:
            try:
                stream.stop()
            except CameraError as exc:
                print(f"카메라 종료 실패: {exc}")
        if gripper is not None:
            gripper.disconnect()
        if robot is not None:
            robot.disconnect()
    return stats
