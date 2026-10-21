"""파지-이적재 데모 루프 — run.py 가 호출한다.

  홈(스캔 자세) → 캡처 → 검출(detector.name) → 공유 후처리(폭 한계·충돌필터·정렬)
  → ROI 필터 → 선정(순위 순, IK·J7·카메라 상향·하강 제한)
  → pre-grasp 이동 중 그리퍼 폭 조절 → 완료 확인 → 접근 → 파지
  → 이송: 복귀·상승 → 상방 경유점 정지 확인 → 릴리즈 자세 → 놓기 → 반복
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
from . import console
from .detectors import build_detector
from .floor import FloorMap, clamp_to_floor
from .motion import connect_robot, go_home, move_tcp, approach_blended, lift_and_transfer_blended, transfer_waypoint
from .payload import load_payload_check
from .postprocess import mask_depth_to_roi, shared_filter, uses_standalone_foundgrasp
from .roi import draw_overlay, exclude_near, filter_roi, project_px
from .se3 import inv_se3, pose_delta
from .selection import select_grasp
from .transforms import load_hand_eye
from .timing import CycleTiming, TaktStats
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
                   grip_check=True, stream=None, viz=None, stop=None, timing=None):
    """파지 → 릴리즈. 반환: ("success"|"fail"|"abort", 사유).

    fail: 파지 실패 (pre-grasp 복귀 완료 또는 릴리즈 자세, 재시도 가능).
    abort: 이동 실패/사용자 취소 — 충돌 가능성이 있으므로 그 자리에 정지.
    grip_check=False: 판정으로 분기하지 않고 이송한다. 핑거 위치는 상세 출력에서 확인.
    """
    def phase(name):
        if timing is not None:
            labels = {
                "Pre-grasp / adjusting width / approach": "접근(폭 조절 병행)",
                "Closing gripper": "파지 닫힘",
                "Retreat / rise / stop / transfer": "복귀·상승",
                "Joint transfer": "조인트 이송",
                "At release pose": "릴리즈 준비",
                "Releasing": "개방",
                "Pre-grasp / opening": "pre-grasp(폭 조절 병행)",
                "Waiting for approach confirmation": "사용자 확인",
                "Approaching": "접근",
                "Reopening / retracting": "실패 후 복귀",
            }
            timing.mark(labels.get(name, name))
        if viz is not None:
            viz.log_stage(name)

    r = cfg["robot"]
    d = cfg["drop"]
    if stream is not None:
        stream.raise_if_failed()

    # Also guard direct calls / disabled per-candidate IK before moving or closing.
    transfer_sign = getattr(sel, "transfer_sign", 1)
    T_transfer, _ = transfer_waypoint(
        sel.T_tcp_pre, sel.T_tcp_lift, d, robot.R_home_tcp, closing_sign=transfer_sign)
    if not robot.check_flange_ik(T_transfer @ inv_se3(robot.T_flange_tcp)):
        print(f"  상승 경유점 사전 IK 실패: TCP {np.round(T_transfer[:3, 3] * 1000, 1).tolist()} mm")
        return "abort", "상승 경유점 사전 IK"
    if cfg.get("selection", {}).get("ik_check_per_candidate", False):
        lo, hi = cfg["selection"].get("j7_limit_deg", [-89., 90.])
        # Floor clamping may have shifted the selected endpoints. Validate the
        # chosen symmetry again before approaching or closing the gripper.
        for T in (sel.T_tcp_pre, sel.T_tcp_grasp, T_transfer):
            joints = robot.flange_ik_joints_deg(T @ inv_se3(robot.T_flange_tcp))
            if (joints is None or len(joints) != 7 or not np.isfinite(joints).all()
                    or not lo <= joints[6] <= hi):
                return "abort", "선정 경로 사전 IK/J7 재검사"
    if timing is not None:
        angle = pose_delta(sel.T_tcp_pre, T_transfer)[1]
        timing.notes.append(f"상승 방향 {'홈' if transfer_sign == 1 else '홈의 180° 대칭'} · "
                            f"TCP 자세 전환 {angle:.1f}°")

    # 기본 모드에서는 pre-grasp 이동 중 폭 조절, 완료 확인 후 최종 접근한다.
    # 최초 활성화 후에는 준비 상태만 확인하며 재활성화 대기를 반복하지 않는다.
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
        approach_metrics = {}
        if not approach_blended(robot, sel.T_tcp_pre, sel.T_tcp_grasp,
                                r, check=check_approach,
                                prepare=lambda: gripper.pre_open(sel.open_width_m, check=check_approach),
                                metrics=approach_metrics):
            return "abort", "연속 접근 이동"
        if timing is not None and approach_metrics:
            timing.notes.append(
                f"접근 분해: 최종 명령 전 {approach_metrics['before_queue_s']:.2f}s · "
                f"명령 후 도착 확인 {approach_metrics['after_queue_s']:.2f}s")
            timing.notes.append(
                f"폭 조절 {approach_metrics['width_s']:.2f}s는 이동과 병행 (위 시간에 포함)")
            timing.notes.append(
                f"파지점 도착 후 정지 확인 {approach_metrics['confirm_s']:.2f}s (접근에 포함)")
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
        console.detail(f"병행: 이동 {t_move:.2f}s / 개방 {res.get('t', float('nan')):.2f}s")
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
        # 확인 모드에서도 설정한 접근 가속도를 적용한다.
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
    console.detail(f"닫힘: finger {pos}% · 판정 {'적용' if grip_check else '미적용'}")
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

    transfer_metrics = {}
    if not lift_and_transfer_blended(robot, sel.T_tcp_pre, sel.T_tcp_lift, r, d,
                                     check=check_transfer, on_stage=phase,
                                     metrics=transfer_metrics, transfer_sign=transfer_sign,
                                     j7_limits=cfg.get("selection", {}).get("j7_limit_deg", [-89., 90.])):
        return "abort", "상승/이송 이동"
    if timing is not None and transfer_metrics:
        timing.notes.append(
            f"복귀·상승 명령~완료 {transfer_metrics['retreat_rise_s']:.2f}s "
            f"(정지 확인 {transfer_metrics['upper_confirm_s']:.2f}s 포함) · "
            f"릴리즈 도착 확인 {transfer_metrics['arrival_confirm_s']:.2f}s (이송에 포함)")

    # 놓기 직전 페이로드 확인 (옵션) — 고정 자세의 관절 중력 토크를 빈 손 기준과
    # 비교. grip_check=False에서도 측정·출력은 유지 (임계값 실측 자료).
    phase("At release pose")
    outcome = ("success", None)
    if payload is not None:
        has_obj, detail = _payload_verdict(robot, payload)
        if detail:
            print(f"  페이로드: {detail} → "
                  f"{'물체 있음' if has_obj else '빈 손'}"
                  + ("" if grip_check else " (판정 미적용)"))
        if not has_obj and grip_check:
            outcome = ("fail", "빈 손(페이로드)")
    phase("Releasing")
    if not gripper.release(sel.open_width_m, check=check_transfer):
        return "abort", "릴리즈 개방 실패"
    return outcome


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
    with console.output_mode(bool(get(cfg, "output.verbose", False))):
        return _run(cfg, auto=auto, dry_run=dry_run)


def _run(cfg, auto=True, dry_run=False):
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
    console.section("FoundGrasp · 실행 준비")
    console.row("설정", cfg["_config_path"])
    console.row("모드", "DRY-RUN · 로봇 이동 없음" if dry_run else (
        "자동 실행" if auto else "접근 전 사용자 확인"))
    console.row("측정", "홈 이동 시작 → 릴리즈 조건 충족 · 초기화/워밍업 제외")
    if not grip_check:
        console.row("판정", "파지 판정 미적용 · 닫힘 완료 후 이송 · 파지 실패 기반 재시도 없음")

    # 빈 박스 하강 제한 (옵션) — dry-run은 실제 로봇 자세가 없어 미적용
    floor_map = None
    fl = cfg.get("floor_limit") or {}
    if fl.get("enabled", False) and not dry_run:
        floor_map = FloorMap.load(fl["map_path"])
        console.row("하강 제한", f"높이맵 {floor_map.z.shape} · "
                    f"여유 {fl.get('clearance_m', 0.008) * 1000:.0f} mm")

    # 페이로드 확인 (옵션) — 판정 미적용 모드에선 기준값 미측정이 실행을 막지 않음
    payload = None
    if not dry_run:
        payload = load_payload_check(cfg, required=grip_check)
        if payload is not None:
            console.row("페이로드", f"임계 {payload.min_delta_nm:.3f} Nm "
                  f"({'방향 투영' if payload.ref_delta is not None else '크기'} 기준)"
                  + ("" if grip_check else " — 출력만"))

    robot = gripper = stream = detector = viz = None
    stop = StopRequest()
    stats = {"cycles": 0, "success": 0, "fail": 0, "empty": 0}
    takt = TaktStats()
    cycle_timing = None

    def finish_cycle(outcome, reason=None):
        if cycle_timing is None or cycle_timing.finished:
            return
        seconds = cycle_timing.finish()
        completed = outcome == "success" and not dry_run
        if completed:
            takt.add(seconds)
        label = {"success": "이송 완료", "fail": "파지 판정에 따른 재시도",
                 "abort": "중단", "no_candidate": "후보 없음", "preview": "미리보기 완료"}[outcome]
        console.row("결과", label + (f" · {reason}" if reason else ""))
        summary = cycle_timing.report(completed=completed, stats=takt)
        if viz is not None:
            viz.log_text(f"사이클 {stats['cycles']}: {label} · {summary}")

    home_on_exit = False  # 제어된 종료(q·취소 등)에서만 True — 예외 시 모션 금지
    try:
        viz = RerunViz(cfg, run_name, out_dir=Path(out_cfg.get("dir", "outputs")))
        console.row("연결", "카메라 / 검출기" + ("" if dry_run else " / 로봇 / 그리퍼"))
        stream = CameraStream(cfg, live=live)
        stream.start()
        console.row("모델", "로딩 중…")
        with console.model_output(outdir / "model_init.log"):
            detector = build_detector(cfg)
        console.row("검출기", detector.describe())
        console.row("모델 로그", str(outdir / "model_init.log"))
        viz.log_text(f"방법: {detector.describe()}  설정: {cfg['_config_path']}")
        warmup_runs = int(get(cfg, "detector.warmup_runs", 1))
        warm_rgb, warm_depth, warm_K = stream.snapshot()
        if warmup_runs > 0:
            console.row("워밍업", f"{warmup_runs}회 · 결과 미사용")
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
                scan_label = "DRY-RUN · 홈 자세 가정" if dry_run else "홈 → 파지 → 릴리즈"
                console.section(f"CYCLE {n:03d}  |  {scan_label}")
                cycle_timing = CycleTiming()
                viz.set_cycle(n)
                viz.log_stage("Homing" if not dry_run else "Home pose preview")
                stream.set_status(f"CYCLE {n}: HOMING")
                if robot is not None:
                    go_home(robot, cfg)
                    T_base_flange, _ = robot.get_flange_pose()
                    robot.R_home_tcp = (T_base_flange @ robot.T_flange_tcp)[:3, :3].copy()
                else:
                    T_base_flange, q_home = _home_flange_pose(cfg)
                    viz.log_robot(q_home)
                T_base_cam = T_base_flange @ T_flange_cam

            cycle_timing.mark("캡처·안정화")
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
            cycle_timing.mark("추론·선정·시각화")
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
            console.row("검출", f"{detect_ms + post_ms:.0f} ms  |  추론 {detect_ms:.0f} ms"
                        + (f" (refiner {extra:.0f} ms 포함)" if extra is not None else "")
                        + f" · 후처리 {post_ms:.0f} ms")
            candidates = (f"{counts['detected']} → score {counts.get('score_ok', counts['detected'])}"
                          f" → 폭 {counts['width_ok']}")
            if not uses_standalone_foundgrasp(cfg) and get(cfg, "postprocess.collision.enabled", True):
                candidates += f" → 충돌 {counts['collision_free']}"
            candidates += f" → ROI {counts['roi']}"
            width_limit = get(cfg, "postprocess.max_open_m")
            if width_limit is not None:
                candidates += f"  |  폭 한계 {width_limit * 1000:g} mm"
            console.row("후보", candidates)
            viz.log_scan(rgb, depth_m, K, T_base_cam, roi)
            viz.log_candidates(gg_roi, K, counts, detect_ms)
            viz.log_text(f"사이클 {n}: 검출 {detect_ms:.0f}ms + 후처리 {post_ms:.0f}ms, 후보 {counts}")

            if len(gg_roi) == 0:
                if inference_retry < max_inference_retries:
                    inference_retry += 1
                    console.row("재추론", f"{inference_retry}/{max_inference_retries} · ROI 후보 없음 · 새 프레임")
                    continue
                finish_cycle("no_candidate", f"추론 {max_inference_retries + 1}회 소진")
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
                console.row("선정", f"실행 가능한 그립 없음 · 탈락 {len(report.rejected)}건")
                console.detail(report.summary())
                viz.log_text(f"실행 가능한 그립 없음 (탈락 {len(report.rejected)}건)", level="WARN")
                if inference_retry < max_inference_retries:
                    inference_retry += 1
                    console.row("재추론", f"{inference_retry}/{max_inference_retries} · 선정 실패 · 새 프레임")
                    continue
                inference_retry = 0
                finish_cycle("no_candidate", "재추론 한도 도달 · 실행 가능한 자세 없음")
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
            console.row("선정", f"#{s.index + 1}/{len(pool)} · score {s.score:.6g} · "
                        f"폭 {s.width_m * 1000:.0f} → 개방 {s.open_width_m * 1000:.0f} mm · "
                        f"기울기 {s.tilt_deg:.0f}°")
            console.row("목표", "  ".join(f"{axis} {value * 1000:.1f}" for axis, value in
                                       zip("XYZ", s.p_base)) + " mm")
            viz.log_selected(g_one, s, K)
            viz.log_text(msg)

            if dry_run:
                finish_cycle("preview")
                viz.log_stage("Preview / waiting")
                if not wait_user(stream, "[dry-run] 선정 완료."):
                    break
                continue

            cycle_timing.mark("모션 준비")
            console.row("실행", "접근·폭 조절 → 닫힘 → 복귀·상승 → 이송 → 개방")
            outcome, reason = pick_and_place(robot, gripper, s, cfg, auto,
                                             payload=payload,
                                             grip_check=grip_check, stream=stream, viz=viz,
                                             stop=stop, timing=cycle_timing)
            finish_cycle(outcome, reason)
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
                console.row("재시도", f"홈 복귀 후 재스캔 · 연속 {fail_streak}회")
                if fail_streak >= cfg["retry"]["max_fail_streak"]:
                    if not wait_user(stream, "연속 파지 실패 — 장면 확인 후 재개하세요."):
                        break
                    failed_pts, fail_streak = [], 0
        stream.raise_if_failed()
        home_on_exit = not unsafe_stop  # 루프의 정상 종료(break 포함)에서만 도달
    except CameraError as exc:
        finish_cycle("abort", str(exc))
        print(f"\n[중단] {exc} — 추가 접근과 자동 홈 복귀를 수행하지 않습니다.")
        if viz is not None:
            viz.log_text(f"카메라 오류로 중단: {exc}", level="ERROR")
    finally:
        finish_cycle("abort", "예외 또는 사용자 중단")
        stop.restore()
        if robot is not None and home_on_exit:
            try:
                print("\n종료 — 홈 위치로 복귀합니다.")
                go_home(robot, cfg)
            except Exception as e:
                print(f"홈 복귀 실패: {e} — 로봇 상태를 수동 확인하세요.")
        console.section("실행 종료")
        console.row("누적", f"사이클 {stats['cycles']} · 이송 완료 {stats['success']}"
                    + (f" · 파지 판정 재시도 {stats['fail']}" if grip_check else ""))
        console.row("택타임", takt.summary())
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
