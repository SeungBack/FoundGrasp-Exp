"""KORAS 그리퍼 제어 — third_party/koras-gripper-wrapper 사본 사용 (원본 무수정).

그립 성공 판정: 전용 상태 비트가 없어, close 후 finger 위치가 완전 닫힘(0%)
전에 멈췄는지(finger_pos > 임계값)로 판정한다.

원본 래퍼를 그대로 쓰지 않고 자체 시퀀스를 두는 이유 (2026-07-21 실측):
- [핵심] Modbus의 Gripper Initialize(101)를 실행하면 이후 grip_close(103)의
  방향 학습이 반대(열림)로 손상된다 — 재현율 100%, Modbus 재초기화로 복구
  불가, 그리퍼 본체 버튼(I) 초기화로만 복구됨. 따라서 이 모듈은 기본적으로
  Modbus 초기화를 절대 보내지 않는다 (버튼 초기화 전제, 학습값은 전원
  차단까지 유지 — 매뉴얼 4.3).
- grip_open/초기화 스윕은 열림 끝단을 700~1200mA로 계속 누르는 미완료
  상태로 남고, 이 상태에서 후속 grip 명령이 무시된다 → 열기 동작은 항상
  set_finger_position(위치 제어, 저전류 정착 실측 확인)만 사용하고,
  grip_close는 실제 파지(접촉 정지·힘 제어)에만 쓴다.
- 원본 connect()는 초기화 스윕(실측 6.6초, 대기 한도 8초)이 끝나지 않았거나
  실패해도 True를 반환한다 → 스윕 도중 후속 명령이 나가면 상태가 꼬인다.
- 이동 명령은 Modbus ACK만으로는 실제 구동을 보장하지 못한다 → 전송 후
  이동 시작을 감지하고, 미감지 시 1회 재전송한다. 연결 시 close 방향을
  실측 점검(probe)해 방향 손상 상태로 로봇이 움직이는 것을 사전 차단한다.
- 개방 계열 이동은 목표 도달을 검증하고, 위치 불변 + 고전류 지속(기구 잼)이면
  모터를 정지시켜 보호한다. close는 물체 접촉으로 멈추는 게 정상이므로
  잼 보호 없이 정착만 기다린다.
"""
import time

from .paths import KORAS_ROOT, add_vendor_paths

add_vendor_paths(KORAS_ROOT)

from koras_gripper_wrapper import KorasGripper  # noqa: E402  (사본)

STALL_CURRENT_MA = 1100   # 이 이상 전류로
STALL_HOLD_S = 1.0        # 이 시간 이상 위치 불변이면 잼으로 판정 (개방 이동)
MOVE_TOL_TICKS = 4        # 위치 변화 감지 임계 (0.1% 단위)
INIT_TIMEOUT_S = 20.0     # 초기화 스윕 대기 한도 (실측 6.6초 + 여유)
CMD_START_TIMEOUT_S = 1.2 # 이동 명령 후 이 시간 내 이동 시작해야 함


class GripperCtl:
    def __init__(self, cfg):
        g = cfg["gripper"]
        self.port = g["port"]
        self.slave_id = int(g["slave_id"])
        self.force_pct = float(g["force_pct"])
        self.speed_pct = float(g["speed_pct"])
        self.close_min_wait_s = float(g.get("close_min_wait_s", 0.3))
        if not 0.15 <= self.close_min_wait_s <= 10.0:
            raise ValueError("gripper.close_min_wait_s는 0.15~10초 범위여야 합니다")
        self.open_min_wait_s = float(g.get("open_min_wait_s", 0.15))
        if not 0.15 <= self.open_min_wait_s <= 10.0:
            raise ValueError("gripper.open_min_wait_s는 0.15~10초 범위여야 합니다")
        self.stroke_mm = float(g["stroke_mm"])
        # 접근 시 개방 폭 여유 (mm). 구 설정(pre_open_margin_pct, 스트로크 %)이
        # 있으면 mm로 환산해 호환 — 전체 개방 접근은 주변 물체와 충돌 위험
        if "pre_open_margin_mm" in g:
            self.pre_open_margin_mm = float(g["pre_open_margin_mm"])
        else:
            self.pre_open_margin_mm = (
                float(g.get("pre_open_margin_pct", 10.0)) / 100.0
                * self.stroke_mm)
        self.grip_success_min_pct = float(g["grip_success_min_pct"])
        # "grip": 펌웨어 grip_open/close 사용(접촉 정지·힘 제어). "position":
        # set_finger_position만 사용 — grip 명령 계열 펌웨어 이상 시 폴백
        self.grip_mode = g.get("grip_mode", "grip")
        # 펌웨어 방향 반전 보정: true면 위치 명령/판독과 open/close를 소프트웨어에서
        # 뒤집는다. scripts/gripper_direction_check.py로 실측 후 설정할 것.
        self.invert = bool(g.get("invert", False))
        # 연결 시 초기화 스윕 실행 여부. 기본 false — Modbus 초기화는 grip_close
        # 방향 학습을 손상시키는 것이 실측 확인됨(모듈 docstring). 반드시 그리퍼
        # 버튼(I)으로 초기화할 것. true로 켜도 직후 방향 점검에서 손상이 걸러진다.
        self.initialize_on_connect = bool(g.get("initialize_on_connect", False))
        # 연결 시 close 방향 실측 점검 (grip 모드 전용, 핑거만 살짝 닫았다 폄)
        self.close_probe_on_connect = bool(g.get("close_probe_on_connect", True))
        self.g = KorasGripper(self.port, slave_id=self.slave_id)
        self._visual_status = None

    # --- 반전 보정 헬퍼: 이 모듈 안에서는 항상 "논리" 값(100=열림)을 쓴다 ---
    def _set_raw_open_pct(self, pct):
        """논리 개방률(0=닫힘, 100=열림)로 위치 명령 전송."""
        raw = round(pct * 10)
        if self.invert:
            raw = 1000 - raw
        return self.g.set_finger_position(max(0, min(1000, raw)))

    def _pos_pct(self, s):
        """상태에서 논리 개방률 판독."""
        return 100.0 - s.position if self.invert else s.position

    def _cmd_open(self):
        return self.g.grip_close() if self.invert else self.g.grip_open()

    def _cmd_close(self):
        return self.g.grip_open() if self.invert else self.g.grip_close()

    # --- 상태/진단 ---
    def _fmt_status(self, s):
        if s is None:
            return "status=None (상태 읽기 실패 — 통신/전원 점검)"
        return (f"pos {self._pos_pct(s):.1f}%, {s.motor_cur}mA, {s.voltage}V, "
                f"enabled={s.enabled}, initBusy={s.initialized}, "
                f"fault={s.fault}, bits=0b{s.states:010b}")

    @property
    def visual_status(self):
        """마지막 정상 판독 (개방 폭 m, monotonic 시각). 시각화에서 추가 Modbus I/O 없음."""
        return self._visual_status

    def _read_status(self):
        s = self.g.read_status()
        if s is not None:
            self._visual_status = (self._pos_pct(s) / 100.0 * self.stroke_mm / 1000.0,
                                   time.monotonic())
        return s

    def _require_status(self, retries=3):
        """상태 읽기 (재시도 포함). 실패 시 RuntimeError."""
        for _ in range(retries):
            s = self._read_status()
            if s is not None:
                return s
            time.sleep(0.2)
        raise RuntimeError(
            f"그리퍼 상태 읽기 실패 ({self.port}) — 전원(24V)·케이블·"
            f"slave_id({self.slave_id}) 점검")

    def probe(self):
        """포트만 열고 상태를 읽는다 (모터 명령 없음). gripper_test --status 용."""
        if not self.g.connect(initialize=False):
            raise RuntimeError(f"그리퍼 포트 열기 실패: {self.port}")
        return self._require_status()

    # --- 연결/초기화 ---
    def connect(self):
        """포트 열기 → 통신 확인 → enable → (옵션) 초기화 → 개방 기능 검증.

        초기화 스윕은 initialize_on_connect: true일 때만 실행한다 (전체 스트로크
        개폐, ~7초 — 핑거 주변 공간 확보 필수). 마지막의 개방 검증이 "명령이
        실제로 그리퍼를 움직이는가"를 확인하므로, 여기서 통과하면 이후 파지
        시퀀스에서 조용히 무동작하는 상황을 사전에 걸러낸다.
        """
        if not self.g.connect(initialize=False):
            raise RuntimeError(f"그리퍼 포트 열기 실패: {self.port}")
        s = self._require_status()
        print(f"   그리퍼 상태: {self._fmt_status(s)}")
        if s.fault:
            raise RuntimeError(f"그리퍼 fault 상태 — 전원 재투입 필요 "
                               f"[{self._fmt_status(s)}]")
        if s.initialized:
            # 초기화 스윕이 아직 도는 중 (직전 세션/버튼 초기화 직후 등) — 완료 대기
            print("   초기화 스윕 진행 중 — 완료 대기...")
            self._wait_init_done(require_busy_seen=True)

        if not self.g.enable():
            time.sleep(0.2)
            if not self.g.enable():
                raise RuntimeError("그리퍼 enable 명령 전송 실패 (Modbus)")
        time.sleep(0.3)
        s = self._require_status()
        if not s.enabled:
            raise RuntimeError(
                f"enable 후에도 모터 비활성 — 펌웨어가 명령을 무시하는 상태. "
                f"전원 재투입 + 버튼(I) 초기화 후 재시도 [{self._fmt_status(s)}]")
        self._send_retry(lambda: self.g.set_speed(self.speed_pct), "속도 설정")
        self._send_retry(lambda: self.g.set_force(self.force_pct), "힘 설정")

        if self.initialize_on_connect:
            self._initialize()

        # 기능 검증 1: 위치 명령이 실제 동작하는지 (이미 열려 있으면 즉시 통과)
        if not self.open():
            s = self._read_status()
            self.g.stop()
            raise RuntimeError(
                f"그리퍼 개방 검증 실패 — 명령은 전송되나 동작하지 않음. "
                f"전원 재투입 + 버튼(I) 초기화 후 재시도 "
                f"[{self._fmt_status(s)}]")
        # 기능 검증 2: grip_close 방향 점검 (방향 손상 상태로 파지 진입 차단)
        if self.grip_mode == "grip" and self.close_probe_on_connect:
            self._probe_close_direction()
        return self

    def _probe_close_direction(self):
        """grip_close가 실제 '닫힘' 방향으로 동작하는지 실측 점검.

        90%에서 close를 보내 5% 이상 닫히기 시작하면 정상 — 즉시 위치 명령으로
        되돌려 개방한다. 열림 방향으로 움직이거나 무동작이면 펌웨어 방향 학습
        손상(Modbus 초기화 이력 등)이므로 즉시 실패시킨다.
        """
        if not (self._set_raw_open_pct(90.0) and
                self._wait_settle(min_wait_s=0.5)[0]):
            raise RuntimeError("close 방향 점검 준비 실패 (90% 이동 불가)")
        p0 = self._pos_pct(self._require_status())
        if not self._cmd_close():
            raise RuntimeError("close 방향 점검 실패 (Modbus 전송 오류)")
        verdict = "무동작"
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.5:
            s = self._read_status()
            if s is not None:
                p = self._pos_pct(s)
                if p <= p0 - 5.0:
                    verdict = "정상"
                    break
                if p >= p0 + 5.0:
                    verdict = "반전"
                    break
            time.sleep(0.05)
        self._set_raw_open_pct(100.0)  # close 중단 + 재개방 (위치 명령이 우선함)
        self._wait_settle(jam_protect=True)
        if verdict != "정상":
            self.g.stop()
            raise RuntimeError(
                f"grip_close 방향 점검 실패 ({verdict}) — 펌웨어 방향 학습 손상. "
                "복구: 그리퍼 본체 버튼(I)로 재초기화 (Modbus 초기화로는 복구 "
                "안 됨, 2026-07-21 실측). 임시 우회: gripper.grip_mode: "
                "\"position\"")
        print("   close 방향 점검: 정상")

    def _initialize(self):
        """초기화 스윕 실행 + 완료까지 자체 대기 (원본 connect()의 8초 한도·
        실패 무시 문제를 피한다). 실패 시 RuntimeError.

        주의: Modbus 초기화는 grip_close 방향 학습을 손상시키는 것이 실측
        확인됨 — 가능하면 버튼(I) 초기화를 쓰고 이 경로는 피할 것. 실행해도
        직후 connect()의 close 방향 점검이 손상을 걸러낸다.
        """
        print("   그리퍼 초기화 스윕 실행 (~7초, 핑거 주변 확보)...")
        print("   경고: Modbus 초기화는 close 방향 학습을 손상시킬 수 있음 "
              "(버튼 초기화 권장)")
        if not self._send_retry(self.g.grip_initialize, "초기화"):
            raise RuntimeError("그리퍼 초기화 명령 전송 실패 (Modbus)")
        self._wait_init_done(require_busy_seen=False)
        time.sleep(0.3)

    def _wait_init_done(self, require_busy_seen):
        """초기화 busy 비트가 서고(관측) 내려갈 때까지 대기."""
        t0 = time.monotonic()
        saw_busy = require_busy_seen
        while time.monotonic() - t0 < INIT_TIMEOUT_S:
            s = self._read_status()
            if s is not None:
                if s.initialized:
                    saw_busy = True
                elif saw_busy:
                    return
            time.sleep(0.05)
        if saw_busy:
            self.g.stop()
            raise RuntimeError(
                f"그리퍼 초기화가 {INIT_TIMEOUT_S:.0f}초 내에 끝나지 않음 — "
                "스윕 걸림(잼) 의심. 전원 재투입 + 핑거 자유 상태에서 "
                "버튼(I) 초기화 권장")
        print("   경고: 초기화 busy 미관측 — 명령이 무시되었을 수 있음 "
              "(이후 개방 검증에서 확인)")

    def _send_retry(self, send_fn, desc):
        """명령 전송, Modbus 오류 시 1회 재시도."""
        if send_fn():
            return True
        print(f"   경고: {desc} 명령 전송 실패 — 재시도")
        time.sleep(0.2)
        return send_fn()

    def _move_cmd(self, send_fn, target_pct, desc, check=None):
        """이동 명령 전송 + 실제 이동 시작 검증 (미시작 시 1회 재전송).

        이미 목표 근처면 이동이 없는 게 정상이므로 전송만 하고 성공 처리.
        """
        if check:
            check()
        s0 = self._read_status()
        if s0 is not None and self._reached(s0, target_pct):
            send_fn()  # 홀드 목표 갱신용 — 이동은 없음
            return True
        for attempt in (1, 2):
            if check:
                check()
            if not send_fn():
                print(f"   경고: {desc} 명령 전송 실패 (Modbus, {attempt}차)")
                time.sleep(0.2)
                continue
            if self._motion_started(s0, check=check):
                return True
            print(f"   경고: {desc} 명령 후 이동 미감지 ({attempt}차) — 재전송")
            s0 = self._read_status()
        print(f"경고: {desc} 명령이 동작하지 않음 — "
              f"[{self._fmt_status(self._read_status())}]")
        return False

    def _motion_started(self, s0, timeout=CMD_START_TIMEOUT_S, check=None):
        """기준 상태 s0 대비 위치가 변하기 시작했는지 감지."""
        ref = None if s0 is None else s0.finger_pos
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if check:
                check()
            s = self._read_status()
            if s is not None:
                if ref is None:
                    ref = s.finger_pos
                elif abs(s.finger_pos - ref) > MOVE_TOL_TICKS:
                    return True
            time.sleep(0.05)
        return False

    def _wait_settle(self, timeout=10.0, jam_protect=False, min_wait_s=1.0, check=None):
        """자체 이동 대기. 반환 (settled, status).

        - 이동 시작 지연을 견디도록 min_wait_s 전에는 정착 판정하지 않음
        - jam_protect: 위치 불변 + 고전류가 STALL_HOLD_S 지속되면 모터 정지 후
          실패 반환 (개방 이동용 — close는 접촉 정지가 정상이라 쓰지 않음)
        """
        t0 = time.monotonic()
        last = None
        stable = 0
        stall_since = None
        s = None
        while time.monotonic() - t0 < timeout:
            if check:
                check()
            s = self._read_status()
            if s is None or s.fault:
                return False, s
            if last is not None:
                if abs(s.finger_pos - last) <= MOVE_TOL_TICKS:
                    stable += 1
                    if jam_protect and abs(s.motor_cur) > STALL_CURRENT_MA:
                        stall_since = stall_since or time.monotonic()
                        if time.monotonic() - stall_since > STALL_HOLD_S:
                            self.g.stop()
                            print("경고: 그리퍼 잼 감지 (위치 불변 + "
                                  f"{s.motor_cur}mA) — 모터 정지")
                            return False, s
                    else:
                        stall_since = None
                    if stable >= 3 and time.monotonic() - t0 >= min_wait_s:
                        return True, s
                else:
                    stable = 0
                    stall_since = None
            last = s.finger_pos
            time.sleep(0.05)
        return False, s

    def _reached(self, s, target_pct, tol_pct=8.0):
        return s is not None and abs(self._pos_pct(s) - target_pct) <= tol_pct

    def open(self):
        """완전 개방. 이동 시작 검증(재전송 1회) + 목표(100%) 도달 검증 + 잼 보호.

        grip 모드에서도 위치 명령을 쓴다 — grip_open은 열림 끝단을 고전류로
        계속 누르는 상태로 남아 이후 grip 명령이 무시되는 것이 실측 확인됨.
        """
        if not self._move_cmd(lambda: self._set_raw_open_pct(100.0),
                              100.0, "개방"):
            return False
        ok, s = self._wait_settle(jam_protect=True,
                                  min_wait_s=self.open_min_wait_s)
        if not (ok and self._reached(s, 100.0)):
            pos = None if s is None else round(self._pos_pct(s), 1)
            print(f"경고: 개방 실패 (finger {pos}%) — 그리퍼 상태 점검 필요")
            return False
        return True

    def pre_open(self, width_m, check=None):
        """목표 그립 폭 + 여유(mm)만큼만 개방 — 접근 중 주변 충돌 최소화.

        시작·도달 검증 + 잼 보호 포함.
        """
        open_mm = width_m * 1000.0 + self.pre_open_margin_mm
        pct = min(100.0, open_mm / self.stroke_mm * 100.0)
        print(f"   pre-open 목표: {pct:.0f}% (그립 폭 {width_m * 1000:.0f}mm + "
              f"여유 {self.pre_open_margin_mm:.0f}mm)")
        if not self._move_cmd(lambda: self._set_raw_open_pct(pct), pct, "pre-open", check=check):
            return False
        ok, s = self._wait_settle(jam_protect=True, check=check)
        if not (ok and self._reached(s, pct)):
            pos = None if s is None else round(self._pos_pct(s), 1)
            print(f"경고: pre-open 실패 (목표 {pct:.0f}%, 실제 {pos}%) — "
                  "그리퍼 상태 점검 필요")
            return False
        return True

    def grip(self):
        """닫아서 파지. 반환: (success, finger_pos_pct).

        이동 시작 검증(재전송 1회) 후, 접촉 정지가 정상이므로 잼 보호 없이
        정착만 기다린다 (파지 유지 전류 허용). position 모드에서는 0%(완전
        닫힘)를 목표로 걸어 물체에서 멈추게 한다.

        방향 감시: close 후 위치가 '열림' 방향으로 5% 이상 움직이면 펌웨어
        방향 손상이므로 즉시 원위치 명령으로 세우고 실패 반환한다.
        """
        send = ((lambda: self._set_raw_open_pct(0.0))
                if self.grip_mode == "position" else self._cmd_close)
        s0 = self._read_status()
        p0 = None if s0 is None else self._pos_pct(s0)
        if not self._move_cmd(send, 0.0, "파지(close)"):
            return False, None
        settled, s = self._wait_settle(jam_protect=False,
                                       min_wait_s=self.close_min_wait_s)
        if not settled:
            # Do not lift while fingers are still moving, even with grip_check off.
            raise RuntimeError("그리퍼 닫힘 안정화 실패 — 상승하지 않습니다")
        if s is None or s.fault:
            return False, None
        pos = self._pos_pct(s)
        if p0 is not None and pos >= p0 + 5.0:
            self._set_raw_open_pct(pos)  # 열림 폭주 중단 (위치 명령이 우선함)
            print(f"경고: close가 열림 방향으로 동작 ({p0:.1f}→{pos:.1f}%) — "
                  "펌웨어 방향 손상. 버튼(I) 초기화 필요")
            return False, round(pos, 1)
        return pos > self.grip_success_min_pct, round(pos, 1)

    def status(self):
        return self._read_status()

    def disconnect(self):
        self.g.disconnect()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.disconnect()
