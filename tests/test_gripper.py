"""GripperCtl 시퀀스 로직 검증 — 모의(Fake) KorasGripper로 하드웨어 없이 테스트.

실측된 실패 모드를 재현한다:
- 명령 1회 유실(Modbus ACK는 성공했지만 모터 무동작) → 재전송으로 복구
- 완전 무반응 펌웨어 → 명확한 실패 반환 (조용한 무동작 금지)
- 초기화 스윕 진행 중 연결 → 완료까지 대기
"""
import time
from unittest.mock import Mock

import pytest

import efg.gripper as gctl  # noqa: E402


class FakeStatus:
    def __init__(self, finger_pos, motor_cur=100, states=0b1, voltage=23):
        self.finger_pos = finger_pos
        self.motor_cur = motor_cur
        self.states = states
        self.voltage = voltage

    @property
    def position(self):
        return self.finger_pos / 10.0

    @property
    def enabled(self):
        return bool(self.states & 1)

    @property
    def initialized(self):
        return bool(self.states & 2)

    @property
    def fault(self):
        return bool(self.states & (1 << 9))


class FakeGripper:
    """단순 1차 동역학: 명령이 '접수'되면 read_status마다 목표로 이동."""

    def __init__(self, port="fake", slave_id=1, pos=1000,
                 drop_first_n=0, dead=False, close_inverted=False):
        self.pos = pos
        self.target = pos
        self.drop_first_n = drop_first_n  # 앞 n개의 이동 명령을 조용히 무시
        self.dead = dead                  # 모든 이동 명령 무시 (ACK는 성공)
        self.close_inverted = close_inverted  # 펌웨어 방향 손상: close가 열림
        self.sent = []
        self.enabled_flag = True
        self.init_busy = False

    def connect(self, initialize=True, init_timeout=8.0):
        return True

    def disconnect(self):
        pass

    def enable(self):
        self.enabled_flag = True
        return True

    def stop(self):
        self.target = self.pos
        return True

    def set_speed(self, v):
        return True

    def set_force(self, v):
        return True

    def _accept_move(self, target):
        self.sent.append(target)
        if self.dead:
            return
        if self.drop_first_n > 0:
            self.drop_first_n -= 1
            return
        self.target = target

    def set_finger_position(self, raw):
        self._accept_move(int(raw))
        return True

    def grip_open(self):
        self._accept_move(1000)
        return True

    def grip_close(self):
        if self.close_inverted:
            self._accept_move(1000)
        else:
            self._accept_move(self.close_stop_at
                              if hasattr(self, "close_stop_at") else 0)
        return True

    def grip_initialize(self):
        self.init_busy = True
        self._init_t0 = time.monotonic()
        return True

    def read_status(self):
        # 초기화 busy는 0.3초 뒤 해제
        if self.init_busy and time.monotonic() - self._init_t0 > 0.3:
            self.init_busy = False
            self.pos = self.target = 1000
        step = 120
        if self.pos < self.target:
            self.pos = min(self.target, self.pos + step)
        elif self.pos > self.target:
            self.pos = max(self.target, self.pos - step)
        states = (1 if self.enabled_flag else 0) | (2 if self.init_busy else 0)
        return FakeStatus(self.pos, states=states)


CFG = {"gripper": {
    "port": "fake", "slave_id": 1, "force_pct": 80, "speed_pct": 100,
    "stroke_mm": 100.0, "pre_open_margin_mm": 10, "grip_success_min_pct": 5.0,
    "grip_mode": "grip", "invert": False, "initialize_on_connect": False,
    "close_probe_on_connect": True,
}}


@pytest.fixture(autouse=True)
def fast_timeouts(monkeypatch):
    monkeypatch.setattr(gctl, "CMD_START_TIMEOUT_S", 0.4)
    monkeypatch.setattr(gctl, "INIT_TIMEOUT_S", 2.0)


def make_ctl(monkeypatch, cfg=None, **fake_kwargs):
    fake = FakeGripper(**fake_kwargs)
    monkeypatch.setattr(gctl, "KorasGripper", lambda *a, **k: fake)
    ctl = gctl.GripperCtl(cfg or CFG)
    return ctl, fake


@pytest.mark.parametrize("invert,expected", [(False, .025), (True, .075)])
def test_visual_feedback_uses_existing_reads_only(monkeypatch, invert, expected):
    cfg = {"gripper": {**CFG["gripper"], "invert": invert}}
    ctl, fake = make_ctl(monkeypatch, cfg=cfg, pos=250)
    assert ctl.visual_status is None
    ctl.status()
    width, stamp = ctl.visual_status
    assert width == pytest.approx(expected)
    assert stamp <= time.monotonic()
    monkeypatch.setattr(fake, "read_status", lambda: pytest.fail("unexpected Modbus I/O"))
    assert ctl.visual_status == (width, stamp)


def test_connect_verifies_open(monkeypatch):
    ctl, fake = make_ctl(monkeypatch, pos=0)  # 닫힘에서 시작
    ctl.connect()
    assert fake.pos == 1000  # 개방 검증으로 실제 열림


def test_connect_already_open_passes(monkeypatch):
    ctl, fake = make_ctl(monkeypatch, pos=1000)
    ctl.connect()  # 이동 없이 통과해야 함


def test_dropped_command_is_resent(monkeypatch):
    ctl, fake = make_ctl(monkeypatch, pos=0, drop_first_n=1)
    ctl.connect()  # 1차 개방 명령 유실 → 재전송으로 복구
    assert fake.pos == 1000
    assert len(fake.sent) >= 2


def test_dead_firmware_raises(monkeypatch):
    ctl, fake = make_ctl(monkeypatch, pos=0, dead=True)
    with pytest.raises(RuntimeError, match="개방 검증 실패"):
        ctl.connect()


def test_grip_success_and_empty(monkeypatch):
    ctl, fake = make_ctl(monkeypatch, pos=1000)
    fake.close_stop_at = 300  # 물체에 걸려 30%에서 정지
    ok, pos = ctl.grip()
    assert ok and pos == 30.0
    del fake.close_stop_at    # 빈 파지 → 0%까지 닫힘
    fake.target = fake.pos = 1000
    ok, pos = ctl.grip()
    assert not ok and pos == 0.0


@pytest.mark.parametrize("min_wait", [.15, .25])
def test_close_minimum_wait_is_configurable(monkeypatch, min_wait):
    cfg = {"gripper": {**CFG["gripper"], "close_min_wait_s": min_wait}}
    ctl, fake = make_ctl(monkeypatch, cfg=cfg)
    monkeypatch.setattr(ctl, "_move_cmd", lambda *a: True)
    waits = []

    def settle(**kwargs):
        waits.append(kwargs["min_wait_s"])
        return True, FakeStatus(300)

    monkeypatch.setattr(ctl, "_wait_settle", settle)
    assert ctl.grip() == (True, 30.)
    assert waits == [min_wait]


def test_unsettled_close_cannot_be_treated_as_success(monkeypatch):
    ctl, _ = make_ctl(monkeypatch)
    monkeypatch.setattr(ctl, "_move_cmd", lambda *a: True)
    monkeypatch.setattr(ctl, "_wait_settle", lambda **kw: (False, FakeStatus(300)))
    with pytest.raises(RuntimeError, match="안정화 실패"):
        ctl.grip()


@pytest.mark.parametrize("position,settled,expected", [
    (1000, True, True), (500, True, False), (1000, False, False),
])
def test_short_open_wait_still_requires_arrival_and_settle(
        monkeypatch, position, settled, expected):
    ctl, _ = make_ctl(monkeypatch)
    monkeypatch.setattr(ctl, "_move_cmd", lambda *a: True)

    def settle(**kwargs):
        assert kwargs == {"jam_protect": True, "min_wait_s": .15}
        return settled, FakeStatus(position)

    monkeypatch.setattr(ctl, "_wait_settle", settle)
    assert ctl.open() is expected


@pytest.mark.parametrize("value", [-1, 0, .1, 11, float("nan")])
def test_invalid_open_wait_rejected(monkeypatch, value):
    cfg = {"gripper": {**CFG["gripper"], "open_min_wait_s": value}}
    with pytest.raises(ValueError, match="open_min_wait_s"):
        make_ctl(monkeypatch, cfg=cfg)


@pytest.mark.parametrize("min_wait", [.15, .3])
def test_minimum_wait_does_not_bypass_position_stability(monkeypatch, min_wait):
    ctl, _ = make_ctl(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(gctl.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gctl.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    monkeypatch.setattr(ctl, "_read_status", lambda: FakeStatus(
        int(1000 - min(clock[0], .6) * 1000)))
    ok, _ = ctl._wait_settle(min_wait_s=min_wait)
    assert ok and clock[0] >= .75


def test_connect_waits_for_running_init(monkeypatch):
    ctl, fake = make_ctl(monkeypatch, pos=500)
    fake.grip_initialize()  # 연결 시점에 이미 스윕 진행 중
    ctl.connect()
    assert not fake.init_busy  # 완료를 기다린 뒤 진행

def test_initialize_on_connect(monkeypatch):
    cfg = {"gripper": dict(CFG["gripper"], initialize_on_connect=True)}
    ctl, fake = make_ctl(monkeypatch, cfg=cfg, pos=0)
    ctl.connect()
    assert fake.pos == 1000


def test_probe_detects_inverted_close(monkeypatch):
    ctl, fake = make_ctl(monkeypatch, pos=1000, close_inverted=True)
    with pytest.raises(RuntimeError, match="방향 점검 실패"):
        ctl.connect()


def test_grip_aborts_on_inverted_close(monkeypatch):
    ctl, fake = make_ctl(monkeypatch, pos=600, close_inverted=True)
    ok, pos = ctl.grip()
    assert not ok  # 열림 방향 폭주를 실패로 판정 (성공 오판 금지)


def test_pre_open_uses_mm_margin(monkeypatch):
    ctl, fake = make_ctl(monkeypatch, pos=1000)
    assert ctl.pre_open(0.05)          # 50mm + 10mm = 60%
    assert fake.pos == 600
    # 구 설정(pct) 호환: 20% of 100mm stroke = 20mm 여유
    legacy = {"gripper": dict(CFG["gripper"])}
    del legacy["gripper"]["pre_open_margin_mm"]
    legacy["gripper"]["pre_open_margin_pct"] = 20
    ctl2, fake2 = make_ctl(monkeypatch, cfg=legacy, pos=1000)
    assert ctl2.pre_open_margin_mm == 20.0


@pytest.mark.parametrize("position,settled,expected", [
    (500, True, True), (900, True, False), (500, False, False),
])
def test_short_pre_open_still_requires_width_and_settle(monkeypatch, position, settled, expected):
    ctl, _ = make_ctl(monkeypatch)
    monkeypatch.setattr(ctl, "_move_cmd", lambda *a, **kw: True)

    def settle(**kwargs):
        assert kwargs["min_wait_s"] == .15
        assert kwargs["jam_protect"] is True
        return settled, FakeStatus(position)

    monkeypatch.setattr(ctl, "_wait_settle", settle)
    assert ctl.pre_open(.04) is expected  # 40 mm + 10 mm margin


def test_position_mode_and_invert(monkeypatch):
    cfg = {"gripper": dict(CFG["gripper"], grip_mode="position", invert=True)}
    ctl, fake = make_ctl(monkeypatch, cfg=cfg, pos=1000)
    # invert: 논리 개방(100%) == raw 0. 현재 raw 1000 = 논리 0%(닫힘)
    assert ctl.open()
    assert fake.pos == 0  # raw 0 = 논리 열림


@pytest.fixture
def release_clock(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(gctl.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gctl.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    return clock


@pytest.mark.parametrize("invert", [False, True])
def test_release_returns_before_fully_open_and_leaves_opening_active(monkeypatch, release_clock, invert):
    cfg = {"gripper": {**CFG["gripper"], "invert": invert}}
    ctl, fake = make_ctl(monkeypatch, cfg=cfg, pos=800 if invert else 200)
    fake.stop = Mock(wraps=fake.stop)
    assert ctl.release(.04)
    pos = (1000 - fake.pos if invert else fake.pos) / 10
    assert 44.5 <= pos < 100
    assert fake.target == (0 if invert else 1000)
    fake.stop.assert_not_called()
    assert release_clock[0] >= .05


def test_release_does_not_accept_current_grip_width_as_clearance(monkeypatch, release_clock):
    ctl, _ = make_ctl(monkeypatch, pos=650)
    monkeypatch.setattr(ctl, "_move_cmd", lambda *a, **kw: True)
    monkeypatch.setattr(ctl, "_read_status", lambda: FakeStatus(650 if release_clock[0] < .1 else 700))
    assert ctl.release(.04)  # actual 65 mm requires 70 mm, even though prediction is 40 mm
    assert release_clock[0] >= .15


def test_release_at_stroke_limit_tolerates_998_readback(monkeypatch, release_clock):
    ctl, fake = make_ctl(monkeypatch, pos=200)
    read = fake.read_status

    def capped():
        status = read()
        status.finger_pos = min(status.finger_pos, 998)
        return status

    monkeypatch.setattr(fake, "read_status", capped)
    assert ctl.release(.100)
    assert fake.pos == 1000


@pytest.mark.parametrize("failure", ["stuck", "jam", "fault", "missing", "reverse", "dead"])
def test_failed_release_never_uses_timeout_as_success(monkeypatch, release_clock, failure):
    ctl, fake = make_ctl(monkeypatch, pos=400, dead=failure == "dead")
    fake.stop = Mock(wraps=fake.stop)
    read = fake.read_status

    def failing_status():
        if fake.sent and failure in ("stuck", "jam"):
            fake.target = 550
        status = read()
        if fake.sent:
            if failure == "fault":
                status.states |= 1 << 9
            elif failure == "missing":
                return None
            elif failure == "jam":
                status.motor_cur = 1200
        return status

    monkeypatch.setattr(fake, "read_status", failing_status)
    if failure == "reverse":
        monkeypatch.setattr(ctl, "_set_raw_open_pct", lambda pct: fake._accept_move(0) or True)
    assert not ctl.release(.06)
    fake.stop.assert_called_once()
    assert fake.target == fake.pos


def test_camera_error_during_release_stops_fingers_and_propagates(monkeypatch, release_clock):
    from efg.camera import CameraError

    ctl, fake = make_ctl(monkeypatch, pos=200)
    fake.stop = Mock(wraps=fake.stop)

    def check():
        if fake.sent:
            raise CameraError("release interrupted")

    with pytest.raises(CameraError, match="release interrupted"):
        ctl.release(.04, check=check)
    fake.stop.assert_called_once()


def test_full_release_mode_waits_for_original_open(monkeypatch):
    cfg = {"gripper": {**CFG["gripper"], "release_mode": "full"}}
    ctl, _ = make_ctl(monkeypatch, cfg=cfg)
    monkeypatch.setattr(ctl, "open", Mock(return_value=False))
    assert not ctl.release(.04)
    ctl.open.assert_called_once()
