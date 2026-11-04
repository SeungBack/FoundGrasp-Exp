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
    monkeypatch.setattr(ctl, "_move_cmd", lambda *a, **kw: True)
    waits = []

    def settle(**kwargs):
        waits.append(kwargs["min_wait_s"])
        return True, FakeStatus(300)

    monkeypatch.setattr(ctl, "_wait_settle", settle)
    assert ctl.grip() == (True, 30.)
    assert waits == [min_wait]


def test_unsettled_close_cannot_be_treated_as_success(monkeypatch):
    ctl, _ = make_ctl(monkeypatch)
    monkeypatch.setattr(ctl, "_move_cmd", lambda *a, **kw: True)
    monkeypatch.setattr(ctl, "_wait_settle", lambda **kw: (False, FakeStatus(300)))
    with pytest.raises(RuntimeError, match="안정화 실패"):
        ctl.grip()


@pytest.mark.parametrize("position,settled,expected", [
    (1000, True, True), (500, True, False), (1000, False, False),
])
def test_short_open_wait_still_requires_arrival_and_settle(
        monkeypatch, position, settled, expected):
    cfg = {"gripper": {**CFG["gripper"], "open_min_wait_s": .15}}
    ctl, _ = make_ctl(monkeypatch, cfg=cfg)
    monkeypatch.setattr(ctl, "_move_cmd", lambda *a: True)

    def settle(**kwargs):
        assert kwargs == {"jam_protect": True, "min_wait_s": .15}
        return settled, FakeStatus(position)

    monkeypatch.setattr(ctl, "_wait_settle", settle)
    assert ctl.open() is expected


@pytest.mark.parametrize("value", [-1, 0, .099, 11, float("nan")])
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
    fake.stop = Mock(wraps=fake.stop)
    with pytest.raises(RuntimeError, match="열림 방향"):
        ctl.grip()
    fake.stop.assert_called_once()
    assert fake.target == fake.pos


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
    cfg = {"gripper": {**CFG["gripper"], "pre_open_min_wait_s": .15}}
    ctl, _ = make_ctl(monkeypatch, cfg=cfg)
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


def test_delayed_close_waits_for_contact_without_stopping_force_hold(monkeypatch, release_clock):
    ctl, fake = make_ctl(monkeypatch, pos=1000)
    fake.close_stop_at = 300
    fake.stop = Mock(wraps=fake.stop)

    def read():
        # A loaded close takes 1.8 s but continues making measurable progress.
        if fake.sent:
            fake.pos = max(300, round(1000 - release_clock[0] * 700 / 1.8))
        return FakeStatus(fake.pos, motor_cur=1400)

    monkeypatch.setattr(fake, "read_status", read)
    assert ctl.grip() == (True, 30.)
    assert 1.9 <= release_clock[0] < 10.
    assert fake.sent == [300]
    fake.stop.assert_not_called()


@pytest.mark.parametrize("phase", ["before", "start", "settling", "after"])
def test_close_guard_interrupts_each_phase_and_stops_squeezing(
        monkeypatch, release_clock, phase):
    ctl, fake = make_ctl(monkeypatch, pos=1000)
    fake.close_stop_at = 300
    fake.stop = Mock(wraps=fake.stop)
    current = ["before"]
    started, settle = ctl._motion_started, ctl._wait_settle

    def motion_started(*args, **kwargs):
        current[0] = "start"
        return started(*args, **kwargs)

    def wait_settle(*args, **kwargs):
        current[0] = "settling"
        result = settle(*args, **kwargs)
        current[0] = "after"
        return result

    error = RuntimeError("robot collision C31")

    def check():
        if current[0] == phase:
            raise error

    monkeypatch.setattr(ctl, "_motion_started", motion_started)
    monkeypatch.setattr(ctl, "_wait_settle", wait_settle)
    with pytest.raises(RuntimeError, match="C31") as caught:
        ctl.grip(check=check)
    assert caught.value is error
    fake.stop.assert_called_once()
    assert fake.sent == ([] if phase == "before" else [300])


def test_close_stop_failure_preserves_original_guard_error(monkeypatch, capsys):
    ctl, fake = make_ctl(monkeypatch)
    fake.stop = Mock(side_effect=OSError("Modbus unavailable"))
    error = RuntimeError("robot collision C31")
    with pytest.raises(RuntimeError, match="C31") as caught:
        ctl.grip(check=Mock(side_effect=error))
    assert caught.value is error
    assert "정지 실패" in capsys.readouterr().out
    assert fake.sent == []


@pytest.mark.parametrize("initial,ack", [(1000, True), (1000, False), (0, False)])
def test_close_command_failure_cannot_be_mistaken_for_empty_grasp(
        monkeypatch, release_clock, initial, ack):
    ctl, fake = make_ctl(monkeypatch, pos=initial, dead=True)
    fake.stop = Mock(wraps=fake.stop)
    if not ack:
        monkeypatch.setattr(fake, "grip_close", Mock(return_value=False))
    with pytest.raises(RuntimeError, match="명령/시작 확인 실패"):
        ctl.grip()
    fake.stop.assert_called_once()


@pytest.mark.parametrize("status", [None, FakeStatus(500, states=0),
                                    FakeStatus(500, states=513), FakeStatus(500, states=3)])
def test_close_bad_initial_status_never_sends_close(monkeypatch, status):
    ctl, fake = make_ctl(monkeypatch)
    fake.stop = Mock(wraps=fake.stop)
    monkeypatch.setattr(fake, "read_status", lambda: status)
    with pytest.raises(RuntimeError, match="시작 상태 오류"):
        ctl.grip()
    assert fake.sent == []
    fake.stop.assert_called_once()


@pytest.mark.parametrize("invert", [False, True])
def test_release_returns_before_fully_open_and_leaves_opening_active(monkeypatch, release_clock, invert):
    # Keep this early-return scenario independent of the production default:
    # the fake advances 10% per read and reaches full opening within 0.1 s.
    cfg = {"gripper": {**CFG["gripper"], "invert": invert, "release_confirm_s": .05}}
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


def test_camera_failure_after_failed_ack_prevents_open_retransmission(
        monkeypatch, release_clock):
    from efg.camera import CameraError

    ctl, fake = make_ctl(monkeypatch, pos=200)
    fake.stop = Mock(wraps=fake.stop)
    send = Mock(return_value=False)
    monkeypatch.setattr(ctl, '_set_raw_open_pct', send)

    def check():
        if send.call_count:
            raise CameraError('camera failed after first write')

    with pytest.raises(CameraError, match='after first write'):
        ctl.release(.040, check=check)
    send.assert_called_once_with(100.)
    fake.stop.assert_called_once()
    assert release_clock[0] == pytest.approx(.2)


def test_full_release_mode_waits_for_original_open(monkeypatch):
    cfg = {"gripper": {**CFG["gripper"], "release_mode": "full"}}
    ctl, _ = make_ctl(monkeypatch, cfg=cfg)
    monkeypatch.setattr(ctl, "open", Mock(return_value=False))
    assert not ctl.release(.04)
    ctl.open.assert_called_once()


@pytest.mark.parametrize('raw,reason', [
    (1001, '위치 판독 범위 오류'), (-1, '위치 판독 범위 오류'),
    (float('nan'), '위치 판독 범위 오류'), (390, '닫힘 방향 이동 감지')])
def test_release_reports_position_failure_values(monkeypatch, release_clock, capsys, raw, reason):
    ctl, fake = make_ctl(monkeypatch, pos=400)
    fake.stop = Mock(wraps=fake.stop)
    initial = True

    def read():
        nonlocal initial
        if initial:
            initial = False
            return FakeStatus(400)
        return FakeStatus(raw)

    monkeypatch.setattr(ctl, '_read_status', read)
    assert not ctl.release(.06)
    fake.stop.assert_called_once()
    output = capsys.readouterr().out
    assert reason in output
    assert '시작 40.00%' in output
    assert f'raw={raw}' in output
    assert '명령 후' in output


@pytest.mark.parametrize('plateau,expected', [(98.9, True), (98.0, False)])
def test_release_full_stroke_replays_measured_endpoint_plateau(
        monkeypatch, release_clock, capsys, plateau, expected):
    cfg = {"gripper": {**CFG["gripper"], "release_confirm_s": .2}}
    ctl, fake = make_ctl(monkeypatch, cfg=cfg, pos=615)
    fake.stop = Mock(wraps=fake.stop)

    def read():
        # The failing recording reached 98.9 mm in 0.87 s and stayed there.
        width = 61.5 + (plateau - 61.5) * min(release_clock[0] / .87, 1.)
        fake.pos = round(width * 10)
        return FakeStatus(fake.pos)

    monkeypatch.setattr(fake, 'read_status', read)
    assert ctl.release(.100) is expected
    if expected:
        assert .9 < release_clock[0] < 1.2
        fake.stop.assert_not_called()
        assert fake.target == 1000
    else:
        fake.stop.assert_called_once()
        output = capsys.readouterr().out
        assert '시간 초과' in output and '98.50mm' in output
        assert '98.00mm' in output and '100mA' in output


@pytest.mark.parametrize('invert', [False, True])
def test_release_small_startup_squeeze_must_recover_into_actual_opening(
        monkeypatch, release_clock, invert):
    cfg = {"gripper": {**CFG["gripper"], "invert": invert}}
    ctl, fake = make_ctl(monkeypatch, cfg=cfg, pos=796 if invert else 204)
    fake.stop = Mock(wraps=fake.stop)

    def read():
        t = release_clock[0]
        width = 20.4 if t < .08 else 19.3 if t < .14 else min(70., 20.4 + (t - .14) * 200)
        fake.pos = round((100 - width if invert else width) * 10)
        return FakeStatus(fake.pos, motor_cur=1462 if .08 <= t < .14 else 100)

    monkeypatch.setattr(fake, 'read_status', read)
    assert ctl.release(.060)
    assert .4 < release_clock[0] < .7
    assert fake.sent == ([0] if invert else [1000])
    fake.stop.assert_not_called()


@pytest.mark.parametrize('scenario', ['sustained_small', 'large', 'after_opening'])
def test_release_real_reverse_stops_even_if_some_opening_was_seen(
        monkeypatch, release_clock, capsys, scenario):
    ctl, fake = make_ctl(monkeypatch, pos=204)
    fake.stop = Mock(wraps=fake.stop)

    def read():
        t = release_clock[0]
        width = 20.4
        if scenario == 'sustained_small' and t >= .1:
            width = 19.3
        elif scenario == 'large' and t >= .1:
            width = 17.3
        elif scenario == 'after_opening' and t >= .1:
            width = 30. if t < .16 else 28.8
        fake.pos = round(width * 10)
        return FakeStatus(fake.pos)

    monkeypatch.setattr(fake, 'read_status', read)
    assert not ctl.release(.060)
    assert .1 <= release_clock[0] < (.15 if scenario == 'large' else .42)
    assert '닫힘 방향 이동 감지' in capsys.readouterr().out
    fake.stop.assert_called_once()
    assert len(fake.sent) == 1  # A reverse command is never retried as an ACK loss.


def test_full_stroke_plateau_with_high_current_is_not_clearance(
        monkeypatch, release_clock, capsys):
    ctl, fake = make_ctl(monkeypatch, pos=615)
    fake.stop = Mock(wraps=fake.stop)

    def read():
        fake.pos = 989 if fake.sent else 615
        return FakeStatus(fake.pos, motor_cur=1300)

    monkeypatch.setattr(fake, 'read_status', read)
    assert not ctl.release(.100)
    assert release_clock[0] >= 1.
    assert '잼' in capsys.readouterr().out
    fake.stop.assert_called_once()


def test_failed_open_ack_cannot_release_even_if_already_near_full_stroke(
        monkeypatch, release_clock):
    ctl, fake = make_ctl(monkeypatch, pos=989)
    fake.stop = Mock(wraps=fake.stop)
    send = Mock(return_value=False)
    monkeypatch.setattr(ctl, '_set_raw_open_pct', send)
    assert not ctl.release(.100)
    assert send.call_count == 2
    fake.stop.assert_called_once()


@pytest.mark.parametrize('start,expected', [(986, False), (998, True)])
def test_looser_full_stroke_tolerance_does_not_accept_unmoved_nearly_open_grip(
        monkeypatch, release_clock, start, expected):
    ctl, fake = make_ctl(monkeypatch, pos=start, dead=True)
    fake.stop = Mock(wraps=fake.stop)
    assert ctl.release(.100) is expected
    assert fake.stop.call_count == int(not expected)
    assert len(fake.sent) == (1 if expected else 2)


def test_status_arriving_after_release_deadline_cannot_report_success(
        monkeypatch, release_clock):
    ctl, fake = make_ctl(monkeypatch, pos=200)
    fake.stop = Mock(wraps=fake.stop)

    def read():
        if fake.sent:
            release_clock[0] = 6.
            fake.pos = 1000
        return FakeStatus(fake.pos)

    monkeypatch.setattr(fake, 'read_status', read)
    assert not ctl.release(.100)
    fake.stop.assert_called_once()


def test_failed_ack_retry_does_not_send_after_overall_deadline(monkeypatch, release_clock):
    cfg = {'gripper': {**CFG['gripper'], 'release_timeout_s': .15}}
    ctl, fake = make_ctl(monkeypatch, cfg=cfg, pos=200)
    fake.stop = Mock(wraps=fake.stop)
    send = Mock(return_value=False)
    monkeypatch.setattr(ctl, '_set_raw_open_pct', send)
    assert not ctl.release(.060)
    send.assert_called_once_with(100.)
    fake.stop.assert_called_once()


@pytest.mark.parametrize('dead', [False, True])
def test_release_retries_once_without_positive_opening_progress(
        monkeypatch, release_clock, dead):
    ctl, fake = make_ctl(monkeypatch, pos=200, drop_first_n=1, dead=dead)
    fake.stop = Mock(wraps=fake.stop)
    assert ctl.release(.040) is not dead
    assert fake.sent == [1000, 1000]
    assert release_clock[0] >= .4
    if dead:
        fake.stop.assert_called_once()
    else:
        fake.stop.assert_not_called()


@pytest.mark.parametrize('plateau,expected', [(64.0, False), (64.5, True)])
def test_non_saturated_clearance_keeps_its_tight_tolerance(
        monkeypatch, release_clock, plateau, expected):
    ctl, fake = make_ctl(monkeypatch, pos=200)
    fake.stop = Mock(wraps=fake.stop)

    def read():
        fake.pos = round(plateau * 10) if fake.sent else 200
        return FakeStatus(fake.pos)

    monkeypatch.setattr(fake, 'read_status', read)
    assert ctl.release(.060) is expected  # 60 + 5 mm requires >=64.5 mm, not 63.5.
    assert fake.stop.call_count == int(not expected)


@pytest.mark.parametrize('value', [-.1, 2.1, float('nan'), float('inf')])
def test_invalid_full_stroke_tolerance_is_rejected(monkeypatch, value):
    cfg = {"gripper": {**CFG["gripper"], "release_full_open_tolerance_mm": value}}
    with pytest.raises(ValueError, match='release_full_open_tolerance_mm'):
        make_ctl(monkeypatch, cfg=cfg)
