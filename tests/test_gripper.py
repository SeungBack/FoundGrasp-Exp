"""GripperCtl 시퀀스 로직 검증 — 모의(Fake) KorasGripper로 하드웨어 없이 테스트.

실측된 실패 모드를 재현한다:
- 명령 1회 유실(Modbus ACK는 성공했지만 모터 무동작) → 재전송으로 복구
- 완전 무반응 펌웨어 → 명확한 실패 반환 (조용한 무동작 금지)
- 초기화 스윕 진행 중 연결 → 완료까지 대기
"""
import time

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


def test_position_mode_and_invert(monkeypatch):
    cfg = {"gripper": dict(CFG["gripper"], grip_mode="position", invert=True)}
    ctl, fake = make_ctl(monkeypatch, cfg=cfg, pos=1000)
    # invert: 논리 개방(100%) == raw 0. 현재 raw 1000 = 논리 0%(닫힘)
    assert ctl.open()
    assert fake.pos == 0  # raw 0 = 논리 열림
