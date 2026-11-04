"""Pre-open USB recovery, without a serial port or motor commands."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import efg.gripper as gctl


def status(position=1000, *, states=1, current=100):
    return SimpleNamespace(
        finger_pos=position, position=position / 10, states=states,
        motor_cur=current, voltage=24, enabled=bool(states & 1),
        initialized=bool(states & 2), fault=bool(states & 512))


class ReenumeratedPort:
    """A stale descriptor stays broken until explicitly closed and reopened."""

    def __init__(self, *, stale=True, position=1000, dead=False):
        self.stale = stale
        self.closed = False
        self.position = self.target = position
        self.dead = dead
        self.events = []
        self.disconnect = Mock(side_effect=self._disconnect)
        self.connect = Mock(side_effect=self._connect)
        self.read_status = Mock(side_effect=self._read_status)
        self.set_finger_position = Mock(side_effect=self._set_finger_position)
        self.stop = Mock(return_value=True)
        self.enable = Mock(side_effect=AssertionError("recovery must not enable"))
        self.grip_initialize = Mock(side_effect=AssertionError("recovery must not initialize"))
        self.grip_close = Mock(side_effect=AssertionError("recovery must not probe close"))
        self.grip_open = Mock(side_effect=AssertionError("recovery must use position control"))

    def _disconnect(self):
        self.events.append("disconnect")
        self.closed = True

    def _connect(self, *, initialize):
        self.events.append("connect")
        assert initialize is False
        if self.closed:
            self.stale = False
            self.closed = False
        return not self.stale

    def _read_status(self):
        self.events.append("read")
        if self.stale:
            return None
        if not self.dead:
            delta = max(-120, min(120, self.target - self.position))
            self.position += delta
        return status(self.position)

    def _set_finger_position(self, position):
        self.events.append("write")
        if self.stale:
            return False
        self.target = position
        return True


@pytest.fixture
def clock(monkeypatch):
    now = [0.]
    monkeypatch.setattr(gctl.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(gctl.time, "sleep", lambda dt: now.__setitem__(0, now[0] + dt))
    return now


@pytest.fixture
def make_ctl(monkeypatch, clock):
    def make(*, settings=None, **port_options):
        fake = ReenumeratedPort(**port_options)
        monkeypatch.setattr(gctl, "KorasGripper", lambda *args, **kwargs: fake)
        cfg = {"gripper": {
            "port": "/dev/serial/by-id/test-gripper", "slave_id": 1,
            "force_pct": 80, "speed_pct": 100, "stroke_mm": 100.,
            "pre_open_margin_mm": 10., "grip_success_min_pct": 5.,
            **(settings or {}),
        }}
        return gctl.GripperCtl(cfg), fake
    return make


def test_stale_usb_descriptor_reopens_and_verifies_requested_width(make_ctl):
    ctl, fake = make_ctl()
    assert ctl.pre_open(.04)
    assert fake.position == 500
    fake.disconnect.assert_called_once_with()
    fake.connect.assert_called_once_with(initialize=False)
    fake.set_finger_position.assert_called_once_with(500)
    assert fake.events.index("disconnect") < fake.events.index("connect") < fake.events.index("write")
    fake.enable.assert_not_called()
    fake.grip_initialize.assert_not_called()
    fake.grip_close.assert_not_called()
    fake.grip_open.assert_not_called()


def test_healthy_pre_open_does_not_reconnect(make_ctl):
    ctl, fake = make_ctl(stale=False)
    assert ctl.pre_open(.04)
    fake.disconnect.assert_not_called()
    fake.connect.assert_not_called()


def test_disabled_recovery_never_reopens_or_blindly_writes(make_ctl):
    ctl, fake = make_ctl(settings={"pre_open_reconnect_timeout_s": 0})
    assert not ctl.pre_open(.04)
    fake.disconnect.assert_not_called()
    fake.connect.assert_not_called()
    fake.set_finger_position.assert_not_called()


@pytest.mark.parametrize("failure", ["absent", "status_missing", "io_exception"])
def test_unavailable_port_stops_at_recovery_budget(make_ctl, clock, failure):
    ctl, fake = make_ctl(settings={"pre_open_reconnect_timeout_s": .35})
    if failure == "absent":
        fake.connect.side_effect = lambda **kwargs: False
    elif failure == "status_missing":
        fake.read_status.side_effect = lambda: None
    else:
        fake.connect.side_effect = OSError(5, "Input/output error")
    assert not ctl.pre_open(.04)
    assert .35 <= clock[0] <= .36
    assert 1 <= fake.connect.call_count <= 5
    fake.set_finger_position.assert_not_called()
    assert all(call.kwargs == {"initialize": False} for call in fake.connect.call_args_list)


def test_port_appearing_during_recovery_can_continue(make_ctl, clock):
    ctl, fake = make_ctl(settings={"pre_open_reconnect_timeout_s": .5})

    def connect(**kwargs):
        if fake.connect.call_count <= 2:
            raise OSError(5, "Input/output error")
        return fake._connect(**kwargs)

    fake.connect.side_effect = connect
    assert ctl.pre_open(.04)
    assert fake.connect.call_count == 3
    assert clock[0] >= .2
    fake.set_finger_position.assert_called_once_with(500)


def test_failed_replay_cannot_start_another_recovery_window(make_ctl):
    ctl, fake = make_ctl()
    recovered_reads = 0

    def read():
        nonlocal recovered_reads
        if fake.stale:
            return None
        recovered_reads += 1
        return status() if recovered_reads == 1 else None

    fake.read_status.side_effect = read
    assert not ctl.pre_open(.04)
    fake.disconnect.assert_called_once_with()
    fake.connect.assert_called_once_with(initialize=False)
    fake.set_finger_position.assert_not_called()


def test_missing_settle_status_recovers_once_then_rechecks_width(make_ctl, monkeypatch):
    ctl, fake = make_ctl(stale=False)
    settle = ctl._wait_settle
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            fake.stale = True
            return False, None
        return settle(**kwargs)

    monkeypatch.setattr(ctl, "_wait_settle", fail_once)
    assert ctl.pre_open(.04)
    assert calls == 2
    assert fake.position == 500
    fake.connect.assert_called_once_with(initialize=False)


def test_target_already_reached_still_recovers_failed_ack(make_ctl):
    ctl, fake = make_ctl(stale=False, position=500)

    def write(position):
        if fake.set_finger_position.call_count == 1:
            fake.stale = True
            return False
        return fake._set_finger_position(position)

    fake.set_finger_position.side_effect = write
    assert ctl.pre_open(.04)
    fake.connect.assert_called_once_with(initialize=False)
    assert fake.set_finger_position.call_count == 2
    assert fake.position == 500


@pytest.mark.parametrize("missing_status", [False, True])
def test_pre_open_lost_writes_reopen_and_verify_width(make_ctl, missing_status):
    ctl, fake = make_ctl(stale=False)

    def write(position):
        if fake.set_finger_position.call_count <= 2:
            fake.stale = missing_status
            return False
        return fake._set_finger_position(position)

    fake.set_finger_position.side_effect = write
    assert ctl.pre_open(.04)
    fake.connect.assert_called_once_with(initialize=False)
    assert fake.set_finger_position.call_count == 3
    assert fake.position == 500


def test_missing_sample_cannot_reclassify_acknowledged_no_motion_as_io(make_ctl):
    ctl, fake = make_ctl(stale=False, dead=True)
    fake.read_status.side_effect = lambda: (
        None if fake.read_status.call_count == 2 else status())
    assert not ctl.pre_open(.04)
    fake.disconnect.assert_not_called()
    fake.connect.assert_not_called()


@pytest.mark.parametrize("states", [0, 3, 513])
@pytest.mark.parametrize("position", [500, 1000])
def test_lost_ack_followed_by_motor_error_is_not_reconnected(make_ctl, states, position):
    ctl, fake = make_ctl(stale=False, position=position)
    fake.read_status.side_effect = lambda: status(
        position, states=states if fake.set_finger_position.call_count else 1)
    fake.set_finger_position.side_effect = lambda position: False
    assert not ctl.pre_open(.04)
    fake.disconnect.assert_not_called()
    fake.connect.assert_not_called()


def test_cached_visual_width_cannot_establish_reconnected_status(make_ctl):
    ctl, fake = make_ctl(stale=False, position=500,
                         settings={"pre_open_reconnect_timeout_s": .2})
    ctl.status()
    assert ctl.visual_status[0] == pytest.approx(.05)
    fake.read_status.side_effect = lambda: None
    assert not ctl.pre_open(.04)
    assert fake.connect.call_count >= 1
    fake.set_finger_position.assert_not_called()


def test_acknowledged_no_motion_never_reopens(make_ctl):
    ctl, fake = make_ctl(stale=False, dead=True)
    assert not ctl.pre_open(.04)
    fake.disconnect.assert_not_called()
    fake.connect.assert_not_called()


@pytest.mark.parametrize("states", [0, 3, 513])
def test_unhealthy_initial_status_never_reopens_or_sends(make_ctl, states):
    ctl, fake = make_ctl(stale=False)
    fake.read_status.side_effect = lambda: status(states=states)
    assert not ctl.pre_open(.04)
    fake.disconnect.assert_not_called()
    fake.connect.assert_not_called()
    fake.set_finger_position.assert_not_called()


@pytest.mark.parametrize("settled,final_status", [
    (False, status(500, states=513)),
    (False, status(700, current=1300)),
    (False, status(500)),
    (True, status(700)),
])
def test_physical_settle_failure_cannot_trigger_reconnect(
        make_ctl, monkeypatch, settled, final_status):
    ctl, fake = make_ctl(stale=False)
    monkeypatch.setattr(ctl, "_wait_settle", lambda **kwargs: (settled, final_status))
    assert not ctl.pre_open(.04)
    fake.disconnect.assert_not_called()
    fake.connect.assert_not_called()


@pytest.mark.parametrize("fresh", [
    status(states=0), status(states=3), status(states=513),
    status(-1), status(1001), status(float("nan")), status(float("inf")),
])
def test_reopened_port_requires_healthy_fresh_status_before_replay(make_ctl, fresh):
    ctl, fake = make_ctl()
    fake.read_status.side_effect = lambda: None if fake.stale else fresh
    assert not ctl.pre_open(.04)
    assert fake.connect.call_count >= 1
    fake.set_finger_position.assert_not_called()
    fake.enable.assert_not_called()
    fake.grip_initialize.assert_not_called()


@pytest.mark.parametrize("boundary", ["disconnect", "connect", "fresh_read"])
def test_camera_abort_at_reconnect_boundaries_prevents_replay(make_ctl, boundary):
    from efg.camera import CameraError

    ctl, fake = make_ctl()
    error = CameraError("camera stopped during recovery")

    def check():
        interrupted = (fake.disconnect.call_count if boundary == "disconnect" else
                       fake.connect.call_count if boundary == "connect" else
                       fake.connect.call_count and fake.events[-1:] == ["read"])
        if interrupted:
            raise error

    with pytest.raises(CameraError, match="camera stopped") as caught:
        ctl.pre_open(.04, check=check)
    assert caught.value is error
    fake.set_finger_position.assert_not_called()
    if boundary == "disconnect":
        fake.connect.assert_not_called()


@pytest.mark.parametrize("slow_call", ["connect", "read"])
def test_status_or_connect_finishing_after_deadline_cannot_replay(make_ctl, clock, slow_call):
    ctl, fake = make_ctl(settings={"pre_open_reconnect_timeout_s": .3})
    original = fake._connect if slow_call == "connect" else fake._read_status

    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        if not fake.stale:
            clock[0] += 1.
        return result

    if slow_call == "connect":
        fake.connect.side_effect = slow
    else:
        fake.read_status.side_effect = slow
    assert not ctl.pre_open(.04)
    assert clock[0] >= 1.
    fake.set_finger_position.assert_not_called()


@pytest.mark.parametrize("operation", ["grip", "release"])
def test_missing_status_during_contact_operations_never_reconnects(make_ctl, operation):
    ctl, fake = make_ctl()
    if operation == "grip":
        with pytest.raises(RuntimeError, match="시작 상태 오류"):
            ctl.grip()
    else:
        assert not ctl.release(.04)
    fake.disconnect.assert_not_called()
    fake.connect.assert_not_called()
    fake.stop.assert_called_once_with()


@pytest.mark.parametrize("value", [-.01, 10.01, float("nan"), float("inf")])
def test_invalid_reconnect_budget_is_rejected(make_ctl, value):
    with pytest.raises(ValueError, match="pre_open_reconnect_timeout_s"):
        make_ctl(settings={"pre_open_reconnect_timeout_s": value})
