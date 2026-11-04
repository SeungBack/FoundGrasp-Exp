"""Measured TCP setup without connecting to or moving a robot."""
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

import numpy as np
import pytest

from efg import motion
from efg import robot as robot_module
from efg.se3 import xarm_pose_to_se3


@pytest.fixture
def scene(monkeypatch):
    robot = object.__new__(robot_module.XArm7)
    robot.arm = Mock(mode=0)
    robot.arm.get_state.return_value = 0, 2
    robot.arm.get_cmdnum.return_value = 0, 0
    robot.arm.get_err_warn_code.return_value = 0, [0, 0]
    robot.arm.set_tcp_offset.return_value = 0
    old = [0., 0., 223., 0., 0., 0.]
    robot.tcp_offset = old.copy()
    robot.T_flange_tcp = xarm_pose_to_se3(old)
    report = PropertyMock(return_value=old)
    type(robot.arm).tcp_offset = report
    clock = [0.]
    monkeypatch.setattr(robot_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(robot_module.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    return SimpleNamespace(robot=robot, arm=robot.arm, old=old, report=report, clock=clock)


def assert_no_motion(arm):
    for name in ("set_position", "set_servo_angle", "motion_enable", "set_state",
                 "set_mode", "clean_error", "clean_warn", "save_conf"):
        getattr(arm, name).assert_not_called()


@pytest.mark.parametrize("requested", [[0, 0, 203, 0, 0, 0], [3, -4, 203, 10, -20, 30]])
def test_measured_offset_waits_for_report_and_updates_both_frames(scene, requested):
    s = scene
    radians = np.asarray(requested, dtype=float)
    radians[3:] = np.radians(radians[3:])
    s.report.side_effect = [s.old, s.old, s.old, radians]
    s.robot.configure_tcp_offset(requested)
    s.arm.set_tcp_offset.assert_called_once_with(requested, is_radian=False, wait=False)
    assert s.clock[0] >= .04
    np.testing.assert_allclose(s.robot.tcp_offset, radians)
    np.testing.assert_allclose(s.robot.T_flange_tcp, xarm_pose_to_se3(radians))
    flange = xarm_pose_to_se3([400, -200, 350, .1, -.2, .3])
    from efg.se3 import se3_to_xarm_pose
    s.arm.get_position.return_value = 0, se3_to_xarm_pose(flange @ s.robot.T_flange_tcp)
    measured, _ = s.robot.get_flange_pose()
    np.testing.assert_allclose(measured, flange)
    assert_no_motion(s.arm)


def test_matching_report_needs_no_setting_command(scene):
    s = scene
    actual = [0., 0., 203., 0., 0., 0.]
    s.report.return_value = actual
    s.robot.configure_tcp_offset(actual)
    s.arm.set_tcp_offset.assert_not_called()
    assert s.robot.tcp_offset == actual
    assert s.robot.T_flange_tcp[2, 3] == .203
    assert_no_motion(s.arm)


@pytest.mark.parametrize("requested", [None, [], [0] * 5, [0] * 7,
                                        [0, 0, float('nan'), 0, 0, 0],
                                        [0, 0, 203, 0, float('inf'), 0], ['bad'] * 6])
def test_invalid_configuration_performs_no_sdk_calls(scene, requested):
    with pytest.raises(ValueError, match="유한한 6개"):
        scene.robot.configure_tcp_offset(requested)
    assert scene.arm.mock_calls == []


@pytest.mark.parametrize("failure", ["moving", "paused", "ready_unknown", "queue", "mode",
                                      "error", "state_read", "queue_read", "error_read"])
def test_nonidle_or_unreadable_start_never_sets_tool_frame(scene, failure):
    s = scene
    if failure in ("moving", "paused", "ready_unknown"):
        s.arm.get_state.return_value = 0, {"moving": 1, "paused": 3, "ready_unknown": 0}[failure]
    elif failure == "queue":
        s.arm.get_cmdnum.return_value = 0, 1
    elif failure == "mode":
        s.arm.mode = 1
    elif failure == "error":
        s.arm.get_err_warn_code.return_value = 0, [31, 0]
    else:
        method = {"state_read": "get_state", "queue_read": "get_cmdnum",
                  "error_read": "get_err_warn_code"}[failure]
        getattr(s.arm, method).return_value = 9, None
    with pytest.raises(RuntimeError, match="정지 상태 확인 실패"):
        s.robot.configure_tcp_offset([0, 0, 203, 0, 0, 0])
    s.arm.set_tcp_offset.assert_not_called()
    assert s.robot.tcp_offset == s.old
    assert_no_motion(s.arm)


@pytest.mark.parametrize("state", [2, 4, 5])
def test_idle_stopped_or_configuration_state_can_apply_tool_frame(scene, state):
    s = scene
    s.arm.get_state.return_value = 0, state
    s.report.side_effect = [s.old, [0, 0, 203, 0, 0, 0]]
    s.robot.configure_tcp_offset([0, 0, 203, 0, 0, 0])
    assert s.robot.tcp_offset[2] == 203
    assert_no_motion(s.arm)


@pytest.mark.parametrize("failure", ["rejected", "stale", "bad_report", "fault_after_set"])
def test_failed_ack_or_readback_keeps_previous_wrapper_frame(scene, failure):
    s = scene
    if failure == "rejected":
        s.arm.set_tcp_offset.return_value = 1
    elif failure == "bad_report":
        s.report.side_effect = [s.old, [0, 0, float('nan'), 0, 0, 0]]
    elif failure == "fault_after_set":
        def fail(*args, **kwargs):
            s.arm.get_err_warn_code.return_value = 0, [31, 0]
            return 0
        s.arm.set_tcp_offset.side_effect = fail
    with pytest.raises(RuntimeError):
        s.robot.configure_tcp_offset([0, 0, 203, 0, 0, 0])
    assert s.robot.tcp_offset == s.old
    np.testing.assert_array_equal(s.robot.T_flange_tcp, xarm_pose_to_se3(s.old))
    assert s.arm.set_tcp_offset.call_count == 1
    assert_no_motion(s.arm)
    if failure == "stale":
        assert 2 <= s.clock[0] < 2.1


def test_connect_applies_tcp_before_payload_without_motion(monkeypatch):
    robot = Mock()
    factory = Mock(return_value=robot)
    monkeypatch.setattr(motion, "XArm7", factory)
    offset = [0, 0, 203, 0, 0, 0]
    cfg = {"robot": {"ip": "test", "tcp_offset_mm_deg": offset,
                     "tcp_payload": {"weight_kg": 1.9, "com_mm": [7, 1, 48]}}}
    assert motion.connect_robot(cfg) is robot
    assert [call[0] for call in robot.mock_calls] == ["configure_tcp_offset", "set_payload"]
    robot.configure_tcp_offset.assert_called_once_with(offset)
    robot.disconnect.assert_not_called()


def test_omitted_tcp_keeps_studio_configuration(monkeypatch):
    robot = Mock()
    monkeypatch.setattr(motion, "XArm7", Mock(return_value=robot))
    assert motion.connect_robot({"robot": {"ip": "test"}}) is robot
    robot.configure_tcp_offset.assert_not_called()
    assert robot.mock_calls == []


@pytest.mark.parametrize("disconnect_fails", [False, True])
def test_configuration_failure_disconnects_without_hiding_original(monkeypatch, disconnect_fails):
    robot = Mock()
    error = RuntimeError("TCP report timeout")
    robot.configure_tcp_offset.side_effect = error
    if disconnect_fails:
        robot.disconnect.side_effect = OSError("closed")
    monkeypatch.setattr(motion, "XArm7", Mock(return_value=robot))
    with pytest.raises(RuntimeError) as caught:
        motion.connect_robot({"robot": {"ip": "test", "tcp_offset_mm_deg": [0, 0, 203, 0, 0, 0]}})
    assert caught.value is error
    robot.disconnect.assert_called_once()
    robot.set_payload.assert_not_called()
    robot.enable_motion.assert_not_called()
