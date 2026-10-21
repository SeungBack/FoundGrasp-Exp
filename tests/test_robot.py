"""Readiness checks without a robot connection."""
from unittest.mock import Mock

import numpy as np
import pytest

from efg import robot as robot_module


@pytest.fixture
def ready_robot(monkeypatch):
    robot = object.__new__(robot_module.XArm7)
    robot._motion_initialized = False
    robot.arm = Mock(mode=0, motor_enable_states=[1] * 7 + [0])
    robot.arm.get_state.return_value = (0, 2)
    robot.arm.get_err_warn_code.return_value = (0, [0, 0])
    robot.arm.get_cmdnum.return_value = (0, 0)
    robot.arm.motion_enable.return_value = 0
    robot.arm.set_mode.return_value = 0
    robot.arm.set_state.return_value = 0
    monkeypatch.setattr(robot_module.time, "sleep", Mock())
    return robot


def test_repeated_enable_checks_readiness_without_reset_or_delay(ready_robot):
    robot = ready_robot
    for _ in range(3):
        robot.enable_motion()
    robot.arm.motion_enable.assert_called_once_with(enable=True)
    robot.arm.set_mode.assert_called_once_with(0)
    robot.arm.set_state.assert_called_once_with(0)
    robot_module.time.sleep.assert_called_once_with(.5)
    assert robot.arm.get_state.call_count == 2


@pytest.mark.parametrize("failure", ["moving", "paused", "stopped", "error",
                                     "state_read", "error_read", "queue_read",
                                     "pending", "mode", "disabled", "missing_motors"])
def test_changed_state_is_not_automatically_reenabled(ready_robot, failure):
    robot = ready_robot
    robot.enable_motion()
    robot.arm.reset_mock()
    robot_module.time.sleep.reset_mock()
    if failure in ("moving", "paused", "stopped"):
        robot.arm.get_state.return_value = (0, {"moving": 1, "paused": 3, "stopped": 4}[failure])
    elif failure == "error":
        robot.arm.get_err_warn_code.return_value = (0, [24, 0])
    elif failure.endswith("_read"):
        method = {"state_read": "get_state", "error_read": "get_err_warn_code",
                  "queue_read": "get_cmdnum"}[failure]
        getattr(robot.arm, method).return_value = (1, 0)
    elif failure == "pending":
        robot.arm.get_cmdnum.return_value = (0, 1)
    elif failure == "mode":
        robot.arm.mode = 1
    else:
        robot.arm.motor_enable_states = [1] * 6 if failure == "missing_motors" else [0] + [1] * 6
    with pytest.raises(RuntimeError, match="준비 상태 변경"):
        robot.enable_motion()
    robot.arm.motion_enable.assert_not_called()
    robot.arm.set_mode.assert_not_called()
    robot.arm.set_state.assert_not_called()
    robot.arm.clean_error.assert_not_called()
    robot_module.time.sleep.assert_not_called()


@pytest.mark.parametrize("command", ["motion_enable", "set_mode", "set_state"])
def test_activation_command_failure_is_not_cached_as_ready(ready_robot, command):
    getattr(ready_robot.arm, command).return_value = 9
    with pytest.raises(RuntimeError, match="활성화 실패"):
        ready_robot.enable_motion()
    assert not ready_robot._motion_initialized
    robot_module.time.sleep.assert_not_called()


@pytest.mark.parametrize('code,angles', [(1, [0.] * 7), (0, [0.] * 6),
                                       (0, [0.] * 6 + [float('nan')])])
def test_joint_read_failure_cannot_supply_symmetry_cost(ready_robot, code, angles):
    ready_robot.arm.get_servo_angle.return_value = code, angles
    with pytest.raises(RuntimeError, match='관절 조회 실패'):
        ready_robot.get_joints_deg()


def test_joint_read_uses_actual_degrees(ready_robot):
    ready_robot.arm.get_servo_angle.return_value = 0, [10.] * 7
    np.testing.assert_array_equal(ready_robot.get_joints_deg(), [10.] * 7)
    ready_robot.arm.get_servo_angle.assert_called_once_with(is_radian=False)
