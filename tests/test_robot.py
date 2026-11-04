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


def test_tcp_offset_is_removed_for_handeye_and_applied_once_for_commands(ready_robot):
    from efg.se3 import xarm_pose_to_se3, se3_to_xarm_pose, inv_se3
    from efg.transforms import grasp_to_base, tool_pose_from_grasp
    from types import SimpleNamespace
    robot = ready_robot
    # Nonzero translation AND rotation expose wrong multiplication order.
    robot.T_flange_tcp = xarm_pose_to_se3([3, -4, 223, .05, -.02, .1])
    flange = xarm_pose_to_se3([400, -200, 450, 2.7, .2, -.5])
    reported_tcp = flange @ robot.T_flange_tcp
    robot.arm.get_position.return_value = 0, se3_to_xarm_pose(reported_tcp)
    measured_flange, _ = robot.get_flange_pose()
    np.testing.assert_allclose(measured_flange, flange, atol=1e-12)
    handeye = xarm_pose_to_se3([130, -34, -37, .01, -.02, 1.56])
    camera = measured_flange @ handeye
    g = SimpleNamespace(rotation_matrix=np.eye(3), translation=np.array([.1, -.05, .5]))
    R, p = grasp_to_base(camera, g)
    target_tcp = tool_pose_from_grasp(R, p, -1, .035)
    target_flange = target_tcp @ inv_se3(robot.T_flange_tcp)
    robot.arm.set_position.return_value = 0
    assert robot.move_flange(target_flange)
    kwargs = robot.arm.set_position.call_args.kwargs
    actual_command = xarm_pose_to_se3([kwargs[k] for k in ('x','y','z','roll','pitch','yaw')])
    np.testing.assert_allclose(actual_command, target_tcp, atol=1e-12)


@pytest.fixture
def ik_robot(ready_robot):
    ready_robot.T_flange_tcp = np.eye(4)
    ready_robot.T_flange_tcp[2, 3] = .223
    return ready_robot


def test_ik_failure_preserves_sdk_code_and_converted_tcp_target(ik_robot):
    from efg.se3 import xarm_pose_to_se3, se3_to_xarm_pose
    flange = xarm_pose_to_se3([400., -200., 350., .1, -.2, .3])
    target = se3_to_xarm_pose(flange @ ik_robot.T_flange_tcp)
    ik_robot.arm.get_inverse_kinematics.return_value = 10, []

    assert ik_robot.flange_ik_joints_deg(flange) is None

    assert "SDK code=10" in ik_robot.last_ik_failure
    assert f"TCP(mm/rad)={np.round(target, 6).tolist()}" in ik_robot.last_ik_failure
    args, kwargs = ik_robot.arm.get_inverse_kinematics.call_args
    np.testing.assert_allclose(args[0], target)
    assert kwargs == dict(input_is_radian=True, return_is_radian=True)
    ik_robot.arm.clean_error.assert_not_called()
    ik_robot.arm.clean_warn.assert_not_called()


def test_successful_ik_returns_degrees_and_clears_previous_failure(ik_robot):
    ik_robot.arm.get_inverse_kinematics.return_value = 10, []
    ik_robot.flange_ik_joints_deg(np.eye(4))
    assert ik_robot.last_ik_failure is not None
    radians = np.array([0., .1, -.2, .3, -.4, .5, -.6])
    ik_robot.arm.get_inverse_kinematics.return_value = 0, radians

    np.testing.assert_allclose(ik_robot.flange_ik_joints_deg(np.eye(4)), np.degrees(radians))
    assert ik_robot.last_ik_failure is None
    with pytest.raises(AttributeError):
        ik_robot.last_ik_failure = "stale"


@pytest.mark.parametrize("angles", [None, [], [0.] * 6, [0.] * 8,
                                    [0.] * 6 + [float('nan')],
                                    [0.] * 6 + [float('inf')],
                                    [0.] * 6 + ['invalid'], [[0.] * 7]])
def test_invalid_ik_angles_are_rejected_with_diagnostic(ik_robot, angles):
    ik_robot.arm.get_inverse_kinematics.return_value = 0, angles
    assert ik_robot.flange_ik_joints_deg(np.eye(4)) is None
    assert "SDK code=0" in ik_robot.last_ik_failure
    assert "7개 관절각" in ik_robot.last_ik_failure


def test_missing_ik_api_has_a_diagnostic_without_commands(ik_robot):
    ik_robot.arm = Mock(spec=[])
    assert ik_robot.flange_ik_joints_deg(np.eye(4)) is None
    assert "API 미지원" in ik_robot.last_ik_failure
    assert "TCP(mm/rad)" in ik_robot.last_ik_failure
    assert ik_robot.arm.mock_calls == []


@pytest.mark.parametrize("error", [RuntimeError("connection lost"),
                                  AttributeError("SDK internal failure")])
def test_unexpected_ik_exception_propagates_without_stale_diagnostic(ik_robot, error):
    ik_robot.arm.get_inverse_kinematics.return_value = 10, []
    ik_robot.flange_ik_joints_deg(np.eye(4))
    ik_robot.arm.get_inverse_kinematics.side_effect = error
    with pytest.raises(type(error)) as caught:
        ik_robot.flange_ik_joints_deg(np.eye(4))
    assert caught.value is error
    assert ik_robot.last_ik_failure is None
    ik_robot.arm.clean_error.assert_not_called()
