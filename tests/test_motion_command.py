"""Native command validation/preflight without a robot connection."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from efg.camera import CameraError
from efg.motion_command import MotionCheckError, MotionCommand, preflight_commands


def line(**kwargs):
    values = dict(kind='cartesian', target=[400, -200, 300, 3.14, 0, 0],
                  speed=400, acceleration=700, radius=10, label='rise')
    values.update(kwargs)
    return MotionCommand(**values)


def joint(**kwargs):
    values = dict(kind='joint', target=[10, 20, 30, 40, 50, 60, 70],
                  speed=120, acceleration=180, radius=0, label='place')
    values.update(kwargs)
    return MotionCommand(**values)


@pytest.fixture
def arm():
    return SimpleNamespace(version_number=(2, 7, 1), only_check_result=0,
                           set_position=Mock(return_value=0),
                           set_servo_angle=Mock(return_value=0),
                           set_only_check_type=Mock())


def test_targets_are_copied_and_commands_are_immutable():
    target = [0] * 6
    command = line(target=target)
    target[0] = 100
    assert command.target == (0.,) * 6
    with pytest.raises(FrozenInstanceError):
        command.radius = 0
    with pytest.raises(TypeError):
        command.target[0] = 100


def test_cartesian_and_joint_dispatch_preserve_units_and_explicit_execution_type(arm):
    cartesian, joints = line(), joint()
    assert cartesian.submit(arm) == 0
    assert joints.submit(arm) == 0
    arm.set_position.assert_called_once_with(
        400., -200., 300., 3.14, 0., 0., is_radian=True, speed=400.,
        mvacc=700., mvtime=0, radius=10., wait=False, only_check_type=0)
    arm.set_servo_angle.assert_called_once_with(
        angle=[10., 20., 30., 40., 50., 60., 70.], is_radian=False,
        speed=120., mvacc=180., mvtime=0, radius=0., wait=False, only_check_type=0)
    arm.set_only_check_type.assert_not_called()


@pytest.mark.parametrize('changes', [
    {'kind': 'unknown'}, {'target': [0] * 7}, {'target': [0] * 5 + [float('nan')]},
    {'speed': 0}, {'speed': -1}, {'speed': float('inf')},
    {'acceleration': 0}, {'acceleration': float('nan')},
    {'radius': -.1}, {'radius': -2}, {'radius': float('inf')},
])
def test_invalid_command_is_rejected_before_submission(changes):
    with pytest.raises(ValueError):
        line(**changes)


def test_joint_command_requires_seven_finite_values():
    with pytest.raises(ValueError):
        joint(target=[0] * 6)
    with pytest.raises(ValueError):
        joint(target=[0] * 6 + [float('inf')])
    assert joint(radius=-1).radius == -1


@pytest.mark.parametrize('value', [-1, 4, None, '1'])
def test_unknown_check_mode_cannot_accidentally_submit_actual_motion(arm, value):
    with pytest.raises(ValueError):
        line().submit(arm, only_check_type=value)
    arm.set_position.assert_not_called()


@pytest.mark.parametrize('count,expected', [(1, [1]), (2, [1, 2]), (4, [1, 3, 3, 2])])
def test_chain_starts_at_actual_and_ends_with_shadow_reset_without_global_mode(arm, count, expected):
    submitted = []
    arm.set_position.side_effect = lambda *args, **kwargs: submitted.append(kwargs) or 0
    arm.set_servo_angle.side_effect = lambda **kwargs: submitted.append(kwargs) or 0
    commands = [line()] * (count - 1) + [joint()]
    preflight_commands(arm, commands)
    assert [command['only_check_type'] for command in submitted] == expected
    assert all(command['wait'] is False for command in submitted)
    arm.set_only_check_type.assert_not_called()


@pytest.mark.parametrize('version', [None, (1, 11, 99), (1, 5, 20), (0, 0, 0),
                                     '2.7.1', (2, 7), (2, 7, 1.0), (True, 11, 100)])
def test_unsupported_or_unknown_firmware_blocks_all_commands(arm, version):
    arm.version_number = version
    with pytest.raises(MotionCheckError, match='FW 1.11.100'):
        preflight_commands(arm, [line(), joint()])
    arm.set_position.assert_not_called()
    arm.set_servo_angle.assert_not_called()


def test_minimum_supported_firmware_and_nonempty_command_contract(arm):
    arm.version_number = (1, 11, 100)
    preflight_commands(arm, [line(), joint()])
    for commands in ([], [object()]):
        with pytest.raises(ValueError):
            preflight_commands(arm, commands)


@pytest.mark.parametrize('code,result', [(1, 24), (0, 24), (-1, 0), (None, 0), (0, None)])
def test_native_failure_is_captured_before_later_commands_and_never_executes(arm, code, result):
    def reject(*args, **kwargs):
        arm.only_check_result = result
        return code

    arm.set_position.side_effect = reject
    with pytest.raises(MotionCheckError, match='rise.*only_check_result') as error:
        preflight_commands(arm, [line(), joint()])
    assert f'code={code}' in str(error.value)
    assert f'only_check_result={result}' in str(error.value)
    assert arm.set_position.call_args.kwargs['only_check_type'] == 1
    arm.set_servo_angle.assert_not_called()
    arm.set_only_check_type.assert_not_called()


def test_interrupted_chain_never_sends_cleanup_motion_and_next_check_restarts_actual(arm):
    calls = []
    arm.set_position.side_effect = lambda *args, **kwargs: calls.append(kwargs['only_check_type']) or 0
    check = Mock(side_effect=[None, CameraError('camera stopped')])
    with pytest.raises(CameraError, match='camera stopped') as error:
        preflight_commands(arm, [line(), line(), joint()], check=check)
    assert not isinstance(error.value, MotionCheckError)
    assert calls == [1]
    arm.set_servo_angle.assert_not_called()
    preflight_commands(arm, [line(), line(), joint()])
    assert calls == [1, 1, 3]
    assert arm.set_servo_angle.call_args.kwargs['only_check_type'] == 2
    arm.set_only_check_type.assert_not_called()


def test_controller_exception_propagates_without_normal_submission(arm):
    arm.set_position.side_effect = RuntimeError('connection lost')
    with pytest.raises(RuntimeError, match='connection lost') as error:
        preflight_commands(arm, [line(), joint()])
    assert not isinstance(error.value, MotionCheckError)
    arm.set_position.assert_called_once()
    assert arm.set_position.call_args.kwargs['only_check_type'] == 1
    arm.set_servo_angle.assert_not_called()
    arm.set_only_check_type.assert_not_called()


@pytest.mark.parametrize('code,result,expected', [
    (1, 21, True), (1, 22, True), (1, 23, True), (1, 24, True),
    (1, 31, False), (1, 0, False), (-1, 21, False), (0, 21, False),
    (None, 21, False), (1, None, False),
])
def test_only_known_native_results_are_candidate_path_rejections(code, result, expected):
    error = MotionCheckError('test', command_index=0, code=code, result=result)
    assert error.is_path_rejection is expected
    assert not MotionCheckError('unknown', code=code, result=result).is_path_rejection


@pytest.mark.parametrize('acceleration_config,expected_accelerations', [
    ({}, [700., 700.]),
    ({'travel_acc_mm_s2': 1800., 'approach_acc_mm_s2': 1100.}, [1800., 1100.]),
])
def test_shared_approach_compiler_bounds_corner_and_stops_at_grasp(
        acceleration_config, expected_accelerations):
    import numpy as np
    from efg.motion_command import approach_commands
    start, pre, grasp = (np.eye(4) for _ in range(3))
    start[2, 3], pre[2, 3], grasp[2, 3] = .5, .2, .1
    cfg = {'travel_speed_mm_s': 600, 'approach_speed_mm_s': 250,
           'approach_acc_mm_s2': 700, 'pregrasp_blend_radius_mm': 50}
    cfg.update(acceleration_config)
    commands = approach_commands(start, pre, grasp, cfg)
    assert commands[0].radius == pytest.approx(25.)
    assert commands[1].radius == -1
    assert [command.target[2] for command in commands] == [200., 100.]
    assert [command.speed for command in commands] == [600., 250.]
    assert [command.acceleration for command in commands] == expected_accelerations
    assert approach_commands(pre, pre, grasp, cfg)[0].radius == -1


@pytest.mark.parametrize('acceleration', [0., -1., float('nan'), float('inf'), -float('inf')])
def test_invalid_travel_acceleration_blocks_all_native_submissions(arm, acceleration):
    import numpy as np
    from efg.motion_command import approach_commands
    start, pre, grasp = (np.eye(4) for _ in range(3))
    start[2, 3], pre[2, 3], grasp[2, 3] = .5, .2, .1
    cfg = {'travel_speed_mm_s': 600, 'approach_speed_mm_s': 250,
           'travel_acc_mm_s2': acceleration, 'approach_acc_mm_s2': 1100}
    with pytest.raises(ValueError):
        preflight_commands(arm, approach_commands(start, pre, grasp, cfg))
    arm.set_position.assert_not_called()
    arm.set_servo_angle.assert_not_called()
