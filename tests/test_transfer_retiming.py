"""Bounded native overspeed recovery without moving or connecting a robot."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg import motion
from efg.camera import CameraError
from efg.motion_command import MotionCheckError, MotionCommand, preflight_with_retiming
from efg.transfer_plan import prepare_transfer


@pytest.fixture
def commands():
    return (
        MotionCommand('cartesian', (400, -200, 200, 0, 0, 0), 600, 700, 10, 'retreat'),
        MotionCommand('cartesian', (400, -200, 370, 0, 0, 0), 500, 600, 20, 'rise'),
        MotionCommand('joint', (10,) * 7, 120, 200, 0, 'place'),
    )


@pytest.fixture
def pick_commands(commands):
    """Approach, retreat/rise and place are three independently stopped phases."""
    return (
        MotionCommand('cartesian', (400, -200, 200, 0, 0, 0), 900, 1100, 10, 'pregrasp'),
        MotionCommand('cartesian', (400, -200, 100, 0, 0, 0), 500, 1100, -1, 'grasp'),
        commands[0],
        replace(commands[1], radius=-1),
        commands[2],
    )


@pytest.fixture
def arm():
    arm = SimpleNamespace(version_number=(2, 7, 1), only_check_result=0,
                          writes=[], responses=[], set_only_check_type=Mock())

    def submit(kind, *args, **kwargs):
        arm.writes.append((kind, args, kwargs))
        code, result = arm.responses.pop(0) if arm.responses else (0, 0)
        arm.only_check_result = result
        return code

    arm.set_position = Mock(side_effect=lambda *a, **kw: submit('cartesian', *a, **kw))
    arm.set_servo_angle = Mock(side_effect=lambda *a, **kw: submit('joint', *a, **kw))
    return arm


def assert_checks_only(arm):
    assert all(call[2]['only_check_type'] in (1, 2, 3) for call in arm.writes)
    arm.set_only_check_type.assert_not_called()


def test_blended_phase_overspeed_rechecks_entire_chain_with_uniform_limits(arm, commands):
    arm.responses = [(0, 0), (1, 24)]
    guard = Mock()

    prepared, scale = preflight_with_retiming(arm, commands, check_start=guard)

    assert scale == .8
    # First failure leaves a shadow state; the whole retry restarts at actual.
    assert [call[2]['only_check_type'] for call in arm.writes] == [1, 3, 1, 3, 2]
    assert guard.call_count == 3  # Both attempts and the successful final check.
    for before, after in zip(commands, prepared):
        assert after.speed == pytest.approx(before.speed * .8)
        assert after.acceleration == pytest.approx(before.acceleration * .8 ** 2)
        assert (after.kind, after.target, after.radius, after.label) == (
            before.kind, before.target, before.radius, before.label)
    assert commands[0].speed == 600 and commands[0].acceleration == 700
    assert_checks_only(arm)


@pytest.mark.parametrize('stop_radius', [-1, 0])
def test_rise_overspeed_does_not_slow_stopped_approach_or_joint_phases(
        arm, pick_commands, stop_radius):
    original = tuple(replace(command, radius=stop_radius) if index in (1, 3) else command
                     for index, command in enumerate(pick_commands))
    arm.responses = [(0, 0)] * 3 + [(1, 24)]

    prepared, scale = preflight_with_retiming(arm, original, check_start=Mock())

    assert scale == .8
    assert [command.speed for command in prepared] == pytest.approx([900, 500, 480, 400, 120])
    assert [command.acceleration for command in prepared] == pytest.approx(
        [1100, 1100, 448, 384, 200])
    assert [call[2]['only_check_type'] for call in arm.writes] == [1, 3, 3, 3, 1, 3, 3, 3, 2]
    assert prepared[0] is original[0] and prepared[1] is original[1]
    assert prepared[4] is original[4]
    assert original[2].speed == 600 and original[3].speed == 500
    assert_checks_only(arm)


def test_separate_phase_failures_reduce_independently_from_original_limits(arm, pick_commands):
    arm.responses = ([(0, 0)] * 3 + [(1, 24)]  # Rise needs 80%.
                     + [(0, 0)] * 4 + [(1, 24)]  # Joint transfer needs 80%.
                     + [(0, 0)] * 3 + [(1, 24)])  # Rise needs 60%.
    guard = Mock()

    prepared, scale = preflight_with_retiming(arm, pick_commands, check_start=guard)

    assert scale == .6  # Summary minimum; the approach is still at nominal speed.
    assert [command.speed for command in prepared] == pytest.approx([900, 500, 360, 300, 96])
    assert [command.acceleration for command in prepared] == pytest.approx(
        [1100, 1100, 252, 216, 128])
    assert [arm.writes[index][2]['only_check_type'] for index in (0, 4, 9, 13)] == [1] * 4
    assert [arm.writes[index][2]['speed'] for index in (0, 4, 9, 13)] == [900] * 4
    assert guard.call_count == 5
    assert_checks_only(arm)


def test_retry_budget_allows_three_reductions_per_stopped_phase(arm, pick_commands):
    # Give pregrasp and retreat distinct targets for the controller stub below.
    original = (replace(pick_commands[0], target=(400, -200, 210, 0, 0, 0)),
                *pick_commands[1:])
    thresholds = {command.target: command.speed * .5 for command in original}

    def submit(kind, target, **kwargs):
        arm.writes.append((kind, target, kwargs))
        arm.only_check_result = 24 if kwargs['speed'] > thresholds[tuple(target)] else 0
        return 1 if arm.only_check_result else 0

    arm.set_position.side_effect = lambda *target, **kw: submit('cartesian', target, **kw)
    arm.set_servo_angle.side_effect = lambda *, angle, **kw: submit('joint', angle, **kw)
    guard = Mock()

    prepared, scale = preflight_with_retiming(arm, original, check_start=guard)

    assert scale == .5
    assert sum(call[2]['only_check_type'] == 1 for call in arm.writes) == 10
    assert guard.call_count == 11
    for before, after in zip(original, prepared):
        assert after.speed == pytest.approx(before.speed * .5)
        assert after.acceleration == pytest.approx(before.acceleration * .25)
    assert_checks_only(arm)


def test_no_scaling_or_extra_check_sequence_when_nominal_commands_pass(arm, commands):
    prepared, scale = preflight_with_retiming(arm, commands, check_start=Mock())
    assert prepared == commands and scale == 1.
    assert [call[2]['only_check_type'] for call in arm.writes] == [1, 3, 2]
    assert_checks_only(arm)


@pytest.mark.parametrize('code,result', [(1, 21), (1, 23), (1, 0), (-1, 24),
                                        (9, 24), (0, 24), (None, 24), (0, None)])
def test_only_native_overspeed_rejection_allows_retiming(arm, commands, code, result):
    arm.responses = [(code, result)]
    with pytest.raises(MotionCheckError) as caught:
        preflight_with_retiming(arm, commands, check_start=Mock())
    assert caught.value.command_index == 0
    assert caught.value.code == code and caught.value.result == result
    assert not caught.value.is_overspeed
    assert len(arm.writes) == 1
    assert_checks_only(arm)


def test_overspeed_budget_is_bounded_and_scales_from_original_not_previous_try(arm, commands):
    arm.responses = [(1, 24)] * 4
    with pytest.raises(MotionCheckError) as caught:
        preflight_with_retiming(arm, commands, check_start=Mock())
    assert caught.value.is_overspeed
    assert caught.value.command_index == 0
    assert caught.value.speed_scale == .5
    assert [call[2]['speed'] for call in arm.writes] == pytest.approx([600, 480, 360, 300])
    assert [call[2]['mvacc'] for call in arm.writes] == pytest.approx([700, 448, 252, 175])
    assert [call[2]['only_check_type'] for call in arm.writes] == [1, 1, 1, 1]
    assert_checks_only(arm)


def test_a_new_non_speed_failure_ends_retiming_immediately(arm, commands):
    arm.responses = [(1, 24), (0, 0), (1, 21)]
    with pytest.raises(MotionCheckError) as caught:
        preflight_with_retiming(arm, commands, check_start=Mock())
    assert caught.value.command_index == 1 and caught.value.result == 21
    assert caught.value.speed_scale == .8
    assert len(arm.writes) == 3
    assert_checks_only(arm)


def test_error_reports_rejected_phase_scale_not_an_unrelated_reduction(arm, pick_commands):
    arm.responses = ([(0, 0)] * 3 + [(1, 24)]
                     + [(0, 0)] * 4 + [(1, 21)])
    with pytest.raises(MotionCheckError) as caught:
        preflight_with_retiming(arm, pick_commands, check_start=Mock())
    assert caught.value.command_index == 4 and caught.value.result == 21
    assert caught.value.speed_scale == 1.  # Only retreat/rise had been reduced.
    assert len(arm.writes) == 9
    assert_checks_only(arm)


@pytest.mark.parametrize('fail_after_success', [False, True])
def test_changed_start_never_returns_or_checks_another_plan(arm, commands, fail_after_success):
    if not fail_after_success:
        arm.responses = [(1, 24)]
    changed = RuntimeError('measured start changed')
    guard = Mock(side_effect=[None, changed])
    with pytest.raises(RuntimeError) as caught:
        preflight_with_retiming(arm, commands, check_start=guard)
    assert caught.value is changed
    assert len(arm.writes) == (3 if fail_after_success else 1)
    assert_checks_only(arm)


def test_camera_failure_between_attempts_cancels_retiming(arm, commands):
    arm.responses = [(1, 24)]

    def camera_check():
        if arm.writes:
            raise CameraError('camera disconnected after check rejection')

    with pytest.raises(CameraError):
        preflight_with_retiming(arm, commands, check_start=Mock(), check=camera_check)
    assert len(arm.writes) == 1
    assert_checks_only(arm)


@pytest.fixture
def scene(arm, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(motion.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(motion.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(motion.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0] + dt))
    start = np.eye(4)
    start[:3, 3] = [.4, -.2, .1]
    pre, lift = start.copy(), start.copy()
    pre[2, 3], lift[2, 3] = .2, .3
    arm.mode = 0
    arm.get_state = Mock(return_value=(0, 2))
    arm.get_cmdnum = Mock(return_value=(0, 0))
    arm.get_err_warn_code = Mock(return_value=(0, [0, 0]))
    arm.get_servo_angle = Mock(return_value=(0, [10.] * 7))
    robot = SimpleNamespace(arm=arm, T_flange_tcp=np.eye(4), R_home_tcp=np.eye(3),
                            get_tcp_pose=Mock(return_value=(start, None)),
                            get_joints_deg=Mock(return_value=[0.] * 7),
                            flange_ik_joints_deg=Mock(return_value=[0.] * 7),
                            stop_motion=Mock())
    cfg = {'lift_speed_mm_s': 600, 'retreat_speed_mm_s': 600,
           'lift_acc_mm_s2': 700, 'reorient_speed_mm_s': 600,
           'reorient_angular_speed_deg_s': 180, 'reorient_acc_mm_s2': 700,
           'retreat_blend_radius_mm': 10, 'rise_ik_step_mm': 20, 'rise_ik_step_deg': 5}
    drop = {'joints_deg': [10.] * 7, 'joint_speed_deg_s': 120,
            'joint_acc_deg_s2': 200, 'via_height_m': .35,
            'transfer_clearance_margin_mm': 10, 'transfer_blend_radius_mm': 20,
            'transfer_timeout_s': .3, 'arrival_confirm_s': .04}
    return SimpleNamespace(robot=robot, arm=arm, pre=pre, lift=lift, cfg=cfg, drop=drop)


def prepare(scene, **kwargs):
    return prepare_transfer(scene.robot, scene.pre, scene.lift, scene.cfg, scene.drop,
                            j7_limits=(-89., 90.), **kwargs)


def test_execution_uses_the_complete_accepted_scaled_sequence(scene):
    s = scene
    s.arm.responses = [(1, 24)]
    plan = prepare(s)
    assert plan.speed_scale == .8
    assert_checks_only(s.arm)
    accepted = s.arm.writes[-3:]
    s.arm.writes.clear()
    # Later config changes cannot replace the already checked limits.
    s.cfg['retreat_speed_mm_s'] = 999
    s.drop['joint_speed_deg_s'] = 179

    assert motion.lift_and_transfer_blended(s.robot, plan, s.cfg, s.drop)

    assert len(s.arm.writes) == 3
    for checked, actual in zip(accepted, s.arm.writes):
        assert checked[:2] == actual[:2]
        assert actual[2]['only_check_type'] == 0
        assert {k: v for k, v in actual[2].items() if k != 'only_check_type'} == {
            k: v for k, v in checked[2].items() if k != 'only_check_type'}
    s.robot.stop_motion.assert_not_called()


def test_exhausted_common_retreat_does_not_try_an_unrelated_upper_symmetry(scene):
    s = scene
    s.arm.responses = [(1, 24)] * 8
    with pytest.raises(RuntimeError, match='only_check_result=24'):
        prepare(s, allow_symmetry=True)
    assert len(s.arm.writes) == 4
    # All requested IK targets belong to the preferred orientation; switching
    # only the later upper orientation cannot change the rejected first leg.
    assert all(call.args[0][0, 0] > 0 for call in s.robot.flange_ik_joints_deg.call_args_list)
    assert_checks_only(s.arm)
    s.robot.stop_motion.assert_not_called()


def test_later_geometric_rejection_can_choose_other_upper_without_retiming(scene):
    s = scene
    s.arm.responses = [(0, 0), (1, 21)]
    plan = prepare(s, allow_symmetry=True)
    assert plan.transfer_sign == -1 and plan.speed_scale == 1.
    assert [call[2]['only_check_type'] for call in s.arm.writes] == [1, 3, 1, 3, 2]
    assert s.arm.writes[0][2]['speed'] == s.arm.writes[2][2]['speed'] == 600
    assert_checks_only(s.arm)


@pytest.mark.parametrize('changed', ['tcp', 'joints'])
def test_measured_start_drift_blocks_scaled_retry_even_when_state_stays_idle(scene, changed):
    s = scene
    native_submit = s.arm.set_position.side_effect

    def reject_then_drift(*args, **kwargs):
        code = native_submit(*args, **kwargs)
        if changed == 'tcp':
            pose = s.robot.get_tcp_pose.return_value[0].copy()
            pose[0, 3] += .003
            s.robot.get_tcp_pose.return_value = pose, None
        else:
            s.robot.get_joints_deg.return_value = [1.] * 7
        return code

    s.arm.responses = [(1, 24)]
    s.arm.set_position.side_effect = reject_then_drift
    with pytest.raises(RuntimeError, match='시작 자세가 변경됨'):
        prepare(s, allow_symmetry=True)
    assert len(s.arm.writes) == 1
    assert_checks_only(s.arm)


@pytest.mark.parametrize('state,error,pending,mode', [(4, 24, 0, 0), (2, 24, 0, 0),
                                                    (2, 0, 1, 0), (2, 0, 0, 1)])
def test_actual_controller_fault_or_motion_prevents_any_scaled_retry(scene, state, error, pending, mode):
    s = scene
    native_submit = s.arm.set_position.side_effect

    def reject_then_change_state(*args, **kwargs):
        result = native_submit(*args, **kwargs)
        s.arm.get_state.return_value = 0, state
        s.arm.get_err_warn_code.return_value = 0, [error, 0]
        s.arm.get_cmdnum.return_value = 0, pending
        s.arm.mode = mode
        return result

    s.arm.responses = [(1, 24)]
    s.arm.set_position.side_effect = reject_then_change_state
    with pytest.raises(RuntimeError, match='정지 확인 실패'):
        prepare(s, allow_symmetry=True)
    assert len(s.arm.writes) == 1
    assert_checks_only(s.arm)

