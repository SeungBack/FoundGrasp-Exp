"""Controller-checked continuous transfer; all robot I/O is simulated."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg import motion
from efg.camera import CameraError
from efg.transfer_plan import prepare_transfer, transfer_waypoint


@pytest.fixture
def scene(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(motion.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(motion.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(motion.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0] + dt))
    start = np.eye(4)
    start[:3, 3] = [.4, -.2, .1]
    pre, lift = start.copy(), start.copy()
    pre[2, 3], lift[2, 3] = .2, .25
    robot = Mock()
    robot.T_flange_tcp = np.eye(4)
    robot.R_home_tcp = np.eye(3)
    robot.get_tcp_pose.return_value = start, None
    robot.get_joints_deg.return_value = [0.] * 7
    robot.flange_ik_joints_deg.return_value = [0.] * 7
    robot.arm.version_number = [2, 7, 1]
    robot.arm.only_check_result = 0
    robot.arm.mode = 0
    robot.arm.get_state.return_value = 0, 2
    robot.arm.get_cmdnum.return_value = 0, 0
    robot.arm.get_err_warn_code.return_value = 0, [0, 0]
    robot.arm.get_servo_angle.return_value = 0, [10.] * 7
    robot.arm.get_position.return_value = 0, [400., -200., 350., 0., 0., 0.]
    writes = []

    def write(kind, *args, **kwargs):
        writes.append((kind, args, kwargs))
        return 0

    robot.arm.set_position.side_effect = lambda *a, **kw: write('cartesian', *a, **kw)
    robot.arm.set_servo_angle.side_effect = lambda *a, **kw: write('joint', *a, **kw)
    cfg = {'lift_speed_mm_s': 600, 'retreat_speed_mm_s': 600,
           'lift_acc_mm_s2': 700, 'reorient_speed_mm_s': 600,
           'reorient_angular_speed_deg_s': 180, 'reorient_acc_mm_s2': 700,
           'retreat_blend_radius_mm': 10, 'rise_ik_step_mm': 20,
           'rise_ik_step_deg': 5}
    drop = {'joints_deg': [10.] * 7, 'joint_speed_deg_s': 120,
            'joint_acc_deg_s2': 180, 'via_height_m': .35,
            'transfer_clearance_margin_mm': 10, 'transfer_blend_radius_mm': 20,
            'transfer_timeout_s': .3, 'transfer_confirm_s': .1,
            'arrival_confirm_s': .04}
    return SimpleNamespace(robot=robot, start=start, pre=pre, lift=lift,
                           cfg=cfg, drop=drop, clock=clock, writes=writes)


def prepare(s, **kwargs):
    return prepare_transfer(s.robot, s.pre, s.lift, s.cfg, s.drop,
                            j7_limits=(-89., 90.), **kwargs)


def actual_writes(s):
    return [write for write in s.writes if not write[2].get('only_check_type', 0)]


def run(s, plan, **kwargs):
    return motion.lift_and_transfer_blended(s.robot, plan, s.cfg, s.drop, **kwargs)


def test_preflight_and_execution_submit_identical_motion_parameters(scene, monkeypatch):
    s = scene
    plan = prepare(s)
    checked = list(s.writes)
    assert plan.continuous
    assert [call[2]['only_check_type'] for call in checked] == [1, 3, 2]
    assert actual_writes(s) == []
    # Enabling a global checking mode would risk leaking it to later commands.
    s.robot.arm.set_only_check_type.assert_not_called()
    monkeypatch.setattr(motion, 'wait_tcp_stopped', Mock(side_effect=AssertionError('upper stop')))
    assert run(s, plan)
    executed = actual_writes(s)
    assert len(executed) == 3
    for preflight, real in zip(checked, executed):
        assert preflight[:2] == real[:2]
        strip_check = lambda kw: {key: value for key, value in kw.items() if key != 'only_check_type'}
        assert strip_check(preflight[2]) == strip_check(real[2])
    assert [call[0] for call in executed] == ['cartesian', 'cartesian', 'joint']
    assert executed[1][2]['radius'] > 0
    assert executed[2][2]['radius'] == 0
    s.robot.stop_motion.assert_not_called()


@pytest.mark.parametrize('intermediate', [False, True])
def test_all_segments_are_queued_before_arrival_polling(scene, monkeypatch, intermediate):
    s = scene
    plan = prepare(s)
    if intermediate:
        from dataclasses import replace
        from efg.se3 import se3_to_xarm_pose
        middle = (plan.pre + plan.upper) / 2
        extra = replace(plan.commands[0], target=tuple(se3_to_xarm_pose(middle)),
                        radius=5., label='상승 중간 경유점')
        plan = replace(plan, commands=(plan.commands[0], extra, *plan.commands[1:]))
    total = len(plan.commands)
    polls = []

    def state():
        count = len(actual_writes(s))
        polls.append(count)
        assert count in (0, total)  # The initial stationary-state guard is permitted.
        return 0, 2

    s.robot.arm.get_state.side_effect = state
    monkeypatch.setattr(motion, 'wait_tcp_stopped', Mock(side_effect=AssertionError('upper stop')))
    assert run(s, plan)
    assert total in polls
    assert s.clock[0] >= s.drop['arrival_confirm_s']


def test_continuous_metrics_report_combined_motion_without_upper_stop(scene):
    s = scene
    plan = prepare(s)
    metrics, stages = {}, []
    assert run(s, plan, metrics=metrics, on_stage=stages.append)
    assert stages == ['Retreat / rise / transfer']
    assert metrics['continuous_transfer_s'] == pytest.approx(s.clock[0])
    assert metrics['arrival_confirm_s'] >= s.drop['arrival_confirm_s']
    assert 'upper_confirm_s' not in metrics and 'retreat_rise_s' not in metrics


def test_config_changes_cannot_change_commands_after_preflight(scene):
    s = scene
    plan = prepare(s)
    checked = list(s.writes)
    s.cfg.update(lift_speed_mm_s=999, retreat_speed_mm_s=999, lift_acc_mm_s2=999)
    s.drop.update(joint_speed_deg_s=999, joint_acc_deg_s2=999, joints_deg=[50.] * 7,
                  transfer_blend_radius_mm=0)
    assert run(s, plan)
    for checked_call, real_call in zip(checked, actual_writes(s)):
        assert checked_call[:2] == real_call[:2]
        assert {k: v for k, v in checked_call[2].items() if k != 'only_check_type'} == {
            k: v for k, v in real_call[2].items() if k != 'only_check_type'}


@pytest.mark.parametrize('state,pending,offset', [(0, 0, 0.), (1, 0, 0.), (2, 1, 0.), (2, 0, .6)])
def test_final_arrival_requires_stopped_empty_queue_and_joint_target(scene, state, pending, offset):
    s = scene
    plan = prepare(s)
    s.robot.arm.get_state.side_effect = lambda: (0, state if len(actual_writes(s)) == 3 else 2)
    s.robot.arm.get_cmdnum.side_effect = lambda: (0, pending if len(actual_writes(s)) == 3 else 0)
    s.robot.arm.get_servo_angle.return_value = 0, [10. + offset] * 7
    assert not run(s, plan)
    assert len(actual_writes(s)) == 3
    s.robot.stop_motion.assert_called_once()


@pytest.mark.parametrize('fail_at', [1, 2, 3])
def test_partial_enqueue_failure_never_attempts_another_route(scene, fail_at):
    s = scene
    plan = prepare(s)
    s.robot.flange_ik_joints_deg.reset_mock()
    s.writes.clear()

    def write(kind, *args, **kwargs):
        s.writes.append((kind, args, kwargs))
        return 9 if len(actual_writes(s)) == fail_at else 0

    s.robot.arm.set_position.side_effect = lambda *a, **kw: write('cartesian', *a, **kw)
    s.robot.arm.set_servo_angle.side_effect = lambda *a, **kw: write('joint', *a, **kw)
    assert not run(s, plan)
    assert len(actual_writes(s)) == fail_at
    s.robot.flange_ik_joints_deg.assert_not_called()
    s.robot.arm.clean_error.assert_not_called()
    s.robot.stop_motion.assert_called_once()


@pytest.mark.parametrize('after', [1, 2, 3])
def test_camera_failure_cancels_partial_or_complete_queue(scene, after):
    s = scene
    plan = prepare(s)

    def check():
        if len(actual_writes(s)) >= after:
            raise CameraError('camera failed while queueing transfer')

    with pytest.raises(CameraError):
        run(s, plan, check=check)
    assert len(actual_writes(s)) == after
    s.robot.stop_motion.assert_called_once()


def test_controller_fault_after_enqueue_stops_without_clearing_error(scene):
    s = scene
    plan = prepare(s)
    s.robot.arm.get_state.side_effect = lambda: (0, 4 if len(actual_writes(s)) == 3 else 2)
    s.robot.arm.get_err_warn_code.side_effect = lambda: (
        0, [24, 0] if len(actual_writes(s)) == 3 else [0, 0])
    assert not run(s, plan)
    assert len(actual_writes(s)) == 3
    s.robot.stop_motion.assert_called_once()
    s.robot.arm.clean_error.assert_not_called()


def test_actual_j7_limit_stops_continuous_transfer_before_release(scene):
    s = scene
    plan = prepare(s)
    s.robot.arm.get_servo_angle.return_value = 0, [10.] * 6 + [91.]
    assert not run(s, plan)
    assert s.clock[0] < s.drop['transfer_timeout_s']
    s.robot.stop_motion.assert_called_once()


@pytest.mark.parametrize('height', [.25, .4])
def test_blend_support_height_preserves_configured_clearance(scene, height):
    s = scene
    s.lift[2, 3] = height
    plan = prepare(s)
    clearance = max(height, s.drop['via_height_m'], s.pre[2, 3] + .05)
    assert plan.upper[2, 3] - plan.commands[1].radius / 1000 >= clearance - 1e-10
    np.testing.assert_array_equal(plan.upper[:2, 3], s.pre[:2, 3])
    # The support waypoint itself is above the clearance plane, not lowered
    # to manufacture an earlier transition through the box rim.
    assert plan.upper[2, 3] == pytest.approx(clearance + .020)


@pytest.mark.parametrize('failure', ['sdk_code', 'controller_result', 'firmware'])
def test_failed_preflight_never_sends_real_motion(scene, failure):
    s = scene
    if failure == 'sdk_code':
        s.robot.arm.set_position.return_value = 9
        s.robot.arm.set_position.side_effect = None
    elif failure == 'controller_result':
        s.robot.arm.only_check_result = 24
    else:
        s.robot.arm.version_number = [1, 11, 99]
    with pytest.raises(RuntimeError):
        prepare(s)
    assert actual_writes(s) == []


def test_camera_failure_during_preflight_does_not_start_execution(scene):
    s = scene

    def check():
        if s.writes:
            raise CameraError('camera failed during preflight')

    with pytest.raises(CameraError):
        prepare(s, check=check, allow_symmetry=True)
    assert len(s.writes) == 1
    assert actual_writes(s) == []


def test_disabled_blend_keeps_existing_upper_waypoint(scene):
    s = scene
    s.drop['transfer_blend_radius_mm'] = 0
    upper, _ = transfer_waypoint(s.pre, s.lift, s.drop, s.robot.R_home_tcp)
    assert upper[2, 3] == pytest.approx(.360)
    plan = prepare(s)
    assert not plan.continuous
    assert [call[2]['only_check_type'] for call in s.writes] == [1, 3, 2]
    assert actual_writes(s) == []
