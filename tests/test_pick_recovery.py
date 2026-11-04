"""Checked open-gripper reverse approach; no hardware is connected."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg import motion
from efg.camera import CameraError
from efg.se3 import xarm_pose_to_se3


@pytest.fixture
def scene(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(motion.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(motion.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(motion.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0] + dt))
    scan = xarm_pose_to_se3([350., -150., 400., np.pi, 0., 0.])
    grasp = xarm_pose_to_se3([400., -200., 50., np.pi, .2, .1])
    pre = grasp.copy()
    pre[:3, 3] -= .1 * pre[:3, 2]
    cfg = dict(travel_speed_mm_s=660., approach_speed_mm_s=400., retreat_speed_mm_s=600.,
               approach_acc_mm_s2=700., lift_acc_mm_s2=350., pregrasp_blend_radius_mm=10.,
               approach_confirm_s=.04, approach_timeout_s=.3)
    plan = SimpleNamespace(start_tcp=scan, pre=pre, grasp=grasp,
                           approach_commands=motion.approach_commands(scan, pre, grasp, cfg))
    arm = SimpleNamespace(version_number=[2, 7, 1], mode=0, only_check_result=0,
                          get_state=Mock(return_value=(0, 2)),
                          get_err_warn_code=Mock(return_value=(0, [0, 0])),
                          get_cmdnum=Mock(return_value=(0, 0)),
                          get_servo_angle=Mock(return_value=(0, [0.] * 7)),
                          get_c31_error_info=Mock(return_value=(0, [4, 17., -7.])),
                          set_servo_angle=Mock(), clean_error=Mock(), set_state=Mock())
    s = SimpleNamespace(clock=clock, cfg=cfg, plan=plan, arm=arm,
                        pose=grasp.copy(), joints=np.zeros(7), writes=[], responses=[], hook=None)
    robot = s.robot = SimpleNamespace(arm=arm, stop_motion=Mock(), enable_motion=Mock(),
                                     get_tcp_pose=lambda: (s.pose.copy(), None),
                                     get_joints_deg=lambda: s.joints.copy())

    def write(*args, **kwargs):
        s.writes.append((args, kwargs))
        if kwargs['only_check_type']:
            code, result = s.responses.pop(0) if s.responses else (0, 0)
            arm.only_check_result = result
        else:
            code = 0
            s.pose = xarm_pose_to_se3(args)
        if s.hook:
            s.hook(kwargs)
        return code

    arm.set_position = Mock(side_effect=write)
    return s


def execute(s, **kwargs):
    return motion.recover_unclosed_pick(s.robot, s.plan, s.cfg,
                                        j7_limits=(-89., 90.), **kwargs)


def physical(s):
    return [call for call in s.writes if call[1]['only_check_type'] == 0]


def test_return_is_checked_before_motion_and_uses_the_reverse_approach(scene):
    s = scene
    metrics = {}
    assert execute(s, metrics=metrics)
    assert [kw['only_check_type'] for _, kw in s.writes] == [1, 2, 0, 0]
    for checked, moved in zip(s.writes[:2], physical(s)):
        assert checked[0] == moved[0]
        assert {k: v for k, v in checked[1].items() if k != 'only_check_type'} == {
            k: v for k, v in moved[1].items() if k != 'only_check_type'}
    np.testing.assert_allclose(xarm_pose_to_se3(physical(s)[0][0]), s.plan.pre, atol=1e-12)
    np.testing.assert_allclose(xarm_pose_to_se3(physical(s)[1][0]), s.plan.start_tcp, atol=1e-12)
    assert all(kw['speed'] <= 600 and kw['mvacc'] <= 350 for _, kw in physical(s))
    assert metrics['confirm_s'] >= .04
    s.arm.set_servo_angle.assert_not_called()
    s.robot.enable_motion.assert_not_called()
    s.arm.clean_error.assert_not_called()
    s.arm.set_state.assert_not_called()


def test_recovery_retains_native_retiming_limits(scene):
    s = scene
    s.responses = [(1, 24)]
    assert execute(s)
    assert [kw['only_check_type'] for _, kw in s.writes] == [1, 1, 2, 0, 0]
    assert physical(s)[0][1]['speed'] == 320.
    assert physical(s)[0][1]['mvacc'] == pytest.approx(224.)


@pytest.mark.parametrize('result', [21, 22, 23, 24])
def test_rejected_reverse_path_never_sends_recovery_motion(scene, result):
    s = scene
    s.responses = [(1, result)] * 4
    assert not execute(s)
    assert not physical(s)
    s.robot.stop_motion.assert_not_called()  # only a shadow plan was rejected
    s.arm.clean_error.assert_not_called()


@pytest.mark.parametrize('failure', ['c31', 'pending', 'moving', 'mode', 'j7', 'pose'])
def test_unhealthy_or_changed_actual_start_blocks_even_preflight(scene, failure):
    s = scene
    if failure == 'c31':
        s.arm.get_err_warn_code.return_value = 0, [31, 0]
    elif failure == 'pending':
        s.arm.get_cmdnum.return_value = 0, 1
    elif failure == 'moving':
        s.arm.get_state.return_value = 0, 1
    elif failure == 'mode':
        s.arm.mode = 1
    elif failure == 'j7':
        s.joints[-1] = 91.
    else:
        s.pose[0, 3] += .003
    assert not execute(s)
    assert not s.writes
    s.robot.enable_motion.assert_not_called()
    s.arm.clean_error.assert_not_called()


@pytest.mark.parametrize('failure', ['c31', 'drift', 'transport'])
def test_fault_or_drift_during_check_blocks_recovery_execution(scene, failure):
    s = scene
    def changed(kwargs):
        if failure == 'c31':
            s.arm.get_err_warn_code.return_value = 0, [31, 0]
        elif failure == 'drift':
            s.joints[0] = 1.
    if failure == 'transport':
        s.responses = [(9, 0)]
    else:
        s.hook = changed
    assert not execute(s)
    assert not physical(s)


def test_camera_failure_propagates_before_recovery_and_never_clears_errors(scene):
    check = Mock(side_effect=CameraError('camera disconnected'))
    with pytest.raises(CameraError, match='camera disconnected'):
        execute(scene, check=check)
    assert not scene.writes
    scene.arm.clean_error.assert_not_called()


def test_user_stop_prevents_recovery_motion(scene):
    check = Mock(side_effect=RuntimeError('user stop'))
    assert not execute(scene, check=check)
    assert not scene.writes


def test_joint_limit_crossing_during_return_stops_motion(scene):
    scene.arm.get_servo_angle.return_value = 0, [0.] * 6 + [91.]
    assert not execute(scene)
    assert len(physical(scene)) == 2
    scene.robot.stop_motion.assert_called_once()
    scene.arm.clean_error.assert_not_called()


@pytest.mark.parametrize('failure', ['c31', 'camera', 'stop'])
def test_recovery_cancels_pending_commands_on_runtime_failure(scene, failure):
    s = scene

    def hook(kwargs):
        if not kwargs['only_check_type'] and failure == 'c31':
            s.arm.get_err_warn_code.return_value = 0, [31, 0]

    def check():
        if physical(s):
            if failure == 'camera':
                raise CameraError('disconnected during return')
            if failure == 'stop':
                raise RuntimeError('user stop during return')

    s.hook = hook
    if failure == 'camera':
        with pytest.raises(CameraError, match='disconnected during return'):
            execute(s, check=check)
    else:
        assert not execute(s, check=check)
    assert len(physical(s)) == (2 if failure == 'c31' else 1)
    s.robot.stop_motion.assert_called_once()
    s.arm.clean_error.assert_not_called()
    s.arm.set_servo_angle.assert_not_called()
