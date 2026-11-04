"""Keep the wrist planning reserve at the measured, unclosed grasp state."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg.camera import CameraError
from efg.motion_command import PlanRejected
from efg.transfer_plan import prepare_transfer


HARD_LIMITS = (-89., 90.)
PLANNING_LIMITS = (-79., 80.)


@pytest.fixture
def scene():
    grasp = np.eye(4)
    grasp[:3, 3] = [.4, -.2, .1]
    pre, lift = grasp.copy(), grasp.copy()
    pre[2, 3], lift[2, 3] = .2, .3
    joints = np.array([0.] * 6 + [70.])
    arm = Mock(mode=0, version_number=[2, 7, 1], only_check_result=0)
    arm.get_state.return_value = 0, 2
    arm.get_cmdnum.return_value = 0, 0
    arm.get_err_warn_code.return_value = 0, [0, 0]
    arm.set_position.return_value = arm.set_servo_angle.return_value = 0
    robot = SimpleNamespace(
        arm=arm, T_flange_tcp=np.eye(4), R_home_tcp=np.eye(3),
        get_tcp_pose=Mock(return_value=(grasp, None)),
        get_joints_deg=Mock(return_value=joints),
        flange_ik_joints_deg=Mock(side_effect=lambda _: joints.copy()),
        stop_motion=Mock(),
    )
    return SimpleNamespace(
        robot=robot, arm=arm, joints=joints, grasp=grasp, pre=pre, lift=lift,
        robot_cfg={'lift_speed_mm_s': 150., 'rise_ik_step_mm': 20.,
                   'rise_ik_step_deg': 5.},
        drop_cfg={'lift_orientation': 'grasp', 'via_height_m': .35,
                  'joints_deg': [0.] * 6 + [85.], 'joint_speed_deg_s': 120.},
    )


def prepare(scene, **kwargs):
    return prepare_transfer(
        scene.robot, scene.pre, scene.lift, scene.robot_cfg, scene.drop_cfg,
        j7_limits=HARD_LIMITS, **kwargs)


def assert_no_native_commands(scene):
    scene.arm.set_position.assert_not_called()
    scene.arm.set_servo_angle.assert_not_called()
    scene.arm.clean_error.assert_not_called()
    scene.robot.stop_motion.assert_not_called()


@pytest.mark.parametrize('actual_j7', [88.514, -84.7])
def test_measured_grasp_must_retain_reserve_even_when_all_ik_is_inside(scene, actual_j7):
    # The actual recorded positive wrist angle was still legal, but had consumed
    # the reserve. A separately solved IK value of 70 degrees must not hide it.
    scene.robot.get_joints_deg.return_value = np.array([0.] * 6 + [actual_j7])

    with pytest.raises(PlanRejected, match='J7'):
        prepare(scene, planning_j7_limits=PLANNING_LIMITS)

    scene.robot.flange_ik_joints_deg.assert_not_called()
    assert scene.robot.get_joints_deg.call_count >= 2  # Rejection validates live state.
    assert_no_native_commands(scene)


@pytest.mark.parametrize('actual_j7', [91., 90.067, -89.001])
def test_actual_hard_limit_breach_is_never_a_retryable_reserve_rejection(scene, actual_j7):
    scene.robot.get_joints_deg.return_value = np.array([0.] * 6 + [actual_j7])

    with pytest.raises(RuntimeError, match='J7') as caught:
        prepare(scene, planning_j7_limits=PLANNING_LIMITS)

    assert not isinstance(caught.value, PlanRejected)
    scene.robot.flange_ik_joints_deg.assert_not_called()
    assert_no_native_commands(scene)


@pytest.mark.parametrize('path_j7', [85., -85.])
def test_retreat_interior_ik_keeps_reserve_when_actual_grasp_and_endpoints_pass(scene, path_j7):
    def ik(target):
        joints = scene.joints.copy()
        if scene.grasp[2, 3] < target[2, 3] < scene.pre[2, 3]:
            joints[6] = path_j7
        return joints

    scene.robot.flange_ik_joints_deg.side_effect = ik

    with pytest.raises(PlanRejected, match='J7'):
        prepare(scene, planning_j7_limits=PLANNING_LIMITS)

    queries = [call.args[0] for call in scene.robot.flange_ik_joints_deg.call_args_list]
    assert any(scene.grasp[2, 3] < target[2, 3] < scene.pre[2, 3] for target in queries)
    assert_no_native_commands(scene)


@pytest.mark.parametrize('failure', ['controller_fault', 'pending_command', 'joint_drift'])
def test_unhealthy_or_changed_state_overrides_reserve_rejection(scene, failure):
    actual = np.array([0.] * 6 + [88.514])
    scene.robot.get_joints_deg.return_value = actual
    if failure == 'controller_fault':
        scene.arm.get_err_warn_code.side_effect = [(0, [0, 0]), (0, [31, 0])]
    elif failure == 'pending_command':
        scene.arm.get_cmdnum.side_effect = [(0, 0), (0, 1)]
    else:
        changed = actual.copy()
        changed[0] += 1.
        scene.robot.get_joints_deg.side_effect = [actual, changed]

    with pytest.raises(RuntimeError) as caught:
        prepare(scene, planning_j7_limits=PLANNING_LIMITS)

    assert not isinstance(caught.value, PlanRejected)
    scene.robot.flange_ik_joints_deg.assert_not_called()
    assert_no_native_commands(scene)


def test_camera_failure_during_reserve_rejection_propagates(scene):
    scene.robot.get_joints_deg.return_value = np.array([0.] * 6 + [88.514])
    failure = CameraError('camera stopped before reserve rejection')
    check = Mock(side_effect=[None, failure])

    with pytest.raises(CameraError) as caught:
        prepare(scene, planning_j7_limits=PLANNING_LIMITS, check=check)

    assert caught.value is failure
    scene.robot.flange_ik_joints_deg.assert_not_called()
    assert_no_native_commands(scene)


@pytest.mark.parametrize('boundary', [-79., 80.])
def test_inner_boundary_is_valid_and_release_and_execution_keep_hard_limits(scene, boundary):
    scene.joints[6] = boundary

    plan = prepare(scene, planning_j7_limits=PLANNING_LIMITS)

    assert plan.start_joints[6] == boundary
    assert plan.commands[-1].target[6] == 85.
    assert plan.j7_limits == HARD_LIMITS
    calls = scene.arm.set_position.call_args_list + scene.arm.set_servo_angle.call_args_list
    assert [call.kwargs['only_check_type'] for call in calls] == [1, 3, 2]
    scene.arm.clean_error.assert_not_called()
    scene.robot.stop_motion.assert_not_called()


@pytest.mark.parametrize('planning_limits', [
    (), (-79.,), (-79., 80., 81.), (0., 0.), (80., -79.),
    (np.nan, 80.), (-79., np.inf), (-90., 80.), (-79., 91.),
])
def test_invalid_planning_interval_cannot_produce_a_plan(scene, planning_limits):
    with pytest.raises(ValueError):
        prepare(scene, planning_j7_limits=planning_limits)

    assert_no_native_commands(scene)


@pytest.mark.parametrize('kwargs', [{}, {'planning_j7_limits': None},
                                    {'planning_j7_limits': HARD_LIMITS}])
def test_omitted_or_unreserved_planning_limits_preserve_hard_limit_behavior(scene, kwargs):
    scene.joints[6] = 88.514

    plan = prepare(scene, **kwargs)

    assert plan.start_joints[6] == 88.514
    assert plan.j7_limits == HARD_LIMITS
    assert plan.commands[-1].target[6] == 85.
