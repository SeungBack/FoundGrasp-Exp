"""Prepare recovery from the measured grasp pose without sending motion."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from efg.camera import CameraError
from efg.transfer_plan import (closest_upright_rotation, prepare_transfer,
                               transfer_waypoint)


@pytest.fixture
def scene():
    grasp = np.eye(4)
    grasp[:3, 3] = [.498, -.310, .060]
    pre = grasp.copy()
    pre[:3, 3] = [.474, -.317, .156]
    lift = pre.copy()
    lift[2, 3] = .350
    joints = np.array([-25., 21., -13., 71., 24., 44., -75.])
    robot = SimpleNamespace(
        T_flange_tcp=np.eye(4), R_home_tcp=np.eye(3),
        get_tcp_pose=Mock(return_value=(grasp, None)),
        get_joints_deg=Mock(return_value=joints),
        flange_ik_joints_deg=Mock(side_effect=lambda _: joints.copy()),
        arm=Mock(), stop_motion=Mock(),
    )
    robot.arm.get_state.return_value = 0, 2
    robot.arm.mode = 0
    robot.arm.version_number = [2, 7, 1]
    robot.arm.only_check_result = 0
    robot.arm.set_position.return_value = 0
    robot.arm.set_servo_angle.return_value = 0
    robot.arm.get_cmdnum.return_value = 0, 0
    robot.arm.get_err_warn_code.return_value = 0, [0, 0]
    return SimpleNamespace(
        robot=robot, grasp=grasp, pre=pre, lift=lift, joints=joints,
        robot_cfg={"rise_ik_step_mm": 20, "rise_ik_step_deg": 15, "lift_speed_mm_s": 150},
        drop_cfg={"via_height_m": .350, "transfer_clearance_margin_mm": 10,
                  "joints_deg": [0.] * 7, "joint_speed_deg_s": 120},
    )


def prepare(scene, **kwargs):
    return prepare_transfer(
        scene.robot, scene.pre, scene.lift, scene.robot_cfg, scene.drop_cfg,
        j7_limits=(-89, 90), **kwargs,
    )


def assert_no_motion(scene):
    allowed_reads = {"get_state", "get_cmdnum", "get_err_warn_code"}
    for call in scene.robot.arm.mock_calls:
        assert (call[0] in allowed_reads or
                (call[0] in {'set_position', 'set_servo_angle'} and
                 call.kwargs.get('only_check_type') in (1, 2, 3)))
    scene.robot.stop_motion.assert_not_called()


def test_current_state_ik_can_choose_other_upper_symmetry_without_changing_retreat(scene):
    """A home-selected upper branch can exceed J7 after reaching the grasp."""
    upper_queries = []

    def ik(target):
        q = scene.joints.copy()
        if np.isclose(target[2, 3], .360):
            upper_queries.append(target.copy())
            q[6] = 104 if target[0, 0] > 0 else -60
        return q

    scene.robot.flange_ik_joints_deg.side_effect = ik
    plan = prepare(scene, allow_symmetry=True)

    assert plan.transfer_sign == -1
    np.testing.assert_array_equal(plan.pre, scene.pre)
    np.testing.assert_allclose(plan.upper[:3, 3], [.474, -.317, .360])
    np.testing.assert_allclose(plan.upper[:3, :3], np.diag([-1., -1., 1.]))
    np.testing.assert_array_equal(plan.start_tcp, scene.grasp)
    np.testing.assert_array_equal(plan.start_joints, scene.joints)
    assert len(upper_queries) == 2
    np.testing.assert_array_equal(upper_queries[0][:3, 3], upper_queries[1][:3, 3])
    assert_no_motion(scene)


def test_common_retreat_failure_cannot_be_repaired_by_upper_symmetry(scene):
    scene.robot.flange_ik_joints_deg.return_value = None
    scene.robot.flange_ik_joints_deg.side_effect = None

    with pytest.raises(RuntimeError, match="IK 불가.*복귀 pre-grasp"):
        prepare(scene, allow_symmetry=True)

    scene.robot.flange_ik_joints_deg.assert_called_once()
    np.testing.assert_array_equal(
        scene.robot.flange_ik_joints_deg.call_args.args[0], scene.pre)
    assert_no_motion(scene)


def test_failed_transfer_retains_original_sdk_ik_diagnostic(scene):
    scene.robot.flange_ik_joints_deg.side_effect = None
    scene.robot.flange_ik_joints_deg.return_value = None
    scene.robot.last_ik_failure = 'SDK code=1, TCP(mm/rad)=[474, -317, 156, 0, 0, 0]'
    with pytest.raises(RuntimeError, match='복귀 pre-grasp, SDK code=1, TCP'):
        prepare(scene, allow_symmetry=True)
    assert_no_motion(scene)


def test_both_upper_branches_rejected_with_diagnostic_ik_and_actual_joints(scene):
    def ik(target):
        q = scene.joints.copy()
        if np.isclose(target[2, 3], .360):
            q[6] = 104 if target[0, 0] > 0 else -102
        return q

    scene.robot.flange_ik_joints_deg.side_effect = ik
    with pytest.raises(RuntimeError) as error:
        prepare(scene, allow_symmetry=True)

    message = str(error.value)
    assert "상승 +1 경유점" in message
    assert "상승 -1 경유점" in message
    assert "104.00" in message and "-102.00" in message
    assert "실측 관절=" in message and "-75.0" in message
    assert_no_motion(scene)


def test_camera_down_symmetry_is_rejected_even_when_its_ik_would_succeed(scene):
    scene.robot.R_home_tcp = Rotation.from_euler("x", 30, degrees=True).as_matrix()

    def ik(target):
        q = scene.joints.copy()
        if np.isclose(target[2, 3], .360) and target[0, 0] > 0:
            q[6] = 104
        return q

    scene.robot.flange_ik_joints_deg.side_effect = ik
    with pytest.raises(RuntimeError, match="상승 -1: 카메라 하향 제한"):
        prepare(scene, allow_symmetry=True, camera_up=(np.array([0., 1., 0.]), 0.))

    # The alternative is discarded before endpoint IK or execution.
    assert scene.robot.flange_ik_joints_deg.call_count == 2
    assert_no_motion(scene)


def test_valid_endpoints_do_not_allow_an_invalid_intermediate_rise(scene):
    def ik(target):
        return None if .200 < target[2, 3] < .300 else scene.joints.copy()

    scene.robot.flange_ik_joints_deg.side_effect = ik
    with pytest.raises(RuntimeError, match="IK 불가.*상승 경로"):
        prepare(scene)

    targets = [call.args[0] for call in scene.robot.flange_ik_joints_deg.call_args_list]
    assert any(np.isclose(target[2, 3], .360) for target in targets)
    assert any(.200 < target[2, 3] < .300 for target in targets)
    assert_no_motion(scene)


def test_camera_failure_during_sampling_propagates_without_trying_another_branch(scene):
    failure = CameraError("camera stopped during rise preparation")
    check = Mock(side_effect=[None, None, failure])

    with pytest.raises(CameraError) as caught:
        prepare(scene, allow_symmetry=True, check=check)

    assert caught.value is failure
    assert check.call_count == 3
    # The common retreat was checked; cancellation precedes further endpoint IK.
    scene.robot.flange_ik_joints_deg.assert_called_once()
    np.testing.assert_array_equal(scene.robot.flange_ik_joints_deg.call_args.args[0], scene.pre)
    assert_no_motion(scene)


def test_plan_copies_inputs_and_disallows_in_place_or_field_mutation(scene):
    plan = prepare(scene)
    initial = {name: getattr(plan, name).copy()
               for name in ("start_tcp", "start_joints", "pre", "upper")}
    scene.grasp[0, 3] += .1
    scene.joints[6] += 5
    scene.pre[0, 3] += .1
    scene.lift[2, 3] += .1

    for name, expected in initial.items():
        np.testing.assert_array_equal(getattr(plan, name), expected)
        with pytest.raises(ValueError, match="read-only"):
            getattr(plan, name).flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        plan.transfer_sign = -1
    assert_no_motion(scene)


@pytest.mark.parametrize("change", ["position", "rotation", "joints", "invalid_joints"])
def test_plan_rejects_changed_actual_start_before_execution(scene, change):
    plan = prepare(scene)
    pose = scene.grasp.copy()
    joints = scene.joints.copy()
    if change == "position":
        pose[0, 3] += .003
    elif change == "rotation":
        pose[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()
    elif change == "joints":
        joints[0] += 1
    else:
        joints[0] = np.nan
    scene.robot.get_tcp_pose.return_value = pose, None
    scene.robot.get_joints_deg.return_value = joints

    with pytest.raises(RuntimeError, match="시작 자세가 변경됨.*이동 미전송"):
        plan.check_start(scene.robot)
    assert_no_motion(scene)


def test_small_measurement_noise_does_not_invalidate_a_stationary_start(scene):
    plan = prepare(scene)
    pose = scene.grasp.copy()
    pose[0, 3] += .0005
    pose[:3, :3] = Rotation.from_euler("z", .2, degrees=True).as_matrix()
    scene.robot.get_tcp_pose.return_value = pose, None
    scene.robot.get_joints_deg.return_value = scene.joints + .1
    plan.check_start(scene.robot)
    assert_no_motion(scene)


@pytest.mark.parametrize("stage", ["preparation", "execution_check"])
@pytest.mark.parametrize("invalid", ["missing", "wrong_shape", "nan_position", "inf_rotation"])
def test_invalid_tcp_read_cannot_produce_or_validate_a_transfer_plan(scene, stage, invalid):
    plan = prepare(scene) if stage == "execution_check" else None
    scene.robot.flange_ik_joints_deg.reset_mock()
    pose = scene.grasp.copy()
    if invalid == "missing":
        pose = None
    elif invalid == "wrong_shape":
        pose = np.eye(3)
    elif invalid == "nan_position":
        pose[0, 3] = np.nan
    else:
        pose[0, 0] = np.inf
    scene.robot.get_tcp_pose.return_value = pose, None

    with pytest.raises(RuntimeError, match="TCP 판독 오류"):
        if plan is None:
            prepare(scene)
        else:
            plan.check_start(scene.robot)

    scene.robot.flange_ik_joints_deg.assert_not_called()
    assert_no_motion(scene)


@pytest.mark.parametrize("stage", ["preparation", "execution_check"])
@pytest.mark.parametrize("method,value", [
    ("get_state", (0, 1)),
    ("get_cmdnum", (0, 1)),
    ("get_err_warn_code", (0, [24, 0])),
    ("get_state", (-1, 2)),
    ("get_cmdnum", (-1, 0)),
    ("get_err_warn_code", (-1, [0, 0])),
], ids=["moving", "pending_command", "controller_error", "state_read_failed",
        "queue_read_failed", "error_read_failed"])
def test_unsettled_or_unreadable_controller_cannot_validate_plan(scene, stage, method, value):
    plan = prepare(scene) if stage == "execution_check" else None
    scene.robot.flange_ik_joints_deg.reset_mock()
    getattr(scene.robot.arm, method).return_value = value

    with pytest.raises(RuntimeError, match="상승 계획.*정지"):
        if plan is None:
            prepare(scene)
        else:
            plan.check_start(scene.robot)

    scene.robot.flange_ik_joints_deg.assert_not_called()
    assert_no_motion(scene)


def test_motion_started_during_preparation_invalidates_completed_ik_plan(scene):
    # A stationary entry snapshot alone cannot authorize executing the plan.
    scene.robot.arm.get_state.side_effect = [(0, 2), (0, 1)]
    with pytest.raises(RuntimeError, match="상승 계획.*정지"):
        prepare(scene)
    assert scene.robot.flange_ik_joints_deg.call_count >= 2
    assert_no_motion(scene)


@pytest.mark.parametrize('prepared', [False, True])
def test_other_control_mode_cannot_prepare_or_execute_plan(scene, prepared):
    plan = prepare(scene) if prepared else None
    scene.robot.arm.mode = 1
    with pytest.raises(RuntimeError, match='mode=1'):
        if prepared:
            plan.check_start(scene.robot)
        else:
            prepare(scene)
    assert_no_motion(scene)


@pytest.mark.parametrize('pre_angles,home_angles', [
    ([75., 17., 165.], [-95., 0., 180.]),
    ([-40., -20., 160.], [115., 0., 180.]),
    ([30., 40., -10.], [-70., 20., 25.]),
])
def test_upright_aligns_tool_axis_with_minimum_rotation(pre_angles, home_angles):
    pre = Rotation.from_euler('zyx', pre_angles, degrees=True).as_matrix()
    home = Rotation.from_euler('zyx', home_angles, degrees=True).as_matrix()
    target = closest_upright_rotation(pre, home)

    np.testing.assert_allclose(target[:, 2], home[:, 2], atol=1e-12)
    np.testing.assert_allclose(target.T @ target, np.eye(3), atol=1e-12)
    assert np.linalg.det(target) == pytest.approx(1.)
    # Any rotation taking one unit axis to the other has at least this angle.
    lower_bound = np.arccos(np.clip(np.dot(pre[:, 2], home[:, 2]), -1., 1.))
    angle = Rotation.from_matrix(target @ pre.T).magnitude()
    assert angle == pytest.approx(lower_bound)
    for yaw in np.linspace(-np.pi, np.pi, 13):
        alternative = Rotation.from_rotvec(home[:, 2] * yaw).as_matrix() @ target
        assert Rotation.from_matrix(alternative @ pre.T).magnitude() >= angle - 1e-12


def test_upright_default_preserves_heading_when_tool_is_already_aligned(scene):
    scene.pre[:3, :3] = Rotation.from_euler('z', 67., degrees=True).as_matrix()
    home = Rotation.from_euler('z', -123., degrees=True).as_matrix()
    before = scene.pre.copy()

    target, _ = transfer_waypoint(scene.pre, scene.lift, scene.drop_cfg, home)

    np.testing.assert_allclose(target[:3, :3], scene.pre[:3, :3], atol=1e-12)
    assert not np.allclose(target[:3, :3], home)
    np.testing.assert_array_equal(scene.pre, before)


def test_home_mode_keeps_legacy_full_orientation_and_same_height_and_blend(scene):
    scene.pre[:3, :3] = Rotation.from_euler('zyx', [72., 14., 162.], degrees=True).as_matrix()
    home = Rotation.from_euler('zyx', [-45., 0., 180.], degrees=True).as_matrix()
    config = scene.drop_cfg | {'transfer_blend_radius_mm': 20}
    upright, upright_radius = transfer_waypoint(scene.pre, scene.lift, config, home)
    legacy, legacy_radius = transfer_waypoint(
        scene.pre, scene.lift, config | {'lift_orientation': 'home'}, home)

    np.testing.assert_allclose(legacy[:3, :3], home, atol=1e-12)
    np.testing.assert_array_equal(upright[:3, 3], legacy[:3, 3])
    assert upright_radius == legacy_radius == 20
    assert not np.allclose(upright[:3, :3], legacy[:3, :3])


def test_grasp_mode_keeps_twenty_degree_tilt_and_existing_clearance_and_blend(scene):
    rotation = Rotation.from_euler('zyx', [31., 20., 180.], degrees=True).as_matrix()
    scene.pre[:3, :3] = rotation
    before = scene.pre.copy()
    config = scene.drop_cfg | {'transfer_blend_radius_mm': 20}
    upright, radius = transfer_waypoint(scene.pre, scene.lift, config, np.diag([1., -1., -1.]))
    grasp, grasp_radius = transfer_waypoint(
        scene.pre, scene.lift, config | {'lift_orientation': 'grasp'}, np.eye(3))

    np.testing.assert_array_equal(grasp[:3, :3], rotation)
    np.testing.assert_array_equal(grasp[:3, 3], upright[:3, 3])
    np.testing.assert_array_equal(scene.pre, before)
    assert grasp_radius == radius == 20
    assert not np.allclose(grasp[:3, :3], upright[:3, :3])


def test_grasp_plan_keeps_rotation_in_both_cartesian_commands_without_extra_stop(scene):
    from efg.se3 import xarm_pose_to_se3
    rotation = Rotation.from_euler('zyx', [31., 20., 180.], degrees=True).as_matrix()
    scene.pre[:3, :3] = scene.grasp[:3, :3] = rotation
    scene.drop_cfg['lift_orientation'] = 'grasp'
    scene.robot_cfg['retreat_blend_radius_mm'] = 50
    plan = prepare(scene, allow_symmetry=True)

    assert plan.transfer_sign == 1
    assert len(plan.commands) == 3
    assert [c.kind for c in plan.commands] == ['cartesian', 'cartesian', 'joint']
    assert plan.commands[0].radius > 0
    assert plan.commands[1].radius == -1
    for command in plan.commands[:2]:
        np.testing.assert_allclose(xarm_pose_to_se3(command.target)[:3, :3], rotation, atol=1e-12)
    targets = [call.args[0] for call in scene.robot.flange_ik_joints_deg.call_args_list]
    for target in targets:
        np.testing.assert_allclose(target[:3, :3], rotation, atol=1e-12)
    assert_no_motion(scene)


def test_grasp_mode_infeasibility_does_not_fall_back_to_a_half_turn(scene):
    scene.drop_cfg['lift_orientation'] = 'grasp'
    queries = []

    def ik(target):
        if np.isclose(target[2, 3], .360):
            queries.append(target.copy())
            return None if target[0, 0] > 0 else scene.joints.copy()
        return scene.joints.copy()

    scene.robot.flange_ik_joints_deg.side_effect = ik
    with pytest.raises(RuntimeError, match='상승 경로 준비 실패'):
        prepare(scene, allow_symmetry=True)
    assert len(queries) == 1
    np.testing.assert_array_equal(queries[0][:3, :3], scene.pre[:3, :3])
    assert_no_motion(scene)


@pytest.mark.parametrize('entry', ['waypoint', 'prepare'])
def test_grasp_mode_rejects_an_explicit_upper_half_turn(scene, entry):
    scene.drop_cfg['lift_orientation'] = 'grasp'
    with pytest.raises(RuntimeError, match='180° 대칭'):
        if entry == 'waypoint':
            transfer_waypoint(scene.pre, scene.lift, scene.drop_cfg, np.eye(3), -1)
        else:
            prepare(scene, transfer_sign=-1, allow_symmetry=True)
    scene.robot.flange_ik_joints_deg.assert_not_called()
    assert_no_motion(scene)


@pytest.mark.parametrize('mode', ['upright', 'home'])
def test_upper_symmetry_is_only_a_tool_axis_half_turn(scene, mode):
    scene.pre[:3, :3] = Rotation.from_euler('zyx', [72., 14., 162.], degrees=True).as_matrix()
    home = Rotation.from_euler('zyx', [-45., 0., 180.], degrees=True).as_matrix()
    config = scene.drop_cfg | {'lift_orientation': mode}
    positive, radius = transfer_waypoint(scene.pre, scene.lift, config, home, 1)
    negative, other_radius = transfer_waypoint(scene.pre, scene.lift, config, home, -1)

    np.testing.assert_allclose(negative[:3, :3], positive[:3, :3] @ np.diag([-1., -1., 1.]))
    np.testing.assert_array_equal(negative[:3, 2:], positive[:3, 2:])
    assert radius == other_radius


def test_antiparallel_axes_use_pre_grasp_x_axis_deterministically():
    pre = Rotation.from_euler('zyx', [65., 20., -40.], degrees=True).as_matrix()
    expected = pre @ np.diag([1., -1., -1.])
    home = expected @ Rotation.from_euler('z', 110., degrees=True).as_matrix()

    target = closest_upright_rotation(pre, home)

    np.testing.assert_allclose(target, expected, atol=1e-12)
    np.testing.assert_allclose(target[:, 2], home[:, 2], atol=1e-12)
    assert Rotation.from_matrix(target @ pre.T).magnitude() == pytest.approx(np.pi)


def test_nearly_antiparallel_axes_are_aligned_without_numerical_instability():
    pre = Rotation.from_euler('zyx', [15., 34., -70.], degrees=True).as_matrix()
    home = Rotation.from_rotvec(pre[:, 0] * (np.pi - 1e-8)).as_matrix() @ pre

    target = closest_upright_rotation(pre, home)

    np.testing.assert_allclose(target[:, 2], home[:, 2], atol=1e-12)
    np.testing.assert_allclose(target.T @ target, np.eye(3), atol=1e-12)
    assert np.linalg.det(target) == pytest.approx(1.)


@pytest.mark.parametrize('invalid', [
    np.eye(2), np.zeros((3, 3)), np.eye(3) * 2,
    np.diag([-1., 1., 1.]), np.eye(3) * np.nan,
    [[1., 0., 0.], [0., 1., 0.], [.1, 0., 1.]],
])
@pytest.mark.parametrize('bad_input', ['pre', 'home'])
def test_upright_rejects_nonfinite_or_improper_rotation_matrices(invalid, bad_input):
    with pytest.raises(RuntimeError, match='회전행렬'):
        closest_upright_rotation(invalid if bad_input == 'pre' else np.eye(3),
                                 invalid if bad_input == 'home' else np.eye(3))


@pytest.mark.parametrize('mode', [None, '', 'vertical', 1, True, {}])
def test_unknown_lift_orientation_is_rejected_before_any_motion(scene, mode):
    scene.drop_cfg['lift_orientation'] = mode
    with pytest.raises(RuntimeError, match='drop.lift_orientation'):
        prepare(scene)
    assert_no_motion(scene)
