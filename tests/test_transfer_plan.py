"""Prepare recovery from the measured grasp pose without sending motion."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from efg.camera import CameraError
from efg.transfer_plan import prepare_transfer


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
    robot.arm.get_cmdnum.return_value = 0, 0
    robot.arm.get_err_warn_code.return_value = 0, [0, 0]
    return SimpleNamespace(
        robot=robot, grasp=grasp, pre=pre, lift=lift, joints=joints,
        robot_cfg={"rise_ik_step_mm": 20, "rise_ik_step_deg": 15},
        drop_cfg={"via_height_m": .350, "transfer_clearance_margin_mm": 10},
    )


def prepare(scene, **kwargs):
    return prepare_transfer(
        scene.robot, scene.pre, scene.lift, scene.robot_cfg, scene.drop_cfg,
        j7_limits=(-89, 90), **kwargs,
    )


def assert_no_motion(scene):
    allowed_reads = {"get_state", "get_cmdnum", "get_err_warn_code"}
    assert all(call[0] in allowed_reads for call in scene.robot.arm.mock_calls)
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
    # Common retreat and preferred upper were checked; no alternate was tried.
    assert scene.robot.flange_ik_joints_deg.call_count == 2
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
