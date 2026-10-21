"""Queued approach tests; no hardware connections."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg import motion
from efg.camera import CameraError


@pytest.fixture
def setup(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(motion.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(motion.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    pre, grasp, start = np.eye(4), np.eye(4), np.eye(4)
    pre[2, 3], grasp[2, 3], start[2, 3] = .2, .1, .4
    robot = Mock()
    robot.get_tcp_pose.side_effect = lambda: (
        grasp if robot.arm.set_position.call_count == 2 else start, None)
    robot.arm.set_position.return_value = 0
    robot.arm.get_state.return_value = (0, 2)
    robot.arm.get_err_warn_code.return_value = (0, [0, 0])
    robot.arm.get_cmdnum.return_value = (0, 0)
    return SimpleNamespace(robot=robot, pre=pre, grasp=grasp, clock=clock,
                           cfg={"travel_speed_mm_s": 700, "approach_speed_mm_s": 150,
                                "approach_acc_mm_s2": 500, "approach_timeout_s": 1})


def test_queues_blend_then_exact_stop_and_waits_for_settle(setup):
    p = setup
    assert motion.approach_blended(p.robot, p.pre, p.grasp, p.cfg)
    first, last = p.robot.arm.set_position.call_args_list
    assert first.kwargs == dict(speed=700., mvacc=500., radius=5., is_radian=True, wait=False)
    assert last.kwargs == dict(speed=150., mvacc=500., radius=-1, is_radian=True, wait=False)
    assert last.args[:3] == (0., 0., 100.)  # TCP, metres -> mm, no double offset
    assert p.clock[0] >= .2
    p.robot.stop_motion.assert_not_called()


@pytest.mark.parametrize("ready", [True, False])
def test_width_adjustment_runs_after_pregrasp_command_before_approach(setup, ready):
    p = setup

    def prepare():
        assert p.robot.arm.set_position.call_count == 1
        assert p.robot.arm.set_position.call_args.kwargs["wait"] is False
        return ready

    assert motion.approach_blended(p.robot, p.pre, p.grasp, p.cfg, prepare=prepare) is ready
    assert p.robot.arm.set_position.call_count == (2 if ready else 1)
    if not ready:
        p.robot.stop_motion.assert_called_once()


def test_camera_failure_while_adjusting_width_cancels_pregrasp(setup):
    p = setup
    def prepare():
        raise CameraError("camera lost during width adjustment")
    with pytest.raises(CameraError):
        motion.approach_blended(p.robot, p.pre, p.grasp, p.cfg, prepare=prepare)
    assert p.robot.arm.set_position.call_count == 1
    p.robot.stop_motion.assert_called_once()


@pytest.mark.parametrize("state,pending,offset", [(0, 0, 0), (1, 0, 0), (2, 1, 0), (2, 0, .01)])
def test_not_complete_before_stopped_empty_queue_and_arrival(setup, state, pending, offset):
    p = setup
    actual = p.grasp.copy()
    actual[0, 3] += offset
    p.robot.get_tcp_pose.return_value = actual, None
    p.robot.get_tcp_pose.side_effect = None
    p.robot.arm.get_state.return_value = 0, state
    p.robot.arm.get_cmdnum.return_value = 0, pending
    assert not motion.approach_blended(p.robot, p.pre, p.grasp, p.cfg)
    p.robot.stop_motion.assert_called_once()


def test_rejected_second_command_cancels_first_without_clearing_error(setup):
    p = setup
    p.robot.arm.set_position.side_effect = [0, 9]
    assert not motion.approach_blended(p.robot, p.pre, p.grasp, p.cfg)
    p.robot.stop_motion.assert_called_once()
    p.robot.arm.clean_error.assert_not_called()


@pytest.mark.parametrize("after", [1, 2])
def test_camera_failure_cancels_pending_motion(setup, after):
    p = setup

    def check():
        if p.robot.arm.set_position.call_count >= after:
            raise CameraError("disconnected")

    with pytest.raises(CameraError):
        motion.approach_blended(p.robot, p.pre, p.grasp, p.cfg, check=check)
    assert p.robot.arm.set_position.call_count == after
    p.robot.stop_motion.assert_called_once()


def test_controller_fault_stops_and_short_leg_limits_radius(setup):
    p = setup
    p.pre[2, 3] = .11
    p.robot.arm.get_err_warn_code.return_value = 0, [31, 0]
    assert not motion.approach_blended(p.robot, p.pre, p.grasp, p.cfg)
    assert p.robot.arm.set_position.call_args_list[0].kwargs["radius"] == pytest.approx(2.5)
    p.robot.stop_motion.assert_called_once()


@pytest.fixture
def transfer(setup):
    p = setup
    p.pre[2, 3] = .5
    p.robot.R_home_tcp = np.diag([1., -1., -1.])
    p.lift = np.eye(4)
    p.lift[2, 3] = .55
    p.drop = {"joints_deg": [1.] * 7, "joint_speed_deg_s": 180,
              "via_height_m": .6, "transfer_timeout_s": 1}
    p.cfg["lift_speed_mm_s"] = 700
    p.robot.T_flange_tcp = np.eye(4)
    p.robot.check_flange_ik.return_value = True
    p.robot.arm.set_servo_angle.return_value = 0
    p.robot.arm.get_servo_angle.return_value = 0, [1.] * 7
    start = np.eye(4)
    start[2, 3] = .4
    target, _ = motion.transfer_waypoint(p.pre, p.lift, p.drop, p.robot.R_home_tcp)
    p.robot.get_tcp_pose.side_effect = lambda: (
        target if p.robot.arm.set_position.call_count == 2 else start, None)
    return p


def test_lift_stops_before_joint_transfer_and_preserves_clearance(transfer):
    p = transfer
    assert motion.lift_and_transfer_blended(p.robot, p.pre, p.lift, p.cfg, p.drop)
    retreat, rise = p.robot.arm.set_position.call_args_list
    np.testing.assert_allclose(retreat.args, motion.se3_to_xarm_pose(p.pre))
    assert retreat.kwargs["radius"] == 5
    assert retreat.kwargs["wait"] is False
    from efg.se3 import xarm_pose_to_se3
    rising_pose = xarm_pose_to_se3(rise.args)
    np.testing.assert_allclose(rising_pose[:3, :3], p.robot.R_home_tcp, atol=1e-10)
    np.testing.assert_allclose(rising_pose[:2, 3], p.pre[:2, 3])
    lift = p.robot.arm.set_position.call_args
    drop = p.robot.arm.set_servo_angle.call_args
    assert lift.args[2] == pytest.approx(610.)
    assert lift.kwargs["radius"] == -1
    assert lift.kwargs["wait"] is False
    assert drop.kwargs["wait"] is False
    assert drop.kwargs["radius"] == -1
    assert p.clock[0] >= .2
    p.robot.stop_motion.assert_not_called()


@pytest.mark.parametrize("failure", ["ik", "lift", "drop", "arrival", "state", "queue", "c24"])
def test_transfer_failures_stop_without_fallback_or_release(transfer, failure):
    p = transfer
    if failure == "ik":
        p.robot.check_flange_ik.return_value = False
    elif failure == "lift":
        p.robot.arm.set_position.return_value = 9
    elif failure == "drop":
        p.robot.arm.set_servo_angle.return_value = 9
    elif failure == "arrival":
        p.robot.arm.get_servo_angle.return_value = 0, [2.] * 7
    elif failure == "state":
        p.robot.arm.get_state.return_value = 0, 1
    elif failure == "c24":
        p.robot.arm.get_state.return_value = 0, 4
        p.robot.arm.get_err_warn_code.return_value = 0, [24, 0]
        p.robot.arm.get_position.return_value = 0, [300., 0., 250., 180., 0., 0.]
    else:
        p.robot.arm.get_cmdnum.return_value = 0, 1
    assert not motion.lift_and_transfer_blended(p.robot, p.pre, p.lift, p.cfg, p.drop)
    p.robot.stop_motion.assert_called_once()
    p.robot.move_joints.assert_not_called()
    p.robot.arm.clean_error.assert_not_called()
    if failure in ("ik", "lift", "state", "queue", "c24"):
        p.robot.arm.set_servo_angle.assert_not_called()


@pytest.mark.parametrize("after_drop", [False, True])
def test_camera_disconnect_cancels_lift_or_queued_transfer(transfer, after_drop):
    p = transfer

    def check():
        count = (p.robot.arm.set_servo_angle.call_count if after_drop
                 else p.robot.arm.set_position.call_count)
        if count:
            raise CameraError("disconnected")

    with pytest.raises(CameraError):
        motion.lift_and_transfer_blended(p.robot, p.pre, p.lift, p.cfg, p.drop, check)
    p.robot.stop_motion.assert_called_once()
    assert p.robot.arm.set_servo_angle.call_count == int(after_drop)


def test_large_rotation_limits_reorientation_speed():
    start, end = np.eye(4), np.eye(4)
    end[2, 3] = .1
    end[:3, :3] = np.diag([1., -1., -1.])
    speed, angle = motion.reorientation_speed(start, end, {"lift_speed_mm_s": 500})
    assert angle == pytest.approx(180.)
    assert speed == pytest.approx(100 * 30 / 180)
    end[:3, :3] = np.eye(3)
    speed, angle = motion.reorientation_speed(start, end, {"lift_speed_mm_s": 500})
    assert speed == 100


def test_retreat_and_reorientation_use_distinct_speeds(transfer):
    p = transfer
    p.cfg.update(retreat_speed_mm_s=140, reorient_speed_mm_s=80,
                 reorient_acc_mm_s2=125, reorient_angular_speed_deg_s=20)
    assert motion.lift_and_transfer_blended(p.robot, p.pre, p.lift, p.cfg, p.drop)
    retreat, rise = p.robot.arm.set_position.call_args_list
    assert retreat.kwargs["speed"] == 140
    assert rise.kwargs["speed"] == pytest.approx(110 * 20 / 180)
    assert rise.kwargs["mvacc"] == 125
    assert retreat.kwargs["radius"] > 0 and rise.kwargs["radius"] == -1


def test_joint_command_is_not_sent_until_cartesian_settled(transfer):
    p = transfer

    def joint_command(**kwargs):
        assert p.clock[0] >= .2
        assert p.robot.arm.set_position.call_count == 2
        assert p.robot.arm.set_position.call_args.kwargs["radius"] == -1
        return 0

    p.robot.arm.set_servo_angle.side_effect = joint_command
    assert motion.lift_and_transfer_blended(p.robot, p.pre, p.lift, p.cfg, p.drop)


def test_stopped_short_of_upper_target_never_queues_joint_motion(transfer):
    p = transfer
    position = np.eye(4)
    position[2, 3] = .4
    p.robot.get_tcp_pose.side_effect = lambda: (position, None)
    assert not motion.lift_and_transfer_blended(p.robot, p.pre, p.lift, p.cfg, p.drop)
    p.robot.arm.set_servo_angle.assert_not_called()
    p.robot.stop_motion.assert_called_once()
