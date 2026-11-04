from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from efg import selection
from efg.config import load_config
from efg.transfer_plan import prepare_transfer


def test_candidate_with_unreachable_recovery_is_skipped(monkeypatch):
    cfg = load_config()
    cfg['drop']['lift_orientation'] = 'upright'
    cfg["selection"]["camera_up"] = {"prefer": False}
    cfg["selection"]["approach_extra_m"] = 0
    home_R = np.diag([1., -1., -1.])
    tested = []

    def ik(T):
        tested.append(T.copy())
        return not (T[0, 3] < .25 and T[2, 3] > .3)

    robot = SimpleNamespace(T_flange_tcp=np.eye(4), R_home_tcp=home_R,
                            check_flange_ik=ik,
                            get_joints_deg=lambda: np.zeros(7),
                            flange_ik_joints_deg=lambda T: [0.] * 7 if ik(T) else None)
    grasps = [SimpleNamespace(score=1., width=.06, depth=0., x=x) for x in (.2, .3)]
    monkeypatch.setattr(selection, "grasp_to_base", lambda _, g: (np.eye(3), np.array([g.x, 0., .1])))
    monkeypatch.setattr(selection, "approach_tilt_deg", lambda R: 0.)

    def pose(R, p, **kwargs):
        T = np.eye(4)
        T[:3, 3] = p
        return T

    monkeypatch.setattr(selection, "tool_pose_from_grasp", pose)
    report = selection.select_grasp(grasps, np.eye(4), cfg, robot)
    assert report.selected.index == 1
    assert any("transfer" in reason for _, reason in report.rejected)
    # Interior samples now include partially rotated poses; endpoints are at max z.
    upper = [T for T in tested if np.isclose(T[2, 3], max(t[2, 3] for t in tested))]
    assert upper
    for T in upper:
        assert any(np.allclose(T[:3, :3], home_R @ np.diag([s, s, 1])) for s in (1, -1))


def test_upright_heading_is_shared_by_selection_and_measured_start_preparation(monkeypatch):
    """A feasible vertical heading need not adopt the home wrist yaw."""
    from scipy.spatial.transform import Rotation
    cfg = load_config()
    cfg['selection'].update(camera_up={'prefer': False}, approach_extra_m=0)
    cfg['drop']['joints_deg'] = [0.] * 7
    home = np.diag([1., -1., -1.])
    heading = Rotation.from_euler('z', 45, degrees=True).as_matrix() @ home

    def ik(T):
        # The synthetic arm cannot reach the two home-heading upper poses.
        return None if T[2, 3] > .3 and abs(T[1, 0]) < .1 else [0.] * 7

    robot = SimpleNamespace(
        T_flange_tcp=np.eye(4), R_home_tcp=home,
        get_joints_deg=lambda: np.zeros(7), flange_ik_joints_deg=ik,
        arm=Mock(mode=0, version_number=[2, 7, 1], only_check_result=0))
    robot.arm.get_state.return_value = 0, 2
    robot.arm.get_cmdnum.return_value = 0, 0
    robot.arm.get_err_warn_code.return_value = 0, [0, 0]
    robot.arm.set_position.return_value = robot.arm.set_servo_angle.return_value = 0
    monkeypatch.setattr(selection, 'grasp_to_base',
                        lambda *_: (np.eye(3), np.array([.3, 0., .1])))
    monkeypatch.setattr(selection, 'approach_tilt_deg', lambda _: 0.)

    def pose(R, p, closing_sign, **kwargs):
        T = np.eye(4)
        T[:3, :3] = heading @ np.diag([closing_sign, closing_sign, 1.])
        T[:3, 3] = p
        return T

    monkeypatch.setattr(selection, 'tool_pose_from_grasp', pose)
    grasps = [SimpleNamespace(score=1., width=.06, depth=0.)]
    cfg['drop']['lift_orientation'] = 'home'
    assert selection.select_grasp(grasps, np.eye(4), cfg, robot).selected is None
    cfg['drop']['lift_orientation'] = 'upright'
    cfg['drop']['transfer_blend_radius_mm'] = 0
    selected = selection.select_grasp(grasps, np.eye(4), cfg, robot).selected
    assert selected is not None
    robot.get_tcp_pose = lambda: (selected.T_tcp_grasp, None)
    plan = prepare_transfer(robot, selected.T_tcp_pre, selected.T_tcp_lift,
                            cfg['robot'], cfg['drop'], transfer_sign=selected.transfer_sign,
                            j7_limits=(-89, 90))
    np.testing.assert_allclose(plan.upper[:3, :3], selected.T_tcp_pre[:3, :3], atol=1e-12)
    np.testing.assert_allclose(plan.upper[:3, 2], home[:, 2], atol=1e-12)
    assert not np.allclose(plan.upper[:3, :3], home)
    # Native full-chain checks still happen before any physical command.
    calls = robot.arm.set_position.call_args_list + robot.arm.set_servo_angle.call_args_list
    assert [call.kwargs['only_check_type'] for call in calls] == [1, 3, 2]


def test_grasp_rotation_is_shared_by_selection_and_measured_start_preparation(monkeypatch):
    from scipy.spatial.transform import Rotation
    cfg = load_config()
    cfg['selection'].update(camera_up={'prefer': False}, approach_extra_m=0)
    cfg['drop'].update(lift_orientation='grasp', transfer_blend_radius_mm=0,
                       joints_deg=[0.] * 7)
    home = np.diag([1., -1., -1.])
    rotation = Rotation.from_euler('y', 20., degrees=True).as_matrix() @ home
    queried = []

    def ik(T):
        queried.append(T.copy())
        return [0.] * 7

    robot = SimpleNamespace(
        T_flange_tcp=np.eye(4), R_home_tcp=home,
        get_joints_deg=lambda: np.zeros(7), flange_ik_joints_deg=ik,
        arm=Mock(mode=0, version_number=[2, 7, 1], only_check_result=0))
    robot.arm.get_state.return_value = 0, 2
    robot.arm.get_cmdnum.return_value = 0, 0
    robot.arm.get_err_warn_code.return_value = 0, [0, 0]
    robot.arm.set_position.return_value = robot.arm.set_servo_angle.return_value = 0
    monkeypatch.setattr(selection, 'grasp_to_base',
                        lambda *_: (np.eye(3), np.array([.3, 0., .1])))
    monkeypatch.setattr(selection, 'approach_tilt_deg', lambda _: 20.)

    def pose(R, p, closing_sign, **kwargs):
        T = np.eye(4)
        T[:3, :3] = rotation @ np.diag([closing_sign, closing_sign, 1.])
        T[:3, 3] = p
        return T

    monkeypatch.setattr(selection, 'tool_pose_from_grasp', pose)
    grasps = [SimpleNamespace(score=1., width=.06, depth=0.)]
    selected = selection.select_grasp(grasps, np.eye(4), cfg, robot).selected
    assert selected is not None and selected.transfer_sign == 1
    robot.get_tcp_pose = lambda: (selected.T_tcp_grasp, None)
    queried.clear()
    plan = prepare_transfer(robot, selected.T_tcp_pre, selected.T_tcp_lift,
                            cfg['robot'], cfg['drop'], transfer_sign=selected.transfer_sign,
                            j7_limits=(-89, 90), allow_symmetry=True)
    np.testing.assert_array_equal(plan.upper[:3, :3], selected.T_tcp_pre[:3, :3])
    assert not np.allclose(plan.upper[:3, 2], home[:, 2])
    assert all(np.allclose(T[:3, :3], plan.upper[:3, :3]) for T in queried)
    assert len(plan.commands) == 3
