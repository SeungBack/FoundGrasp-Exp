"""Bounded upper-heading alternatives, using mock feasibility rather than hardware IK."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from efg.motion_command import PlanRejected
from efg.pick_plan import prepare_pick
from efg.se3 import xarm_pose_to_se3
from efg.transfer_plan import (closest_upright_rotation, prepare_transfer,
                               transfer_route_candidates)


@pytest.fixture
def scene():
    home = np.diag([1., -1., -1.])
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_euler('y', 24, degrees=True).as_matrix() @ home
    grasp[:3, 3] = [.46, -.32, .05]
    pre = grasp.copy()
    pre[:3, 3] -= .1 * grasp[:3, 2]
    lift = pre.copy()
    lift[2, 3] = .15
    joints = np.array([-20., 25., -21., 80., 35., 60., 45.])
    arm = SimpleNamespace(mode=0, version_number=(2, 7, 1), only_check_result=0,
                          writes=[], responses=[],
                          get_state=Mock(return_value=(0, 2)),
                          get_cmdnum=Mock(return_value=(0, 0)),
                          get_err_warn_code=Mock(return_value=(0, [0, 0])),
                          clean_error=Mock())

    def submit(kind, *args, **kwargs):
        arm.writes.append((kind, args, kwargs))
        code, result = arm.responses.pop(0) if arm.responses else (0, 0)
        arm.only_check_result = result
        return code

    arm.set_position = Mock(side_effect=lambda *a, **kw: submit('cartesian', *a, **kw))
    arm.set_servo_angle = Mock(side_effect=lambda *a, **kw: submit('joint', *a, **kw))
    robot = SimpleNamespace(
        arm=arm, T_flange_tcp=np.eye(4), R_home_tcp=home,
        get_tcp_pose=Mock(return_value=(grasp, None)),
        get_joints_deg=Mock(return_value=joints),
        flange_ik_joints_deg=Mock(side_effect=lambda _: joints.copy()),
        stop_motion=Mock())
    cfg = {'travel_speed_mm_s': 1000., 'travel_acc_mm_s2': 1800.,
           'approach_speed_mm_s': 250., 'approach_acc_mm_s2': 1100.,
           'lift_speed_mm_s': 1000., 'retreat_speed_mm_s': 850.,
           'lift_acc_mm_s2': 800., 'reorient_speed_mm_s': 1000.,
           'reorient_angular_speed_deg_s': 300., 'reorient_acc_mm_s2': 1800.,
           'retreat_blend_radius_mm': 50., 'rise_ik_step_mm': 20., 'rise_ik_step_deg': 5.}
    drop = {'lift_orientation': 'grasp', 'route_search': True,
            'via_height_m': .35, 'min_transfer_height_m': .25, 'height_search_step_m': .02,
            'upright_yaw_offsets_deg': [15., -15., 30., -30.],
            'transfer_clearance_margin_mm': 10., 'transfer_blend_radius_mm': 0.,
            'joints_deg': [0.] * 7, 'joint_speed_deg_s': 180., 'joint_acc_deg_s2': 350.}
    return SimpleNamespace(robot=robot, arm=arm, grasp=grasp, pre=pre, lift=lift,
                           joints=joints, cfg=cfg, drop=drop)


def routes(s):
    return transfer_route_candidates(s.grasp, s.pre, s.lift, s.cfg, s.drop,
                                     s.robot.R_home_tcp)


def prepare(s):
    return prepare_transfer(s.robot, s.pre, s.lift, s.cfg, s.drop,
                            j7_limits=(-89., 90.), planning_j7_limits=(-79., 80.))


def assert_checks_only(s):
    assert all(call[2]['only_check_type'] in (1, 2, 3) for call in s.arm.writes)
    s.robot.stop_motion.assert_not_called()
    s.arm.clean_error.assert_not_called()


def reject_legacy_endpoints(s):
    """Synthetic geometry: all original upper poses exceed the planning limit."""
    original_upper = [r.upper for r in routes(s) if 'yaw' not in r.name]

    def ik(target):
        q = s.joints.copy()
        if any(np.allclose(target, upper, atol=1e-8, rtol=0) for upper in original_upper):
            q[6] = 95.
        return q

    s.robot.flange_ik_joints_deg.side_effect = ik
    return ik


def single_height(s):
    s.drop.update(via_height_m=.25, upright_yaw_offsets_deg=[15.])


def test_all_existing_heights_keep_priority_before_heading_alternatives(scene):
    s = scene
    enabled = routes(s)
    s.drop['upright_yaw_offsets_deg'] = []
    original = routes(s)
    assert len(original) == 18
    assert [r.name for r in enabled[:18]] == [r.name for r in original]
    assert [r.commands for r in enabled[:18]] == [r.commands for r in original]
    assert len(enabled) == 42
    assert [r.name for r in enabled[18:22]] == [
        'lower_upright_yaw_p15', 'lower_upright_yaw_m15',
        'lower_upright_yaw_p30', 'lower_upright_yaw_m30']
    assert enabled[-1].name == 'upright_yaw_m30'


@pytest.mark.parametrize('blend', [0., 10., 50.])
def test_heading_routes_preserve_escape_and_delay_rotation_above_clearance(scene, blend):
    s = scene
    s.cfg['retreat_blend_radius_mm'] = blend
    upright = closest_upright_rotation(s.pre[:3, :3], s.robot.R_home_tcp)
    found = [r for r in routes(s) if 'yaw' in r.name]
    for route, angle in zip(found, s.drop['upright_yaw_offsets_deg'] * 6):
        assert route.transfer_sign == 1
        np.testing.assert_allclose(route.pre, s.pre)
        np.testing.assert_allclose(route.start, s.grasp)
        np.testing.assert_allclose(route.upper[:3, :3],
                                   upright @ Rotation.from_euler('z', angle, degrees=True).as_matrix(),
                                   atol=1e-8)
        pre, guard, upper = [xarm_pose_to_se3(c.target) for c in route.commands[:-1]]
        np.testing.assert_allclose(pre, s.pre, atol=1e-8)
        np.testing.assert_allclose(guard[:3, :3], s.pre[:3, :3], atol=1e-8)
        np.testing.assert_allclose(upper, route.upper, atol=1e-8)
        for point in (pre, guard, upper):
            np.testing.assert_allclose(point[:2, 3], s.pre[:2, 3])
        floor = s.drop['min_transfer_height_m']
        guard_radius = max(route.commands[1].radius, 0.) / 1000
        assert s.pre[2, 3] < guard[2, 3] < upper[2, 3]
        assert guard[2, 3] >= floor + .5 * (upper[2, 3] - floor) - 1e-10
        assert guard[2, 3] >= s.pre[2, 3] + .2 * (upper[2, 3] - s.pre[2, 3]) - 1e-10
        assert guard_radius <= .25 * (guard[2, 3] - floor) + 1e-10
        assert guard[2, 3] - guard_radius > floor
        assert route.commands[-1].target == tuple(s.drop['joints_deg'])


@pytest.mark.parametrize('disable', ['missing_offsets', 'empty_offsets', 'no_search'])
def test_heading_fallback_is_opt_in(scene, disable):
    s = scene
    if disable == 'missing_offsets':
        s.drop.pop('upright_yaw_offsets_deg')
    elif disable == 'empty_offsets':
        s.drop['upright_yaw_offsets_deg'] = []
    else:
        s.drop['route_search'] = False
        s.drop.pop('min_transfer_height_m')
    assert not any('yaw' in route.name for route in routes(s))


def test_upright_grasp_still_has_heading_alternatives(scene):
    s = scene
    s.grasp[:3, :3] = s.pre[:3, :3] = s.robot.R_home_tcp
    found = routes(s)
    assert sum('yaw' in r.name for r in found) == 24
    assert sum('yaw' not in r.name for r in found) == 6


@pytest.mark.parametrize('offsets', [
    [True], [False], [np.nan], [np.inf], [-np.inf], [0.], [-0.],
    [45.01], [-45.01], ['15'], [15., 15.], list(range(1, 10)), 15., '15',
])
def test_invalid_heading_offsets_cannot_submit_commands(scene, offsets):
    scene.drop['upright_yaw_offsets_deg'] = offsets
    with pytest.raises((ValueError, TypeError)):
        prepare(scene)
    assert not scene.arm.writes


@pytest.mark.parametrize('minimum', [None, 0., -1., np.nan, np.inf])
def test_enabled_heading_search_requires_explicit_valid_clearance(scene, minimum):
    scene.drop['min_transfer_height_m'] = minimum
    with pytest.raises((ValueError, RuntimeError)):
        prepare(scene)
    assert not scene.arm.writes


def test_heading_alternative_can_pass_after_all_legacy_endpoint_rejections(scene):
    s = scene
    reject_legacy_endpoints(s)
    plan = prepare(s)
    assert plan.route_name == 'lower_upright_yaw_p15'
    assert plan.j7_limits == (-89., 90.)
    assert len(s.arm.writes) == 4
    assert [call[2]['only_check_type'] for call in s.arm.writes] == [1, 3, 3, 2]
    assert_checks_only(s)


@pytest.mark.parametrize('location', ['guard', 'rotation_interior'])
def test_heading_guard_and_rotation_interior_keep_j7_reserve(scene, location):
    s = scene
    single_height(s)
    base_ik = reject_legacy_endpoints(s)
    yaw = next(r for r in routes(s) if 'yaw' in r.name)
    guard = xarm_pose_to_se3(yaw.commands[1].target)

    def ik(target):
        q = base_ik(target)
        reject = (np.allclose(target, guard, atol=1e-8, rtol=0) if location == 'guard'
                  else guard[2, 3] + 1e-8 < target[2, 3] < yaw.upper[2, 3] - 1e-8)
        if reject:
            q[6] = 85.  # Still inside hard limit; planning reserve must reject.
        return q

    s.robot.flange_ik_joints_deg.side_effect = ik
    with pytest.raises(PlanRejected, match='J7'):
        prepare(s)
    assert not s.arm.writes


def test_real_fault_during_heading_check_prevents_next_alternative(scene):
    s = scene
    reject_legacy_endpoints(s)

    def reject_and_fault(*args, **kwargs):
        s.arm.writes.append(('cartesian', args, kwargs))
        s.arm.only_check_result = 21
        s.arm.get_err_warn_code.return_value = 0, [31, 14]
        return 1

    s.arm.set_position.side_effect = reject_and_fault
    with pytest.raises(RuntimeError) as caught:
        prepare(s)
    assert not isinstance(caught.value, PlanRejected)
    assert len(s.arm.writes) == 1
    assert_checks_only(s)


def test_all_heading_native_rejections_remain_plan_rejections(scene):
    s = scene
    single_height(s)
    reject_legacy_endpoints(s)
    s.arm.responses = [(0, 0), (1, 21)]
    with pytest.raises(PlanRejected, match='only_check_result=21'):
        prepare(s)
    assert [call[2]['only_check_type'] for call in s.arm.writes] == [1, 3]
    assert_checks_only(s)


def test_same_heading_routes_are_checked_before_approach(scene):
    s = scene
    reject_legacy_endpoints(s)
    start = s.pre.copy()
    start[2, 3] = .5
    s.robot.get_tcp_pose.return_value = start, None
    candidate = SimpleNamespace(T_tcp_pre=s.pre, T_tcp_grasp=s.grasp,
                                T_tcp_lift=s.lift, transfer_sign=1)
    cfg = {'robot': s.cfg, 'drop': s.drop,
           'selection': {'j7_limit_deg': [-89., 90.], 'camera_up': {'prefer': False}}}
    plan = prepare_pick(s.robot, candidate, cfg, j7_limits=(-79., 80.))
    assert plan.transfer_route.name == 'lower_upright_yaw_p15'
    assert plan.transfer_route.commands == next(
        r.commands for r in routes(s) if r.name == plan.transfer_route.name)
    assert len(s.arm.writes) == 6
    assert [call[2]['only_check_type'] for call in s.arm.writes] == [1, 3, 3, 3, 3, 2]
    assert plan.approach_commands[-1].radius == -1
    assert_checks_only(s)
