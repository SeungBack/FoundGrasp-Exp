"""Search alternative Cartesian return paths without sending real motion."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from efg.camera import CameraError
from efg.motion_command import MotionCheckError, PlanRejected
from efg.se3 import xarm_pose_to_se3
from efg.transfer_plan import (TransferRoute, closest_upright_rotation,
                               prepare_transfer, route_path_error, transfer_commands,
                               transfer_route_candidates, transfer_waypoint)


@pytest.fixture
def scene():
    home = np.diag([1., -1., -1.])
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_euler('y', 24, degrees=True).as_matrix() @ home
    grasp[:3, 3] = [.46, -.32, .05]
    pre = grasp.copy()
    pre[:3, 3] -= .1 * grasp[:3, 2]
    lift = pre.copy()
    lift[2, 3] = .35
    joints = np.array([-20., 25., -21., 80., 35., 60., -47.])
    arm = SimpleNamespace(version_number=(2, 7, 1), only_check_result=0, mode=0,
                          writes=[], responses=[],
                          get_state=Mock(return_value=(0, 2)),
                          get_cmdnum=Mock(return_value=(0, 0)),
                          get_err_warn_code=Mock(return_value=(0, [0, 0])))

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
    cfg = {'lift_speed_mm_s': 600, 'retreat_speed_mm_s': 400,
           'lift_acc_mm_s2': 700, 'reorient_speed_mm_s': 500,
           'reorient_angular_speed_deg_s': 180, 'reorient_acc_mm_s2': 700,
           'retreat_blend_radius_mm': 10, 'rise_ik_step_mm': 20, 'rise_ik_step_deg': 5}
    drop = {'lift_orientation': 'grasp', 'route_search': True,
            'via_height_m': .35, 'transfer_clearance_margin_mm': 10,
            'transfer_blend_radius_mm': 0, 'joints_deg': [0.] * 7,
            'joint_speed_deg_s': 120, 'joint_acc_deg_s2': 200}
    return SimpleNamespace(robot=robot, arm=arm, grasp=grasp, pre=pre, lift=lift,
                           joints=joints, cfg=cfg, drop=drop)


def routes(s, **kwargs):
    return transfer_route_candidates(s.grasp, s.pre, s.lift, s.cfg, s.drop,
                                     s.robot.R_home_tcp, **kwargs)


def prepare(s, **kwargs):
    return prepare_transfer(s.robot, s.pre, s.lift, s.cfg, s.drop,
                            j7_limits=(-89., 90.), **kwargs)


def assert_checks_only(s):
    assert all(call[2]['only_check_type'] in (1, 2, 3) for call in s.arm.writes)
    s.robot.stop_motion.assert_not_called()


def test_disabled_search_preserves_original_three_commands(scene):
    s = scene
    s.drop.pop('route_search')
    found = routes(s)
    assert len(found) == 1 and found[0].name == 'configured'
    upper, radius = transfer_waypoint(s.pre, s.lift, s.drop, s.robot.R_home_tcp)
    assert found[0].commands == transfer_commands(s.grasp, s.pre, upper, radius, s.cfg, s.drop)


def test_route_family_preserves_clearance_and_delays_rotation_above_pre(scene):
    s = scene
    found = routes(s, allow_symmetry=True)
    assert [r.name for r in found] == [
        'configured', 'half_tilt', 'upright']
    assert [r.commands for r in found] == [r.commands for r in routes(s)]
    upright = closest_upright_rotation(s.pre[:3, :3], s.robot.R_home_tcp)
    for route in found:
        assert route.transfer_sign == 1  # No 180-degree upper fallback in grasp mode.
        np.testing.assert_allclose(route.pre, s.pre)
        np.testing.assert_allclose(route.upper[:3, 3], found[0].upper[:3, 3])
        points = [xarm_pose_to_se3(c.target) for c in route.commands[:-1]]
        assert np.all(np.diff([p[2, 3] for p in points]) > 0)
        for point in points:
            np.testing.assert_allclose(point[:2, 3], s.pre[:2, 3])
        if route.name == 'configured':
            continue
        guard = points[1]
        np.testing.assert_allclose(guard[:3, :3], s.pre[:3, :3], atol=1e-8)
        assert guard[2, 3] - route.commands[1].radius / 1000 > s.pre[2, 3]
        tilt = Rotation.from_matrix(route.upper[:3, :3].T @ upright).magnitude()
        assert np.rad2deg(tilt) == pytest.approx(12 if 'half_tilt' in route.name else 0)


def test_native_c21_on_configured_rise_selects_alternative_for_same_grasp(scene):
    s = scene
    s.arm.responses = [(0, 0), (1, 21)]
    plan = prepare(s)
    assert plan.route_name == 'half_tilt' and len(plan.commands) == 4
    np.testing.assert_allclose(plan.start_tcp, s.grasp)
    np.testing.assert_allclose(plan.pre, s.pre)
    assert [c[2]['only_check_type'] for c in s.arm.writes] == [1, 3, 1, 3, 3, 2]
    assert not plan.continuous  # The last Cartesian command controls this, not the guard.
    assert_checks_only(s)


def test_geometry_rejection_searches_upright_without_lowering_clearance(scene):
    s = scene
    nominal_height = routes(s)[0].upper[2, 3]

    def ik(target):
        tilt = Rotation.from_matrix(target[:3, :3].T @ s.robot.R_home_tcp).magnitude()
        return None if target[2, 3] >= nominal_height - 1e-8 and tilt > .01 else s.joints

    s.robot.flange_ik_joints_deg.side_effect = ik
    plan = prepare(s)
    assert plan.route_name == 'upright'
    assert plan.upper[2, 3] == pytest.approx(nominal_height)
    assert len(s.arm.writes) == 4
    assert_checks_only(s)


def test_all_native_geometric_rejections_remain_clean_plan_rejections(scene):
    s = scene
    # Each route succeeds at retreat; its first subsequent segment fails.
    s.arm.responses = [(0, 0), (1, 21)] * 3
    with pytest.raises(PlanRejected, match='only_check_result=21'):
        prepare(s)
    assert len(s.arm.writes) == 6
    assert_checks_only(s)


@pytest.mark.parametrize('code,result', [(-1, 21), (1, 31), (1, 0), (0, 21)])
def test_native_transport_unknown_or_real_fault_never_search_another_route(scene, code, result):
    s = scene
    s.arm.responses = [(code, result)]
    with pytest.raises(MotionCheckError) as caught:
        prepare(s)
    assert not isinstance(caught.value, PlanRejected)
    assert len(s.arm.writes) == 1
    assert_checks_only(s)


def test_latched_fault_after_clean_check_rejection_prevents_search(scene):
    s = scene
    submit = s.arm.set_position.side_effect

    def reject_then_fault(*args, **kwargs):
        code = submit(*args, **kwargs)
        s.arm.get_err_warn_code.return_value = 0, [31, 14]
        return code

    s.arm.set_position.side_effect = reject_then_fault
    s.arm.responses = [(1, 21)]
    with pytest.raises(RuntimeError, match='정지 확인 실패') as caught:
        prepare(s)
    assert not isinstance(caught.value, PlanRejected)
    assert len(s.arm.writes) == 1
    assert_checks_only(s)


def test_camera_loss_during_native_rejection_is_never_a_retryable_plan(scene):
    s = scene
    s.arm.responses = [(1, 21)]

    def check():
        if s.arm.writes:
            raise CameraError('camera disconnected during native planning')

    with pytest.raises(CameraError):
        prepare(s, check=check)
    assert len(s.arm.writes) == 1
    assert_checks_only(s)


@pytest.mark.parametrize('dense', [False, True])
def test_preferred_retimed_route_keeps_exact_commands_without_nominal_duplicate(scene, dense):
    s = scene
    if dense:
        s.lift[2, 3] = .20
        s.drop['min_transfer_height_m'] = .25
    original = routes(s)[1]
    slowed = replace(original, commands=tuple(
        replace(c, speed=c.speed * .6, acceleration=c.acceleration * .36)
        for c in original.commands))
    found = routes(s, preferred_route=slowed)
    assert found[0].commands == slowed.commands
    assert len(found) == (18 if dense else 3)
    assert sum(r.name == original.name for r in found) == 1
    plan = prepare(s, preferred_route=slowed)
    assert plan.commands == slowed.commands
    assert_checks_only(s)


def test_hard_j7_constraint_applies_to_route_interiors(scene):
    s = scene
    route = routes(s)[1]
    guard_z = xarm_pose_to_se3(route.commands[1].target)[2, 3]

    def ik(target):
        q = s.joints.copy()
        if guard_z + .02 < target[2, 3] < route.upper[2, 3] - .02:
            q[6] = 91.
        return q

    s.robot.flange_ik_joints_deg.side_effect = ik
    reason = route_path_error(s.robot, route, s.cfg, (-89., 90.))
    assert 'J7 91.00' in reason and '상승 경로' in reason
    assert_checks_only(s)


def test_camera_constraint_applies_to_interiors_even_if_both_endpoints_pass(scene):
    s = scene
    axis = np.array([np.sqrt(.75), 0., .5])
    pre, upper = s.pre.copy(), s.pre.copy()
    pre[:3, :3] = Rotation.from_rotvec(axis * np.deg2rad(100)).as_matrix()
    upper[:3, :3] = Rotation.from_rotvec(axis * np.deg2rad(260)).as_matrix()
    upper[2, 3] = .36
    commands = transfer_commands(pre, pre, upper, 0, s.cfg, s.drop)
    route = TransferRoute('camera_arc', pre, upper, 1, commands)
    direction = np.array([0., 0., 1.])
    assert (pre[:3, :3] @ direction)[2] > 0
    assert (upper[:3, :3] @ direction)[2] > 0
    assert '카메라 하향 제한 (상승 경로' in route_path_error(
        s.robot, route, s.cfg, camera_up=(direction, 0.))
    assert_checks_only(s)


@pytest.mark.parametrize('code', [-1, 1, 3, 9, 11, 12])
def test_ik_transport_failure_is_not_reported_as_geometric_rejection(scene, code):
    s = scene
    s.robot.flange_ik_joints_deg.side_effect = None
    s.robot.flange_ik_joints_deg.return_value = None
    s.robot.last_ik_failure = f'SDK code={code}, TCP=[...]'
    with pytest.raises(RuntimeError, match='IK 조회 실패'):
        route_path_error(s.robot, routes(s)[0], s.cfg)
    assert_checks_only(s)


def test_sdk_10_at_lower_height_searches_higher_route_without_motion(scene):
    s = scene
    s.lift[2, 3] = .20
    s.drop.update(min_transfer_height_m=.25, height_search_step_m=.02)
    s.arm.get_err_warn_code.return_value = 0, [0, 14]  # no-solution warning, no actual fault

    def ik(target):
        s.robot.last_ik_failure = None
        if np.isclose(target[2, 3], .26):
            s.robot.last_ik_failure = 'SDK code=10, TCP(mm/rad)=[454, -343, 260, -2.81, .11, .43]'
            return None
        return s.joints.copy()

    s.robot.flange_ik_joints_deg.side_effect = ik
    plan = prepare(s)
    assert plan.route_name == 'level_1_configured'
    assert plan.upper[2, 3] == pytest.approx(.28)
    assert_checks_only(s)


def test_all_sdk_10_routes_are_clean_plan_rejection(scene):
    s = scene

    def ik(target):
        s.robot.last_ik_failure = None
        if target[2, 3] > s.pre[2, 3] + .01:
            s.robot.last_ik_failure = 'SDK code=10, TCP=[...]'
            return None
        return s.joints.copy()

    s.robot.flange_ik_joints_deg.side_effect = ik
    with pytest.raises(PlanRejected, match='SDK code=10'):
        prepare(s)
    assert not s.arm.writes
    assert_checks_only(s)


@pytest.mark.parametrize('fault', ['controller', 'state', 'read'])
def test_sdk_10_does_not_hide_actual_fault_or_failed_state_query(scene, fault):
    s = scene
    s.robot.flange_ik_joints_deg.side_effect = None
    s.robot.flange_ik_joints_deg.return_value = None
    s.robot.last_ik_failure = 'SDK code=10, TCP=[...]'
    if fault == 'controller':
        s.arm.get_err_warn_code.return_value = 0, [31, 14]
    elif fault == 'state':
        s.arm.get_state.return_value = 0, 5
    else:
        s.arm.get_state.return_value = 3, 2
    with pytest.raises(RuntimeError, match='정지 확인 실패'):
        route_path_error(s.robot, routes(s)[0], s.cfg)
    assert not s.arm.writes
    assert_checks_only(s)


def test_route_owns_readonly_geometry_and_rejects_mismatched_commands(scene):
    s = scene
    route = routes(s)[0]
    for name in ('pre', 'upper', 'start'):
        with pytest.raises(ValueError, match='read-only'):
            getattr(route, name)[0, 3] = 0
    with pytest.raises(FrozenInstanceError):
        route.name = 'changed'
    changed = s.pre.copy()
    changed[0, 3] += .1
    with pytest.raises(ValueError, match='명령 목표가 일치하지'):
        replace(route, pre=changed)


def test_explicit_minimum_prioritizes_low_routes_with_clearance_margin(scene):
    s = scene
    s.lift[2, 3] = .20
    s.drop['min_transfer_height_m'] = .25
    found = routes(s)
    assert len(found) == 18
    assert [r.name for r in found[:3]] == [
        'lower_configured', 'lower_half_tilt', 'lower_upright']
    assert [r.name for r in found[-3:]] == ['configured', 'half_tilt', 'upright']
    assert [r.upper[2, 3] for r in found] == pytest.approx(
        np.repeat([.26, .28, .30, .32, .34, .36], 3))
    plan = prepare(s)
    assert plan.route_name == 'lower_configured'
    assert plan.upper[2, 3] == pytest.approx(.26)
    assert len(s.arm.writes) == 3
    assert_checks_only(s)


@pytest.mark.parametrize('bound', ['object_escape', 'pre_grasp'])
def test_minimum_height_never_reduces_required_object_escape_height(scene, bound):
    s = scene
    s.drop['min_transfer_height_m'] = .20
    if bound == 'object_escape':
        s.lift[2, 3] = .29
        clearance = .29
    else:
        s.pre[2, 3] = .26
        s.lift[2, 3] = .28
        clearance = .31
    found = routes(s)
    assert found[0].upper[2, 3] == pytest.approx(clearance + .01)
    assert all(r.upper[2, 3] >= clearance + .01 - 1e-10 for r in found)


@pytest.mark.parametrize('step,expected', [
    (.01, np.linspace(.26, .36, 11)),
    (.027, [.26, .287, .314, .341, .36]),
    (1e20, [.26, .36]),
])
def test_height_spacing_is_configurable_and_both_endpoints_are_retained(scene, step, expected):
    s = scene
    s.lift[2, 3] = .20
    s.drop.update(min_transfer_height_m=.25, height_search_step_m=step)
    found = routes(s)
    actual = [r.upper[2, 3] for r in found[::3]]
    assert actual == pytest.approx(expected)
    assert np.all(np.diff(actual) > 0)
    assert np.max(np.diff(actual)) <= step + 1e-10
    assert found[-1].name == 'upright'


@pytest.mark.parametrize('step', [0, -.01, float('nan'), float('inf')])
def test_invalid_height_spacing_fails_before_native_commands(scene, step):
    s = scene
    s.drop['height_search_step_m'] = step
    with pytest.raises(ValueError, match='height_search_step_m'):
        prepare(s)
    assert not s.arm.writes


@pytest.mark.parametrize('step', [1e-5, 1e-320])
def test_excessive_height_search_is_rejected_instead_of_silently_truncated(scene, step):
    s = scene
    s.lift[2, 3] = .20
    s.drop.update(min_transfer_height_m=.25, height_search_step_m=step)
    with pytest.raises(ValueError, match='64개를 초과'):
        prepare(s)
    assert not s.arm.writes


def test_null_minimum_never_invents_a_lower_clearance(scene):
    s = scene
    s.lift[2, 3] = .20
    s.drop['min_transfer_height_m'] = None
    assert all(r.upper[2, 3] == pytest.approx(.36) for r in routes(s))


def test_minimum_above_nominal_is_honored_even_without_search(scene):
    s = scene
    s.drop.update(min_transfer_height_m=.40, route_search=False)
    found = routes(s)
    assert len(found) == 1
    assert found[0].upper[2, 3] == pytest.approx(.41)


@pytest.mark.parametrize('value', [0, -.1, float('nan'), float('inf')])
def test_invalid_minimum_height_fails_before_any_native_command(scene, value):
    s = scene
    s.drop['min_transfer_height_m'] = value
    with pytest.raises(ValueError, match='min_transfer_height_m'):
        prepare(s)
    assert not s.arm.writes


def test_rejected_low_routes_progress_to_next_height_without_mutating_grasp(scene):
    s = scene
    s.lift[2, 3] = .20
    s.drop['min_transfer_height_m'] = .25
    s.arm.responses = [(0, 0), (1, 21)] * 3
    plan = prepare(s)
    assert plan.route_name == 'level_1_configured'
    assert plan.upper[2, 3] == pytest.approx(.28)
    np.testing.assert_allclose(plan.pre, s.pre)
    np.testing.assert_allclose(plan.start_tcp, s.grasp)
    assert_checks_only(s)


def test_dual_sign_dense_search_keeps_nominal_upright_after_all_lower_routes(scene):
    s = scene
    s.lift[2, 3] = .20
    s.drop.update(min_transfer_height_m=.25, lift_orientation='home')
    s.robot.R_home_tcp = Rotation.from_euler('z', 90, degrees=True).as_matrix() @ s.robot.R_home_tcp
    upright = closest_upright_rotation(s.pre[:3, :3], s.robot.R_home_tcp)
    found = routes(s, allow_symmetry=True)
    assert len(found) == 18
    assert found[-1].name == 'upright'

    def ik(target):
        if target[2, 3] > .25:
            if target[2, 3] < .359 or not np.allclose(target[:3, :3], upright):
                return None
        return s.joints

    # Endpoint rejection is enough to establish the required candidate exists;
    # permit interior poses for its final geometric check.
    endpoints = [r.upper for r in found]

    def endpoint_ik(target):
        return ik(target) if any(np.allclose(target, p) for p in endpoints) else s.joints

    s.robot.flange_ik_joints_deg.side_effect = endpoint_ik
    plan = prepare(s, allow_symmetry=True)
    assert plan.route_name == 'upright'
    assert plan.upper[2, 3] == pytest.approx(.36)
    assert_checks_only(s)
