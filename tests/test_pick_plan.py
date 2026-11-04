"""Whole-cycle native planning and candidate rejection without hardware."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg import pick_plan
from efg.camera import CameraError
from efg.motion_command import MotionCommand, MotionCheckError, PlanRejected, approach_commands
from efg.se3 import se3_to_xarm_pose
from efg.transfer_plan import TransferRoute


@pytest.fixture
def scene(monkeypatch):
    start = np.eye(4)
    start[:3, :3] = np.diag([1., -1., -1.])
    start[:3, 3] = [.4, -.2, .5]
    grasp, pre, upper = (start.copy() for _ in range(3))
    grasp[2, 3], pre[2, 3], upper[2, 3] = .05, .15, .36
    cfg = {
        'robot': {'travel_speed_mm_s': 600., 'approach_speed_mm_s': 400.,
                  'approach_acc_mm_s2': 700., 'pregrasp_blend_radius_mm': 10.,
                  'rise_ik_step_mm': 20., 'rise_ik_step_deg': 5.},
        'selection': {'j7_limit_deg': [-89., 90.], 'camera_up': {'prefer': False}},
        'drop': {},
    }
    arm = SimpleNamespace(mode=0, version_number=[2, 7, 1], only_check_result=0,
                          errors=[0, 0], state=2, pending=0, writes=[], reject=None)
    arm.get_state = lambda: (0, arm.state)
    arm.get_cmdnum = lambda: (0, arm.pending)
    arm.get_err_warn_code = lambda: (0, list(arm.errors))

    def submit(kind, *args, **kw):
        arm.writes.append((kind, args, kw))
        code, result = arm.reject(kind, args, kw) if arm.reject else (0, 0)
        arm.only_check_result = result
        return code

    arm.set_position = Mock(side_effect=lambda *a, **k: submit('cartesian', *a, **k))
    arm.set_servo_angle = Mock(side_effect=lambda **k: submit('joint', **k))
    robot = SimpleNamespace(
        arm=arm, T_flange_tcp=np.eye(4), R_home_tcp=start[:3, :3],
        get_tcp_pose=Mock(return_value=(start, None)),
        get_joints_deg=Mock(return_value=np.zeros(7)),
        flange_ik_joints_deg=Mock(return_value=[0.] * 7), stop_motion=Mock())
    selected = SimpleNamespace(T_tcp_grasp=grasp, T_tcp_pre=pre,
                               T_tcp_lift=upper, transfer_sign=1)

    def route(name, upper_x):
        target = upper.copy()
        target[0, 3] = upper_x
        commands = (
            MotionCommand('cartesian', tuple(se3_to_xarm_pose(pre)), 600, 350, 10, '복귀'),
            MotionCommand('cartesian', tuple(se3_to_xarm_pose(target)), 660, 770, -1, '상승'),
            MotionCommand('joint', (10.,) * 7, 135, 220, -1, '이송'),
        )
        return TransferRoute(name, pre, target, 1, commands, grasp)

    routes = (route('configured', .4), route('alternative', .45))
    monkeypatch.setattr(pick_plan, 'transfer_route_candidates', lambda *a, **kw: iter(routes))
    return SimpleNamespace(robot=robot, arm=arm, cfg=cfg, selected=selected,
                           routes=routes, start=start, grasp=grasp, pre=pre)


def prepare(scene, **kw):
    return pick_plan.prepare_pick(scene.robot, scene.selected, scene.cfg, **kw)


def assert_checks_only(scene):
    assert scene.arm.writes
    assert all(item[2]['only_check_type'] in (1, 2, 3) for item in scene.arm.writes)
    scene.robot.stop_motion.assert_not_called()


def test_native_c21_rejects_first_whole_route_and_selects_alternative_before_approach(scene):
    s = scene
    s.arm.reject = lambda kind, args, kw: (
        (1, 21) if kind == 'cartesian' and args[0] == 400 and args[2] == 360 else (0, 0))
    plan = prepare(s)
    assert plan.transfer_route.name == 'alternative'
    assert [item[2]['only_check_type'] for item in s.arm.writes] == [1, 3, 3, 3, 1, 3, 3, 3, 2]
    assert plan.approach_commands[-1].radius == -1  # closure still occurs at an exact stop
    assert_checks_only(s)


def test_all_native_c21_routes_are_expected_candidate_rejection(scene):
    scene.arm.reject = lambda kind, args, kw: (
        (1, 21) if kind == 'cartesian' and args[2] == 360 else (0, 0))
    with pytest.raises(PlanRejected, match='configured.*alternative'):
        prepare(scene)
    assert len(scene.arm.writes) == 8
    assert_checks_only(scene)


@pytest.mark.parametrize('fault', ['error', 'state', 'pending'])
def test_native_rejection_with_actual_robot_fault_is_fatal_not_candidate_rejection(scene, fault):
    def reject(*_):
        if fault == 'error':
            scene.arm.errors[0] = 31
        elif fault == 'state':
            scene.arm.state = 5
        else:
            scene.arm.pending = 1
        return 1, 21
    scene.arm.reject = reject
    with pytest.raises(RuntimeError) as caught:
        prepare(scene)
    assert not isinstance(caught.value, PlanRejected)
    assert len(scene.arm.writes) == 1
    assert_checks_only(scene)


@pytest.mark.parametrize('code,result', [(-1, 0), (1, 31), (0, 21), (1, None)])
def test_transport_or_unknown_result_does_not_try_another_route(scene, code, result):
    scene.arm.reject = lambda *_: (code, result)
    with pytest.raises(MotionCheckError) as caught:
        prepare(scene)
    assert not isinstance(caught.value, PlanRejected)
    assert len(scene.arm.writes) == 1
    assert_checks_only(scene)


def test_full_chain_retiming_keeps_exact_accepted_approach_and_transfer_commands(scene):
    first = [True]
    def reject_once(kind, args, kw):
        if first[0] and kind == 'cartesian' and args[2] == 360:
            first[0] = False
            return 1, 24
        return 0, 0
    scene.arm.reject = reject_once
    plan = prepare(scene)
    assert plan.speed_scale == .8
    original = approach_commands(scene.start, scene.pre, scene.grasp, scene.cfg['robot']) + scene.routes[0].commands
    accepted = plan.approach_commands + plan.transfer_route.commands
    for nominal, actual, written in zip(original, accepted, scene.arm.writes[-5:]):
        assert actual.speed == pytest.approx(nominal.speed * .8)
        assert actual.acceleration == pytest.approx(nominal.acceleration * .64)
        assert actual.target == nominal.target and actual.radius == nominal.radius
        assert written[2]['speed'] == actual.speed
        assert written[2]['mvacc'] == actual.acceleration
        assert written[2]['mvtime'] == 0
    assert_checks_only(scene)


def test_plan_snapshots_are_immutable_and_detached_from_inputs(scene):
    plan = prepare(scene)
    original_grasp = plan.grasp.copy()
    scene.selected.T_tcp_grasp[2, 3] = 99.
    scene.cfg['robot']['travel_speed_mm_s'] = 999.
    np.testing.assert_array_equal(plan.grasp, original_grasp)
    assert plan.approach_commands[0].speed == 600.
    with pytest.raises(ValueError):
        plan.start_tcp[0, 3] = 0.
    with pytest.raises(FrozenInstanceError):
        plan.speed_scale = .1


def test_changed_actual_start_after_success_prevents_returning_a_plan(scene):
    def submit(kind, args, kw):
        if kind == 'joint':
            scene.robot.get_joints_deg.return_value = [1.] * 7
        return 0, 0
    scene.arm.reject = submit
    with pytest.raises(RuntimeError, match='시작 자세가 변경') as caught:
        prepare(scene)
    assert not isinstance(caught.value, PlanRejected)
    assert_checks_only(scene)


def test_camera_failure_is_not_converted_into_candidate_rejection(scene):
    with pytest.raises(CameraError, match='camera stopped'):
        prepare(scene, check=Mock(side_effect=CameraError('camera stopped')))
    assert not scene.arm.writes


def test_inner_j7_limits_are_applied_before_native_commands(scene):
    scene.robot.flange_ik_joints_deg.return_value = [0.] * 6 + [84.]
    with pytest.raises(PlanRejected, match='J7'):
        prepare(scene, j7_limits=(-79., 80.))
    assert not scene.arm.writes


def test_actual_start_j7_outside_hard_limit_is_a_state_failure(scene):
    scene.robot.get_joints_deg.return_value = [0.] * 6 + [91.]
    with pytest.raises(RuntimeError, match='실측 J7') as caught:
        prepare(scene)
    assert not isinstance(caught.value, PlanRejected)
    assert not scene.arm.writes


def test_release_joint_outside_hard_limit_is_not_a_retryable_grasp_failure(scene, monkeypatch):
    from dataclasses import replace
    route = scene.routes[0]
    commands = (*route.commands[:-1], replace(route.commands[-1], target=[0.] * 6 + [91.]))
    invalid = replace(route, commands=commands)
    monkeypatch.setattr(pick_plan, 'transfer_route_candidates', lambda *a, **kw: [invalid])
    with pytest.raises(RuntimeError, match='릴리즈 목표 J7') as caught:
        prepare(scene)
    assert not isinstance(caught.value, PlanRejected)
    assert not scene.arm.writes
