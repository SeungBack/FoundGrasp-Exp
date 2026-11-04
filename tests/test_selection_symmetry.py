"""Endpoint route selection using synthetic IK, without a robot connection."""
from types import SimpleNamespace

import numpy as np
import pytest

from efg import selection
from efg.config import load_config
from efg.motion_command import PlanRejected


@pytest.fixture
def scene(monkeypatch):
    cfg = load_config()
    cfg['drop']['lift_orientation'] = 'upright'
    cfg['selection'].update(camera_up={'prefer': False}, approach_extra_m=0)
    cfg['drop']['joints_deg'] = [0.] * 7
    home = np.diag([1., -1., -1.])
    current = np.zeros(7)
    joints = {(phase, sign): np.zeros(7) for phase in ('pre', 'grasp', 'upper')
              for sign in (1, -1)}

    def ik(T):
        phase = 'upper' if T[2, 3] > .3 else ('pre' if T[2, 3] > .15 else 'grasp')
        return joints[phase, 1 if T[0, 0] > 0 else -1]

    robot = SimpleNamespace(T_flange_tcp=np.eye(4), R_home_tcp=home,
                            get_joints_deg=lambda: current.copy(), flange_ik_joints_deg=ik)
    grasps = [SimpleNamespace(score=1., width=.06, depth=0.)]
    monkeypatch.setattr(selection, 'grasp_to_base',
                        lambda *_: (np.eye(3), np.array([.3, 0., .1])))
    monkeypatch.setattr(selection, 'approach_tilt_deg', lambda _: 0.)

    def pose(R, p, closing_sign, **kwargs):
        T = np.eye(4)
        T[:3, :3] = home @ np.diag([closing_sign, closing_sign, 1.])
        T[:3, 3] = p
        return T

    monkeypatch.setattr(selection, 'tool_pose_from_grasp', pose)
    return SimpleNamespace(cfg=cfg, robot=robot, joints=joints, current=current,
                           select=lambda: selection.select_grasp(grasps, np.eye(4), cfg, robot))


def test_uses_actual_all_joint_angles_not_home_j7(scene):
    p = scene
    p.current[0] = 100
    p.cfg['drop']['joints_deg'][0] = 100
    for phase in ('pre', 'grasp', 'upper'):
        p.joints[phase, 1][0] = 0
        p.joints[phase, -1][0] = 90
    s = p.select().selected
    assert s.closing_sign == -1
    # Upright's base heading follows this grasp's sign; compare the physical
    # upper orientation rather than its old home-relative sign label.
    target, _ = selection.transfer_waypoint(
        s.T_tcp_pre, s.T_tcp_lift, p.cfg['drop'], p.robot.R_home_tcp, s.transfer_sign)
    assert target[0, 0] < 0


def test_nearer_upper_symmetry_is_selected(scene):
    p = scene
    for phase in ('pre', 'grasp'):
        for sign in (1, -1):
            p.joints[phase, sign][0] = 70
    p.joints['upper', 1][0] = -60
    p.joints['upper', -1][0] = 80
    p.cfg['drop']['joints_deg'][0] = 90
    assert p.select().selected.transfer_sign == -1


def test_drop_leg_is_included_even_if_upper_leg_alone_is_longer(scene):
    p = scene
    for phase in ('pre', 'grasp'):
        for sign in (1, -1):
            p.joints[phase, sign][0] = 20
    p.joints['upper', 1][0] = 0
    p.joints['upper', -1][0] = 50
    p.cfg['drop']['joints_deg'][0] = 80
    assert p.select().selected.transfer_sign == -1


@pytest.mark.parametrize('invalid', ['ik', 'j7'])
def test_invalid_upper_alternative_is_excluded(scene, invalid):
    p = scene
    if invalid == 'ik':
        p.joints['upper', 1] = None
    else:
        p.joints['upper', 1][6] = 100
    assert p.select().selected.transfer_sign == -1
    assert p.select().rejected
    p.joints['upper', -1] = None
    assert p.select().selected is None


def test_malformed_ik_read_aborts_instead_of_trying_other_candidates(scene):
    scene.joints['pre', 1][6] = np.nan
    with pytest.raises(RuntimeError, match='IK 관절 판독 오류'):
        scene.select()


def test_ik_transport_failure_is_not_reported_as_no_candidate(scene):
    scene.robot.flange_ik_joints_deg = lambda _: None
    scene.robot.last_ik_failure = 'SDK code=-1, TCP=[0,0,0,0,0,0]'
    with pytest.raises(RuntimeError, match='IK 조회 실패'):
        scene.select()


def test_sdk_10_at_grasp_selects_other_symmetry(scene):
    scene.robot.arm = SimpleNamespace(
        mode=0, get_state=lambda: (0, 2), get_cmdnum=lambda: (0, 0),
        get_err_warn_code=lambda: (0, [0, 14]))
    original = scene.robot.flange_ik_joints_deg

    def ik(target):
        scene.robot.last_ik_failure = None
        if target[0, 0] > 0 and target[2, 3] < .15:
            scene.robot.last_ik_failure = 'SDK code=10, TCP=[...]'
            return None
        return original(target)

    scene.robot.flange_ik_joints_deg = ik
    report = scene.select()
    assert report.selected.closing_sign == -1
    assert any('IK 불가' in reason for _, reason in report.rejected)


def test_camera_down_limit_overrides_shorter_route(scene, monkeypatch):
    p = scene
    p.cfg['selection']['camera_up'] = {'prefer': True, 'max_down_deg': 20}
    # Synthetic calibration direction makes the -1 alternative point down.
    monkeypatch.setattr(selection, '_camera_lateral_dir_tcp', lambda *_: np.array([1., 0., 0.]))
    original = selection.tool_pose_from_grasp

    def tilted(*args, **kwargs):
        T = original(*args, **kwargs)
        theta = np.radians(-30)
        R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
        T[:3, :3] = R @ T[:3, :3]
        return T

    monkeypatch.setattr(selection, 'tool_pose_from_grasp', tilted)
    p.joints['pre', 1][0] = p.joints['grasp', 1][0] = 50
    report = p.select()
    assert report.selected.closing_sign == 1
    assert any('카메라 하향' in why for _, why in report.rejected)


def test_camera_up_is_only_tiebreaker_when_both_directions_allowed(scene, monkeypatch):
    p = scene
    p.cfg['selection']['camera_up'] = {'prefer': True, 'max_down_deg': 20}
    monkeypatch.setattr(selection, '_camera_lateral_dir_tcp', lambda *_: np.array([1., 0., 0.]))
    original = selection.tool_pose_from_grasp

    def tilted(*args, **kwargs):
        T = original(*args, **kwargs)
        theta = np.radians(-10)
        R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
        T[:3, :3] = R @ T[:3, :3]
        return T

    monkeypatch.setattr(selection, 'tool_pose_from_grasp', tilted)
    p.joints['pre', 1][0] = p.joints['grasp', 1][0] = 50
    assert p.select().selected.closing_sign == -1
    p.joints['pre', 1][0] = p.joints['grasp', 1][0] = 0
    assert p.select().selected.closing_sign == 1


def test_joint_cost_does_not_wrap_away_cable_rotation():
    a, b = np.zeros(7), np.zeros(7)
    a[6], b[6] = -170, 170
    assert selection._joint_route_cost(a, b) == (340., 340.)


def test_whole_path_rejection_tries_other_grasp_symmetry_before_approach(scene):
    visited = []

    def prepare(candidate):
        visited.append(candidate.closing_sign)
        if candidate.closing_sign == 1:
            raise PlanRejected('native check C21')
        return candidate

    grasps = [SimpleNamespace(score=1., width=.06, depth=0.)]
    report = selection.select_grasp(grasps, np.eye(4), scene.cfg, scene.robot,
                                    candidate_prepare=prepare)
    assert visited == [1, -1]
    assert report.selected.closing_sign == -1
    assert any('C21' in reason for _, reason in report.rejected)


def test_alternative_route_planner_is_not_vetoed_by_single_configured_upper(scene):
    for sign in (1, -1):
        scene.joints['upper', sign] = None
    grasps = [SimpleNamespace(score=1., width=.06, depth=0.)]
    assert scene.select().selected is None
    report = selection.select_grasp(grasps, np.eye(4), scene.cfg, scene.robot,
                                    candidate_prepare=lambda candidate: candidate)
    assert report.selected is not None


def test_planning_hardware_failure_does_not_try_other_candidates(scene):
    visited = []

    def prepare(candidate):
        visited.append(candidate.closing_sign)
        raise RuntimeError('actual controller C31')

    grasps = [SimpleNamespace(score=1., width=.06, depth=0.)]
    with pytest.raises(RuntimeError, match='C31'):
        selection.select_grasp(grasps, np.eye(4), scene.cfg, scene.robot,
                               candidate_prepare=prepare)
    assert visited == [1]


def test_dry_run_does_not_claim_joint_optimized_upper(scene):
    s = selection.select_grasp([SimpleNamespace(score=1., width=.06, depth=0.)],
                               np.eye(4), scene.cfg).selected
    assert s.transfer_sign == 1
    assert s.j7_range_deg is None


def test_endpoint_feasible_but_midrise_impossible_is_rejected(scene):
    p = scene
    original = p.robot.flange_ik_joints_deg
    p.robot.flange_ik_joints_deg = lambda T: (
        None if .25 < T[2, 3] < .30 else original(T))
    report = p.select()
    assert report.selected is None
    assert any('상승 경로' in why for _, why in report.rejected)


def test_midrise_j7_limit_is_enforced(scene):
    p = scene
    original = p.robot.flange_ik_joints_deg
    p.robot.flange_ik_joints_deg = lambda T: (
        [0.] * 6 + [100.] if .25 < T[2, 3] < .30 else original(T))
    report = p.select()
    assert report.selected is None
    assert any('상승 경로' in why and 'J7' in why for _, why in report.rejected)


def test_grasp_mode_selects_other_grasp_instead_of_turning_the_loaded_wrist(scene, monkeypatch):
    p = scene
    p.cfg['drop']['lift_orientation'] = 'grasp'
    # Only the upper orientation belonging to the other grasp sign is feasible.
    p.joints['upper', 1] = None
    waypoint = selection.transfer_waypoint
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs.get('closing_sign', 1))
        return waypoint(*args, **kwargs)

    monkeypatch.setattr(selection, 'transfer_waypoint', observed)
    selected = p.select().selected
    assert selected.closing_sign == -1
    assert selected.transfer_sign == 1
    assert calls and set(calls) == {1}
    target, _ = waypoint(selected.T_tcp_pre, selected.T_tcp_lift, p.cfg['drop'], p.robot.R_home_tcp)
    np.testing.assert_array_equal(target[:3, :3], selected.T_tcp_pre[:3, :3])


def test_grasp_mode_rejects_candidate_when_only_upper_half_turn_would_work(scene):
    p = scene
    p.cfg['drop']['lift_orientation'] = 'grasp'
    p.joints['upper', 1] = None
    # The alternative upper could be reached but its corresponding grasp cannot.
    p.joints['grasp', -1] = None
    assert p.select().selected is None


@pytest.mark.parametrize('phase', ['pre', 'grasp', 'upper'])
@pytest.mark.parametrize('near_limit', [84.74, -84.74])
def test_selection_reserve_rejects_near_limit_endpoint_before_approach(
        scene, phase, near_limit):
    """A shorter route near the cable limit loses to its feasible jaw symmetry."""
    p = scene
    p.cfg['drop']['lift_orientation'] = 'grasp'
    p.cfg['selection']['j7_selection_margin_deg'] = 10
    p.joints[phase, 1][6] = near_limit
    for waypoint in ('pre', 'grasp', 'upper'):
        p.joints[waypoint, -1][0] = 180

    report = p.select()

    assert report.selected.closing_sign == -1
    assert any('J7' in reason for _, reason in report.rejected)
    assert p.cfg['selection']['j7_limit_deg'] == [-89, 90]


@pytest.mark.parametrize('near_limit', [84.74, -84.74])
def test_selection_reserve_covers_interior_rise_samples(scene, near_limit):
    p = scene
    p.cfg['drop']['lift_orientation'] = 'grasp'
    p.cfg['selection']['j7_selection_margin_deg'] = 10
    original = p.robot.flange_ik_joints_deg
    for waypoint in ('pre', 'grasp', 'upper'):
        p.joints[waypoint, -1][0] = 180

    def ik(target):
        joints = original(target).copy()
        if target[0, 0] > 0 and .25 < target[2, 3] < .30:
            joints[6] = near_limit
        return joints

    p.robot.flange_ik_joints_deg = ik
    report = p.select()

    assert report.selected.closing_sign == -1
    assert any('상승 경로' in reason and 'J7' in reason
               for _, reason in report.rejected)


@pytest.mark.parametrize('boundary', [-79., 80.])
def test_selection_reserve_accepts_exact_inner_boundary(scene, boundary):
    p = scene
    p.cfg['selection']['j7_selection_margin_deg'] = 10
    for joints in p.joints.values():
        joints[6] = boundary
    assert p.select().selected is not None


@pytest.mark.parametrize('near_limit', [-84.74, 84.74])
def test_selection_rejects_candidate_when_all_symmetries_lack_reserve(
        scene, near_limit):
    p = scene
    p.cfg['selection']['j7_selection_margin_deg'] = 10
    for joints in p.joints.values():
        joints[6] = near_limit
    assert p.select().selected is None


def test_selection_skips_insufficient_reserve_without_reordering_safe_scores(
        scene, monkeypatch):
    p = scene
    p.cfg['selection']['j7_selection_margin_deg'] = 10
    grasps = [SimpleNamespace(score=score, width=.06, depth=0., x=x)
              for score, x in ((.9, .2), (.8, .3), (.7, .4))]
    monkeypatch.setattr(selection, 'grasp_to_base',
                        lambda _, g: (np.eye(3), np.array([g.x, 0., .1])))

    def ik(target):
        joints = np.zeros(7)
        if target[0, 3] < .25:
            joints[6] = 84.74
        elif target[0, 3] < .35:
            joints[0] = 150  # Longer than candidate #3, but a higher score.
        return joints

    p.robot.flange_ik_joints_deg = ik
    report = selection.select_grasp(grasps, np.eye(4), p.cfg, p.robot)

    assert report.selected.index == 1
    assert report.selected.score == .8
    assert any(index == 0 and 'J7' in reason for index, reason in report.rejected)


@pytest.mark.parametrize('margin', [None, 0])
def test_absent_or_zero_selection_margin_preserves_hard_limits(scene, margin):
    p = scene
    if margin is None:
        p.cfg['selection'].pop('j7_selection_margin_deg', None)
    else:
        p.cfg['selection']['j7_selection_margin_deg'] = margin
    for joints in p.joints.values():
        joints[6] = 84.74
    assert p.select().selected is not None
    for joints in p.joints.values():
        joints[6] = 90.22
    assert p.select().selected is None


@pytest.mark.parametrize('margin', [-1, np.nan, np.inf, 89.5, 100])
def test_invalid_selection_margin_is_rejected(scene, margin):
    scene.cfg['selection']['j7_selection_margin_deg'] = margin
    with pytest.raises(ValueError, match='j7_selection_margin_deg'):
        scene.select()


def test_selected_j7_diagnostic_includes_interior_extrema(scene):
    p = scene
    p.cfg['drop']['lift_orientation'] = 'grasp'
    p.cfg['selection']['j7_selection_margin_deg'] = 10
    p.joints['grasp', -1] = None
    for phase, j7 in (('pre', 10), ('grasp', 20), ('upper', 30)):
        p.joints[phase, 1][6] = j7
    original = p.robot.flange_ik_joints_deg

    def ik(target):
        joints = original(target)
        if joints is None:
            return None
        joints = joints.copy()
        if .25 < target[2, 3] < .27:
            joints[6] = -60
        elif .27 < target[2, 3] < .31:
            joints[6] = 70
        return joints

    p.robot.flange_ik_joints_deg = ik
    selected = p.select().selected

    assert selected.closing_sign == 1
    assert selected.j7_range_deg == (-60., 70.)
