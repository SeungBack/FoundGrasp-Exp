"""Endpoint route selection using synthetic IK, without a robot connection."""
from types import SimpleNamespace

import numpy as np
import pytest

from efg import selection
from efg.config import load_config


@pytest.fixture
def scene(monkeypatch):
    cfg = load_config()
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
    assert s.transfer_sign == -1


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


@pytest.mark.parametrize('invalid', ['ik', 'nan', 'j7'])
def test_invalid_upper_alternative_is_excluded(scene, invalid):
    p = scene
    if invalid == 'ik':
        p.joints['upper', 1] = None
    else:
        p.joints['upper', 1][6] = np.nan if invalid == 'nan' else 100
    assert p.select().selected.transfer_sign == -1
    assert p.select().rejected
    p.joints['upper', -1] = None
    assert p.select().selected is None


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


def test_dry_run_does_not_claim_joint_optimized_upper(scene):
    s = selection.select_grasp([SimpleNamespace(score=1., width=.06, depth=0.)],
                               np.eye(4), scene.cfg).selected
    assert s.transfer_sign == 1
