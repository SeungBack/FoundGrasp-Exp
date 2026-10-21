from types import SimpleNamespace

import numpy as np

from efg import selection
from efg.config import load_config


def test_candidate_with_unreachable_recovery_is_skipped(monkeypatch):
    cfg = load_config()
    cfg["selection"]["camera_up"] = {"prefer": False}
    cfg["selection"]["approach_extra_m"] = 0
    home_R = np.diag([1., -1., -1.])
    tested = []

    def ik(T):
        tested.append(T.copy())
        return not (T[0, 3] < .25 and T[2, 3] > .3)

    robot = SimpleNamespace(T_flange_tcp=np.eye(4), R_home_tcp=home_R,
                            check_flange_ik=ik,
                            flange_ik_joints_deg=lambda T: [0.] * 7)
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
    upper = [T for T in tested if T[2, 3] > .3]
    assert upper
    for T in upper:
        np.testing.assert_array_equal(T[:3, :3], home_R)
