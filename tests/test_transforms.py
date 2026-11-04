"""transforms/selection 오프라인 검증 (로봇/카메라/추론 불필요)."""

import numpy as np
import pytest

from efg import transforms as T  # noqa: E402


def _grasp(R, t, score=0.9, width=0.05):
    """graspnetAPI Grasp 대역 — .rotation_matrix/.translation/.score/.width."""
    class G:
        pass
    g = G()
    g.rotation_matrix = np.asarray(R, float)
    g.translation = np.asarray(t, float)
    g.score = score
    g.width = width
    return g


def test_grasp_to_base_identity():
    R = np.eye(3)
    t = np.array([0.1, 0.2, 0.5])
    Rb, pb = T.grasp_to_base(np.eye(4), _grasp(R, t))
    assert np.allclose(pb, t)
    assert np.allclose(Rb, R)


def test_grasp_to_base_with_transform():
    Tbc = np.eye(4)
    Tbc[:3, 3] = [1.0, 0.0, 0.0]
    _, pb = T.grasp_to_base(Tbc, _grasp(np.eye(3), [0.1, 0.2, 0.5]))
    assert np.allclose(pb, [1.1, 0.2, 0.5])


def test_tool_pose_orthonormal():
    R = np.eye(3)
    Tt = T.tool_pose_from_grasp(R, np.array([0.3, 0.0, 0.1]))
    Rt = Tt[:3, :3]
    assert np.allclose(Rt @ Rt.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(Rt), 1.0, atol=1e-9)
    # 툴 z축은 grasp 접근 방향(x축)과 정렬
    assert np.allclose(Tt[:3, 2], R[:, 0])


def test_insertion_along_approach():
    # 접근 방향(grasp x=[1,0,0])으로 insertion만큼 깊이 진입
    R = np.eye(3)
    Tt = T.tool_pose_from_grasp(R, np.array([0.3, 0.0, 0.1]),
                                insertion_m=0.025)
    assert np.allclose(Tt[:3, 3], [0.325, 0.0, 0.1])


def test_pre_grasp_offset_along_z():
    R = np.eye(3)
    Tt = T.tool_pose_from_grasp(R, np.array([0.3, 0.0, 0.1]))
    Tpre = T.offset_along_tool_z(Tt, 0.1)
    # 접근 방향(z=grasp x=[1,0,0]) 반대로 0.1m 후퇴
    assert np.allclose(Tpre[:3, 3], [0.2, 0.0, 0.1])


def test_lift_raises_z():
    Tt = np.eye(4)
    Tt[:3, 3] = [0.3, 0.0, 0.1]
    assert np.allclose(T.lifted(Tt, 0.15)[:3, 3], [0.3, 0.0, 0.25])


def test_approach_tilt():
    # 접근 벡터가 base -z (수직 아래)면 기울임 0도
    R = np.column_stack([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    assert T.approach_tilt_deg(R) < 1e-6
    # 접근 벡터가 수평이면 90도
    R2 = np.column_stack([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert abs(T.approach_tilt_deg(R2) - 90.0) < 1e-6


@pytest.mark.parametrize("tilt", [0., 20.])
@pytest.mark.parametrize("sign", [-1, 1])
def test_measured_tool_and_insertion_preserve_existing_pick_flange_target(tilt, sign):
    """223/70mm and 203/50mm calibrations produce the same flange target.

    These are fixed calibration cases, independent of later insertion tuning.
    """
    from scipy.spatial.transform import Rotation
    from efg.se3 import inv_se3, xarm_pose_to_se3

    downward = np.column_stack([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    R = Rotation.from_euler('y', tilt, degrees=True).as_matrix() @ downward
    p = np.array([.4195, -.1055, .1272])
    old_tcp = xarm_pose_to_se3([0., 0., 223., 0., 0., 0.])
    new_tcp = xarm_pose_to_se3([0., 0., 203., 0., 0., 0.])
    for depth in (.015, .04):
        old = T.tool_pose_from_grasp(R, p, sign, depth + .07)
        new = T.tool_pose_from_grasp(R, p, sign, depth + .05)
        for retreat in (0., .1):
            np.testing.assert_allclose(
                T.offset_along_tool_z(old, retreat) @ inv_se3(old_tcp),
                T.offset_along_tool_z(new, retreat) @ inv_se3(new_tcp), atol=1e-12)
