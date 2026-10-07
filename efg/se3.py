"""SE(3) helpers. All poses are 4x4 float64, meters/radians unless noted."""

import numpy as np
from scipy.spatial.transform import Rotation


def make_se3(rot_mat, trans):
    T = np.eye(4)
    T[:3, :3] = np.asarray(rot_mat, dtype=np.float64)
    T[:3, 3] = np.asarray(trans, dtype=np.float64).reshape(3)
    return T


def inv_se3(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def xarm_pose_to_se3(pose_mm_rad):
    """xArm [x, y, z (mm), roll, pitch, yaw (rad)] -> 4x4 SE(3) in meters.

    xArm RPY convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll),
    which is scipy's extrinsic 'xyz' order.
    """
    x, y, z, roll, pitch, yaw = [float(v) for v in pose_mm_rad]
    R = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return make_se3(R, [x / 1000.0, y / 1000.0, z / 1000.0])


def se3_to_xarm_pose(T):
    """4x4 SE(3) in meters -> xArm [x, y, z (mm), roll, pitch, yaw (rad)]."""
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    t = T[:3, 3] * 1000.0
    return [t[0], t[1], t[2], rpy[0], rpy[1], rpy[2]]


def rvec_tvec_to_se3(rvec, tvec):
    """OpenCV solvePnP output -> 4x4 SE(3)."""
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return make_se3(R, np.asarray(tvec, dtype=np.float64).reshape(3))


def rotation_angle_deg(R):
    """Geodesic rotation angle of a rotation matrix, in degrees."""
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def pose_delta(T_a, T_b):
    """(translation [m], rotation [deg]) difference between two poses."""
    d = inv_se3(T_a) @ T_b
    return float(np.linalg.norm(d[:3, 3])), rotation_angle_deg(d[:3, :3])


def se3_to_pos_quat_xyzw(T):
    pos = T[:3, 3].tolist()
    quat = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()  # xyzw
    return pos, quat


def average_translations(Ts):
    """Mean translation and per-axis std over a list of SE(3) poses."""
    ts = np.stack([T[:3, 3] for T in Ts])
    return ts.mean(axis=0), ts.std(axis=0)
