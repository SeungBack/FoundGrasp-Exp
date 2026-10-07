"""PayloadCheck 판정 로직 — 가짜 컨트롤러로 검증 (로봇 불필요)."""
import numpy as np
import pytest

from efg.payload import PayloadCheck

EMPTY = np.array([0.0, -12.021, 3.293, 19.960, -0.112, -2.594, 0.0])  # 실측 빈 손
LOAD = np.array([0.0, -0.90, 0.05, 1.30, 0.01, 0.12, 0.0])            # 샘플 ~300g 상당
POSE = np.array([17.72, 7.1, 13.13, 82.38, -1.26, 76.13, 30.97])


class FakeArm:
    def __init__(self, torque, pose=POSE, spike_every=0, fail=0):
        self.torque, self.pose = np.asarray(torque), np.asarray(pose)
        self.spike_every, self.fail, self.n = spike_every, fail, 0

    def get_joints_torque(self):
        self.n += 1
        if self.n <= self.fail:
            return 1, None
        t = self.torque.copy()
        if self.spike_every and self.n % self.spike_every == 0:
            t[3] += 5.7  # 실측된 단발 이상치 재현
        return 0, t.tolist()

    def get_servo_angle(self, is_radian=False):
        return 0, self.pose.tolist()


class FakeRobot:
    def __init__(self, *a, **kw):
        self.arm = FakeArm(*a, **kw)


def make(ref=LOAD, thr=None):
    return PayloadCheck(EMPTY, POSE, ref_delta=ref,
                        min_delta_nm=(float(np.linalg.norm(LOAD)) * 0.35
                                      if thr is None else thr),
                        samples=9, settle_s=0.0)


def test_empty_and_loaded():
    pc = make()
    assert pc.measure(FakeRobot(EMPTY))[0] is False
    has, d = pc.measure(FakeRobot(EMPTY + LOAD))
    assert has is True and abs(d - np.linalg.norm(LOAD)) < 1e-6


def test_threshold_band():
    pc = make()
    assert pc.measure(FakeRobot(EMPTY + LOAD * 0.5))[0] is True
    assert pc.measure(FakeRobot(EMPTY + LOAD * 0.2))[0] is False


def test_projection_rejects_opposite_drift():
    """반대 방향 편차는 물체로 오판하지 않음 — 크기(L2) 기준과의 차이."""
    has, d = make().measure(FakeRobot(EMPTY - LOAD))
    assert has is False and d < 0
    norm_only = PayloadCheck(EMPTY, POSE, ref_delta=None,
                             min_delta_nm=make().min_delta_nm, samples=9, settle_s=0.0)
    assert norm_only.measure(FakeRobot(EMPTY - LOAD))[0] is True  # 투영이 필요한 이유


def test_pose_mismatch_returns_none():
    bad = POSE.copy(); bad[0] += 3.0
    has, err = make().measure(FakeRobot(EMPTY + LOAD, pose=bad))
    assert has is None and abs(err - 3.0) < 1e-6
    near = POSE.copy(); near[0] += 0.5
    assert make().measure(FakeRobot(EMPTY + LOAD, pose=near))[0] is True


def test_median_tolerates_spikes():
    assert make().measure(FakeRobot(EMPTY, spike_every=3))[0] is False


def test_read_failure_raises():
    with pytest.raises(RuntimeError):
        make().measure(FakeRobot(EMPTY, fail=99))


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "pl.npz"
    make().save(p)
    lo = PayloadCheck.load(p, {})
    assert np.allclose(lo.baseline, EMPTY) and np.allclose(lo.ref_delta, LOAD)
    assert abs(lo.min_delta_nm - make().min_delta_nm) < 1e-9
    assert abs(PayloadCheck.load(p, {"min_delta_nm": 0.99}).min_delta_nm - 0.99) < 1e-9
    PayloadCheck(EMPTY, POSE, ref_delta=None, min_delta_nm=0.2).save(p)
    lo3 = PayloadCheck.load(p, {})
    assert lo3.ref_delta is None and lo3.measure(FakeRobot(EMPTY + LOAD))[0] is True
