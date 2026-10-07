"""릴리즈 자세 관절 토크로 파지 성공을 판정한다 (페이로드 확인).

FESTO 플렉스 팁으로 교체한 뒤(2026-08-20) 기존 판정(close 후 finger 간격)이
무력해졌다 — 팁이 물체 형상대로 휘면서 최종 간격이 빈 손과 구분되지 않는다.
대신 **고정된 릴리즈 자세에서 관절 중력 토크를 빈 손 기준값과 비교**해
"손에 질량이 있는가"를 직접 본다. 팁 변형과 무관하고, 이송 경로가 이미 이
자세를 지나므로 추가 동작이 없다.

실측 근거 (2026-08-20, xArm7 v2.7.1): 정지 자세에서 get_joints_torque()는
2초 블록 6회 동안 비트 단위로 동일했다 (블록평균 재현성 std 0.000 Nm,
분해능 0.001 Nm). 100g을 레버암 0.45m에서 들면 J2 토크가 ~0.44 Nm 변하므로
여유가 크다. 다만 보고 스트림이 드물게 튀는 것이 관측돼(J4 단발 5.7 Nm)
중앙값으로 읽는다. F/T 센서는 없다 (get_ft_sensor_data 전부 0).

한계: 기준값은 측정 당시 자세에서만 유효하다 — 릴리즈 자세를 다시 티칭하면
scripts/calib_payload.py 를 재실행해야 한다 (자세 불일치는 실행 중 검사).
"""
from pathlib import Path
import time

import numpy as np

POSE_TOL_DEG = 1.0      # 기준값 측정 자세와 이 이상 다르면 판정 무효
MIN_SAMPLES = 3         # 중앙값을 낼 최소 표본 수


def read_torque(robot, samples=12, period_s=0.05, settle_s=0.4):
    """정지 대기 후 관절 토크를 여러 번 읽어 관절별 중앙값 반환 (shape (7,)).

    중앙값: 정지 판독 자체는 재현성이 매우 높지만 단발 이상치가 섞이는 것이
    실측돼 평균 대신 사용한다. settle_s는 이동 직후 잔진동이 가라앉을 시간.
    """
    if settle_s > 0:
        time.sleep(settle_s)
    vals = []
    for _ in range(int(samples)):
        code, t = robot.arm.get_joints_torque()
        if code == 0 and t is not None:
            vals.append([float(v) for v in t[:7]])
        time.sleep(period_s)
    if len(vals) < MIN_SAMPLES:
        raise RuntimeError(
            f"관절 토크 읽기 실패 (표본 {len(vals)}개 < {MIN_SAMPLES}) — "
            "컨트롤러 연결 상태를 확인하세요")
    return np.median(np.asarray(vals, float), axis=0)


def joints_now(robot):
    """현재 조인트 각도 (deg, shape (7,))."""
    code, ang = robot.arm.get_servo_angle(is_radian=False)
    if code != 0:
        raise RuntimeError(f"조인트 읽기 실패 (code={code})")
    return np.asarray([float(a) for a in ang[:7]], float)


class PayloadCheck:
    """빈 손 기준 토크와의 차이로 파지 성공을 판정한다.

    판정량(delta):
    - ref_delta(샘플 물체를 든 상태의 기준 차이 벡터)가 있으면 그 방향으로의
      **부호 있는 투영**을 쓴다 — 자세 미세 차이·드리프트가 만드는 엉뚱한
      방향의 편차를 "물체 있음"으로 오판하지 않는다.
    - 없으면 차이 벡터의 크기(L2)를 쓴다.
    """

    def __init__(self, baseline, joints_deg, ref_delta=None,
                 min_delta_nm=0.1, samples=12, settle_s=0.4):
        self.baseline = np.asarray(baseline, float)
        self.joints_deg = np.asarray(joints_deg, float)
        self.ref_delta = (None if ref_delta is None or not len(ref_delta)
                          else np.asarray(ref_delta, float))
        self.min_delta_nm = float(min_delta_nm)
        self.samples = int(samples)
        self.settle_s = float(settle_s)

    # --- 저장/로드 (floor_limit의 높이맵과 동일한 npz 규약) ---
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, baseline=self.baseline, joints_deg=self.joints_deg,
                 ref_delta=(np.zeros(0) if self.ref_delta is None
                            else self.ref_delta),
                 min_delta_nm=np.float64(self.min_delta_nm))
        return path

    @classmethod
    def load(cls, path, cfg_pc=None):
        path = Path(path)
        if not path.exists():
            raise SystemExit(
                f"페이로드 기준값 없음: {path} — "
                "scripts/calib_payload.py 로 빈 손 기준을 측정하세요")
        d = np.load(path)
        cfg_pc = cfg_pc or {}
        # 임계값은 설정 파일이 우선 (미지정 시 캘리브레이션 권장값)
        thr = cfg_pc.get("min_delta_nm")
        return cls(
            baseline=d["baseline"], joints_deg=d["joints_deg"],
            ref_delta=d["ref_delta"],
            min_delta_nm=(float(d["min_delta_nm"]) if thr is None else thr),
            samples=cfg_pc.get("samples", 12),
            settle_s=cfg_pc.get("settle_s", 0.4))

    def metric(self, torque):
        """토크 판독 → 판정량 (Nm)."""
        d = np.asarray(torque, float) - self.baseline
        if self.ref_delta is not None:
            n = float(np.linalg.norm(self.ref_delta))
            if n > 1e-9:
                return float(d @ (self.ref_delta / n))
        return float(np.linalg.norm(d))

    def pose_error_deg(self, robot):
        return float(np.max(np.abs(joints_now(robot) - self.joints_deg)))

    def measure(self, robot):
        """현재 자세에서 판정. 반환 (has_object|None, delta_nm).

        has_object=None: 자세가 기준값 측정 자세와 달라 판정 불가 (호출부에서
        경고 후 통과시킬 것 — 실행 중인 데모를 이 이유로 중단하지 않는다).
        """
        err = self.pose_error_deg(robot)
        if err > POSE_TOL_DEG:
            return None, err
        t = read_torque(robot, samples=self.samples, settle_s=self.settle_s)
        delta = self.metric(t)
        return delta >= self.min_delta_nm, delta


def load_payload_check(cfg, enabled=None, required=True):
    """설정에서 PayloadCheck 로드. 비활성이면 None.

    enabled: 강제값 (None이면 설정 payload_check.enabled).
    required: False면 기준값 파일이 없어도 경고 후 None (판정에 쓰지 않고
        참고 출력만 하는 경우 — 미측정을 이유로 실행을 막지 않는다).
    """
    pc = cfg.get("payload_check") or {}
    on = pc.get("enabled", False) if enabled is None else enabled
    if not on:
        return None
    path = Path(pc.get("baseline_path", "configs/calib/payload_baseline.npz"))
    if not required and not path.exists():
        print(f"경고: 페이로드 기준값 없음 ({path.name}) — 페이로드 출력 생략. "
              "필요하면 scripts/calib_payload.py 로 측정하세요")
        return None
    return PayloadCheck.load(path, pc)
