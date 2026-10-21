"""AnyGrasp SDK 워커 — 논문 sim_worker.py 와 같은 형식 (efg/worker_protocol.py).

입력 프레임 → 카메라 좌표계 점군(depth (0, 1.5m), EconomicGrasp 워커와 같은 범위)
→ detector.get_grasp(points, dense_grasp=False, collision_detection=True)
→ (N,17) GraspGroup 배열, width 는 raw 폭으로 환산(÷ --width_divisor).

width 규약: SDK 문서는 width 를 "필요한 개방 폭"이라 설명하고, AnyGrasp 의 전신
GSNet 계열 decode 는 예측 폭에 x1.2 마진을 곱한다. 공유 실행 계층이 x1.2 를 한 번
적용하므로 여기서 1.2 로 나눠 다른 방법과 같은 raw 폭으로 맞춘다 (SDK 가 비공개라
추정 — 마진이 없다고 확인되면 --width_divisor 1.0).
충돌검사: SDK 내장 CD 를 켠다 (RNGNet·HGGD 내장 CD 와 같은 "방법 고유 CD" 위치).

라이선스는 <sdk_dir>/license_registration/license 를 쓴다 (작업 디렉터리를 그곳으로).
실행 (efg-anygrasp env):
  python workers/anygrasp_worker.py --work_dir /tmp/w --sdk_dir ~/Workspaces/anygrasp_sdk \
      --checkpoint <checkpoint_detection.tar>
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from efg.worker_protocol import serve  # noqa: E402

DEPTH_MAX_M = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--sdk_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max_gripper_width", type=float, default=0.1)
    ap.add_argument("--gripper_height", type=float, default=0.03)
    ap.add_argument("--width_divisor", type=float, default=1.2)
    args = ap.parse_args()

    sdk = Path(args.sdk_dir).expanduser()
    if not Path(args.checkpoint).expanduser().exists():
        raise SystemExit(f"AnyGrasp 체크포인트 없음: {args.checkpoint}")
    work_dir = os.path.abspath(args.work_dir)
    sys.path.insert(0, str(sdk / "grasp_detection" / "gsnet_versions"))  # gsnet.cpython-3XX*.so
    os.chdir(sdk / "license_registration")                               # ./license
    from easydict import EasyDict
    from gsnet import create_detector

    detector = create_detector(EasyDict(
        checkpoint_path=str(Path(args.checkpoint).expanduser()),
        max_gripper_width=args.max_gripper_width, gripper_height=args.gripper_height))
    if detector is None:
        raise SystemExit("AnyGrasp 라이선스 검증 실패 (license_registration/license 확인)")
    print(f"-> AnyGrasp loaded {args.checkpoint}", flush=True)

    def predict(rgb, depth, K):
        H, W = depth.shape
        us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        z = depth.astype(np.float32)
        mask = (z > 0) & (z < DEPTH_MAX_M)
        pts = np.stack([(us - K[0, 2]) / K[0, 0] * z, (vs - K[1, 2]) / K[1, 1] * z, z], -1)[mask]
        gg = detector.get_grasp(pts.astype(np.float32), {
            "dense_grasp": False, "collision_detection": True})
        if gg is None or len(gg) == 0:
            return np.zeros((0, 17))
        arr = np.asarray(gg.grasp_group_array, dtype=np.float64).copy()
        arr[:, 1] /= args.width_divisor
        return arr

    serve(work_dir, predict)


if __name__ == "__main__":
    main()
