#!/usr/bin/env python3
"""[사전 준비, 선택] 파지 성공 판정용 페이로드 기준값 측정 (payload_check).

릴리즈 자세(drop.joints_deg)로 이동해 **빈 손** 관절 토크를 기준값으로 잡고,
이어서 샘플 물체를 물린 상태의 토크를 재 차이 벡터(방향)와 권장 임계값을
구한다. 결과는 payload_check.baseline_path 에 저장된다.

플렉스 팁은 물체마다 다르게 휘어 finger 간격으로는 파지 여부를 알 수 없다 —
대신 손에 실린 질량을 관절 중력 토크로 본다 (efg/payload.py 참조).

  python scripts/calib_payload.py                        # 빈 손 + 샘플 물체 (권장)
  python scripts/calib_payload.py --empty-only --min-delta 0.15
  python scripts/calib_payload.py --no-save              # 측정만

주의: 릴리즈 자세를 다시 티칭하면(teach_drop.py) 기준값이 무효 — 재실행할 것.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from efg.config import add_config_args, config_from_args, require  # noqa: E402
from efg.payload import PayloadCheck, joints_now, read_torque  # noqa: E402

# 권장 임계값 = 샘플 물체 신호의 이 비율. 더 가벼운 물체도 잡히도록 낮게 잡되
# 자세 재현 오차보다는 충분히 크게
THRESHOLD_RATIO = 0.35


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--samples", type=int, default=20, help="자세당 토크 표본 수 (기본 20)")
    ap.add_argument("--empty-only", action="store_true",
                    help="샘플 물체 단계 생략 — --min-delta 로 임계값 지정")
    ap.add_argument("--min-delta", type=float, default=None,
                    help="판정 임계값 (Nm). 미지정 시 샘플 물체에서 자동 산출")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    drop = require(cfg, "drop.joints_deg", "scripts/teach_drop.py 로 릴리즈 자세를 티칭하세요")
    if args.empty_only and args.min_delta is None:
        raise SystemExit("--empty-only 는 --min-delta 가 필요합니다")

    from efg.gripper import GripperCtl
    from efg.motion import connect_robot, go_home

    robot = connect_robot(cfg)
    gripper = None
    try:
        gripper = GripperCtl(cfg).connect()

        print("\n== 릴리즈 자세로 이동 (홈 경유) ==")
        go_home(robot, cfg)
        robot.move_joints(drop, speed_deg_s=cfg["drop"]["joint_speed_deg_s"])
        pose = joints_now(robot)
        print(f"   자세: {np.round(pose, 2).tolist()}")

        print("\n== 1) 빈 손 기준값 ==")
        input("   그리퍼 사이가 비었는지 확인 후 Enter... ")
        gripper.grip()  # 실제 파지 실패와 같은 상태(완전 닫힘)로 맞춘다
        base = read_torque(robot, samples=args.samples, settle_s=0.6)
        print(f"   토크: {np.round(base, 3).tolist()} Nm")

        ref = None
        if not args.empty_only:
            print("\n== 2) 샘플 물체 ==")
            gripper.open()
            input("   실험에 쓸 대표 물체를 핑거 사이에 넣고 Enter... ")
            ok, pos = gripper.grip()
            print(f"   파지: finger {pos}% (간격 판정: {'성공' if ok else '실패'} — 플렉스 팁에선 참고용)")
            loaded = read_torque(robot, samples=args.samples, settle_s=0.6)
            ref = loaded - base
            print(f"   토크: {np.round(loaded, 3).tolist()} Nm")
            print(f"   차이: {np.round(ref, 3).tolist()} Nm (크기 {np.linalg.norm(ref):.3f})")
            input("   물체를 회수할 준비가 되면 Enter (그리퍼를 엽니다)... ")
            gripper.open()
    finally:
        if gripper is not None:
            gripper.disconnect()
        robot.disconnect()

    if ref is not None and np.linalg.norm(ref) < 1e-3:
        raise SystemExit("샘플 물체 신호가 잡히지 않습니다 (차이 < 0.001 Nm) — 물체가 너무 "
                         "가볍거나 파지되지 않았습니다. 더 무거운 샘플로 재시도하세요.")

    thr = args.min_delta
    if thr is None:
        thr = float(np.linalg.norm(ref)) * THRESHOLD_RATIO
    print(f"\n== 판정 임계값: {thr:.3f} Nm ==")
    if ref is not None:
        print(f"   (샘플 신호 {np.linalg.norm(ref):.3f} Nm 의 {THRESHOLD_RATIO * 100:.0f}%)")

    pcheck = PayloadCheck(baseline=base, joints_deg=pose, ref_delta=ref,
                          min_delta_nm=thr, samples=args.samples)
    if args.no_save:
        return
    path = pcheck.save(cfg["payload_check"]["baseline_path"])
    print(f"저장됨: {path}")
    print("설정에서 payload_check.enabled: true 로 활성화하세요 "
          "(또는 run.py --set payload_check.enabled=true).")


if __name__ == "__main__":
    main()
