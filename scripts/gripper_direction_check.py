#!/usr/bin/env python3
"""[점검] 그리퍼 방향/매핑 실측 점검 — 눈으로 보면서 확인.

펌웨어의 위치 매핑·open/close 방향이 뒤집혔는지(잼/전원 이상 후 발생 가능)
실측으로 판정하고, 설정 gripper 항목에 넣을 값을 추천한다.

  python scripts/gripper_direction_check.py
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time

from efg.config import add_config_args, config_from_args  # noqa: E402
from efg.paths import KORAS_ROOT, add_vendor_paths  # noqa: E402

add_vendor_paths(KORAS_ROOT)
from koras_gripper_wrapper import KorasGripper  # noqa: E402


def ask(q):
    return input(f"{q} [y/N] ").strip().lower() == "y"


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    cfg = config_from_args(args)["gripper"]
    g = KorasGripper(cfg["port"], slave_id=int(cfg["slave_id"]))
    if not g.connect(initialize=False):
        raise SystemExit("연결 실패")
    g.enable()
    g.set_speed(50)
    try:
        print("\n[1/3] 위치 매핑 확인 — raw 1000(=상태값 100%)으로 이동합니다.")
        g.set_finger_position(1000)
        time.sleep(3.0)
        print(f"   상태값: {g.read_status().position:.1f}%")
        open_at_1000 = ask("   지금 그리퍼가 '열려' 있습니까?")

        print("\n[2/3] raw 0(=상태값 0%)으로 이동합니다.")
        g.set_finger_position(0)
        time.sleep(3.0)
        print(f"   상태값: {g.read_status().position:.1f}%")
        closed_at_0 = ask("   지금 그리퍼가 '닫혀' 있습니까?")
        invert = not (open_at_1000 and closed_at_0)

        print("\n[3/3] grip 명령 방향 확인 — 중간(raw 500)에서 grip_close를 보냅니다.")
        g.set_finger_position(500)
        time.sleep(2.5)
        g.grip_close()
        time.sleep(2.5)
        print(f"   상태값: {g.read_status().position:.1f}%")
        closed_by_close = ask("   grip_close 후 그리퍼가 '닫혔'습니까?")
        g.stop()

        print("\n===== 판정 =====")
        if not invert and closed_by_close:
            print("정상: grip_mode: \"grip\", invert: false")
        elif not invert and not closed_by_close:
            print("위치 매핑 정상, grip 명령만 이상 → grip_mode: \"position\", invert: false")
        elif invert and not closed_by_close:
            print("위치 매핑·grip 방향 모두 반전 → grip_mode: \"grip\" 유지 + invert: true")
        else:
            print("위치 매핑 반전, grip 방향 정상 (드묾) → invert: false 유지, "
                  "pre-open만 이상하면 invert: true 시도")
        print("설정의 gripper 항목에 반영 후 gripper_test.py 로 재확인하세요.")
        print("\n주의: 근본 원인은 잼/이상 상태의 초기화. 24V 전원 재투입 + 핑거 자유 "
              "상태에서 버튼(I) 재초기화가 가장 깔끔한 해결책입니다.")
    finally:
        g.disconnect()


if __name__ == "__main__":
    main()
