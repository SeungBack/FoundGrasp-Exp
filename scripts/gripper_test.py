#!/usr/bin/env python3
"""[점검] KORAS 그리퍼 단독 점검 — 연결/개폐/상태 확인.

connect()는 개방 동작 + close 방향 점검으로 기능을 검증한다. 전원 투입 후
그리퍼 버튼(I) 초기화가 선행되어야 한다.

  python scripts/gripper_test.py            # 연결(검증) → close → open
  python scripts/gripper_test.py --status   # 상태만 읽고 종료 (모터 명령 없음)
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from efg.config import add_config_args, config_from_args  # noqa: E402
from efg.gripper import GripperCtl  # noqa: E402


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    if args.status:
        gc = GripperCtl(cfg)
        s = gc.probe()
        print(f"상태: {gc._fmt_status(s)}")
        gc.disconnect()
        return

    print("그리퍼 연결 중 (개방 동작으로 기능 검증)...")
    with GripperCtl(cfg) as gc:
        s = gc.status()
        print(f"연결됨. 상태: {gc._fmt_status(s)}")
        print("파지(close)...")
        ok, pos = gc.grip()
        print(f"  결과: {'성공(물체 감지)' if ok else '완전 닫힘(빈 파지)'} (finger {pos}%)")
        print("개방...")
        gc.open()
    print("완료.")


if __name__ == "__main__":
    main()
