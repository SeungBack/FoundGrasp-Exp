#!/usr/bin/env python3
"""exp_foundgrasp — 파지-이적재 데모 실행.

  python run.py                 # 확인 없이 연속 접근
  python run.py --auto          # 기존 명령 호환 (기본 동작과 동일)
  python run.py --dry-run       # 스캔·검출·선정만 (로봇/그리퍼 미사용)

모든 동작 옵션은 configs/default.yaml 에 있다. 실행 시 일시 변경은 --set:
  python run.py --set floor_limit.enabled=true --set live.enabled=false
다른 설정 파일(비교 실험 등): --config configs/my_exp.yaml  (base: 로 상속 가능)
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from efg.config import add_config_args, config_from_args  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--auto", action="store_true", help="기본 자동 접근과 동일 (호환 옵션)")
    mode.add_argument("--confirm", action="store_true", help="pre-grasp에서 정지하고 접근 확인")
    ap.add_argument("--dry-run", action="store_true",
                    help="스캔·검출·선정만, 로봇/그리퍼 미사용")
    ap.add_argument("--verbose", action="store_true",
                    help="모델 초기화·refiner·필터·모션 진단 상세 출력")
    args = ap.parse_args()

    cfg = config_from_args(args)
    if args.verbose:
        cfg.setdefault("output", {})["verbose"] = True

    from efg.demo import run
    run(cfg, auto=not args.confirm, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
