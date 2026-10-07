#!/usr/bin/env python3
"""exp_foundgrasp — 파지-이적재 데모 실행.

  python run.py                 # 접근 직전 확인 프롬프트 유지
  python run.py --auto          # 무인 연속 실행
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
    ap.add_argument("--auto", action="store_true", help="접근 확인 프롬프트 생략 (무인)")
    ap.add_argument("--dry-run", action="store_true",
                    help="스캔·검출·선정만, 로봇/그리퍼 미사용")
    args = ap.parse_args()

    cfg = config_from_args(args)
    print(f"설정: {cfg['_config_path']}"
          + (f"  (+{len(args.set)}개 --set)" if args.set else ""))

    from efg.demo import run
    run(cfg, auto=args.auto, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
