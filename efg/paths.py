"""프로젝트 경로 및 서드파티(원본 사본) import 경로.

third_party/ 아래 원본은 수정하지 않는다. FoundGrasp는 루트에 utils/, model/
같은 일반 이름의 패키지를 두므로, 이 프로젝트 루트에 같은 이름의 디렉터리를
만들면 import 충돌이 난다 — 만들지 말 것.
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = PROJECT_ROOT / "third_party"
FOUNDGRASP_ROOT = THIRD_PARTY / "FoundGrasp"
KORAS_ROOT = THIRD_PARTY / "koras-gripper-wrapper"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def add_vendor_paths(*roots):
    """서드파티 루트를 sys.path 맨 앞에 등록 (기본: koras + FoundGrasp).

    koras_gripper_wrapper는 다른 곳에 editable 설치가 있을 수 있어 사본이
    우선되도록 앞에 넣는다.
    """
    for p in roots or (KORAS_ROOT, FOUNDGRASP_ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def resolve(path, base=PROJECT_ROOT):
    """설정의 상대 경로를 프로젝트 루트 기준 절대 경로로. None은 그대로."""
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (Path(base) / p).resolve()


def import_grasp_group():
    """graspnetAPI.GraspGroup — 의존성 autolab_core가 import 시 root logger로
    내는 ROS 미설치 경고를 억제한다 (이 환경에선 무의미)."""
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        from graspnetAPI.grasp import GraspGroup
    finally:
        root.setLevel(level)
    return GraspGroup
