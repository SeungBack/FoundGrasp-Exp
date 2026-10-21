"""설정 로드 — 단일 YAML(configs/default.yaml) + 상속(base) + CLI 오버라이드.

- `base: other.yaml` 키가 있으면 그 파일을 먼저 읽고 위에 덮어쓴다 (비교 실험용
  설정 파일이 기본값을 통째로 복사하지 않도록). null 값은 "미설정"으로 보고
  base 값을 유지한다.
- `--set a.b.c=value` 로 어떤 키든 실행 시 일시 변경 (value는 YAML로 해석).
- 파일 경로 항목은 프로젝트 루트 기준 상대 경로를 허용하며 절대 경로로 바꿔 둔다.
"""
from pathlib import Path
import re

import yaml

from .paths import DEFAULT_CONFIG, resolve

# 프로젝트 루트 기준으로 해석할 경로 키 (점 표기)
PATH_KEYS = (
    "calib.hand_eye", "calib.intrinsics",
    "detector.foundgrasp.config", "detector.foundgrasp.checkpoint",
    "detector.foundgrasp.refiner.weights",
    "floor_limit.map_path", "payload_check.baseline_path", "output.dir",
)


def _deep_merge(base, over):
    for k, v in (over or {}).items():
        if v is None:
            continue  # null = 미설정 → base 값 유지
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml_chain(path, _seen=()):
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"설정 base 순환 참조: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        parent = _load_yaml_chain(resolve(base, path.parent), _seen + (path,))
        cfg = _deep_merge(parent, cfg)
    return cfg


def get(cfg, dotted, default=None):
    v = cfg
    for k in dotted.split("."):
        if not isinstance(v, dict) or k not in v:
            return default
        v = v[k]
    return v


def set_(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
        if not isinstance(d, dict):
            raise KeyError(f"'{dotted}': '{k}'는 딕셔너리가 아님")
    d[keys[-1]] = value


def apply_overrides(cfg, items):
    """['a.b=1', 'x.y=true', ...] 적용. 값은 YAML 스칼라/리스트로 해석."""
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set 형식 오류: '{item}' (key.path=value)")
        key, raw = item.split("=", 1)
        set_(cfg, key.strip(), yaml.safe_load(raw.strip()))
    return cfg


def load_config(path=None, overrides=None):
    """설정 로드 → base 상속 → --set 적용 → 경로 절대화. 반환: dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    cfg = _load_yaml_chain(cfg_path)
    apply_overrides(cfg, overrides)
    for key in PATH_KEYS:
        v = get(cfg, key)
        if v is not None:
            set_(cfg, key, str(resolve(v)))
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def require(cfg, key, hint):
    """필수 설정 검사 — 미설정이면 설정 방법을 안내하고 종료."""
    v = get(cfg, key)
    if v is None:
        raise SystemExit(f"설정 누락: {key} — {hint}")
    return v


def set_yaml_line(path, key, value):
    """YAML 파일에서 `key:` 로 시작하는 첫 줄의 값을 교체 (들여쓰기 유지).

    사전 준비 스크립트가 측정값(ROI·자세)을 설정 파일에 기록할 때 사용.
    주석·구조를 보존하려고 파서 대신 한 줄 치환을 쓴다. 반환: 성공 여부.
    """
    path = Path(path)
    text = path.read_text()
    new, n = re.subn(rf"^(\s*){re.escape(key)}:.*$",
                     lambda m: f"{m.group(1)}{key}: {value}",
                     text, count=1, flags=re.M)
    if n == 0:
        return False
    path.write_text(new)
    return True


def add_config_args(ap):
    """argparse에 --config / --set 공통 인자 추가 (run.py·scripts 공용)."""
    ap.add_argument("--config", default=None,
                    help="설정 파일 (기본 configs/default.yaml)")
    ap.add_argument("--set", metavar="KEY=VALUE", action="append", default=[],
                    help="설정 항목 일시 변경 (반복 가능, 예: --set live.enabled=false)")
    return ap


def config_from_args(args):
    return load_config(args.config, args.set)
