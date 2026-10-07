"""설정 로드·상속·--set 오버라이드·경로 해석·YAML 한 줄 치환."""
from pathlib import Path

import pytest
import yaml

from efg.config import apply_overrides, load_config, set_yaml_line
from efg.paths import DEFAULT_CONFIG, PROJECT_ROOT


def test_default_config_loads_and_resolves_paths():
    cfg = load_config()
    assert cfg["detector"]["name"] == "foundgrasp"
    assert Path(cfg["calib"]["hand_eye"]).is_absolute()
    assert Path(cfg["calib"]["hand_eye"]).exists()
    assert Path(cfg["detector"]["foundgrasp"]["config"]).exists()
    assert cfg["_config_path"] == str(DEFAULT_CONFIG)


def test_requested_defaults_are_off():
    cfg = load_config()
    assert cfg["floor_limit"]["enabled"] is False
    assert cfg["payload_check"]["enabled"] is False
    assert cfg["grip_check"]["enabled"] is False  # 파지 판정 미적용이 기본 (플렉스 팁)


def test_set_override_parses_yaml_values():
    cfg = load_config(overrides=["floor_limit.enabled=true",
                                 "selection.approach_max_tilt_deg=30",
                                 "robot.home_joints_deg=[0,0,0,0,0,0,0]",
                                 "detector.name=foo"])
    assert cfg["floor_limit"]["enabled"] is True
    assert cfg["selection"]["approach_max_tilt_deg"] == 30
    assert cfg["robot"]["home_joints_deg"] == [0] * 7
    assert cfg["detector"]["name"] == "foo"
    with pytest.raises(SystemExit):
        apply_overrides({}, ["novalue"])


def test_base_inheritance_and_null(tmp_path):
    child = tmp_path / "exp.yaml"
    child.write_text(f"base: {DEFAULT_CONFIG}\n"
                     "selection: {approach_max_tilt_deg: 15, min_score: null}\n"
                     "live: {enabled: false}\n")
    cfg = load_config(child)
    assert cfg["selection"]["approach_max_tilt_deg"] == 15
    assert cfg["selection"]["min_score"] == 0.1  # null = 미설정 → base 유지
    assert cfg["live"]["enabled"] is False
    assert cfg["robot"]["ip"]  # 나머지는 base에서


def test_set_yaml_line_keeps_indent_and_other_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("robot:\n  ip: x\n  home_joints_deg: [1, 2]\ndrop:\n  joints_deg: [3]\n")
    assert set_yaml_line(p, "home_joints_deg", [7, 8, 9])
    assert set_yaml_line(p, "joints_deg", [4, 5])
    d = yaml.safe_load(p.read_text())
    assert d["robot"]["home_joints_deg"] == [7, 8, 9]
    assert d["drop"]["joints_deg"] == [4, 5]
    assert d["robot"]["ip"] == "x"
    assert not set_yaml_line(p, "missing_key", 1)


def test_no_generic_toplevel_dirs():
    """FoundGrasp 루트의 utils/·model/ 과 충돌하는 디렉터리를 만들지 않았는지."""
    for name in ("utils", "model"):
        assert not (PROJECT_ROOT / name).exists()
