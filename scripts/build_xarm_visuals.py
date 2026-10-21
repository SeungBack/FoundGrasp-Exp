"""제조사 xArm7 모델의 부품별 재질을 보존한 Rerun용 GLB 생성.

개발 시에만 필요: pip install trimesh pycollada
실행: python scripts/build_xarm_visuals.py
런타임은 생성된 GLB를 직접 사용한다.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import trimesh

COMMIT = "cf51617d98bc55a88d9aa735acfa43a0bbb8e425"
BASE = f"https://raw.githubusercontent.com/xArm-Developer/xarm_ros/{COMMIT}"
DEST = Path(__file__).resolve().parents[1] / "third_party/xarm7_visual"
LINKS = ["link_base"] + [f"link{i}" for i in range(1, 8)]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{BASE}/LICENSE", timeout=30) as response:
        (DEST / "LICENSE").write_bytes(response.read())
    with TemporaryDirectory() as tmp:
        for name in LINKS:
            source = f"xarm_description/meshes/xarm7_1305/visual/{name}.dae"
            path = Path(tmp) / f"{name}.dae"
            with urlopen(f"{BASE}/{source}", timeout=30) as response:
                path.write_bytes(response.read())
            # Scene graph transforms and distinct material groups are retained.
            # Do not concatenate into a single uncolored mesh.
            scene = trimesh.load_scene(path)
            scene.export(DEST / f"{name}.glb")
            print(f"{name}: {len(scene.geometry)} material groups")
    (DEST / "VENDORED_FROM.txt").write_text(
        "Source : https://github.com/xArm-Developer/xarm_ros\n"
        f"Commit : {COMMIT}\n"
        "Files  : xarm_description/meshes/xarm7_1305/visual/*.dae\n"
        "License: BSD-3-Clause (UFACTORY), see LICENSE\n"
        "Derived: link_base, link1..link7 converted to GLB with trimesh/pycollada.\n"
        "         Original scene transforms and white/gray materials retained.\n"
        "         Source assets have no image texture maps.\n"
        "Rebuild: python scripts/build_xarm_visuals.py (requires trimesh, pycollada)\n"
        "Use    : visualization only; MJCF kinematics and robot control unchanged.\n"
    )


if __name__ == "__main__":
    main()
