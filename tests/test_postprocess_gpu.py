"""Shared collision backend equivalence, including chunk boundaries."""
import numpy as np
import pytest

from efg.paths import import_grasp_group
from efg.postprocess import backproject, shared_filter


@pytest.mark.parametrize("empty_scene", [False, True])
def test_gpu_matches_cpu_collision_candidates(empty_scene):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(77)
    depth = rng.uniform(.38, .42, (48, 64)).astype(np.float32)
    K = np.array([[100., 0, 32], [0, 100., 24], [0, 0, 1.]])
    points = backproject(depth, K)
    arr = np.zeros((530, 17))
    arr[:, 0] = rng.uniform(.1, 1, len(arr))
    arr[:, 1] = rng.uniform(.02, .10, len(arr))
    arr[:, 2] = .02
    arr[:, 3] = rng.uniform(.01, .06, len(arr))
    arr[:, 4:13] = Rotation.random(len(arr), random_state=rng).as_matrix().reshape(-1, 9)
    arr[:, 13:16] = points[rng.integers(len(points), size=len(arr))]
    arr[:100, 15] -= .15  # known free-space candidates
    if empty_scene:
        depth[:] = 0
    gg = import_grasp_group()(arr)
    cfg = {"postprocess": {"max_open_m": .095, "collision": {
        "enabled": True, "backend": "cpu", "voxel_size_m": .01,
        "approach_dist_m": .05, "thresh": .01}}}
    cpu, cpu_counts = shared_filter(gg, depth, K, cfg)
    cfg["postprocess"]["collision"]["backend"] = "cuda"
    gpu, gpu_counts = shared_filter(gg, depth, K, cfg)
    assert cpu_counts == gpu_counts
    np.testing.assert_array_equal(cpu.grasp_group_array, gpu.grasp_group_array)
    if not empty_scene:
        assert 0 < len(cpu) < cpu_counts["width_ok"]
