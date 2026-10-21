import json
from pathlib import Path
import sys

import numpy as np
import pytest

from efg import worker_protocol as protocol
from efg.detectors.worker import WorkerDetector


@pytest.mark.parametrize('with_metrics', [False, True])
def test_legacy_and_timed_worker_responses_are_compatible(tmp_path, monkeypatch, with_metrics):
    def fake_worker(path, **arrays):
        assert arrays['rgb'].dtype == np.uint8
        assert arrays['depth'].dtype == np.float32
        stem = Path(path).stem.replace('req_', 'res_')
        if with_metrics:
            (tmp_path / (stem + '.json')).write_text(json.dumps({'version': 1, 'inference_ms': 12.}))
        np.save(tmp_path / (stem + '.npy'), np.empty((0, 17)))
    monkeypatch.setattr(protocol, 'atomic_save_npz', fake_worker)
    client = protocol.WorkerClient(str(tmp_path))
    client.last_metrics = {'stale': 1}
    result = client.request(np.zeros((2, 2, 3)), np.zeros((2, 2)), np.eye(3))
    assert result.shape == (0, 17)
    assert client.last_metrics == ({'version': 1, 'inference_ms': 12.} if with_metrics else None)
    assert not list(tmp_path.iterdir())


def test_failed_worker_request_is_not_an_empty_success(tmp_path, monkeypatch):
    def fail(path, **arrays):
        (tmp_path / 'err_000000.txt').write_text('SDK failure')
    monkeypatch.setattr(protocol, 'atomic_save_npz', fail)
    client = protocol.WorkerClient(str(tmp_path))
    with pytest.raises(RuntimeError, match='SDK failure'):
        client.request(np.zeros((2, 2, 3)), np.zeros((2, 2)), np.eye(3))


def test_startup_failure_reports_log_and_reaps_child(tmp_path):
    script = tmp_path / 'fail.py'
    script.write_text('raise RuntimeError("test worker startup failure")')
    with pytest.raises(SystemExit, match='test worker startup failure'):
        WorkerDetector({'python': sys.executable, 'script': str(script), 'startup_timeout_s': 5},
                       name='fake')


def test_roundtrip_includes_cpu_metadata_and_clean_shutdown(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'worker.py'
    script.write_text(f'''import sys
sys.path.insert(0, {str(root)!r})
import argparse
import numpy as np
from efg.worker_protocol import serve
ap=argparse.ArgumentParser()
ap.add_argument('--work_dir')
args=ap.parse_args()
serve(args.work_dir, lambda *a: np.empty((0,17)), poll_interval=.005)
''')
    det = WorkerDetector({'python': sys.executable, 'script': str(script),
                          'startup_timeout_s': 10, 'poll_interval_s': .005}, name='fake')
    try:
        result = det.detect(np.zeros((2, 2, 3), dtype=np.uint8),
                            np.zeros((2, 2), dtype=np.float32), np.eye(3))
        assert len(result) == 0
        assert 0 <= det.last_inference_ms <= det.last_roundtrip_ms
    finally:
        det.close()
    assert det.proc.poll() is not None
    assert det._log.closed
