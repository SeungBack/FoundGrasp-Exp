"""Reproducible perception artifacts, shared by offline replay and live cycles."""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .paths import PROJECT_ROOT, resolve


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(cfg, path, detector=None):
    name = cfg['detector']['name']
    block = cfg['detector'][name]
    weights = block.get('checkpoint')
    if not weights and '--checkpoint' in block.get('args', []):
        args = block['args']
        weights = args[args.index('--checkpoint') + 1]
    checkpoint = resolve(weights)
    def git(*args):
        p = subprocess.run(['git', '-C', str(PROJECT_ROOT), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    sources = [*PROJECT_ROOT.glob('efg/*.py'), *PROJECT_ROOT.glob('efg/detectors/*.py'),
               *PROJECT_ROOT.glob('workers/*.py')]
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': cfg, 'git_commit': git('rev-parse', 'HEAD'),
        'git_status': git('status', '--short'), 'driver_python': platform.python_version(),
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): file_digest(p) for p in sources},
        'calibration_sha256': {k: file_digest(resolve(v)) for k, v in cfg.get('calib', {}).items()
                               if v and resolve(v).is_file()},
        'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_sha256': file_digest(checkpoint) if checkpoint and checkpoint.is_file() else None,
        'width_semantics': 'required_opening_m' if name in ('foundgrasp', 'anygrasp') else 'raw_width_m',
        'timing_boundary': 'detect_ms includes IPC; inference_ms is worker preprocessing+SDK+decode+native filters, excluding IPC/shared filters',
        'success_measurement': 'transfer completion is not physical grasp success',
    }
    if detector is not None:
        metadata['detector_description'] = detector.describe()
        metadata['worker_runtime'] = getattr(detector, 'runtime_metadata', None)
    refiner = block.get('refiner', {})
    if refiner.get('enabled') and refiner.get('weights'):
        metadata['refiner_sha256'] = file_digest(resolve(refiner['weights']))
    if name == 'anygrasp' and '--sdk_dir' in block.get('args', []):
        args = block['args']
        sdk = resolve(args[args.index('--sdk_dir') + 1])
        metadata['sdk_binary_sha256'] = {
            p.name: file_digest(p) for p in (sdk / 'grasp_detection/gsnet_versions').glob('gsnet.cpython-310-*.so')}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


def save_detection(prefix, rgb, depth_m, K, detection):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(prefix) + '_frame.npz', rgb=rgb, depth_m=depth_m, K=K)
    np.save(str(prefix) + '_grasps.npy', detection.grasps.grasp_group_array)
    if detection.raw_grasps is not None:
        np.save(str(prefix) + '_raw_grasps.npy', detection.raw_grasps)
    info = {'counts': detection.counts, 'detect_ms': detection.detect_ms,
            'post_ms': detection.post_ms, 'inference_ms': detection.inference_ms}
    Path(str(prefix) + '_metrics.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
