"""Reproducible perception artifacts, shared by offline replay and live cycles."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .paths import FOUNDGRASP_ROOT, PROJECT_ROOT, resolve


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _foundgrasp_provenance(block, detector):
    """Snapshot the loaded recipe and native code, without importing a model.

    The vendored checkout has no separate Git history. Hash its entry points and
    already loaded modules rather than traversing checkpoints/data/the whole repo.
    """
    import yaml

    root = FOUNDGRASP_ROOT.resolve()
    sources = {root / name for name in (
        'grasp_detector.py', 'model/found_grasp.py', 'utils/config.py',
        'utils/collision_detector.py', 'scripts/verifier_points.py')}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if isinstance(filename, str):
            source = Path(filename).resolve()
            if source.suffix == '.py' and source.is_relative_to(root):
                sources.add(source)
    config_hashes = {}
    config = resolve(block.get('config'))
    while config is not None and config.is_file() and str(config) not in config_hashes:
        config_hashes[str(config)] = file_digest(config)
        parent = (yaml.safe_load(config.read_text()) or {}).get('__base__')
        config = resolve(parent, config.parent) if parent else None
    info = {
        'model_config_sha256': config_hashes,
        'native_source_sha256': {str(p.relative_to(root)): file_digest(p)
                                 for p in sorted(sources) if p.is_file()},
    }
    if detector is None:
        return info
    from omegaconf import OmegaConf

    loaded = getattr(getattr(detector, 'impl', None), 'config', None)
    info['effective_model_config'] = (OmegaConf.to_container(loaded, resolve=True)
                                      if OmegaConf.is_config(loaded) else loaded)
    info['refiner_enabled'] = getattr(detector, 'refiner', None) is not None
    info['refiner_env'] = dict(getattr(detector, 'refiner_env', {}))
    # Native verifier also reads optional VERIFIER_* settings from the process;
    # these are not necessarily present in the adapter's merged default map.
    info['refiner_process_env'] = {k: v for k, v in sorted(os.environ.items())
                                   if k.startswith('VERIFIER_')}
    info['final_nms'] = getattr(detector, 'final_nms', None)
    info['top_k'] = getattr(detector, 'top_k', None)
    torch = getattr(detector, 'torch', None)
    if torch is not None:
        info['runtime'] = {
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'device': str(detector.device),
            'gpu': (torch.cuda.get_device_name(detector.device)
                    if str(detector.device).startswith('cuda') and torch.cuda.is_initialized()
                    else None),
        }
    return info


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
               *PROJECT_ROOT.glob('workers/*.py'), *PROJECT_ROOT.glob('workers/shims/*.py')]
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': cfg, 'git_commit': git('rev-parse', 'HEAD'),
        'git_status': git('status', '--short'), 'driver_python': platform.python_version(),
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): file_digest(p) for p in sources},
        'calibration_sha256': {k: file_digest(resolve(v)) for k, v in cfg.get('calib', {}).items()
                               if isinstance(v, (str, Path)) and v and resolve(v).is_file()},
        'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_sha256': file_digest(checkpoint) if checkpoint and checkpoint.is_file() else None,
        'width_semantics': 'required_opening_m' if name in ('foundgrasp', 'anygrasp', 'gfla') else 'raw_width_m',
        'execution_open_multiplier': (1.0 if name == 'foundgrasp' else
                                      cfg['postprocess'].get('open_multiplier', 1.2)),
        'additional_detector_input': ({'T_base_cam': 'capture-time camera optical frame to robot base',
                                      'assumption': 'table frame equals robot base'} if name == 'gfla' else {}),
        'timing_boundary': 'detect_ms includes IPC; inference_ms is worker preprocessing+SDK+decode+native filters, excluding IPC/shared filters',
        'success_measurement': 'transfer completion is not physical grasp success',
    }
    if detector is not None:
        metadata['detector_description'] = detector.describe()
        metadata['worker_runtime'] = getattr(detector, 'runtime_metadata', None)
    if name == 'foundgrasp':
        metadata['foundgrasp_runtime'] = _foundgrasp_provenance(block, detector)
    worker_args = block.get('args', [])
    if '--repo_dir' in worker_args:
        repo = resolve(worker_args[worker_args.index('--repo_dir') + 1])
        result = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True)
        metadata['detector_repository'] = str(repo)
        metadata['detector_repository_commit'] = result.stdout.strip() if result.returncode == 0 else None
        status = subprocess.run(['git', '-C', str(repo), 'status', '--short'],
                                capture_output=True, text=True)
        metadata['detector_repository_status'] = status.stdout.strip() if status.returncode == 0 else None
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


def save_detection(prefix, rgb, depth_m, K, detection, *, rectified=False):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    context = {}
    if detection.T_base_cam is not None:
        from .worker_protocol import validate_camera_pose
        context = {'T_base_cam': validate_camera_pose(detection.T_base_cam),
                   'camera_pose_source': str(detection.camera_pose_source or 'provided')}
    np.savez(str(prefix) + '_frame.npz', rgb=rgb, depth_m=depth_m, K=K,
             rectified=bool(rectified), **context)
    np.save(str(prefix) + '_grasps.npy', detection.grasps.grasp_group_array)
    if detection.raw_grasps is not None:
        np.save(str(prefix) + '_raw_grasps.npy', detection.raw_grasps)
    info = {'counts': detection.counts, 'detect_ms': detection.detect_ms,
            'post_ms': detection.post_ms, 'inference_ms': detection.inference_ms}
    Path(str(prefix) + '_metrics.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
