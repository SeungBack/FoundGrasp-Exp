"""GFLA single-frame worker using the author's CFGNet prediction pipeline.

The model operates in a table-aligned frame. The experiment explicitly defines
that frame as the robot base and supplies the capture-time T_base_cam for every
request. Neither an identity transform nor an inferred plane is substituted.
"""
import argparse
import copy
import hashlib
import importlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from efg.worker_protocol import serve, validate_camera_pose


class NoContactFamilies(Exception):
    """The native contact detector produced no valid input to volume sampling."""


# The release checkpoint includes the older learned CSF decoder, while the
# published inference source uses only contact coordinates (CSSA.CSF.forward).
# Those layers are commented out upstream; test.py defaults to strict=False.
# Permit these inactive keys explicitly and strictly load every active layer.
UNUSED_CSF_CHECKPOINT_KEYS = frozenset(
    f'csf_net.{layer}.{field}'
    for layer, fields in [
        *[(f'feature_projector.mlp.{index}', ('weight', 'bias')) for index in (0, 2, 4)],
        *[(layer, ('weight', 'bias')) for layer in
          ('conv1', 'conv2', 'offset_regressor.0', 'score_regressor.0')],
        *[(layer, ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'))
          for layer in ('bn1', 'bn2')],
    ]
    for field in fields
)


def load_checkpoint(model, state):
    """Strictly load every active weight; permit only the unused release heads."""
    extra = set(state) - set(model.state_dict())
    if extra:
        if (not model.csf_net.return_coord_only
                or extra - UNUSED_CSF_CHECKPOINT_KEYS):
            raise RuntimeError(f'GFLA checkpoint contains unexpected active/unknown keys: {sorted(extra)}')
        state = copy.copy(state)  # Preserve state-dict metadata and caller input.
        for key in extra:
            del state[key]
    model.load_state_dict(state, strict=True)
    return sorted(extra)


def validate_table_transform(T_base_cam):
    if T_base_cam is None:
        raise ValueError("GFLA requires capture-time T_base_cam: table frame is robot base")
    return validate_camera_pose(T_base_cam)


def prepare_rgbd(rgb, depth_m, K):
    """Match GraspNet preprocessing: half size, then 32-aligned edge padding.

    RGB remains RGB as in upstream loadRGB(), with float bilinear interpolation.
    Depth stays in metres, uses nearest sampling and retains invalid zeros. The
    default 1280x720 capture becomes 640x384 (12 padded rows on each side).
    Intrinsics follow every resize/padding operation, including nondefault sizes.
    """
    import cv2

    rgb = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    intr = np.asarray(K, dtype=np.float64)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or depth.shape != rgb.shape[:2] or min(depth.shape) < 2):
        raise ValueError("GFLA input must be RGB uint8 HxWx3 and matching depth HxW")
    if (intr.shape != (3, 3) or not np.isfinite(intr).all()
            or intr[0, 0] <= 0 or intr[1, 1] <= 0
            or not np.allclose(intr[2], [0., 0., 1.])):
        raise ValueError("GFLA K must be finite pinhole intrinsics with positive focal lengths")
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
    height, width = depth.shape
    out_h, out_w = height // 2, width // 2
    resized_rgb = cv2.resize(rgb.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    resized_depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    scaled_K = intr.copy()
    scaled_K[0] *= out_w / width
    scaled_K[1] *= out_h / height
    # Upstream's 0.6*width padding equals 384x640 for the default capture.
    # Pad both dimensions to the network stride for other aspect ratios, rather
    # than letting FastSAM add unaccounted padding behind the projection code.
    padded_w = (out_w + 31) // 32 * 32
    padded_h = (max(out_h, int(out_w * .6)) + 31) // 32 * 32
    left = (padded_w - out_w) // 2
    top = (padded_h - out_h) // 2
    padding = ((top, padded_h - out_h - top), (left, padded_w - out_w - left))
    resized_rgb = np.pad(resized_rgb, (*padding, (0, 0)), mode='edge')
    resized_depth = np.pad(resized_depth, padding, mode='edge')
    scaled_K[0, 2] += left
    scaled_K[1, 2] += top
    return (np.ascontiguousarray(resized_rgb, dtype=np.float32),
            np.ascontiguousarray(resized_depth, dtype=np.float32),
            np.asarray(scaled_K, dtype=np.float32))


def predict(run_model, rgb, depth_m, K, *, T_base_cam=None):
    """Run native synthesis/CD/NMS and retain its width, score and frame output."""
    import torch

    table_from_camera = validate_table_transform(T_base_cam)
    image, depth, intr = prepare_rgbd(rgb, depth_m, K)
    if not np.any(depth > 0):
        return np.empty((0, 17), dtype=np.float64)
    # _run_model's four-input API requires RGB as a tensor; otherwise its
    # raw_rgb local is uninitialized in the author's current implementation.
    inputs = (torch.from_numpy(image[None]), torch.from_numpy(depth[None, None]),
              torch.from_numpy(intr[None]),
              torch.from_numpy(table_from_camera.astype(np.float32)[None]))
    try:
        with torch.no_grad():
            results = run_model(inputs)
    except NoContactFamilies:
        return np.empty((0, 17), dtype=np.float64)
    if not isinstance(results, (tuple, list)) or len(results) != 1:
        raise RuntimeError("GFLA predictor must return one grasp array for one input frame")
    rows = np.asarray(results[0], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 17 or not np.isfinite(rows).all():
        raise RuntimeError(f"GFLA returned invalid grasps: expected finite Nx17, got {rows.shape}")
    # Native predictor already restores the camera frame, adds WIDTH_ADD and
    # applies its custom 3cm/30deg NMS. No undocumented inverse scaling here.
    return rows.copy()


def prepare_runtime(repo_dir, work_dir, segment_checkpoint, fsam_size):
    """Satisfy upstream's Windows-style relative weight path in a private CWD."""
    repo = Path(repo_dir).expanduser().resolve()
    for relative in ('models/CFG_main.py', 'cfg/config.yaml', 'cfg/trainer_config.yaml'):
        if not (repo / relative).is_file():
            raise FileNotFoundError(f"GFLA source/config missing: {repo / relative}")
    if fsam_size not in ('s', 'x'):
        raise ValueError("GFLA fsam_size must be s or x")
    segment = Path(segment_checkpoint).expanduser().resolve()
    if not segment.is_file():
        raise FileNotFoundError(f"GFLA FastSAM checkpoint missing: {segment}")
    runtime = Path(work_dir).resolve() / 'gfla_runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    weight_link = runtime / f'base_weights\\FastSAM-{fsam_size}.pt'
    if not weight_link.is_symlink() or weight_link.resolve() != segment:
        weight_link.symlink_to(segment)
    return runtime


def _repository_revision(repo):
    result = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _package_revision(name):
    """Record pip's resolved Git revision for the pinned helper packages."""
    try:
        distribution = package_metadata.distribution(name)
    except package_metadata.PackageNotFoundError:
        return {'version': None}
    result = {'version': distribution.version}
    direct_url = distribution.read_text('direct_url.json')
    if direct_url:
        result['source'] = json.loads(direct_url)
    return result


def import_launcher(torch_module):
    """Keep the training helper's import-time GPU selection inside this worker.

    Upstream calls nvidia-smi as one executable string (Windows convention) and
    chooses a global GPU index. Give that exact query a CUDA-visible device-0
    response, then restore subprocess immediately. No inference code is patched.
    """
    query = 'nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,nounits,noheader'
    original = subprocess.check_output
    optional_modules = {}
    if (sys.platform != 'win32' and 'win32com' not in sys.modules
            and importlib.util.find_spec('win32com') is None):
        # Both training-helper modules unconditionally import this Windows-only
        # package to create .lnk shortcuts. Inference never uses that function.
        # Supply just its import boundary; attempting any COM action still fails.
        package = ModuleType('win32com')
        client = ModuleType('win32com.client')

        def unsupported_shortcut(*args, **kwargs):
            raise RuntimeError('Windows shortcut creation is unavailable in the GFLA inference worker')

        client.Dispatch = unsupported_shortcut
        package.client = client
        optional_modules = {'win32com': package, 'win32com.client': client}
        sys.modules.update(optional_modules)

    def visible_gpu(command, *args, **kwargs):
        if command == query:
            free, total = torch_module.cuda.mem_get_info(0)
            name = torch_module.cuda.get_device_name(0).replace(',', ' ')
            return f'0, {name}, {(total - free) // 2**20}, {total // 2**20}\n'.encode()
        return original(command, *args, **kwargs)

    subprocess.check_output = visible_gpu
    try:
        return importlib.import_module('LauncherTemplate')
    finally:
        subprocess.check_output = original
        for name, module in optional_modules.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]


def load_native_model(repo, checkpoint, model_cfg, predictor_cfg):
    """Import and use the original model and postprocessing, without a dataset."""
    import torch
    import typing

    # DatasetTemplate and CFD import these Python 3.11 annotation names.
    # Backport only these names in the isolated worker, leaving the original
    # model and installed helper source files unchanged.
    import typing_extensions
    for name in ('Self', 'TypeVarTuple'):
        if not hasattr(typing, name):
            setattr(typing, name, getattr(typing_extensions, name))

    sys.path.insert(0, str(repo))
    import_launcher(torch)
    # The source checkout's namespace expects its own C++ submodule; the setup
    # script installs the unmodified extension into the isolated environment.
    extension = importlib.import_module('grasp_nms_cpp')
    sys.modules['models.my_grasp_nms.grasp_nms_cpp'] = extension
    from models.CFG_main import CFGNet, CFG_Predictor

    class SingleFrameCFGNet(CFGNet):
        def _calc_contact_family(self, *args, **kwargs):
            families, valid = super()._calc_contact_family(*args, **kwargs)
            if not valid.any().item():
                # Native argmax over an empty family is undefined; an empty
                # detection result is valid and does not require fake contacts.
                raise NoContactFamilies()
            return families, valid

    model = SingleFrameCFGNet(model_cfg).cuda()
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
    unused_checkpoint_keys = load_checkpoint(model, state)
    model.eval()
    # run_model only needs these four inference attributes. Avoid constructing
    # the dataset/training launcher (filesystem logging, training state, workers).
    predictor = SimpleNamespace(model=model, config=predictor_cfg,
                                plot_mode=False, _visualize_grasp_in_scene=False)
    return (lambda inputs: CFG_Predictor.run_model(predictor, inputs)), unused_checkpoint_keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work_dir', required=True)
    ap.add_argument('--repo_dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--segment_checkpoint', help='Default: repo/base_weights/FastSAM-{fsam_size}.pt')
    ap.add_argument('--volume_per_batch', type=int, default=8, help='Upstream test.py default')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    repo = Path(args.repo_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GFLA checkpoint missing: {checkpoint}")
    if not 1 <= args.volume_per_batch <= 32:
        raise ValueError('GFLA volume_per_batch must be in [1,32]')
    work_dir = Path(args.work_dir).resolve()
    import yaml
    with open(repo / 'cfg/config.yaml') as f:
        model_cfg = yaml.safe_load(f)
    with open(repo / 'cfg/trainer_config.yaml') as f:
        predictor_cfg = yaml.safe_load(f)
    model_cfg['volume_per_batch'] = args.volume_per_batch
    model_cfg['print_timing'] = False
    predictor_cfg['restore_to_Cam'] = True
    predictor_cfg['print_timing'] = False
    size = model_cfg['fsam_size']
    segment = (Path(args.segment_checkpoint).expanduser().resolve() if args.segment_checkpoint
               else repo / 'base_weights' / f'FastSAM-{size}.pt')
    runtime = prepare_runtime(repo, work_dir, segment, size)
    # Old Ultralytics loads its published pickled nn.Module checkpoint. Limit
    # legacy loading compatibility to this isolated process; CFG weights above
    # explicitly use the safe state-dict loader.
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    os.chdir(runtime)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GFLA requires CUDA')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_model, unused_checkpoint_keys = load_native_model(repo, checkpoint, model_cfg, predictor_cfg)
    metadata = {
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'seed': args.seed, 'repo': str(repo), 'repo_commit': _repository_revision(repo),
        'python_compatibility': (['typing.Self and TypeVarTuple from typing_extensions']
                                 if sys.version_info < (3, 11) else []),
        'platform_compatibility': ['private Windows-style weight symlink',
                                   'import-time GPU query uses visible device',
                                   'unused Windows shortcut import; COM calls explicitly unsupported'],
        'checkpoint': str(checkpoint), 'segment_checkpoint': str(segment),
        'checkpoint_loading': {'strict_active_parameters': True,
                               'ignored_unused_csf_decoder_keys': unused_checkpoint_keys},
        'segment_checkpoint_sha256': _sha256(segment),
        'native_config_sha256': {
            name: _sha256(repo / 'cfg' / name)
            for name in ('config.yaml', 'trainer_config.yaml')},
        'effective_model_config': model_cfg,
        'effective_predictor_config': predictor_cfg,
        'packages': {name: _package_revision(name) for name in
                     ('LauncherTemplate', 'DatasetTemplate', 'ultralytics', 'pytorch3d')},
        'volume_per_batch': args.volume_per_batch,
        'table_frame_source': 'robot_base', 'calibration_input': 'T_base_cam',
        'launcher_device_selection': 'CUDA-visible device 0',
        'table_frame_assumption': 'table frame equals robot base; table surface is base Z=0',
        'width_semantics': 'required_opening_m',
        'native_width_add_m': float(predictor_cfg['WIDTH_ADD']),
        'native_postprocessing': {'collision': 'completed SDF',
                                  'collision_params': predictor_cfg['collision_params'],
                                  'nms_translation_m': .03, 'nms_rotation_deg': 30.,
                                  'gripper_offset_candidates_m': predictor_cfg['gripper_offset_candi'],
                                  'max_approaching_roll_rad': predictor_cfg['max_approaching_roll']},
        'preprocessing': 'half-resolution RGB bilinear / depth nearest; edge padding to 32-pixel stride; transformed K',
    }
    (work_dir / 'runtime.json').write_text(json.dumps(metadata, indent=2))
    print(f'-> GFLA loaded {checkpoint}; native opening margin={metadata["native_width_add_m"]:g}m; table=robot base', flush=True)

    def infer(rgb, depth, K, *, T_base_cam=None):
        return predict(run_model, rgb, depth, K, T_base_cam=T_base_cam)

    serve(str(work_dir), infer, poll_interval=.005, synchronize=torch.cuda.synchronize)


if __name__ == '__main__':
    main()
