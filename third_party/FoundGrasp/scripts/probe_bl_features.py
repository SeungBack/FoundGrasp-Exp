"""Diagnostic probe: B vs L trained FoundGrasp models on one real GraspNet image.

Measures, per backbone block:
  1. ||adapter delta|| / ||MLP residual||  (relative injection magnitude)
  2. residual stream norm growth
And per DPT tap:
  3. cosine similarity between adapted tap features and pristine frozen tap
     features (adapter zeroed) -> manifold drift caused by GeoFormer
  4. token-norm outlier stats of pristine frozen features (artifact check)

Read-only analysis; no training, no file writes outside stdout.
"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scipy.io as scio

from model.found_grasp import FoundGrasp

DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SCENE = '/data/Grasp/GraspNet-1Billion/scenes/scene_0130/realsense'
IMG_ID = '0000'
H_T, W_T = 360, 640
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_sample():
    rgb = np.asarray(Image.open(f'{SCENE}/rgb/{IMG_ID}.png'), dtype=np.float32) / 255.0  # (720,1280,3)
    depth = np.asarray(Image.open(f'{SCENE}/depth/{IMG_ID}.png'), dtype=np.float32)      # mm
    meta = scio.loadmat(f'{SCENE}/meta/{IMG_ID}.mat')
    K = meta['intrinsic_matrix']
    factor = float(meta['factor_depth'])
    H0, W0 = depth.shape
    # backproject at native res
    u, v = np.meshgrid(np.arange(W0), np.arange(H0))
    z = depth / factor
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    xyz = np.stack([x, y, z], 0)  # (3,H,W)

    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)[None]
    xyz_t = torch.from_numpy(xyz)[None].float()
    dep_t = torch.from_numpy(z)[None, None].float()
    rgb_t = F.interpolate(rgb_t, size=(H_T, W_T), mode='bilinear', align_corners=False)
    xyz_t = F.interpolate(xyz_t, size=(H_T, W_T), mode='nearest')
    dep_t = F.interpolate(dep_t, size=(H_T, W_T), mode='nearest')
    rgb_t = (rgb_t - MEAN) / STD
    return {
        'pixel_values': rgb_t.to(DEV),
        'xyz': xyz_t.to(DEV),
        'depth': dep_t.to(DEV),
        'depth_mask': (dep_t > 0).float().to(DEV),
    }


def build(ckpt_path, zero_adapter=False):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    cfg = ck['config'] if isinstance(ck, dict) and 'config' in ck else None
    sd = ck.get('model_state_dict') or ck.get('state_dict') or ck
    from omegaconf import OmegaConf
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    # normalize legacy peft type name
    if cfg.model.peft.type == 'rgbd_adaptformer':
        cfg.model.peft.type = 'geoformer'
    model = FoundGrasp(cfg.model)
    if zero_adapter:
        sd = {k: (torch.zeros_like(v) if ('peft.blocks' in k and ('up.weight' in k or 'up.bias' in k)) else v)
              for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    mk = [k for k in missing if 'backbone' in k or 'peft' in k]
    uk = [k for k in unexpected if 'backbone' in k or 'peft' in k]
    if mk or uk:
        print(f'  !! backbone/peft key mismatch: missing={mk[:5]} unexpected={uk[:5]}')
    model.eval().to(DEV)
    return model, cfg


@torch.no_grad()
def probe(name, final_path):
    print('=' * 30, name, '=' * 30)
    data = load_sample()
    model, cfg = build(final_path)

    # --- hooks: adapter delta norm + mlp residual norm per layer ---
    stats = {}
    def mk_adapter_hook(i):
        def h(mod, inp, out):
            stats.setdefault(i, {})['delta'] = out.float().norm(dim=-1).mean().item()
        return h
    def mk_mlp_hook(i):
        def h(mod, inp, out):
            stats.setdefault(i, {})['mlp'] = out.float().norm(dim=-1).mean().item()
        return h
    def mk_block_hook(i):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            stats.setdefault(i, {})['stream'] = o.float().norm(dim=-1).mean().item()
        return h
    handles = []
    for i, blk in enumerate(model.layers):
        handles.append(model.peft.blocks[i].register_forward_hook(mk_adapter_hook(i)))
        # layer_scale2 output = final MLP residual contribution (pre drop_path; eval -> identity)
        handles.append(blk.layer_scale2.register_forward_hook(mk_mlp_hook(i)))
        handles.append(blk.register_forward_hook(mk_block_hook(i)))

    taps_adapted = {}
    orig_norm = model._backbone_norm
    def tap_capture(model, store):
        idxs = model.out_indices
        feats = []
        hs_store = []
        def blk_out_hook(i):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                if i in idxs:
                    store[i] = o.detach()
            return h
        return blk_out_hook
    cap = {}
    mk = tap_capture(model, cap)
    for i, blk in enumerate(model.layers):
        handles.append(blk.register_forward_hook(mk(i)))

    model(data)
    for h in handles:
        h.remove()

    nl = len(model.layers)
    print(f'layers={nl}  taps={model.out_indices}  embed={model.embed_dim}')
    print('layer | ||delta||/tok | ||mlp_res||/tok | delta/mlp | ||stream||/tok')
    for i in range(nl):
        s = stats[i]
        print(f'{i:5d} | {s["delta"]:13.3f} | {s["mlp"]:15.3f} | {s["delta"]/max(s["mlp"],1e-9):9.3f} | {s["stream"]:14.1f}')
    tot_rel = sum(stats[i]['delta'] / stats[i]['stream'] for i in range(nl))
    print(f'sum over layers of ||delta||/||stream|| = {tot_rel:.4f}')

    # --- pristine (adapter-zeroed) model: tap drift + artifact stats ---
    model_z, _ = build(final_path, zero_adapter=True)
    cap0 = {}
    handles = []
    mk0 = tap_capture(model_z, cap0)
    for i, blk in enumerate(model_z.layers):
        handles.append(blk.register_forward_hook(mk0(i)))
    model_z(data)
    for h in handles:
        h.remove()

    print('tap | cos(adapted, frozen) | frozen token-norm med/p99/max (pre-LN)')
    for idx in model.out_indices:
        a = cap[idx][:, 1:, :].float()   # drop cls
        b = cap0[idx][:, 1:, :].float()
        cos = F.cosine_similarity(a.flatten(0, 1), b.flatten(0, 1), dim=-1)
        tn = b.norm(dim=-1).flatten()
        med = tn.median().item(); p99 = tn.quantile(0.99).item(); mx = tn.max().item()
        print(f'{idx:3d} | mean={cos.mean().item():.4f} p05={cos.quantile(0.05).item():.4f} | {med:7.1f} / {p99:7.1f} / {mx:7.1f} (max/med={mx/med:.1f})')

    del model, model_z
    torch.cuda.empty_cache()


which = sys.argv[1] if len(sys.argv) > 1 else 'both'
if which in ('base', 'both'):
    probe('BASE  (foundgrasp_vitb, bn96 dp0.2)', 'logs/tmech/foundgrasp_vitb/checkpoints/checkpoint_epoch_150.pth')
if which in ('large', 'both'):
    probe('LARGE (foundgrasp_vitl_0.2, bn192 dp0.2)', 'logs/tmech/foundgrasp_vitl_0.2/checkpoints/checkpoint_epoch_150.pth')
