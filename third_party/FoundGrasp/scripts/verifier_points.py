"""Point-wise collision verifier on the frozen FoundGrasp features (post hoc; the detector is unchanged).

Each candidate is represented by lattice points of the evaluator's gripper boxes (two fingers, back
plate, closing region; the geometry of graspnetAPI collision_detection). Every point is projected
into the image and reads
  - the frozen backbone tokens at its pixel (768-d, learned linear map to 32-d; --feature vfm),
    the trained head feature for the ablation (--feature feat), or nothing (--feature none)
  - its camera depth z, its signed distance s = z - D(pixel) to the measured surface along the
    camera ray (0 where depth is missing), and a depth-valid flag
  - its gripper-frame coordinates and region (finger / back plate / closing).
A shared point network (per-point MLP -> max-pool over the gripper -> per-point MLP) yields a
point logit "inside an object". collision = logsumexp over the body points (any point inside),
empty = -logsumexp over the closing points (no point inside), and their logsumexp is the rejection
logit, trained with BCE against the graspnetAPI verdict of verifier_labels.py.

  python scripts/verifier_points.py cache --split train              # compact per-frame cache (CPU)
  CUDA_VISIBLE_DEVICES=2 python scripts/verifier_points.py train --tag P_vfm --feature vfm
  CUDA_VISIBLE_DEVICES=2 python scripts/verifier_points.py predict --tag P_vfm --split val9 --bench
Predictions go to logs/_verifier/models/<tag>_<split>_pred.pt in the order verifier_ap_val9.py expects.
"""
import argparse, glob, json, math, os, random, sys, time
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = "/data/Grasp/verifier"
MODELS = os.path.join(ROOT, "logs/_verifier/models")
PATCH = 14   # backbone patch size: the feature maps cover the input padded to a multiple of it; set_patch() from the data / detector
HEIGHT, DEPTH_BASE, FW = 0.02, 0.02, 0.01          # graspnetAPI gripper: thickness, base depth, finger width
N_FX, N_BY, N_CX, N_CY = 8, 16, 6, 4                # lattice counts: finger x, back-plate y, closing x, closing y
MAX_W = {"g1b": 0.1, "gc6d": 0.14}                  # the evaluators' width clip


def sample_files(split):
    return sorted(f for f in glob.glob(f"{DATA}/{split}/*.pt") if not f.endswith(".sdf.pt") and os.path.exists(f[:-3] + ".lbl.npy"))


# ----------------------------------------------------------------------------- cache (CPU)
_DS = {}


def _init(dataset, camera, split):
    torch.set_num_threads(1)
    from utils.config import load_config_with_base
    cfg = load_config_with_base(os.path.join(ROOT, "configs/ours_base.yaml"))
    if dataset == "g1b":
        from datasets.graspnet1b import GraspNet1BDataset
        ds = GraspNet1BDataset(root=cfg.datasets.GraspNet1B.root, camera=camera, split="train" if split.startswith("train") else "test",
                               image_size=tuple(cfg.dataset.image_size), backbone_name=cfg.model.backbone.name,
                               depth_corruption=None, pixelwise_gt_dir=cfg.dataset.get("pixelwise_gt_dir", "pixelwise_grasp_gt_top300"))
        index = {(s, int(f)): i for i, (s, f) in enumerate(zip(ds.scenename, ds.frameid))}
    else:
        from datasets.graspclutter6d import GraspClutter6DDataset
        ds = GraspClutter6DDataset(root=cfg.datasets.GraspClutter6D.root, camera=camera, split="test",
                                   image_size=tuple(cfg.dataset.image_size), backbone_name=cfg.model.backbone.name)
        index = {(f"{s:06d}", int(f)): i for i, (s, f) in enumerate(zip(ds.scene_ids, ds.img_ids))}
    _DS.update(ds=ds, index=index)


def _item(f):
    d = torch.load(f, weights_only=False, map_location="cpu")
    xyz = _DS["ds"][_DS["index"][(d["scene"], int(d["frame"]))]]["xyz"]
    g1b = d.get("dataset", "g1b") == "g1b"
    return dict(name=os.path.basename(f), dataset="g1b" if g1b else "gc6d",
                scene_int=int(d["scene"].split("_")[1]) if g1b else int(d["scene_int"]), levels=d["levels"],
                grasps=d["grasps"].float(), level=d["level"].to(torch.int8), y=torch.from_numpy(np.load(f[:-3] + ".lbl.npy")),
                K=d["K"].float(), depth=(xyz[2] * (xyz[2] > 0.1)).half(), vfm=d["vfm"], feat=d.get("feat"))


def _item_extra(f):
    d = torch.load(f, weights_only=False, map_location="cpu")
    return dict(name=os.path.basename(f), dpt=d["dpt"], geo=d.get("geo"))


def build_cache(a):
    files = sample_files(a.split)
    if a.mode == "cache_extra":  # DPT-decoder output (64x104x184) and geometry-encoder features (64x91x161)
        with Pool(a.proc) as p:
            items = p.map(_item_extra, files, chunksize=4)
        torch.save(items, f"{DATA}/cache/points_extra_{a.split}.pt")
        print(f"cached {len(items)} extra frames of {a.split}")
        return
    d0 = torch.load(files[0], weights_only=False, map_location="cpu")
    with Pool(a.proc, initializer=_init, initargs=(d0.get("dataset", "g1b"), d0["camera"], a.split)) as p:
        items = p.map(_item, files, chunksize=4)
    os.makedirs(f"{DATA}/cache", exist_ok=True)
    torch.save(items, f"{DATA}/cache/points_{a.split}.pt")
    print(f"cached {len(items)} frames of {a.split}")


# ----------------------------------------------------------------------------- geometry
CTX = (0.03, 0.03, 6, 12, 4)   # context box: 3 cm beyond the fingertips along the approach, 3 cm outside each finger; counts x, y, z


CG = (3, 48, 9, 0.0025, 0.02)   # contact grid: samples along approach x closing x thickness, spacing (m), margin beyond each finger (m)


def contact_grid(g, max_w):
    """(n, p, 3) gripper-frame points of a fine regular grid around the evaluator's contact line: x = fingertip depth
    (+-1 spacing), y over [-(w/2 + margin), +(w/2 + margin)], z over +-4 spacings. The benchmark closes the fingers along this
    line (dexnet centre = t + depth * approach) and judges force closure from the surface normals at the two contacts."""
    kx, ky, kz, sp, mg = CG
    n, dev = len(g), g.device
    w = g[:, 1].clamp(0, max_w)[:, None]; d = g[:, 3][:, None]
    x = d + (torch.arange(kx, device=dev, dtype=torch.float32) - kx // 2)[None] * sp
    y = -(w / 2 + mg) + (torch.arange(ky, device=dev, dtype=torch.float32) + 0.5)[None] / ky * (w + 2 * mg)
    z = ((torch.arange(kz, device=dev, dtype=torch.float32) - kz // 2) * sp)[None].expand(n, -1)
    shape = (n, kx, ky, kz)
    return torch.stack([x[:, :, None, None].expand(shape), y[:, None, :, None].expand(shape), z[:, None, None, :].expand(shape)], -1).reshape(n, -1, 3)


def lattice(g, max_w, counts=(N_FX, N_BY, N_CX, N_CY), ctx=False):
    """(n, p, 3) gripper-frame lattice points of the evaluator boxes and their region (0 finger, 1 back plate, 2 closing).
    counts = points along the finger, across the back plate, and along / across the closing region (fingers 2 x 4 and the
    back plate 2 x 4 in the other two axes, the closing region 2 in thickness); the default gives 304 points."""
    N_FX, N_BY, N_CX, N_CY = counts
    n, dev = len(g), g.device
    w = g[:, 1].clamp(0, max_w)[:, None]; d = g[:, 3][:, None]
    lin = lambda k: (torch.arange(k, device=dev, dtype=torch.float32) + 0.5) / k
    z4 = (lin(4) - 0.5) * HEIGHT
    fx = -DEPTH_BASE + lin(N_FX) * (d + DEPTH_BASE)
    fy = w / 2 + FW * torch.tensor([0.25, 0.75], device=dev)
    bx = (-DEPTH_BASE - FW * torch.tensor([0.75, 0.25], device=dev)).expand(n, 2)
    by = -(w / 2 + FW) + lin(N_BY) * (w + 2 * FW)
    cx = -DEPTH_BASE + lin(N_CX) * (d + DEPTH_BASE)
    cy = -w / 2 + lin(N_CY) * w
    z2 = torch.tensor([-0.005, 0.005], device=dev)

    def box(x, y, z):  # (n,a), (n,b), (c,) -> (n, a*b*c, 3)
        shape = (n, x.shape[1], y.shape[1], len(z))
        return torch.stack([x[:, :, None, None].expand(shape), y[:, None, :, None].expand(shape), z[None, None, None].expand(shape)], -1).reshape(n, -1, 3)
    f = box(fx, fy, z4)
    P = torch.cat([f, f * torch.tensor([1.0, -1.0, 1.0], device=dev), box(bx, by, z4), box(cx, cy, z2)], 1)
    region = torch.cat([torch.zeros(2 * f.shape[1]), torch.ones(2 * N_BY * 4), torch.full((N_CX * N_CY * 2,), 2.0)]).long().to(dev)
    if ctx:  # context points outside the evaluator's boxes (region 3): the amodal cross-section below and beside the fingers;
        dx, dy, kx, ky, kz = CTX   # they feed the pooled context only, never the collision logits
        ex = -DEPTH_BASE + lin(kx) * (d + DEPTH_BASE + dx)
        ey = -(w / 2 + FW + dy) + lin(ky) * (w + 2 * FW + 2 * dy)
        ez = (lin(kz) - 0.5) * (HEIGHT + 0.02)
        P = torch.cat([P, box(ex, ey, ez)], 1)
        region = torch.cat([region, torch.full((kx * ky * kz,), 3, device=dev, dtype=torch.long)])
    return P, region


def depth_normals(depth, K):
    """Unit surface normals (3, H, W) of the measured depth by central differences of the back-projected point map; 0 where invalid."""
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    z = depth.float(); X = torch.stack([(u - K[2]) * z / K[0], (v - K[3]) * z / K[1], z])
    dx = torch.zeros_like(X); dy = torch.zeros_like(X)
    dx[:, :, 1:-1] = X[:, :, 2:] - X[:, :, :-2]; dy[:, 1:-1] = X[:, 2:] - X[:, :-2]
    nrm = torch.cross(dx, dy, dim=0)
    ok = (z > 0.1); ok = ok & torch.roll(ok, 1, 1) & torch.roll(ok, -1, 1) & torch.roll(ok, 1, 0) & torch.roll(ok, -1, 0)
    nrm = nrm / nrm.norm(dim=0, keepdim=True).clamp(min=1e-6)
    nrm = torch.where((nrm * X).sum(0, keepdim=True) > 0, -nrm, nrm)   # orient toward the camera
    return nrm * ok[None]


def point_inputs(g, P, region, K, depth, fmap, proj, use_ray=True, use_pos=True, use_normal=False, n_reg=None):
    """Per-point inputs (n, p, d): ray geometry [z, 10*s, valid], position [gripper xyz scaled, region one-hot],
    the measured surface normal at the point's pixel in the gripper frame (optional), projected map feature."""
    n, p, _ = P.shape
    R = g[:, 4:13].reshape(n, 3, 3); t = g[:, 13:16]
    X = torch.einsum("nij,npj->npi", R, P) + t[:, None]
    z = X[..., 2].clamp(min=1e-3)
    u = K[0] * X[..., 0] / z + K[2]; v = K[1] * X[..., 1] / z + K[3]
    H, W = depth.shape
    ui, vi = u.round().long(), v.round().long()
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    D = torch.zeros_like(z); D[ok] = depth[vi[ok], ui[ok]].float()
    valid = ok & (D > 0.1)
    s = torch.where(valid, X[..., 2] - D, torch.zeros_like(z))
    parts = []
    if use_ray:
        parts.append(torch.stack([X[..., 2], 10 * s, valid.float()], -1))
    if use_pos:
        n_reg = n_reg or (4 if bool((region == 3).any()) else 3)
        parts.append(torch.cat([P / torch.tensor([0.06, 0.07, 0.01], device=P.device), F.one_hot(region, n_reg).float()[None].expand(n, -1, -1)], -1))
    if use_normal:  # measured surface normal at the pixel, expressed in the gripper frame (x approach, y closing, z thickness)
        N = depth_normals(depth, K)
        nc = torch.zeros_like(X); nc[ok] = N[:, vi[ok], ui[ok]].T
        parts.append(torch.einsum("nji,npj->npi", R, nc))   # R^T n: camera -> gripper frame
    if fmap is not None:  # one or more maps (e.g. frozen tokens and head feature), each with its own linear projection
        Hp, Wp = math.ceil(H / PATCH) * PATCH, math.ceil(W / PATCH) * PATCH   # the maps cover the padded input
        grid = torch.stack([(u + 0.5) / Wp * 2 - 1, (v + 0.5) / Hp * 2 - 1], -1).reshape(1, -1, 1, 2)
        for fm, pr in zip(fmap, proj):
            m = pr(fm.float().permute(1, 2, 0)).permute(2, 0, 1)[None]
            parts.append(F.grid_sample(m, grid, mode="bilinear", align_corners=False)[0, :, :, 0].T.reshape(n, p, -1))
    return torch.cat(parts, -1)


def n_points(counts, ctx):
    N_FX, N_BY, N_CX, N_CY = counts
    return 2 * N_FX * 2 * 4 + 2 * N_BY * 4 + N_CX * N_CY * 2 + (CTX[2] * CTX[3] * CTX[4] if ctx else 0)


CROP_K, CROP_MARGIN = 8, 0.03   # token crop: K x K samples over the gripper footprint plus a 3 cm margin, aligned to the closing axis


def crop_inputs(g, K, depth, fmap, proj, maps=None, k=CROP_K, nodepth=False):
    """(n, C, K, K) token map sampled on a gripper-aligned image grid around the projected grasp: axis a along the projected
    closing direction, b perpendicular; half extent w/2 + margin (metric, converted with the centre depth). Reads the
    object context around the grasp directly, independent of the occupancy lattice."""
    n = len(g); H, W = depth.shape
    R = g[:, 4:13].reshape(n, 3, 3); t = g[:, 13:16]; y = R[:, :, 1]
    z0 = t[:, 2].clamp(min=1e-3); u0 = K[0] * t[:, 0] / z0 + K[2]; v0 = K[1] * t[:, 1] / z0 + K[3]
    t1 = t + 0.005 * y; z1 = t1[:, 2].clamp(min=1e-3)
    d2 = torch.stack([K[0] * t1[:, 0] / z1 + K[2] - u0, K[1] * t1[:, 1] / z1 + K[3] - v0], -1)
    d2 = d2 / d2.norm(dim=1, keepdim=True).clamp(min=1e-6); p2 = torch.stack([-d2[:, 1], d2[:, 0]], -1)
    half = (g[:, 1].clamp(0, 0.14) / 2 + CROP_MARGIN) * K[0] / z0                     # pixels
    lin = (torch.arange(k, device=g.device, dtype=torch.float32) + 0.5) / k * 2 - 1
    a, b = torch.meshgrid(lin, lin, indexing="ij")                                     # (K, K): a along the closing axis
    u = u0[:, None, None] + half[:, None, None] * (a[None] * d2[:, 0, None, None] + b[None] * p2[:, 0, None, None])
    v = v0[:, None, None] + half[:, None, None] * (a[None] * d2[:, 1, None, None] + b[None] * p2[:, 1, None, None])
    Hp, Wp = math.ceil(H / PATCH) * PATCH, math.ceil(W / PATCH) * PATCH
    grid = torch.stack([(u + 0.5) / Wp * 2 - 1, (v + 0.5) / Hp * 2 - 1], -1).reshape(1, -1, 1, 2)
    m = proj(fmap.float().permute(1, 2, 0)).permute(2, 0, 1)[None]
    out = F.grid_sample(m, grid, mode="bilinear", align_corners=False)[0, :, :, 0].T.reshape(n, k, k, -1).permute(0, 3, 1, 2)
    if maps is not None:   # fine-resolution channels from the input-size maps: relative depth (x10), depth valid, objectness, graspness, score
        ui, vi = u.round().long().clamp(0, W - 1), v.round().long().clamp(0, H - 1)
        dz = depth[vi, ui].float(); valid = (dz > 0.1).float()
        rel = ((dz - z0[:, None, None]) * 10).clamp(-0.5, 0.5) * valid
        if nodepth:   # sensor-agnostic variant: keep only the detector's maps
            rel = torch.zeros_like(rel); valid = torch.zeros_like(valid)
        sc = maps[:, vi, ui].float() / 255.0                                            # (3, n, k, k)
        out = torch.cat([out, rel[:, None], valid[:, None], sc.permute(1, 0, 2, 3)], 1)
    return out


# ----------------------------------------------------------------------------- model
class FieldConv(nn.Module):
    """Convolutional decoder of the contact-surface distance field on the regular contact grid (kx, ky, kz): the kx approach
    slices are folded into channels and 3x3 convolutions dilated along the closing axis give every cell its neighbourhood
    (a distance field is 1-Lipschitz, so the neighbouring observations constrain it); the gripper-wide lattice context
    (max-pooled phi of the collision lattice, 128-d) enters the per-cell head."""

    def __init__(self, d_in, hidden=64, d_ctx=128, dil=(1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Conv2d(d_in, hidden, 1)
        self.blocks = nn.ModuleList([nn.Conv2d(hidden, hidden, 3, padding=(d, 1), dilation=(d, 1)) for d in dil])
        self.head = nn.Sequential(nn.Conv2d(hidden + d_ctx, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, CG[0], 1))

    def forward(self, xg, ctx):
        """xg (n, kx * d, ky, kz) per-cell inputs; ctx (n, d_ctx) -> distance logits (n, kx, ky, kz)."""
        h = F.relu(self.inp(xg))
        for b in self.blocks:
            h = h + F.relu(b(h))
        return self.head(torch.cat([h, ctx[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])], 1))


class PointVerifier(nn.Module):
    MU = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)   # the benchmark's friction thresholds

    def __init__(self, c_map, use_ray=True, use_pos=True, pool=True, counts=(N_FX, N_BY, N_CX, N_CY), d_tok=32, quality=False, use_normal=False, refine=False, ctx=False, crop=False, field=False, crop_maps=False, crop_k=CROP_K, crop_nodepth=False, cgrid=False, cg_hidden=64, readout_tok=False, cg_fine=False, objctx=False, cg_conv=False, cg_mirror=False, readout_obs=False):
        super().__init__()
        cs = [c for c in ([c_map] if isinstance(c_map, int) else list(c_map)) if c]   # channels of each sampled map
        self.cfg = dict(c_map=cs, use_ray=use_ray, use_pos=use_pos, pool=pool, counts=tuple(counts), quality=quality, use_normal=use_normal, refine=refine, ctx=ctx, crop=crop, field=field, crop_maps=crop_maps, crop_k=crop_k, crop_nodepth=crop_nodepth, d_tok=d_tok, cgrid=cgrid, cg_hidden=cg_hidden, readout_tok=readout_tok, cg_fine=cg_fine, objctx=objctx, cg_conv=cg_conv, cg_mirror=cg_mirror, readout_obs=readout_obs)
        # contact-grid field: per-point distance-to-surface regression on the fine grid around the contact line (CAD supervision),
        # read by a geometry-only contact readout (scripts/verifier_readout.py) that yields the force-closure logits
        if cgrid:
            from verifier_readout import ContactReadout
            d_cg = 3 * use_ray + (7 if ctx else 6) * use_pos + 3 * use_normal + d_tok * len(cs)
            h = cg_hidden
            if cg_conv:   # convolutional decoder over the regular contact grid instead of the per-cell PointNet
                # cg_mirror: every cell also receives the inputs of its mirror cell across the grasp centre (y -> -y), a bilateral
                # symmetry prior: the observed near-side contact constrains the occluded far-side one
                self.cg_net = FieldConv(CG[0] * d_cg * (2 if cg_mirror else 1), h)
            else:
                self.phi_cg = nn.Sequential(nn.Linear(d_cg, h), nn.ReLU(), nn.Linear(h, 2 * h), nn.ReLU())
            # objctx: the gripper-aligned image crop (tokens + detector maps) summarised by the crop CNN joins the field context,
            # giving every cell one object-level shape prior instead of only the front-surface token of its own ray
            if not cg_conv:
                self.psi_cg = nn.Sequential(nn.Linear(d_cg + 2 * h + 128 + 64 * objctx, 2 * h), nn.ReLU(), nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
            self.readout = ContactReadout(ctx_dim=128 if readout_tok else 0, obs=readout_obs)   # readout_obs: observed-cell flags join the readout (geometry only)
        # token crop branch (quality only): a gripper-aligned k x k crop of the projected token map (+ depth / score maps) -> 64-d
        self.cnet = nn.Sequential(nn.Conv2d(d_tok + 5 * crop_maps, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(32 * (crop_k // 2) ** 2, 64), nn.ReLU()) if crop else None
        # pose-refinement head: residual [rotation vector (gripper frame), 10 x width, 10 x depth] toward the labelled grasp
        # of the candidate's pixel (scripts/verifier_pose_targets.py), read from the same gripper-wide context
        self.dhead = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5)) if refine else None
        # optional grasp-quality head: P(force closure at friction mu_j) from the gripper-wide context (needs pool)
        # field readout: the quality head also reads the per-point occupancy field in its fixed lattice order (amodal cross-section)
        self.qhead = nn.Sequential(nn.Linear(128 + 64 * crop + n_points(counts, ctx) * field, 128 if field else 64), nn.ReLU(),
                                   nn.Linear(128 if field else 64, len(self.MU))) if quality else None
        # single-scalar variant (control): one logit for the expected success rate over the six thresholds, collision counted as 0
        self.shead = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)) if quality == "scalar" else None
        # rank variant: residual re-scoring log s' = log s + delta([context, log s]), trained with a listwise loss per frame
        self.rhead = nn.Sequential(nn.Linear(129, 64), nn.ReLU(), nn.Linear(64, 1)) if quality == "rank" else None
        self.proj = nn.ModuleList([nn.Linear(c, d_tok) for c in cs]) if cs else None
        d = 3 * use_ray + (7 if ctx else 6) * use_pos + 3 * use_normal + d_tok * len(cs)
        self.phi = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU()) if pool else None
        self.psi = nn.Sequential(nn.Linear(d + 128 * pool, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.empty_bias = nn.Parameter(torch.zeros(()))

    def forward(self, x, region, logs=None, crop=None):
        """x (n, p, d) -> rejection logit (n,), point logits (n, p), quality logits (n, 6) / (n, 1) or None.
        logs: (n,) log detector score, used by the rank variant."""
        q = None
        if self.phi is not None:  # gripper-wide context: max over the points
            g = self.phi(x).max(1).values
            x = torch.cat([x, g[:, None].expand(-1, x.shape[1], -1)], -1)
            if self.cnet is not None and crop is not None:   # the crop branch joins the quality head only
                g = torch.cat([g, self.cnet(crop)], 1)
            if self.cfg.get("field") and self.qhead is not None:
                g = torch.cat([g, torch.sigmoid(self.psi(x)[..., 0])], 1)
            if self.rhead is not None:
                q = self.rhead(torch.cat([g[:, :128], logs[:, None]], 1))   # (n, 1) residual delta on the log score
            elif self.shead is not None:
                q = self.shead(g[:, :128])   # (n, 1) expected-success logit
            elif self.qhead is not None and not (self.cnet is not None and crop is None):   # (skipped when the crop branch is not computed: cgrid overrides q)
                q = self.qhead(g)
            elif self.dhead is not None:
                q = self.dhead(g[:, :128])   # (n, 5) pose residual
        l = self.psi(x)[..., 0]
        body = region < 2; closing = region == 2
        collision = torch.logsumexp(l[:, body], 1)
        empty = -torch.logsumexp(l[:, closing], 1) + self.empty_bias
        return torch.logsumexp(torch.stack([collision, empty], 1), 1), l, q


def set_patch(p):
    global PATCH
    PATCH = int(p)


def patch_from_tokens(H, W, th, tw):
    """The patch size that maps an (H, W) input to a (th, tw) token grid (14 for DINOv2 / DAv2, 16 for SAM / DAv3)."""
    for p in range(8, 33):
        if -(-H // p) == th and -(-W // p) == tw:
            return p
    raise ValueError(f"no patch size maps {(H, W)} to a {(th, tw)} token grid")


def load_frames(split, feature, maps=False):
    """The per-frame cache, merged with the extra cache when a decoder or geometry-encoder feature is requested,
    and with the detector's objectness / graspness / score maps (scripts/verifier_mask_cache.py) when maps=True."""
    frames = torch.load(f"{DATA}/cache/points_{split}.pt", weights_only=False)
    if "vfm" in frames[0]:   # the cached token grid fixes the backbone's patch size
        set_patch(patch_from_tokens(*frames[0]["depth"].shape, *frames[0]["vfm"].shape[-2:]))
    if maps:
        extra = torch.load(f"{DATA}/cache/points_obj_{split}.pt", weights_only=False)
        for c, e in zip(frames, extra):
            assert c["name"] == e["name"]
            c["maps"] = e["maps"]
    if feature != "none" and set(feature.split("+")) & {"dpt", "geo"}:
        extra = torch.load(f"{DATA}/cache/points_extra_{split}.pt", weights_only=False)
        for c, e in zip(frames, extra):
            assert c["name"] == e["name"]
            c.update(dpt=e["dpt"], geo=e["geo"])
    if feature != "none" and "dav2" in feature.split("+"):   # pure DAv2-B tokens (scripts/verifier_add_dav2.py)
        extra = torch.load(f"{DATA}/cache/points_extra_dav2_{split}.pt", weights_only=False)
        for c, e in zip(frames, extra):
            assert c["name"] == e["name"]
            c.update(dav2=e["dav2"])
    return frames


def to_gpu(c, feature):
    out = dict(g=c["grasps"].cuda(), lvl=c["level"].cuda(), y=c["y"].cuda().float(), K=c["K"].cuda(), depth=c["depth"].cuda(),
               fmap=None if feature == "none" else [c[f].cuda() for f in feature.split("+")], levels=c["levels"], scene=c["scene_int"], max_w=MAX_W[c["dataset"]])
    if "tgt" in c:
        out["tgt"] = c["tgt"].cuda(); out["tgt_has"] = c["tgt_has"].cuda()
    if "maps" in c:
        out["maps"] = c["maps"].cuda()
    return out


def apply_delta(g, delta, clearance=0.010):
    """Refined grasps: rotate about the pixel point (centre + clearance x approach) by the gripper-frame rotation vector,
    then re-apply the palm clearance along the new approach; width and depth take their residuals (clamped to the decoder's ranges)."""
    n = len(g); R = g[:, 4:13].reshape(n, 3, 3)
    p = g[:, 13:16] + clearance * R[:, :, 0]
    rv = delta[:, :3]; ang = rv.norm(dim=1, keepdim=True).clamp(min=1e-8); k = rv / ang
    Kx = torch.zeros(n, 3, 3, device=g.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    s, c = torch.sin(ang)[:, :, None], torch.cos(ang)[:, :, None]
    Rn = R @ (torch.eye(3, device=g.device) + s * Kx + (1 - c) * (Kx @ Kx))
    out = g.clone()
    out[:, 4:13] = Rn.reshape(n, 9)
    out[:, 13:16] = p - clearance * Rn[:, :, 0]
    out[:, 1] = (g[:, 1] + delta[:, 3] / 10).clamp(0.0, 0.1)
    out[:, 3] = (g[:, 3] + delta[:, 4] / 10).clamp(0.01, 0.05)
    return out


DROP = dict(tok=0.0, pt=0.0)   # training-time input dropout: token channels (per candidate) and lattice points (per candidate)


def forward_frame(model, c, idx, with_quality=False, with_points=False):
    g = c["g"][idx]
    P, region = lattice(g, c["max_w"], model.cfg["counts"], model.cfg.get("ctx", False))
    x = point_inputs(g, P, region, c["K"], c["depth"], c["fmap"], model.proj, model.cfg["use_ray"], model.cfg["use_pos"], model.cfg.get("use_normal", False))
    crop = crop_inputs(g, c["K"], c["depth"], c["fmap"][0], model.proj[0], c.get("maps") if model.cfg.get("crop_maps") else None,
                       model.cfg.get("crop_k", CROP_K), model.cfg.get("crop_nodepth", False)) if model.cfg.get("crop") and (not model.cfg.get("cgrid") or model.cfg.get("objctx")) else None
    # (with cgrid the crop branch only feeds the unused quality head, so it is skipped: inference then needs no detector maps)
    if model.training and (DROP["tok"] > 0 or DROP["pt"] > 0):
        n, p, d = x.shape; d_tok = model.cfg.get("d_tok", 32); geo = d - d_tok * len(model.cfg["c_map"])
        if DROP["tok"] > 0:   # drop token channels per candidate (same channels for all its points), keep geometry
            keep = (torch.rand(n, 1, d - geo, device=x.device) > DROP["tok"]).float() / (1 - DROP["tok"])
            x = torch.cat([x[..., :geo], x[..., geo:] * keep], -1)
            if crop is not None:
                kc = (torch.rand(n, d_tok, 1, 1, device=x.device) > DROP["tok"]).float() / (1 - DROP["tok"])
                crop = torch.cat([crop[:, :d_tok] * kc, crop[:, d_tok:]], 1)
        if DROP["pt"] > 0:    # drop whole lattice points per candidate
            x = x * (torch.rand(n, p, 1, device=x.device) > DROP["pt"]).float()
    L, l, q = model(x, region, torch.log(g[:, 0].clamp(min=1e-6)), crop)
    if model.cfg.get("cgrid"):   # predicted contact-grid distance field (mm) and its geometry-only readout
        kx, ky, kz, _, _ = CG
        Pc = contact_grid(g, c["max_w"])
        xc = point_inputs(g, Pc, torch.full((Pc.shape[1],), 2, device=g.device, dtype=torch.long), c["K"], c["depth"], c["fmap"], model.proj,
                          model.cfg["use_ray"], model.cfg["use_pos"], model.cfg.get("use_normal", False), n_reg=4 if model.cfg.get("ctx") else 3)
        if model.cfg.get("cg_conv"):   # (n, kx, ky, kz, d) -> (n, kx * d, ky, kz)
            xg = xc.reshape(len(g), kx, ky, kz, -1).permute(0, 1, 4, 2, 3).reshape(len(g), -1, ky, kz)
            if model.cfg.get("cg_mirror"):
                xg = torch.cat([xg, xg.flip(2)], 1)
            Dh = torch.sigmoid(model.cg_net(xg, model.phi(x).max(1).values)) * model.readout.d_max   # (n, kx, ky, kz) mm
        else:
            hc = model.phi_cg(xc); ctx = torch.cat([hc.max(1).values, model.phi(x).max(1).values] + ([model.cnet(crop)] if model.cfg.get("objctx") else []), 1)
            Dh = torch.sigmoid(model.psi_cg(torch.cat([xc, ctx[:, None].expand(-1, xc.shape[1], -1)], -1))[..., 0]) * model.readout.d_max   # (n, p) mm
            Dh = Dh.reshape(len(g), kx, ky, kz)
        model._last_ctx = model.phi(x).max(1).values if model.cfg.get("readout_tok") else None
        # observed cells: valid pixel and the cell lies in front of / at the measured surface (10 s <= 0.02 <-> s <= 2 mm)
        model._last_obs = ((xc[..., 2] > 0) & (xc[..., 1] <= 0.02)).float().reshape(len(g), kx, ky, kz) if model.cfg.get("readout_obs") else None
        q = model.readout(Dh, g[:, 1].clamp(0, c["max_w"]), model._last_ctx, model._last_obs)
        model._last_field = Dh
    if with_points:
        return L, q, l
    return (L, q) if with_quality else L


@torch.no_grad()
def predict_frames(model, frames, chunk=2048):
    out = []
    for c in frames:
        idx = c.get("allowed", torch.arange(len(c["g"]), device="cuda"))
        out.append((torch.cat([forward_frame(model, c, idx[i:i + chunk]) for i in range(0, len(idx), chunk)]), c["y"][idx]))
    return out


def auroc(res):
    from sklearn.metrics import roc_auc_score
    L = torch.cat([r[0] for r in res]).cpu().numpy(); y = torch.cat([r[1] for r in res]).cpu().numpy()
    return float(roc_auc_score(y, -L)) if 0 < y.mean() < 1 else float("nan")


def train(a):
    torch.manual_seed(a.seed); random.seed(a.seed)
    raw = load_frames(a.train_split, a.feature, maps=a.crop_maps)
    if a.quality:   # frames without a quality label file are left out (train_kinect: 1414 of 1600 labelled)
        raw = [r for r in raw if os.path.exists(f"{DATA}/{a.train_split}/{r['name'][:-3]}.q.npy")]
    frames = [to_gpu(c, a.feature) for c in raw]
    if a.obj_shift_max > 0:  # object masks for the object-only depth shift (scratchpad/mask_cache.py -> points_mask_<split>.pt)
        masks = torch.load(f"{DATA}/cache/points_mask_{a.train_split}.pt", weights_only=False)
        for c, r, m in zip(frames, raw, masks):
            assert r["name"] == m["name"]
            c["mask"] = m["mask"].cuda().bool()
    names = a.levels.split(",")
    for c in frames:
        c["allowed"] = torch.isin(c["lvl"], torch.tensor([c["levels"].index(l) for l in names], device="cuda", dtype=torch.int8)).nonzero()[:, 0]
    tr = [c for c in frames if c["scene"] < a.holdout]; ho = [c for c in frames if c["scene"] >= a.holdout]
    mu = torch.tensor(PointVerifier.MU, device="cuda")
    if a.quality:  # benchmark friction quality per candidate (scripts/verifier_quality_labels.py: <sample>.q.npy)
        for c, r in zip(frames, raw):
            c["q"] = torch.from_numpy(np.load(f"{DATA}/{a.train_split}/{r['name'][:-3]}.q.npy")).cuda()
    if a.soft:  # robust success: the group mean (a decoded candidate and its jittered copies) of [collision-free and force closure at mu]
        assert a.quality is True and not a.anno_mix
        for c, r in zip(frames, raw):
            d = torch.load(f"{DATA}/{a.train_split}/{r['name']}", weights_only=False, map_location="cpu")
            bi = d["base_idx"].cuda(); al = c["allowed"]
            succ = ((c["q"][:, None] > 0) & (c["q"][:, None] <= mu[None])).float()   # (n, 6), 0 for colliding / unlabelled
            cnt = torch.zeros(len(c["g"]), device="cuda").index_add_(0, bi[al], torch.ones(len(al), device="cuda"))
            acc = torch.zeros(len(c["g"]), 6, device="cuda").index_add_(0, bi[al], succ[al])
            c["qsoft"] = (acc / cnt.clamp(min=1)[:, None])[bi]                     # every candidate takes its group's mean
    if a.cgrid:  # contact-grid distance labels (scripts/verifier_occ_labels.py --grid contact: <sample>.cg.npy, mm, decoded+mid)
        assert a.levels == "decoded,mid" and not a.anno_mix
        for c, r in zip(frames, raw):
            cg = torch.from_numpy(np.load(f"{DATA}/{a.train_split}/{r['name'][:-3]}" + (".cgf.npy" if a.cg_fine else ".cg.npy")))
            assert len(cg) == len(c["allowed"]), (r["name"], cg.shape, len(c["allowed"]))
            row = torch.full((len(c["g"]),), -1, dtype=torch.long, device="cuda"); row[c["allowed"]] = torch.arange(len(cg), device="cuda")
            c["cg"] = cg.cuda(); c["cg_row"] = row
    if a.occ:   # per-point surface distance of the training levels (scripts/verifier_occ_labels.py: <sample>.occ.npy, mm)
        assert a.levels == "decoded,mid" and not a.anno_mix, "occupancy labels cover the decoded and mid candidates"
        for c, r in zip(frames, raw):
            occ = torch.from_numpy(np.load(f"{DATA}/{a.train_split}/{r['name'][:-3]}.occ.npy"))
            assert len(occ) == len(c["allowed"]), (r["name"], occ.shape, len(c["allowed"]))
            row = torch.full((len(c["g"]),), -1, dtype=torch.long, device="cuda"); row[c["allowed"]] = torch.arange(len(occ), device="cuda")
            c["occ"] = occ.cuda(); c["occ_row"] = row
    if a.refine:   # pose residual toward the labelled grasp of the candidate's pixel (scripts/verifier_pose_targets.py)
        tg = {t["name"]: t for t in torch.load(f"{DATA}/cache/pose_targets_{a.train_split}.pt", weights_only=False)}
        for c, r in zip(frames, raw):
            t = tg[r["name"]]
            c["tgt"] = torch.from_numpy(np.concatenate([t["drot"], 10 * t["dw"][:, None], 10 * t["dd"][:, None]], 1)).cuda()
            c["tgt_has"] = torch.from_numpy(t["has"]).cuda()
    if a.anno_mix:  # annotated grasps of the same frame (scripts/verifier_anno_data.py) appended to the decoded candidates
        for c, r in zip(frames, raw):
            an = torch.load(f"{DATA}/train_anno/{r['name'][:-3]}.pt", weights_only=False)
            keep = c["allowed"]                                   # decoded + selected jitter levels
            c["g"] = torch.cat([c["g"][keep], an["grasps"].cuda()]); c["y"] = torch.cat([c["y"][keep], an["free"].cuda().float()])
            if a.quality:
                c["q"] = torch.cat([c["q"][keep], an["q"].cuda()])
            c["allowed"] = torch.arange(len(c["g"]), device="cuda")
    c_map = 0 if a.feature == "none" else [m.shape[0] for m in frames[0]["fmap"]]
    model = PointVerifier(c_map, use_ray=not a.no_ray, use_pos=not a.no_pos, pool=not a.no_pool,
                          counts=tuple(int(v) for v in a.counts.split(",")), quality=a.quality or False, use_normal=a.use_normal, refine=a.refine, ctx=a.ctx, crop=a.crop, field=a.field, crop_maps=a.crop_maps, crop_k=a.crop_k, crop_nodepth=a.crop_nodepth, d_tok=a.d_tok, cgrid=a.cgrid, cg_hidden=a.cg_hidden,
                          readout_tok=a.readout_tok, cg_fine=a.cg_fine, objctx=a.objctx, cg_conv=a.cg_conv, cg_mirror=a.cg_mirror, readout_obs=a.readout_obs).cuda()
    DROP.update(tok=a.drop_tok, pt=a.drop_pt)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    n_train = sum(len(c["allowed"]) for c in tr)
    print(f"{len(tr)} train frames, {n_train} grasps, {len(ho)} holdout frames, feature {a.feature} ({c_map} channels), "
          f"{sum(p.numel() for p in model.parameters())} params", flush=True)
    t0 = time.time()
    for step in range(a.steps):
        Ls, ys, qls = [], [], []
        for c in random.sample(tr, a.frames):
            idx = c["allowed"][torch.randint(len(c["allowed"]), (a.per_frame,), device="cuda")]
            if a.depth_shift_max > 0 and random.random() < a.depth_shift_p:  # sensor-offset augmentation: the measured
                shift = random.uniform(0.0, a.depth_shift_max) / 100.0        # surface moves farther by 0..max cm, labels unchanged
                c = dict(c, depth=c["depth"] + shift * (c["depth"] > 0.1))
            elif a.obj_shift_max > 0 and random.random() < a.depth_shift_p:  # object-only variant: only object pixels move farther
                shift = random.uniform(0.0, a.obj_shift_max) / 100.0
                c = dict(c, depth=c["depth"] + shift * (c["mask"] & (c["depth"] > 0.1)))
            if a.quality:
                L, q, lp = forward_frame(model, c, idx, with_quality=True, with_points=True)
                if a.cgrid:
                    kx, ky, kz, _, _ = CG
                    Dt = c["cg"][c["cg_row"][idx]].float().reshape(len(idx), kx, ky, kz)
                    wgt = 1 + a.cg_near * (Dt <= 6).float()                       # emphasise the near-surface cells the readout depends on
                    qls.append(a.cg_weight * (wgt * (model._last_field - Dt.clamp(0, model.readout.d_max)).abs()).mean() / model.readout.d_max / (1 + a.cg_near * 0.2))
                    qt0 = c["q"][idx]; ytrue = ((qt0[:, None] > 0) & (qt0[:, None] <= mu[None])).float()
                    qls.append(F.binary_cross_entropy_with_logits(model.readout(Dt, c["g"][idx, 1].clamp(0, c["max_w"]), model._last_ctx, model._last_obs), ytrue))   # readout on the true field
                if a.occ:
                    tgt = (c["occ"][c["occ_row"][idx]] < a.occ_mm).float()
                    qls.append(a.occ_weight * F.binary_cross_entropy_with_logits(lp, tgt))
                qt = c["q"][idx]; free = qt > 0                                   # collision-free candidates carry a quality label
                if a.quality == "scalar":  # target = expected success rate: 0 for colliding, share of thresholds passed otherwise
                    known = qt > -2
                    tgt = torch.where(free, (qt[:, None] <= mu[None]).float().mean(1), torch.zeros_like(qt))
                    qls.append(F.binary_cross_entropy_with_logits(q[known, 0], tgt[known]))
                elif a.quality == "rank":  # listwise (ListNet) loss within the frame: relevance = expected success rate
                    known = qt > -2
                    rel = torch.where(free, (qt[:, None] <= mu[None]).float().mean(1), torch.zeros_like(qt))[known]
                    r = torch.log(c["g"][idx, 0].clamp(min=1e-6))[known] + q[known, 0]
                    qls.append(-(F.softmax(rel / a.rank_temp, 0) * F.log_softmax(r, 0)).sum())
                elif a.cgrid:                                                     # predicted field -> readout, exact success (collision = 0)
                    qls.append(F.binary_cross_entropy_with_logits(q, ((qt[:, None] > 0) & (qt[:, None] <= mu[None])).float()))
                elif a.soft:                                                      # robust success target on every candidate
                    qls.append(F.binary_cross_entropy_with_logits(q, c["qsoft"][idx]))
                elif free.any():                                                  # success at each friction threshold (q = 9: none)
                    qls.append(F.binary_cross_entropy_with_logits(q[free], (qt[free][:, None] <= mu[None]).float()))
            elif a.refine:
                L, q = forward_frame(model, c, idx, with_quality=True)
                has = c["tgt_has"][idx]
                if has.any():
                    qls.append(F.smooth_l1_loss(q[has], c["tgt"][idx][has], beta=0.05))
            else:
                L = forward_frame(model, c, idx)
            Ls.append(L); ys.append(c["y"][idx])
        loss = F.binary_cross_entropy_with_logits(torch.cat(Ls), 1 - torch.cat(ys))
        if qls:
            loss = loss + a.quality_weight * torch.stack(qls).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    ho_auroc = auroc(predict_frames(model, ho)) if ho else float("nan")
    os.makedirs(MODELS, exist_ok=True)
    torch.save(dict(state=model.state_dict(), feature=a.feature, cfg=model.cfg), f"{MODELS}/{a.tag}.pt")
    print("RESULT", json.dumps(dict(tag=a.tag, feature=a.feature, **model.cfg, levels=names, n_train=n_train, steps=a.steps, train_split=a.train_split,
                                    depth_shift=[a.depth_shift_max, a.depth_shift_p], obj_shift=a.obj_shift_max,
                                    batch=a.frames * a.per_frame, params=sum(p.numel() for p in model.parameters()),
                                    holdout_auroc=ho_auroc, train_s=round(time.time() - t0))), flush=True)


def predict(a):
    ck = torch.load(f"{MODELS}/{a.tag}.pt", weights_only=False)
    state = {k.replace("proj.", "proj.0.") if k.startswith("proj.") and not k[5].isdigit() else k: v for k, v in ck["state"].items()}  # single-map checkpoints
    model = PointVerifier(**ck["cfg"]).cuda(); model.load_state_dict(state); model.eval()
    frames = [to_gpu(c, ck["feature"]) for c in load_frames(a.split, ck["feature"], maps=bool(model.cfg.get("crop_maps")))]
    res = predict_frames(model, frames)
    pred = torch.cat([torch.sigmoid(-L) for L, _ in res]).cpu()
    saved = dict(pred=pred)
    if model.cfg.get("quality"):
        with torch.no_grad():
            qs = []
            for c in frames:
                idx = torch.arange(len(c["g"]), device="cuda")
                qs.append(torch.cat([torch.sigmoid(forward_frame(model, c, idx[i:i + 2048], with_quality=True)[1]).mean(1) for i in range(0, len(idx), 2048)]))
        if model.cfg.get("quality") == "rank":
            qs = []
            with torch.no_grad():
                for c in frames:
                    idx = torch.arange(len(c["g"]), device="cuda")
                    qs.append(torch.cat([torch.exp(forward_frame(model, c, idx[i:i + 2048], with_quality=True)[1][:, 0].clamp(-10, 10)) for i in range(0, len(idx), 2048)]))
        saved["qpred"] = torch.cat(qs).cpu()   # expected success (quality heads) or exp(delta) (rank head)
    if model.cfg.get("refine"):
        with torch.no_grad():
            ds = []
            for c in frames:
                idx = torch.arange(len(c["g"]), device="cuda")
                ds.append(torch.cat([forward_frame(model, c, idx[i:i + 2048], with_quality=True)[1] for i in range(0, len(idx), 2048)]))
        saved["delta"] = torch.cat(ds).cpu()   # (n, 5) pose residual
    torch.save(saved, f"{MODELS}/{a.tag}_{a.split}_pred.pt")
    out = dict(tag=a.tag, split=a.split, n=len(pred), auroc=auroc(res))
    if a.split == "val9":
        for name, (s0, s1) in {"seen": (100, 130), "similar": (130, 160), "novel": (160, 190)}.items():
            out[f"auroc_{name}"] = auroc([r for r, c in zip(res, frames) if s0 <= c["scene"] < s1])
    if a.bench:  # verifier time for 2048 candidates of one frame, projection of the map included
        c = frames[0]; idx = torch.arange(min(2048, len(c["g"])), device="cuda")
        with torch.no_grad():
            for _ in range(5):
                forward_frame(model, c, idx)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(20):
                forward_frame(model, c, idx)
            torch.cuda.synchronize()
        out["ms_per_frame"] = round((time.time() - t0) / 20 * 1000, 2); out["bench_n"] = len(idx)
    print("RESULT", json.dumps(out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cache", "cache_extra", "train", "predict"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--proc", type=int, default=32)
    ap.add_argument("--tag", default="P_vfm")
    ap.add_argument("--feature", default="vfm", help="sampled map(s) among none, vfm (frozen tokens), feat (grasp head), dpt (DPT decoder), geo (geometry encoder); '+' joins several")
    ap.add_argument("--levels", default="decoded,mid")
    ap.add_argument("--holdout", type=int, default=90, help="train scenes < holdout, holdout scenes >= holdout")
    ap.add_argument("--steps", type=int, default=2880)
    ap.add_argument("--frames", type=int, default=16, help="frames per step")
    ap.add_argument("--per_frame", type=int, default=512, help="grasps per frame per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_split", default="train", help="cache split used for training (train = RealSense scenes)")
    ap.add_argument("--depth_shift_max", type=float, default=0.0, help="augmentation: shift the measured depth farther by U(0, max) cm")
    ap.add_argument("--depth_shift_p", type=float, default=0.5, help="share of frames per step that receive the depth shift")
    ap.add_argument("--obj_shift_max", type=float, default=0.0, help="augmentation: shift only object pixels farther by U(0, max) cm")
    ap.add_argument("--quality", nargs="?", const=True, default=False,
                    help="also predict the benchmark quality (needs .q.npy labels): flag alone = force closure per friction threshold; 'scalar' = one expected-success output")
    ap.add_argument("--quality_weight", type=float, default=1.0)
    ap.add_argument("--rank_temp", type=float, default=0.1, help="ListNet temperature on the relevance (rank variant)")
    ap.add_argument("--anno_mix", action="store_true", help="add the annotated-grasp candidates of each frame (train_anno) to the training set")
    ap.add_argument("--use_normal", action="store_true", help="add the measured surface normal at each point (gripper frame) to the inputs")
    ap.add_argument("--refine", action="store_true", help="add the pose-refinement head (targets: cache/pose_targets_<train_split>.pt)")
    ap.add_argument("--ctx", action="store_true", help="add context lattice points beyond the gripper boxes (pooled context only)")
    ap.add_argument("--crop", action="store_true", help="add the gripper-aligned token-crop branch to the quality head")
    ap.add_argument("--occ", action="store_true", help="dense per-point occupancy supervision (needs <sample>.occ.npy; quality mode)")
    ap.add_argument("--occ_weight", type=float, default=1.0)
    ap.add_argument("--occ_mm", type=float, default=4.0, help="a lattice point counts as occupied below this surface distance (mm)")
    ap.add_argument("--field", action="store_true", help="the quality head reads the per-point occupancy field (fixed lattice order)")
    ap.add_argument("--crop_maps", action="store_true", help="add relative depth, validity, objectness, graspness and score channels to the crop")
    ap.add_argument("--crop_k", type=int, default=CROP_K, help="crop resolution (k x k)")
    ap.add_argument("--crop_nodepth", action="store_true", help="crop maps without the relative-depth and validity channels")
    ap.add_argument("--d_tok", type=int, default=32, help="width of the learned token projection")
    ap.add_argument("--drop_tok", type=float, default=0.0, help="training-time dropout of token channels per candidate")
    ap.add_argument("--drop_pt", type=float, default=0.0, help="training-time dropout of lattice points per candidate")
    ap.add_argument("--cgrid", action="store_true", help="contact-grid distance regression + geometry-only readout (needs <sample>.cg.npy)")
    ap.add_argument("--cg_weight", type=float, default=5.0)
    ap.add_argument("--cg_hidden", type=int, default=64, help="width of the contact-grid field network")
    ap.add_argument("--cg_near", type=float, default=0.0, help="extra L1 weight on cells within 6 mm of the surface")
    ap.add_argument("--readout_tok", action="store_true", help="ablation: the contact readout also receives the appearance context")
    ap.add_argument("--cg_fine", action="store_true", help="use the fine-sampled float16 contact labels (.cgf.npy)")
    ap.add_argument("--cg_conv", action="store_true", help="convolutional field decoder over the contact grid (FieldConv) instead of the per-cell PointNet")
    ap.add_argument("--cg_mirror", action="store_true", help="with --cg_conv: concatenate the mirror cell's inputs (y -> -y) to every cell (bilateral symmetry prior)")
    ap.add_argument("--readout_obs", action="store_true", help="the readout also receives per-cell observed flags (valid pixel, cell at or in front of the measured surface)")
    ap.add_argument("--objctx", action="store_true", help="feed the crop-CNN summary (object-level context) into the contact-field network")
    ap.add_argument("--soft", action="store_true", help="quality target = robust success (mean over the candidate's jitter group, collision = 0)")
    ap.add_argument("--no_ray", action="store_true", help="drop z, s, valid (appearance only)")
    ap.add_argument("--no_pos", action="store_true", help="drop the gripper-frame position and region inputs")
    ap.add_argument("--no_pool", action="store_true", help="no gripper-wide max-pool: a pure per-point function")
    ap.add_argument("--counts", default=f"{N_FX},{N_BY},{N_CX},{N_CY}", help="lattice points: along finger, across back plate, along and across the closing region")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    assert a.feature == "none" or set(a.feature.split("+")) <= {"vfm", "feat", "dpt", "geo", "dav2"}, a.feature
    {"cache": build_cache, "cache_extra": build_cache, "train": train, "predict": predict}[a.mode](a)


# ----------------------------------------------------------------------------- inference (test.py --verifier TAG)
def intrinsics_from_xyz(xyz):
    """verifier_data.fit_intrinsics in closed form: the same per-axis least-squares line u = fx X/Z + cx (v likewise) over valid
    pixels, from masked sums in float64 -- no boolean indexing (host sync) and no QR."""
    _, H, W = xyz.shape
    Z = xyz[2].double(); w = (Z > 0.1).double(); Zs = torch.where(w > 0, Z, torch.ones_like(Z))
    vv, uu = torch.meshgrid(torch.arange(H, device=xyz.device, dtype=torch.float64), torch.arange(W, device=xyz.device, dtype=torch.float64), indexing="ij")
    fc = []
    for x, t in ((xyz[0].double() / Zs, uu), (xyz[1].double() / Zs, vv)):
        n = w.sum(); mx = (w * x).sum() / n; mt = (w * t).sum() / n
        f = (w * (x - mx) * (t - mt)).sum() / (w * (x - mx) ** 2).sum()
        fc.append((f, mt - f * mx))
    return torch.stack([fc[0][0], fc[1][0], fc[0][1], fc[1][1]]).float()   # fx, fy, cx, cy


def _mex_layers(n, rows, cols):
    """Greedy NMS layers in one pass over score order: layer_i = the smallest layer used by none of i's higher-scored neighbours
    (rows[k] > cols[k], rows ascending). Identical to peeling greedy NMS again and again on what is left."""
    layer = np.zeros(n, np.int64); seen = np.zeros(n + 1, np.bool_); p = 0
    for i in range(n):
        q = p
        while q < len(rows) and rows[q] == i:
            seen[layer[cols[q]]] = True; q += 1
        k = 0
        while seen[k]:
            k += 1
        layer[i] = k
        for r in range(p, q):
            seen[layer[cols[r]]] = False
        p = q
    return layer


try:
    import numba
    _mex_layers = numba.njit(cache=True)(_mex_layers)
except ImportError:
    pass


PRETEST = os.environ.get("VERIFIER_PRETEST", "own")   # "lattice": a candidate survives the pretest if ANY opening the width choice
                                                      # can use (its own width, every 5 mm opening up to max_w; own depth) is uncrossed


def _cross_chunk(gi, K, depth, Ps, ix, iy, fz, G, ax, mu, tau, max_w, box):
    """box = (x0, nx, y0, ny, nz): the projected cells are the (x, y, z) box x0..x0+nx-1, y0..y0+ny-1, all nz heights, x-major, so
    the crossed cells fold to an (x, y) map by a reshape. Body memberships are 0/1 per axis, so a sum of products is > 0 exactly
    when some body cell is crossed."""
    x0, nx, y0, ny, nz = box; n = len(gi); d = gi[:, 3]
    _, _, s, valid, _ = G.project_cells(gi, K, depth, mu, ax, Ps)
    hit = (s.abs() * mu <= tau) & (valid > 0)                                          # body-reachable cells crossed by the surface
    H = (hit.reshape(n, nx, ny, nz) & (fz.reshape(nx, ny, nz) > 0)).any(3).float()      # crossed (x, y) columns within the finger height
    w0 = gi[:, 1].clamp(0, max_w)
    mx_f, mx_b, my_f, my_b, _ = G.body_factors(w0, d, ax)
    A = torch.einsum("nxy,nx->ny", H, mx_f[:, x0:x0 + nx]); B = torch.einsum("nxy,nx->ny", H, mx_b[:, x0:x0 + nx])
    out = ((A * my_f[:, y0:y0 + ny]).sum(1) + (B * my_b[:, y0:y0 + ny]).sum(1)) > 0    # crossed at the own width
    if PRETEST == "lattice":   # and at every 5 mm opening: the x and z memberships do not depend on the width
        sy = G.UG["y"][2]
        Wl = torch.arange(1, int(round(max_w / sy)) + 1, device=gi.device, dtype=w0.dtype) * sy
        _, _, lf, lb, _ = G.body_factors(Wl, d[:1].expand(len(Wl)), ax)                   # (J, ky) finger / palm memberships
        lat = ((A @ lf[:, y0:y0 + ny].T + B @ lb[:, y0:y0 + ny].T) > 0).all(1)             # crossed at every lattice opening
        out = (out & lat) if G.OWN_W else lat                                              # survives if any opening is uncrossed
    return out


_SUB = {}


def surface_cross(g, K, depth, G, ax, P, mu, tau, max_w, chunk=2048, fn=None):
    """(n,) bool: a gripper-body cell of the candidate is crossed by the measured surface (|D - z| <= tau at its pixel, depth valid) --
    the observed-cloud collision check expressed on the verifier's own grid; a certain collision for any sensor that measures the
    surface at or behind the true one. No network involved. Only the cells any body can occupy (about 1/6 of the grid) are
    projected, and the body membership is evaluated on them from the separable per-axis overlaps of verifier_grid.gripper_masks.
    fn: a torch.compile'd _cross_chunk (G, ax bound); chunks are then padded to one shape."""
    key = (float(max_w), str(g.device), id(G), len(P))
    if key not in _SUB:   # the projected cells (a box of the grid, the same for every frame at this opening range)
        kx, ky, kz = (len(a) for a in ax)
        sub = ((P[:, 0] >= -0.035) & (P[:, 0] <= 0.055) & (P[:, 1].abs() <= max_w / 2 + 0.015) & (P[:, 2].abs() <= 0.015)).nonzero()[:, 0]
        ix, iy, iz = sub // (ky * kz), (sub // kz) % ky, sub % kz
        fz = G.member(ax[2], torch.full((1, 1), -G.HALF_H, device=g.device), torch.full((1, 1), G.HALF_H, device=g.device), G.UG["z"][2])[0][iz]
        x0, y0, z0 = int(ix.min()), int(iy.min()), int(iz.min()); nx, ny, nz = int(ix.max()) - x0 + 1, int(iy.max()) - y0 + 1, int(iz.max()) - z0 + 1
        assert nx * ny * nz == len(sub), "the projected cells must form a box"
        _SUB[key] = (P[sub], ix, iy, fz, (x0, nx, y0, ny, nz))
    Psub, ix, iy, fz, box = _SUB[key]
    out = []
    for i in range(0, len(g), chunk):
        gi = g[i:i + chunk]; m = len(gi)
        if fn is None:
            out.append(_cross_chunk(gi, K, depth, Psub, ix, iy, fz, G, ax, mu, tau, max_w, box))
            continue
        if m < chunk:
            gi = torch.cat([gi, gi[:1].expand(chunk - m, -1)])
        out.append(fn(gi, K, depth, Psub, ix, iy, fz, mu, tau, max_w, box)[:m])
    return torch.cat(out)


def budget_select(g, n_keep, t_thr=0.01, deg=10.0):
    """Indices of the n_keep candidates to verify, in (greedy NMS layer, detector score) order, NMS at t_thr and deg.
    g (n, 17) GraspGroup rows on the device; the neighbour pass runs there, the layering on the CPU."""
    order = torch.argsort(-g[:, 0], stable=True); h = g[order]; R = h[:, 4:13]
    near = (torch.cdist(h[:, 13:16], h[:, 13:16]) < t_thr) & (R @ R.T > 1 + 2 * math.cos(math.radians(deg)))
    ij = torch.nonzero(torch.tril(near, -1)).cpu().numpy()                      # row-major: rows ascending
    layer = _mex_layers(len(h), ij[:, 0].copy(), ij[:, 1].copy())
    keep = np.lexsort((np.arange(len(h)), layer))[:n_keep]
    return order[torch.from_numpy(keep).to(g.device)]


class Verifier:
    """Scores the decoded grasps of one frame with a trained point verifier: p_free in [0, 1] per grasp."""

    def __init__(self, tag, device="cuda", patch=14, tof=False):
        set_patch(patch)                                                               # the detector's backbone patch size
        self.flag = float(tof)                                                         # depth sensor type, read by --sensor_flag models
        ck = torch.load(f"{MODELS}/{tag}.pt", weights_only=False, map_location="cpu")
        self.features = [] if ck["feature"] == "none" else ck["feature"].split("+")   # maps to capture from the detector
        self.grid = "design" in ck["cfg"]                                              # scripts/verifier_grid.py checkpoint
        if self.grid:
            import verifier_grid as G
            self.G = G; self.model = G.GridVerifier(**ck["cfg"]).to(device); self.model.load_state_dict(ck["state"]); self.model.eval()
            self.ax = G.cell_axes(device); self.P = G.cell_centres(device)
            if os.environ.get("VERIFIER_SAMPLER", "triton") == "triton" and len(self.features) in (1, 2):   # verifier_sample: channel-last, one pass
                self.model.sampler = "triton"
            self.model.surf_tau = float(os.environ.get("VERIFIER_SURF", "0"))   # metres; > 0 = grid surface test inside the verdict (no separate cloud check)
            if os.environ.get("VERIFIER_SELECT", "0") == "1":   # in-frame choice: depth +-5 mm (one cell) by the verdict, the opening by collision
                self.model.select = ([float(x) for x in os.environ.get("VERIFIER_SELECT_DEPTHS", "-0.005,0,0.005").split(",")], None)   # depth offsets (m); None: every opening of the gripper on the 5 mm lattice
                self.model.select_alpha = float(os.environ.get("VERIFIER_PF_EXP", "1"))
                self.model.select_top = int(os.environ.get("VERIFIER_SELECT_TOP", "0"))  # depth choice only for the K best at their own depth (0: all)
            self.compiled = os.environ.get("VERIFIER_COMPILE", "0") == "1"   # torch.compile of the fast path; chunks padded to one fixed shape
            self.fast = torch.compile(self.model.forward_fast, dynamic=False, mode="max-autotune-no-cudagraphs") if self.compiled else self.model.forward_fast
            ax = self.ax
            self.cross = torch.compile(lambda gi, K, depth, Ps, ix, iy, fz, mu, tau, max_w, box: _cross_chunk(gi, K, depth, Ps, ix, iy, fz, G, ax, mu, tau, max_w, box),
                                       dynamic=False) if self.compiled else None                  # the surface pretest, fused
            return
        state = {k.replace("proj.", "proj.0.") if k.startswith("proj.") and not k[5].isdigit() else k: v for k, v in ck["state"].items()}
        self.model = PointVerifier(**ck["cfg"]).to(device); self.model.load_state_dict(state); self.model.eval()

    @torch.no_grad()
    def p_free(self, grasps, xyz, fmaps, max_w=0.1, chunk=2048):
        """grasps (n, 17) on the device, xyz (3, H, W) network-input point map, fmaps {name: (C, h, w)} from the same forward pass.
        With VERIFIER_SELECT the width and depth the verifier chose are written into grasps (columns 1 and 3) in place."""
        self.last_stats = None  # diagnostics only; no candidate/score changes
        depth = xyz[2] * (xyz[2] > 0.1)
        c = dict(g=grasps.float(), K=intrinsics_from_xyz(xyz), depth=depth, fmap=[fmaps[f] for f in self.features] or None, max_w=max_w)
        idx = torch.arange(len(grasps), device=grasps.device)
        if self.grid:   # grid verifier: p_free x expected force-closure score, both from the predicted TSDF
            import types
            G = self.G; a = types.SimpleNamespace(mu=self.model.cfg["mu"])
            cg = dict(g=c["g"], K=c["K"], depth=c["depth"], fmaps=[fmaps[f] for f in self.features], max_w=max_w, q=None,
                      t_row=torch.full((len(grasps),), -1, dtype=torch.long, device=grasps.device), y=torch.zeros(len(grasps), device=grasps.device))
            out = []; alpha = float(os.environ.get("VERIFIER_PF_EXP", "1"))              # p_free^alpha (alpha chosen on the training holdout)
            if self.model.cfg.get("encoder") in ("pillar", "pillar2", "vfe"):             # folded fast path (exact up to fp16), 1024 per chunk
                budget = int(os.environ.get("VERIFIER_BUDGET", "0"))
                tau = getattr(self.model, "surf_tau", 0.0)
                idx_keep = idx if tau <= 0 else surface_cross(c["g"], c["K"], c["depth"], G, self.ax, self.P, a.mu, tau, max_w, fn=self.cross).logical_not().nonzero()[:, 0]
                self.last_stats = dict(input=len(idx), surface_rejected=len(idx) - len(idx_keep),
                                       budget_skipped=0, evaluated=0, zero_after_network=0)
                if os.environ.get("VERIFIER_NONET", "0") == "1":   # ladder row: the measured-surface test alone (no network, no selection)
                    return torch.zeros(len(idx), device=grasps.device).index_put_((idx_keep,), torch.ones(len(idx_keep), device=grasps.device))
                if 0 < budget < len(idx_keep):                                             # verify the first `budget` survivors in NMS-layer order
                    sel = idx_keep[budget_select(c["g"][idx_keep], budget, float(os.environ.get("VERIFIER_BUDGET_T", "0.01")),
                                                 float(os.environ.get("VERIFIER_BUDGET_DEG", "10")))]
                else:
                    sel = None if tau <= 0 else idx_keep                                   # unselected candidates score 0
                gv = c["g"] if sel is None else c["g"][sel]
                self.last_stats.update(budget_skipped=len(idx_keep) - len(gv), evaluated=len(gv))
                if len(gv) == 0:                                                           # every candidate crossed by the measured surface
                    return torch.zeros(len(idx), device=grasps.device)
                with torch.autocast("cuda", dtype=torch.float16):
                    ck = int(os.environ.get("VERIFIER_CHUNK", "1024"))
                    folded = self.model.fold(cg["fmaps"])                                  # per-frame part, once
                    if getattr(self.model, "sampler", "") == "triton":                     # channel-last maps for verifier_sample
                        folded = (folded[0], folded[1], [m.permute(1, 2, 0).contiguous() for m in folded[2]])
                    wd = []
                    for i in range(0, len(gv), ck):
                        gi = gv[i:i + ck]; m = len(gi)
                        if self.compiled and m < ck:                                       # one shape for the compiled graph
                            gi = torch.cat([gi, gi[:1].expand(ck - m, -1)])
                        reject, q, _, wdi = self.fast(gi, c["K"], c["depth"], cg["fmaps"], a.mu, self.ax, self.P, max_w, flag=self.flag, folded=folded)
                        out.append((torch.sigmoid(-reject.float()) ** alpha * G.fc_score(q.float(), self.model.cfg["fc_target"]))[:m]); wd.append(wdi[:m])
                if getattr(self.model, "select", None) is not None:                        # the chosen configuration, where it moved
                    rows = idx if sel is None else sel; wd = torch.cat(wd).to(grasps.dtype)
                    moved = ((wd[:, 0] - gv[:, 1].clamp(0, max_w)).abs() > 1e-9) | ((wd[:, 1] - gv[:, 3]).abs() > 1e-9)
                    grasps[rows[moved], 1] = wd[moved, 0]; grasps[rows[moved], 3] = wd[moved, 1]
                result = torch.cat(out)
                self.last_stats["zero_after_network"] = int((result <= 0).sum().item())
                if sel is None:
                    return result
                return torch.zeros(len(idx), device=grasps.device).index_put_((sel,), result)
            if self.model.cfg.get("feat_cells"):
                chunk = min(chunk, 512)                                   # per-cell features: 15x wider inputs, keep the chunk small
            for i in range(0, len(idx), chunk):
                b = G.batch(self.model, [cg], a, self.ax, self.P, idxs=[idx[i:i + chunk]])
                reject, q, _ = self.model(b["img"], b["body"], b["close"], b["open"], b["d"])
                out.append(torch.sigmoid(-reject) ** alpha * G.fc_score(q, self.model.cfg["fc_target"]))
            return torch.cat(out)
        if self.model.cfg.get("quality"):  # collision-free probability x expected force-closure success over the six thresholds
            out = [forward_frame(self.model, c, idx[i:i + chunk], with_quality=True) for i in range(0, len(idx), chunk)]
            return torch.sigmoid(-torch.cat([L for L, _ in out])) * torch.sigmoid(torch.cat([q for _, q in out])).mean(1)
        return torch.sigmoid(-torch.cat([forward_frame(self.model, c, idx[i:i + chunk]) for i in range(0, len(idx), chunk)]))
