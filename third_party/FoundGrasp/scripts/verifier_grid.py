"""Gripper-frame grid verifier: one uniform grid fixed in the gripper frame, read as a 2-D image whose channels stack
the cells along the approach axis (design agreed 2026-09-23; docs/HANDOVER.md).

Grid (must match scripts/verifier_occ_labels.py UG): x (approach) [-6, 8] cm, y (closing) [-9, 9] cm, z (thickness)
[-4, 4] cm, 5 mm cells -> (kx, ky, kz) = (28, 36, 16). Image = (y, z) 36 x 16; channels = the 28 cells along x.

Input image of a candidate (R, t, w, d):
  kx ch  projective TSDF clamp(D(u, v) - z_cam, -mu, mu) / mu of each cell (D = measured depth at the cell's pixel,
         positive in front of the observed surface); 0 where the depth is invalid
  kx ch  depth valid
  64 ch  per feature map: one detector feature per pixel (y, z), bilinearly sampled at the pixel of the column's cell
         nearest to the observed surface (--feature none | vfm | dpt | geo | feat; '+' joins maps)
Backbone: 1x1 stem -> 4 residual 3x3 conv blocks (dilation 1, 2, 4, 8) -> h (128 ch, 36 x 16).
Heads:
  TSDF           1x1 conv -> kx ch = T_hat, the scene TSDF / mu on every cell; L1 to the truncated CAD TSDF
  collision      per-cell occupancy logits (two 1x1 convs, output bias at the prior -6), pooled over the evaluator's boxes for (w, d) with soft cell
                 membership: collision = LSE over finger and back-plate cells, empty = -LSE over closing cells + b,
                 reject = LSE(collision, empty); BCE to graspnetAPI collision_detection (<sample>.lbl.npy)
  force closure  2-D conv over (y, z) with the opening mask (|y| <= w/2) and d, max + mean pool -> 6 logits for
                 mu_j = 0.2 ... 1.2, symmetrised over the y and z flips; BCE to 0 < q <= mu_j on collision-free
                 candidates (<sample>.q.npy)
  --design A  both heads read h (parallel multi-task heads)
  --design B  both heads read only clamp(T_hat) (collision: all cells; force closure: x = d - 5 mm, d, d + 5 mm),
              and are also trained on the true TSDF of the labelled candidates
Score = s x p_free x mean_j sigmoid(q_j).

  python scripts/verifier_grid.py train --tag G_A_dpt --design A --feature dpt
  python scripts/verifier_grid.py predict --tag G_A_dpt --split val9
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verifier_points as VP  # noqa: E402
from verifier_points import load_frames, DATA, MODELS, MAX_W  # noqa: E402

GRIDS = {"g5": dict(x=(-0.06, 0.08, 0.005), y=(-0.09, 0.09, 0.005), z=(-0.04, 0.04, 0.005)),      # (28, 36, 16), the first grid
         "y25": dict(x=(-0.05, 0.07, 0.005), y=(-0.08, 0.08, 0.0025), z=(-0.03, 0.03, 0.005)),     # (24, 64, 12): 2.5 mm along the closing axis, 2 cm margin
         "f25": dict(x=(-0.06, 0.05, 0.005), y=(-0.09, 0.09, 0.0025), z=(-0.02, 0.02, 0.005)),     # (22, 72, 8): the volume trimmed to what a gripper box can reach
         "t5": dict(x=(-0.06, 0.05, 0.005), y=(-0.09, 0.09, 0.005), z=(-0.02, 0.02, 0.005)),       # (22, 36, 8): f25's trimmed volume at 5 mm; its cells are a subset of g5's
         "z6": dict(x=(-0.06, 0.08, 0.005), y=(-0.09, 0.09, 0.005), z=(-0.06, 0.06, 0.005)),      # (28, 36, 24): z extended to +-6 cm
         "g5f": dict(x=(-0.06, 0.08, 0.005), y=(-0.09, 0.09, 0.005), z=(-0.04, 0.04, 0.005)),     # g5 geometry, fine (1.5 mm, float16) labels
         "z5": dict(x=(-0.06, 0.08, 0.005), y=(-0.09, 0.09, 0.005), z=(-0.02, 0.02, 0.005)),      # (28, 36, 8): only the z shell no finger touches removed
         "g5w": dict(x=(-0.06, 0.10, 0.005), y=(-0.12, 0.12, 0.005), z=(-0.04, 0.04, 0.005))}     # (32, 48, 16): +3 cm context on each side along the closing axis, +2 cm ahead (large objects)
         # f25 rationale (measured 2026-09-25): only 1456 of g5's 16128 cells can lie inside any gripper box for any
         # (width, depth), and the predicted field's near-surface error is 4.33 mm against a 5 mm cell.  So g5 spends
         # its cells on volume no verdict reads while under-resolving the closing axis, where contact is decided.
         # f25 drops the 3 cm beyond the deepest fingertip and the z shell no finger touches, and halves the closing
         # cell: 12672 cells, 21% fewer than g5, at twice the resolution on the axis that decides the grasp.
UG = dict(GRIDS["g5"])


def set_grid(name):
    """Select the grid of the current process (labels, inputs and masks all follow it)."""
    UG.clear(); UG.update(GRIDS[name]); UG["name"] = name


def tsdf_path(split, name, level="decoded"):
    g = UG.get("name", "g5"); base = f"{DATA}/{split}/{name}.tsdf" + ("" if g == "g5" else f"_{g}")
    return base + (".npy" if level == "decoded" else f"_{level}.npy")
MU = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
DEPTH_BASE, FW, HALF_H = 0.02, 0.01, 0.01          # graspnetAPI gripper: base depth, finger width, half height
# Inference-only switches of the measured-surface test; all unset = the final recipe, bit-identical.
# VERIFIER_OVL_SNAP=1: overlap() zeroes the float32 residue (< 1e-3 of a cell) left on cells whose edge lies on a box face.
# VERIFIER_BODY=center: a cell counts as gripper body in the surface test only if its centre lies inside the box (default
# "overlap": any overlap counts, which widens a finger by up to one cell).
OVL_SNAP = os.environ.get("VERIFIER_OVL_SNAP", "0") == "1"
BODY_RULE = os.environ.get("VERIFIER_BODY", "overlap")
OWN_W = os.environ.get("VERIFIER_OWN_W", "1") == "1"      # 0: width options are the lattice openings only (ties: nearest the decoded width)
EXACT = os.environ.get("VERIFIER_EXACT", "0") == "1"      # 1: a configuration whose body the measured surface crosses is excluded (+inf)


def excluded(r, crossed):
    """Reject logit with the measured-surface test applied: +inf (EXACT) or the legacy +30 shift."""
    return torch.where(crossed, torch.full_like(r, float("inf")), r) if EXACT else torch.where(crossed, r + 30.0, r)
SPLITS = {"seen": (100, 130), "similar": (130, 160), "novel": (160, 190)}


def cell_axes(device):
    return [lo + (torch.arange(round((hi - lo) / sp), device=device, dtype=torch.float32) + 0.5) * sp for lo, hi, sp in (UG["x"], UG["y"], UG["z"])]


def cell_centres(device):
    X, Y, Z = torch.meshgrid(*cell_axes(device), indexing="ij")          # same order as the labels: x-major (x, y, z)
    return torch.stack([X, Y, Z], -1).reshape(-1, 3)


def overlap(c, a, b, sp):
    """Fraction of each cell [c - sp/2, c + sp/2] (c: (K,)) inside [a, b] (a, b: (n, 1)) -> (n, K)."""
    h = sp / 2
    o = (torch.minimum(b, c + h) - torch.maximum(a, c - h)).clamp(min=0) / sp
    return torch.where(o < 1e-3, torch.zeros_like(o), o) if OVL_SNAP else o


def member(c, a, b, sp):
    """Hard membership (float 0/1, same shape as overlap) of each cell in [a, b] for the measured-surface test: "overlap" = the
    cell touches the interval (overlap > 0, the default); "center" = its centre lies inside (inclusive, 1 um tolerance)."""
    if BODY_RULE == "center":
        return ((c >= a - 1e-6) & (c <= b + 1e-6)).float()
    return (overlap(c, a, b, sp) > 0).float()


def body_factors(w, d, ax):
    """Per-axis hard memberships of the gripper body for the surface test: mx_f, mx_b (n, kx), my_f, my_b (n, ky), mz (kz,)."""
    xc, yc, zc = ax
    w, d = w[:, None], d[:, None]
    o = torch.ones_like(w)
    sx, sy, sz = UG["x"][2], UG["y"][2], UG["z"][2]
    mz = member(zc, -HALF_H * o[:1], HALF_H * o[:1], sz)[0]
    mx_f = member(xc, -DEPTH_BASE * o, d, sx)
    mx_b = member(xc, -(DEPTH_BASE + FW) * o, -DEPTH_BASE * o, sx)
    my_f = torch.maximum(member(yc, -(w / 2 + FW), -w / 2, sy), member(yc, w / 2, w / 2 + FW, sy))
    my_b = member(yc, -(w / 2 + FW), w / 2 + FW, sy)
    return mx_f, mx_b, my_f, my_b, mz


def body_member(w, d, ax):
    """(n, kx, ky, kz) bool: the cells the surface test counts as gripper body at width w and depth d."""
    mx_f, mx_b, my_f, my_b, mz = body_factors(w, d, ax)
    return ((mx_f[:, :, None] * my_f[:, None] + mx_b[:, :, None] * my_b[:, None]) > 0)[..., None] & (mz > 0)


def gripper_factors(w, d, ax):
    """Per-axis overlaps of the evaluator's boxes, whose products are the masks of gripper_masks: fx_f, fx_b (n, kx) along the
    approach (fingers, back plate), fy_f, fy_b, fy_in (n, ky) along the closing axis (fingers, back plate, opening), fz (kz,)."""
    xc, yc, zc = ax
    w, d = w[:, None], d[:, None]
    o = torch.ones_like(w)
    sx, sy, sz = UG["x"][2], UG["y"][2], UG["z"][2]
    fz = overlap(zc, -HALF_H * o[:1], HALF_H * o[:1], sz)[0]
    fx_f = overlap(xc, -DEPTH_BASE * o, d, sx)
    fx_b = overlap(xc, -(DEPTH_BASE + FW) * o, -DEPTH_BASE * o, sx)
    fy_f = overlap(yc, -(w / 2 + FW), -w / 2, sy) + overlap(yc, w / 2, w / 2 + FW, sy)
    fy_b = overlap(yc, -(w / 2 + FW), w / 2 + FW, sy)
    fy_in = overlap(yc, -w / 2, w / 2, sy)
    return fx_f, fx_b, fy_f, fy_b, fy_in, fz


def gripper_masks(w, d, ax):
    """Soft membership (n, kx, ky, kz) of each cell in the evaluator's boxes (graspnetAPI collision_detection):
    body = two fingers (x in [-base, d], w/2 <= |y| <= w/2 + finger width) + back plate (x in [-base - finger width, -base],
    |y| <= w/2 + finger width); closing region = x in [-base, d], |y| <= w/2; all boxes |z| <= height / 2.
    Also returns the opening mask along y (n, ky)."""
    fx_f, fx_b, fy_f, fy_b, fy_in, fz = gripper_factors(w, d, ax)
    body = (fx_f[:, :, None] * fy_f[:, None] + fx_b[:, :, None] * fy_b[:, None])[..., None] * fz
    close = (fx_f[:, :, None] * fy_in[:, None])[..., None] * fz
    return body, close, fy_in


def kinect_noise(depth, mask=None, sigma_a=0.004, edge_thr=0.02, edge_p=0.5, hole_p=0.005, bias_max=0.008):
    """Parametric ToF-like sensor noise on a metric depth map: Gaussian noise with sigma = sigma_a * z^2 (2 mm at 0.7 m),
    dropout of a 1-px band at depth discontinuities (flying pixels), sparse holes, and an object-surface bias U(0, bias_max)
    farther from the camera (ToF over-estimation on object surfaces). Applied per frame; CAD targets are unchanged."""
    d = depth.float(); valid = d > 0.1
    out = d + torch.randn_like(d) * sigma_a * d ** 2
    gx = torch.zeros_like(d); gy = torch.zeros_like(d)
    gx[:, 1:] = (d[:, 1:] - d[:, :-1]).abs(); gy[1:] = (d[1:] - d[:-1]).abs()
    edge = ((gx > edge_thr) | (gy > edge_thr)) & valid
    edge = F.max_pool2d(edge.float()[None, None], 3, stride=1, padding=1)[0, 0] > 0
    drop = (edge & (torch.rand_like(d) < edge_p)) | (torch.rand_like(d) < hole_p)
    if mask is not None:
        out = out + random.uniform(0.0, bias_max) * mask.float()
    out = torch.where(valid & ~drop, out, torch.zeros_like(d))
    return out.to(depth.dtype)


OCC_SCALE = 0.10          # metres; the range over which the "how far behind the measured surface" channel is read


def project_cells(g, K, depth, mu, ax, P):
    """Cells of the candidates g (n, 17) -> pixel coordinates u, v (n, p), projective TSDF s / mu clamped (n, p), valid (n, p),
    occlusion depth (n, p): how far behind the measured surface the cell lies, over OCC_SCALE, which the clamped TSDF saturates away."""
    n = len(g)
    R = g[:, 4:13].reshape(n, 3, 3); t = g[:, 13:16]
    X = torch.einsum("nij,pj->npi", R, P) + t[:, None]
    z = X[..., 2].clamp(min=1e-3)
    u = K[0] * X[..., 0] / z + K[2]; v = K[1] * X[..., 1] / z + K[3]
    H, W = depth.shape
    ui, vi = u.round().long(), v.round().long()
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    D = torch.where(ok, depth[vi.clamp(0, H - 1), ui.clamp(0, W - 1)].float(), torch.zeros_like(z))   # no boolean gather: no host sync
    valid = ok & (D > 0.1)
    s = torch.where(valid, D - X[..., 2], torch.zeros_like(z))
    return u, v, (s / mu).clamp(-1, 1), valid.float(), (-s / OCC_SCALE).clamp(0, 1) * valid


def grid_inputs(g, K, depth, fmaps, projs, mu, ax, P, feat_cells=False, occ_ch=False, back=None, nrm=None, flag=None):
    """Input image (n, 2 kx + 64 len(fmaps), ky, kz) of the candidates g (n, 17) of one frame."""
    kx, ky, kz = (len(a) for a in ax)
    n = len(g)
    with torch.no_grad():
        R = g[:, 4:13].reshape(n, 3, 3); t = g[:, 13:16]
        X = torch.einsum("nij,pj->npi", R, P) + t[:, None]
        z = X[..., 2].clamp(min=1e-3)
        u = K[0] * X[..., 0] / z + K[2]; v = K[1] * X[..., 1] / z + K[3]
        H, W = depth.shape
        ui, vi = u.round().long(), v.round().long()
        ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        D = torch.zeros_like(z); D[ok] = depth[vi[ok], ui[ok]].float()
        valid = ok & (D > 0.1)
        s = torch.where(valid, D - X[..., 2], torch.zeros_like(z))      # projective signed distance (m), + in front
        parts = [(s / mu).clamp(-1, 1).reshape(n, kx, ky, kz), valid.float().reshape(n, kx, ky, kz)]
        if occ_ch:                                      # unsaturated occlusion depth: the clamped TSDF is -1 for every cell >= mu behind
            parts.append(((-s / OCC_SCALE).clamp(0, 1) * valid).reshape(n, kx, ky, kz))
        if back is not None:   # per-pixel back surface along the ray (layer 0 first exit, layer 1 last exit of the first-hit object)
            B = torch.zeros(len(back), *z.shape, device=z.device); B[:, ok] = back[:, vi[ok], ui[ok]].float()
            hb = ok & (B[0] > 0)                                                   # the ray's first hit is an object
            for L in range(2):   # (z_cell - z_back) / mu: < 0 in front of the back surface (inside the object if also behind the front)
                parts.append(torch.where(hb, (X[..., 2] - B[L]) / mu, torch.ones_like(z)).clamp(-1, 1).reshape(n, kx, ky, kz))
            parts.append(hb.float().reshape(n, kx, ky, kz))
            if len(back) == 4:   # predicted maps with their Laplace scale per layer (back_unc): how far to trust the two channels above
                parts += [((B[2 + L] / mu).clamp(0, 4) / 4).reshape(n, kx, ky, kz) for L in range(2)]
        if flag is not None:   # sensor type (0 stereo / structured light, 1 time-of-flight), constant over the grid (--sensor_flag)
            parts.append(torch.full((n, kx, ky, kz), float(flag), device=z.device))
        if nrm is not None:    # per-pixel surface normals (first hit, first exit, last exit) rotated into the gripper frame
            N = torch.zeros(9, *z.shape, device=z.device); N[:, ok] = nrm[:, vi[ok], ui[ok]].float()
            for L in range(3):
                ng = torch.einsum("nji,jnp->inp", R, N[3 * L:3 * L + 3])                # R^T n per cell -> (3, n, p)
                parts += [ng[i].reshape(n, kx, ky, kz) for i in range(3)]
        if fmaps:   # the column's cell nearest to the observed surface; the grasp-centre cell when the column has no depth
            a = torch.where(valid, s.abs(), torch.full_like(s, float("inf"))).reshape(n, kx, ky, kz)
            k = a.argmin(1)
            k = torch.where(torch.isinf(a.amin(1)), torch.full_like(k, int(ax[0].abs().argmin())), k)
            Hp, Wp = math.ceil(H / VP.PATCH) * VP.PATCH, math.ceil(W / VP.PATCH) * VP.PATCH   # the maps cover the padded network input
            if feat_cells:                                                   # every cell reads its own pixel -> (n, F * kx, ky, kz)
                grid = torch.stack([(u + 0.5) / Wp * 2 - 1, (v + 0.5) / Hp * 2 - 1], -1).reshape(1, n * kx * ky, kz, 2)
            else:
                uu = u.reshape(n, kx, ky, kz).gather(1, k[:, None])[:, 0]; vv = v.reshape(n, kx, ky, kz).gather(1, k[:, None])[:, 0]
                grid = torch.stack([(uu + 0.5) / Wp * 2 - 1, (vv + 0.5) / Hp * 2 - 1], -1).reshape(1, n * ky, kz, 2)
    for fm, pr in zip(fmaps or [], projs):
        m = pr(fm.float()[None])                                         # (1, 64, h, w)
        f = F.grid_sample(m, grid, mode="bilinear", align_corners=False)[0]
        parts.append(f.reshape(-1, n, kx * ky, kz).transpose(0, 1).reshape(n, -1, ky, kz) if feat_cells else f.reshape(-1, n, ky, kz).transpose(0, 1))
    return torch.cat(parts, 1)


def slices_at(T, xq):
    """T (n, kx, ky, kz) on the x cells; xq (n, k) query x (m) -> (n, k, ky, kz), linear along x."""
    f = ((xq - UG["x"][0]) / UG["x"][2] - 0.5).clamp(0, T.shape[1] - 1)
    k0 = f.floor().long().clamp(max=T.shape[1] - 2); a = (f - k0)[..., None, None]
    e = (-1, -1, T.shape[2], T.shape[3])
    return (1 - a) * T.gather(1, k0[..., None, None].expand(*e)) + a * T.gather(1, (k0 + 1)[..., None, None].expand(*e))


def surface_delta(T, ax, max_shift=0.03):
    """Approach-axis offset (n,) in metres from the grasp centre to the predicted object surface: along the centre column
    (y = 0, z = 0; mean of the 4 nearest columns) of the TSDF T (n, kx, ky, kz; sign +outside), the first crossing from
    outside to inside when scanning from the palm toward the fingertips, linearly interpolated. 0 when there is none."""
    xc = ax[0]; ky, kz = T.shape[2], T.shape[3]
    col = T[:, :, ky // 2 - 1:ky // 2 + 1, kz // 2 - 1:kz // 2 + 1].mean((2, 3))           # (n, kx)
    a, b = col[:, :-1], col[:, 1:]
    cross = (a > 0) & (b <= 0)
    first = torch.where(cross.any(1), cross.float().argmax(1), torch.zeros(len(T), dtype=torch.long, device=T.device))
    a1 = a.gather(1, first[:, None])[:, 0]; b1 = b.gather(1, first[:, None])[:, 0]
    frac = (a1 / (a1 - b1).clamp(min=1e-6)).clamp(0, 1)
    x = xc[first] + frac * (xc[1] - xc[0])
    return torch.where(cross.any(1), x.clamp(-max_shift, max_shift), torch.zeros_like(x))


def object_extent(T, ax, d):
    """Object extent (n,) in metres along the closing axis at the fingertip plane x = d, z = 0 (mean of the 2 centre rows):
    from the centre outward, the first inside->outside crossing on each side, linearly interpolated. NaN when the centre
    cell is not inside the object or a side has no crossing."""
    yc = ax[1]; kz = T.shape[3]
    row = slices_at(T, d[:, None])[:, 0][:, :, kz // 2 - 1:kz // 2 + 1].mean(2)             # (n, ky)
    ky = row.shape[1]; n = len(T)
    out = torch.full((n,), float("nan"), device=T.device)
    c0, c1 = row[:, ky // 2 - 1], row[:, ky // 2]
    inside = (c0 <= 0) & (c1 <= 0)
    right = row[:, ky // 2:]; left = row[:, :ky // 2].flip(1)                                  # from the centre outward
    def edge(seg):                                                                            # first inside->outside crossing
        a, b = seg[:, :-1], seg[:, 1:]; cross = (a <= 0) & (b > 0)
        i = torch.where(cross.any(1), cross.float().argmax(1), torch.zeros(n, dtype=torch.long, device=T.device))
        a1 = a.gather(1, i[:, None])[:, 0]; b1 = b.gather(1, i[:, None])[:, 0]
        frac = (-a1 / (b1 - a1).clamp(min=1e-6)).clamp(0, 1)
        return (i.float() + frac) * UG["y"][2], cross.any(1)
    er, okr = edge(right); el, okl = edge(left)
    ok = inside & okr & okl
    out[ok] = (er + el)[ok]
    return out


class FCHead(nn.Module):
    """Force-closure head: 2-D conv over (y, z) (kernels 5 x 3, dilation 1, 2, 4 along y), max + mean pool -> 6 logits,
    symmetrised over the finger swap (y flip) and the thickness flip (z flip)."""

    def __init__(self, c_in, hidden=32, n_out=len(MU), n_vec=0, zrows=None):
        super().__init__()
        self.zrows = zrows                                                 # (z0, z1): pool only over these thickness rows (--fc_zband)
        # the last 9 n_vec input channels are gripper-frame normal slices ordered (vector, component x/y/z, slice): a finger
        # swap negates their y component and a thickness flip their z component
        sy, sz = torch.ones(c_in), torch.ones(c_in)
        for v in range(n_vec):
            o = c_in - 9 * n_vec + 9 * v
            sy[o + 3:o + 6] = -1; sz[o + 6:o + 9] = -1
        self.register_buffer("sy", sy[None, :, None, None], persistent=False); self.register_buffer("sz", sz[None, :, None, None], persistent=False)
        self.net = nn.Sequential(nn.Conv2d(c_in, hidden, 1), nn.ReLU(),
                                 nn.Conv2d(hidden, hidden, (5, 3), padding=(2, 1)), nn.ReLU(),
                                 nn.Conv2d(hidden, hidden, (5, 3), padding=(4, 1), dilation=(2, 1)), nn.ReLU(),
                                 nn.Conv2d(hidden, hidden, (5, 3), padding=(8, 1), dilation=(4, 1)), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def _once(self, x):
        h = self.net(x)
        if self.zrows is not None:                                         # the evaluator decides force closure on the fingertip line: read there
            h = h[:, :, :, self.zrows[0]:self.zrows[1]]
        return self.out(torch.cat([h.amax((2, 3)), h.mean((2, 3))], 1))

    def forward(self, x):
        n = len(x); k = getattr(self, "flips", 4)                             # the flips in one batch: same values, one kernel launch
        xs = [x, x.flip(2) * self.sy, x.flip(3) * self.sz, x.flip(2).flip(3) * self.sy * self.sz][:k]
        return self._once(torch.cat(xs, 0)).reshape(k, n, -1).mean(0)


class GridVerifier(nn.Module):
    def __init__(self, c_maps, design="A", hidden=128, d_feat=64, mu=0.02, fc_target="cls", udf=False, fusion="early", feat_cells=False, heads_hT=False, dilation=True, ordinal=False, grid=None, tsdf_unc=False, encoder="stack", neck="serial", c_cell=0, tsdf_res=False, occ_ch=False, n_blocks=4, xyz_coord=False, c3_early_stride=False, fc_flips=4, mirror_geo=False, back_ch=False, nrm_ch=False, back_unc=False, sensor_flag=False, fc_zband=0.0, fc_no_open=False, fc_field_only=False):
        super().__init__()
        if grid is not None:
            set_grid(grid)
        if encoder in ("pillar", "pillar2", "conv3d", "vfe", "triplane", "axial"):
            feat_cells = True                                                # every cell carries its own feature
        c_cell = c_cell or (64 if encoder in ("pillar", "pillar2") else 16)
        self._tri = None                                                     # transient tri-plane features, set by the encoder   # width of the per-cell tensor inference streams
        self.cfg = dict(c_maps=list(c_maps), design=design, hidden=hidden, d_feat=d_feat, mu=mu, fc_target=fc_target, udf=udf,
                        fusion=fusion, feat_cells=feat_cells, heads_hT=heads_hT, dilation=dilation, ordinal=ordinal, grid=UG.get("name", "g5"), tsdf_unc=tsdf_unc, encoder=encoder, neck=neck, c_cell=c_cell, tsdf_res=tsdf_res, occ_ch=occ_ch, n_blocks=n_blocks, xyz_coord=xyz_coord, c3_early_stride=c3_early_stride, fc_flips=fc_flips, mirror_geo=mirror_geo, back_ch=back_ch, nrm_ch=nrm_ch, back_unc=back_unc, sensor_flag=sensor_flag, fc_zband=fc_zband, fc_no_open=fc_no_open, fc_field_only=fc_field_only)
        kx = round((UG["x"][1] - UG["x"][0]) / UG["x"][2]); self.kx = kx
        zc = UG["z"][0] + (np.arange(round((UG["z"][1] - UG["z"][0]) / UG["z"][2])) + 0.5) * UG["z"][2]   # thickness-row centres
        zr = np.flatnonzero(np.abs(zc) <= fc_zband + 1e-9)
        self.fc_zrows = (int(zr[0]), int(zr[-1]) + 1) if fc_zband > 0 else None
        self.n_geo = (3 if occ_ch else 2) + (3 if back_ch else 0) + (2 if back_unc else 0) + (1 if sensor_flag else 0) + (9 if nrm_ch else 0)   # + back layers 0, 1, flag (+ their scales); + 3 normals
        self.n_coord = 3 if xyz_coord else 1
        self.n_mirror = 2 if mirror_geo else 0                               # tsdf, valid of the cell mirrored across the closing axis                                 # cell coordinate channels: x only, or all three                                      # geometric maps per x cell: tsdf, valid (, occlusion depth)
        self.proj = nn.ModuleList([nn.Conv2d(c, d_feat, 1) for c in c_maps])
        n_g, n_f = self.n_geo * kx, d_feat * len(c_maps) * (kx if feat_cells else 1)
        dil = ((1, 2, 4, 8) if dilation else (1, 1, 1, 1))[:n_blocks]
        block = lambda c, r: nn.Conv2d(c, c, 3, padding=r, dilation=r)
        d_cell = self.n_coord + self.n_geo + self.n_mirror + d_feat * len(c_maps)                      # per-cell attributes [x, tsdf, valid (, occlusion depth), feature]
        if encoder == "pillar":                                            # PointPillars-style column encoder: shared MLP over the
            self.pillar = nn.Sequential(nn.Conv2d(d_cell, c_cell, 1), nn.ReLU(), nn.Conv2d(c_cell, 64, 1), nn.ReLU())   # cells of a column, max-pooled
            self.stem = nn.Conv2d(64, hidden, 1)
        elif encoder == "pillar2":                                         # faithful PFN: Linear-BN-ReLU, column max concatenated, second layer, max
            self.pfn1 = nn.Sequential(nn.Conv2d(d_cell, 32, 1, bias=False), nn.BatchNorm2d(32, eps=1e-3, momentum=0.01), nn.ReLU())
            self.pfn2 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64, eps=1e-3, momentum=0.01), nn.ReLU())
            self.stem = nn.Conv2d(64, hidden, 1)
        elif encoder == "conv3d":                                          # ordered x: two 3-D convs over (x, y, z), then x folded into channels
            self.c3 = nn.Sequential(nn.Conv3d(d_cell, 32, 3, padding=1), nn.ReLU(), nn.Conv3d(32, 16, 3, padding=1), nn.ReLU())
            self.stem = nn.Conv2d(16 * kx, hidden, 1)
        elif encoder == "vfe":                                             # VoxelNet / SECOND: a linear VFE narrows the cell attributes,
            self.vfe = nn.Sequential(nn.Conv2d(d_cell, c_cell, 1, bias=False),   # then 3-D convs widen while striding the approach axis x
                                     nn.BatchNorm2d(c_cell, eps=1e-3, momentum=0.01), nn.ReLU())
            s1 = 4 if c3_early_stride else 2                                          # stride the approach axis early: the wide layer
            self.c3 = nn.Sequential(nn.Conv3d(c_cell, 32, 3, stride=(s1, 1, 1), padding=1, bias=False),   # then runs at a quarter resolution
                                    nn.BatchNorm3d(32, eps=1e-3, momentum=0.01), nn.ReLU(),
                                    nn.Conv3d(32, 64, 3, stride=(2, 1, 1) if not c3_early_stride else (1, 1, 1), padding=1, bias=False),
                                    nn.BatchNorm3d(64, eps=1e-3, momentum=0.01), nn.ReLU())
            self.kx_out = (kx + s1 - 1) // s1
            if not c3_early_stride:
                self.kx_out = (self.kx_out + 1) // 2
            self.stem = nn.Conv2d(64 * self.kx_out, hidden, 1)
        elif encoder == "axial":        # Transformer over the ordered cells of the approach axis (SST / DSVT put attention on
            self.vfe = nn.Sequential(nn.Conv2d(d_cell, c_cell, 1, bias=False),      # voxels; the axis factorisation is Axial-DeepLab's)
                                     nn.BatchNorm2d(c_cell, eps=1e-3, momentum=0.01), nn.ReLU())
            self.pos = nn.Parameter(torch.zeros(1, kx, c_cell)); nn.init.trunc_normal_(self.pos, std=0.02)
            self.attn = nn.ModuleList([nn.MultiheadAttention(c_cell, 4, batch_first=True) for _ in range(2)])
            self.ln = nn.ModuleList([nn.LayerNorm(c_cell) for _ in range(4)])
            self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(c_cell, 2 * c_cell), nn.GELU(), nn.Linear(2 * c_cell, c_cell)) for _ in range(2)])
            self.stem = nn.Conv2d(c_cell * kx, hidden, 1)                           # x order kept, folded into channels
        elif encoder == "triplane":     # ConvONet / EG3D tri-plane: the cells are pooled onto three orthogonal planes, each convolved
            self.vfe = nn.Sequential(nn.Conv2d(d_cell, c_cell, 1, bias=False),         # in 2-D (the cheapest primitive in this regime),
                                     nn.BatchNorm2d(c_cell, eps=1e-3, momentum=0.01), nn.ReLU())    # and every cell reads the sum of its
            pl = lambda: nn.Sequential(nn.Conv2d(c_cell, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64, eps=1e-3, momentum=0.01), nn.ReLU(),   # three
                                       nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(64, eps=1e-3, momentum=0.01), nn.ReLU())
            self.planes = nn.ModuleList([pl() for _ in range(3)])                      # yz (x pooled), xz (y pooled), xy (z pooled)
            self.stem = nn.Conv2d(64, hidden, 1)                                       # neck and both verdicts stay on the yz plane
            self.tri_yz = nn.Conv2d(hidden, 32, 1); self.tri_xz = nn.Conv2d(64, 32, 1); self.tri_xy = nn.Conv2d(64, 32, 1)
            self.tri_dec = nn.Conv3d(32, 1, 1)
        if encoder in ("pillar", "pillar2", "conv3d", "vfe", "triplane", "axial"):
            if neck == "aspp":                                               # PillarNeXt-style ASPP: one 3x3 weight at dilations 1,2,4,8 + 1x1, fused, then 2 residual blocks
                self.aspp_w = nn.Parameter(torch.empty(hidden, hidden, 3, 3)); nn.init.kaiming_normal_(self.aspp_w)
                self.aspp_1x1 = nn.Conv2d(hidden, hidden, 1); self.aspp_fuse = nn.Conv2d(5 * hidden, hidden, 1)
                self.blocks = nn.ModuleList([block(hidden, 1) for _ in range(2)])
            else:
                self.blocks = nn.ModuleList([block(hidden, r) for r in dil])
        elif fusion == "early":                                            # one stem over all channels
            self.stem = nn.Conv2d(n_g + n_f, hidden, 1)
            if neck == "aspp":
                self.aspp_w = nn.Parameter(torch.empty(hidden, hidden, 3, 3)); nn.init.kaiming_normal_(self.aspp_w)
                self.aspp_1x1 = nn.Conv2d(hidden, hidden, 1); self.aspp_fuse = nn.Conv2d(5 * hidden, hidden, 1)
                self.blocks = nn.ModuleList([block(hidden, 1) for _ in range(2)])
            else:
                self.blocks = nn.ModuleList([block(hidden, r) for r in dil])
        elif fusion == "mid":                                              # a stem + one conv per modality, then shared blocks
            h2 = hidden // 2
            self.stem_g = nn.Sequential(nn.Conv2d(n_g, h2, 1), nn.ReLU(), nn.Conv2d(h2, h2, 3, padding=1))
            self.stem_f = nn.Sequential(nn.Conv2d(n_f, h2, 1), nn.ReLU(), nn.Conv2d(h2, h2, 3, padding=1))
            self.blocks = nn.ModuleList([block(hidden, r) for r in dil[1:]])
        else:                                                              # late: two full streams, concatenated for the heads
            h2 = hidden // 2
            self.stem_g = nn.Conv2d(n_g, h2, 1); self.stem_f = nn.Conv2d(n_f, h2, 1)
            self.blocks_g = nn.ModuleList([block(h2, r) for r in dil]); self.blocks_f = nn.ModuleList([block(h2, r) for r in dil])
        self.tsdf = nn.Conv2d(hidden, kx * (2 if tsdf_unc else 1), 1)     # tsdf_unc: per-cell mean and log-scale (Kendall & Gal)
        extra = (hidden if heads_hT else 0) + (kx if tsdf_unc else 0)       # heads may also read the backbone features / the TSDF scale
        # per-cell occupancy logits: the same two-layer 1x1 head in both designs; the output bias starts at the prior
        # -log(cells of the gripper boxes) ~ -6, so the pooled collision / empty logits start near zero (as a detector prior)
        self.occ = nn.Sequential(nn.Conv2d(hidden if design == "A" else kx + extra, 64, 1), nn.ReLU(), nn.Conv2d(64, kx, 1))
        nn.init.constant_(self.occ[-1].bias, -6.0)
        self._fc_flips = fc_flips
        n_vec = 3 if (nrm_ch and design == "B") else 0                  # design B: the force-closure head also reads the normals
        self.fc = FCHead(hidden + 2 if design == "A" else 3 + (0 if fc_field_only else 1 if fc_no_open else 2) + (hidden if heads_hT else 0) + (3 if tsdf_unc else 0) + 9 * n_vec,
                         n_out=(1 if ordinal else len(MU)) if fc_target == "cls" else 1, n_vec=n_vec, zrows=self.fc_zrows)
        if ordinal:   # CORAL: one shared score, ordered cut points (success at mu_j implies success at mu_j+1 -> logits increase in j)
            self.cut0 = nn.Parameter(torch.tensor(-1.0)); self.cut_d = nn.Parameter(torch.zeros(len(MU) - 1))
        self.fc.flips = fc_flips
        self.b_empty = nn.Parameter(torch.zeros(()))

    def reject_logit(self, o, body, close):
        col = torch.logsumexp((o + torch.log(body.clamp(min=1e-12))).flatten(1), 1)
        emp = -torch.logsumexp((o + torch.log(close.clamp(min=1e-12))).flatten(1), 1) + self.b_empty
        return torch.logsumexp(torch.stack([col, emp], 1), 1).clamp(-30, 30)

    def choose_config(self, content, w, d, score, ax, max_w, offsets, surf=None, h=None, sig=None, nrm=None, alpha=3.0, top=0):
        """In-frame configuration choice (design B). The predicted field lives in the gripper frame, so it does not depend on
        the opening width w or the approach depth d, which enter only through the gripper boxes. A parallel jaw closes until
        contact: the width decides only whether the fingers collide, so at each depth option the width is the option with the
        lowest reject logit (the collision head's per-cell logits once; the boxes are products of per-axis factors, so an option
        costs two bilinear forms over (x, y)). The depth moves the closing line and with it the contacts, so it is chosen by the
        full verdict score * sigmoid(-reject)^alpha * force closure (one force-closure pass per depth option), for the `top`
        candidates with the best verdict at their own depth (0: all). offsets: (depth offsets, width offsets) in metres, width
        offsets None = every opening of the gripper on the grid lattice; ties keep the candidate's own values. surf (n, kx, ky, kz) bool: cells crossed by the measured surface -- an option whose
        body touches one is excluded, as in the surface test. Returns w, d, reject, q at the chosen configuration."""
        dd, dw = offsets; n = len(w); i0 = dd.index(0.0)
        W_ = self._widths(self.occ(content).float(), w.float(), d.float(), ax, max_w, dd, dw, surf)          # (len(dd), n)
        D_ = torch.stack([(d + a).clamp(0.0, 0.05) for a in dd])
        hT = self.cfg.get("heads_hT"); unc = self.cfg.get("tsdf_unc")

        def verdict(rows, ws, ds):   # the model's reject logit (with the surface test) and force-closure logits at (ws, ds); rows None: all
            k = len(ws) // (n if rows is None else len(rows))
            rep = lambda x: None if x is None else (x if rows is None else x[rows].repeat(k, *([1] * (x.dim() - 1))))
            body, close, open_y = gripper_masks(ws, ds, ax)
            r, q = self.heads(rep(content), body, close, open_y, ds, h=rep(h) if hT else None, sig=rep(sig) if unc else None, nrm=rep(nrm))
            if surf is not None:
                r = excluded(r, (rep(surf).reshape(len(ws), -1) & body_member(ws, ds, ax).reshape(len(ws), -1)).any(1))
            return r, q

        ar = torch.arange(n, device=w.device)
        r0, q0 = verdict(None, W_[i0], D_[i0])
        v0 = torch.sigmoid(-r0.float()) ** alpha * fc_score(q0.float(), self.cfg["fc_target"])
        rows = torch.topk(score * v0, top).indices if 0 < top < n else ar
        m = len(rows); other = [i for i in range(len(dd)) if i != i0]
        r1, q1 = verdict(rows, W_[other][:, rows].reshape(-1), D_[other][:, rows].reshape(-1))
        R = torch.cat([r0[rows][None], r1.reshape(len(other), m)]); Q = torch.cat([q0[rows][None], q1.reshape(len(other), m, -1)])
        V = torch.sigmoid(-R.float()) ** alpha * fc_score(Q.reshape(-1, Q.shape[-1]).float(), self.cfg["fc_target"]).reshape(-1, m)
        if not EXACT:
            V[0] += 1e-9                                                                                         # ties keep the own depth
        # (EXACT: argmax returns the first maximum, and row 0 is the own depth)
        j = V.argmax(0); idx = torch.tensor([i0] + other, device=w.device)[j]; am = torch.arange(m, device=w.device)
        w_out = W_[i0].clone(); d_out = D_[i0].clone(); r_out = r0.clone(); q_out = q0.clone()
        w_out[rows] = W_[idx, rows]; d_out[rows] = D_[idx, rows]; r_out[rows] = R[j, am]; q_out[rows] = Q[j, am]
        return w_out, d_out, r_out, q_out

    @torch.autocast("cuda", enabled=False)
    def _widths(self, o, w, d, ax, max_w, dd, dw, surf):
        """(len(dd), n): at each depth option, the width option with the lowest reject logit (separable collision re-read,
        all options in one batch; ties keep the own width)."""
        n = len(w); m = o.flatten(1).amax(1)
        fz = gripper_factors(w[:1], d[:1], ax)[5]
        e = torch.exp(o - m[:, None, None, None])
        Z = 1e-12 * e.flatten(1).sum(1)                                   # reject_logit's 1e-12 mask floor, summed over the cells
        E = torch.einsum("nxyz,z->nxy", e, fz)
        if dw is None:   # every opening of the gripper on the grid lattice (5 mm, ..., max_w), and the candidate's own
            sp = UG["y"][2]; Wl = ([w] if OWN_W else []) + [torch.full_like(w, k * sp) for k in range(1, int(round(max_w / sp)) + 1)]
            own = 0 if OWN_W else None
        else:
            Wl = [(w + b).clamp(0.0, max_w) for b in dw]; own = list(dw).index(0.0)
        D = torch.stack([(d + a).clamp(0.0, 0.05) for a in dd for _ in Wl]).reshape(-1)          # options depth-major
        Wo = torch.stack([x for a in dd for x in Wl]).reshape(-1)
        fx_f, fx_b, fy_f, fy_b, fy_in, _ = (t.reshape(-1, n, t.shape[-1]) if t.dim() == 2 else t for t in gripper_factors(Wo, D, ax))
        bil = lambda a, M, b: torch.einsum("knx,nxy,kny->kn", a, M, b)
        col = torch.log(bil(fx_f, E, fy_f) + bil(fx_b, E, fy_b) + Z) + m
        emp = -(torch.log(bil(fx_f, E, fy_in) + Z) + m) + self.b_empty.float()
        r = torch.logsumexp(torch.stack([col, emp]), 0).clamp(-30, 30)                         # (k, n)
        if surf is not None:
            mx_f, mx_b, my_f, my_b, mz = (t.reshape(-1, n, t.shape[-1]) if t.dim() == 2 else t for t in body_factors(Wo, D, ax))
            S = (surf & (mz > 0)).any(3).float()
            hit = bil(mx_f, S, my_f) + bil(mx_b, S, my_b)
            r = excluded(r, hit > 0)
        nw = len(Wl); r = r.reshape(len(dd), nw, n); Wr = Wo.reshape(len(dd), nw, n)
        if own is None:   # lattice only: lowest reject logit, ties to the opening nearest the decoded width (no epsilon)
            dist = torch.where(r == r.amin(1, keepdim=True), (Wr - w[None, None]).abs(), torch.full_like(r, float("inf")))
            return Wr.gather(1, dist.argmin(1, keepdim=True))[:, 0]
        r = r - 1e-4 * (torch.arange(nw, device=r.device) == own).float()[None, :, None]                # ties keep the own width
        return Wr.gather(1, r.argmin(1, keepdim=True))[:, 0]

    def heads(self, content, body, close, open_y, d, h=None, sig=None, nrm=None):
        """content: h (design A) or the clamped TSDF (design B) -> reject logit (n,), force-closure logits (n, 6).
        h: backbone features, appended to both head inputs when cfg heads_hT (design B only); sig: per-cell TSDF scale
        (tsdf_unc), appended likewise (zeros when the true TSDF is fed)."""
        n, _, ky, kz = content.shape
        grip = torch.cat([open_y[:, None, :, None].expand(n, 1, ky, kz), (d / 0.05)[:, None, None, None].expand(n, 1, ky, kz)], 1)
        if self.cfg["design"] == "A":
            fc_in = torch.cat([content, grip], 1); occ_in = content
        else:
            sp = UG["x"][2]; xq = torch.stack([d - sp, d, d + sp], 1)
            fc_in = torch.cat([slices_at(content, xq)] + ([] if self.cfg.get("fc_field_only") else [grip[:, 1:] if self.cfg.get("fc_no_open") else grip]), 1); occ_in = content   # fc_no_open / fc_field_only: the evaluator's force closure depends only on the surface along the fingertip line
            if self.cfg.get("heads_hT") and h is not None:
                fc_in = torch.cat([fc_in, h], 1); occ_in = torch.cat([occ_in, h], 1)
            if self.cfg.get("tsdf_unc"):
                sg = sig if sig is not None else torch.zeros_like(content)
                fc_in = torch.cat([fc_in, slices_at(sg, xq)], 1); occ_in = torch.cat([occ_in, sg], 1)
            if self.cfg.get("nrm_ch"):   # nrm (n, 9, kx, ky, kz): the three normals' gripper-frame components on every cell
                fc_in = torch.cat([fc_in] + [slices_at(nrm[:, c], xq) for c in range(9)], 1)
        q = self.fc(fc_in)
        if self.cfg.get("ordinal"):
            q = q + torch.cat([self.cut0[None], self.cut0 + torch.cumsum(F.softplus(self.cut_d), 0)])[None]
        return self.reject_logit(self.occ(occ_in), body, close), q

    def backbone(self, img):
        n_g = self.n_geo * self.kx; fu = self.cfg.get("fusion", "early")
        enc = self.cfg.get("encoder", "stack")
        if enc in ("pillar", "pillar2", "conv3d", "vfe", "triplane", "axial"):
            n, _, ky, kz = img.shape; kx = self.kx; ng = self.n_geo
            geo = [img[:, i * kx:(i + 1) * kx, None] for i in range(ng)]                      # tsdf, valid (, occlusion depth)
            feat = img[:, ng * kx:].reshape(n, -1, kx, ky, kz).permute(0, 2, 1, 3, 4)          # (n, kx, F, ky, kz)
            ca = cell_axes(img.device)
            co = [(ca[0] / 0.07)[None, :, None, None, None].expand(n, kx, 1, ky, kz)]
            if self.n_coord == 3:                                            # absolute cell position is not recoverable from a
                co.append((ca[1] / 0.09)[None, None, None, :, None].expand(n, kx, 1, ky, kz))   # translation-equivariant conv
                co.append((ca[2] / 0.04)[None, None, None, None, :].expand(n, kx, 1, ky, kz))
            mir = [t.flip(3) for t in geo[:2]] if self.n_mirror else []   # the observed near side constrains the occluded far side
            cells = torch.cat(co + geo + mir + [feat], 2)                                          # (n, kx, d_cell, ky, kz)
            if enc == "pillar":
                h = self.pillar(cells.reshape(n * kx, -1, ky, kz)).reshape(n, kx, -1, ky, kz).amax(1)
            elif enc == "pillar2":
                x1 = self.pfn1(cells.reshape(n * kx, -1, ky, kz)).reshape(n, kx, -1, ky, kz)
                x1 = torch.cat([x1, x1.amax(1, keepdim=True).expand(-1, kx, -1, -1, -1)], 2)   # cell feature + column max
                h = self.pfn2(x1.reshape(n * kx, -1, ky, kz)).reshape(n, kx, -1, ky, kz).amax(1)
            elif enc == "vfe":
                v = self.vfe(cells.reshape(n * kx, -1, ky, kz)).reshape(n, kx, -1, ky, kz).permute(0, 2, 1, 3, 4)
                h = self.c3(v).reshape(n, -1, ky, kz)                                          # (n, 64 * kx_out, ky, kz), x order kept
            elif enc == "triplane":
                v = self.vfe(cells.reshape(n * kx, -1, ky, kz)).reshape(n, kx, -1, ky, kz).permute(0, 2, 1, 3, 4)
                h = self.tri_planes(v)
            elif enc == "axial":
                v = self.vfe(cells.reshape(n * kx, -1, ky, kz)).reshape(n, kx, -1, ky, kz)
                h = self.axial_tokens(v, n, kx, ky, kz)
            else:
                h = self.c3(cells.permute(0, 2, 1, 3, 4)).reshape(n, -1, ky, kz)               # (n, 16 * kx, ky, kz), x order kept
            h = F.relu(self.stem(h))
        elif fu == "early":
            h = F.relu(self.stem(img))
        elif fu == "mid":
            h = torch.cat([F.relu(self.stem_g(img[:, :n_g])), F.relu(self.stem_f(img[:, n_g:]))], 1)
        else:
            hg = F.relu(self.stem_g(img[:, :n_g])); hf = F.relu(self.stem_f(img[:, n_g:]))
            for bg, bf in zip(self.blocks_g, self.blocks_f):
                hg = hg + F.relu(bg(hg)); hf = hf + F.relu(bf(hf))
            return torch.cat([hg, hf], 1)
        if self.cfg.get("neck") == "aspp":
            br = [self.aspp_1x1(h)] + [F.conv2d(h, self.aspp_w, padding=r, dilation=r) for r in (1, 2, 4, 8)]
            h = F.relu(self.aspp_fuse(torch.cat(br, 1)))
        for b in self.blocks:
            h = h + F.relu(b(h))
        return h

    @torch.no_grad()
    @torch.no_grad()
    def c3_as_2d(self, v):
        """The x-strided 3-D encoder (Conv3d k3 stride (2,1,1) pad 1 + eval BN + ReLU, twice) evaluated exactly as 2-D convs over
        triplets of x-slices: each output x-slice is a 3x3 conv over (y, z) of the 3 input slices stacked as channels. Weights are
        reshaped (and BN folded) once and cached. v (n, C, kx, ky, kz) -> (n, 64 * kx_out, ky, kz)."""
        if getattr(self, "_c3_2d", None) is None:
            layers = []
            for conv, bn in ((self.c3[0], self.c3[1]), (self.c3[3], self.c3[4])):
                W = conv.weight                                                     # (Co, Ci, 3, 3, 3) over (x, y, z)
                sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                W2 = (W * sc[:, None, None, None, None]).permute(0, 1, 2, 3, 4).reshape(W.shape[0], W.shape[1] * 3, 3, 3)   # channel = ci * 3 + kx
                b2 = bn.bias - bn.running_mean * sc
                layers.append((W2, b2, conv.stride[0]))
            self._c3_2d = layers
        n = v.shape[0]
        for W2, b2, st in self._c3_2d:
            C, kx, ky, kz = v.shape[1:]
            vp = F.pad(v, (0, 0, 0, 0, 1, 1))                                       # pad x by one on each side
            t = vp.unfold(2, 3, st)                                                 # (n, C, kx_out, ky, kz, 3)
            kxo = t.shape[2]
            t = t.permute(0, 2, 1, 5, 3, 4).reshape(n * kxo, C * 3, ky, kz)        # channels (c, kx) as the weights
            v = F.relu(F.conv2d(t, W2.to(t.dtype), b2.to(t.dtype), padding=1)).reshape(n, kxo, -1, ky, kz).permute(0, 2, 1, 3, 4)
        return v.reshape(n, -1, ky, kz) if False else v.permute(0, 1, 2, 3, 4).reshape(n, -1, v.shape[3], v.shape[4])

    @torch.no_grad()
    def fold(self, fmaps):
        """Per-frame part of the fast path: the first cell layer's weights with eval BatchNorm folded in, and the feature maps
        pushed through the map projections and that layer's feature columns (bilinear sampling commutes with linear maps)."""
        enc = self.cfg.get("encoder")
        if enc == "pillar":
            lin1, bn1 = self.pillar[0], None
        elif enc in ("vfe", "triplane", "axial"):
            lin1, bn1 = self.vfe[0], self.vfe[1]
        else:
            lin1, bn1 = self.pfn1[0], self.pfn1[1]
        W1 = lin1.weight[:, :, 0, 0]; b1 = lin1.bias if lin1.bias is not None else torch.zeros(W1.shape[0], device=W1.device)
        if bn1 is not None:                                                             # eval BN: per-channel affine
            sc = bn1.weight / torch.sqrt(bn1.running_var + bn1.eps); W1 = W1 * sc[:, None]; b1 = (b1 - bn1.running_mean) * sc + bn1.bias
        d_feat = self.cfg["d_feat"]; o0 = self.n_coord + self.n_geo + self.n_mirror
        maps = [torch.einsum("hd,dyx->hyx", W1[:, o0 + i * d_feat:o0 + (i + 1) * d_feat], pr(fm.float()[None])[0])
                for i, (fm, pr) in enumerate(zip(fmaps, self.proj))]
        return W1, b1, maps

    def forward_fast(self, g, K, depth, fmaps, mu, ax, P, max_w, back=None, nrm=None, flag=0.0, folded=None):
        """Inference path for the pillar encoders without the per-cell feature image: the first pillar layer is linear in the
        sampled features, and bilinear sampling commutes with linear maps, so its feature weights (and the map projections,
        and BatchNorm in eval mode) are applied to the feature maps once per frame; each cell then samples the 64-d result.
        back / nrm: the frame's back-surface (2, H, W) and normal (9, H, W) maps when the model reads them (back_ch / nrm_ch).
        Returns (reject, q, T) as forward(), and the (n, 2) width and depth read (the candidate's own unless `select` is set)."""
        enc = self.cfg.get("encoder")
        kx, ky, kz = (len(a) for a in ax); n = len(g)
        u, v, tsdf, valid, occ = project_cells(g, K, depth, mu, ax, P)                   # (n, p)
        ray = []                                                                        # per-pixel ray channels, as grid_inputs builds them
        if self.cfg.get("back_ch") or self.cfg.get("nrm_ch") or self.cfg.get("sensor_flag"):
            R = g[:, 4:13].reshape(n, 3, 3); zc = (torch.einsum("nij,pj->npi", R, P) + g[:, None, 13:16])[..., 2]
            ui, vi = u.round().long(), v.round().long()
            ok = (ui >= 0) & (ui < depth.shape[1]) & (vi >= 0) & (vi < depth.shape[0])
            uc, vc = ui.clamp(0, depth.shape[1] - 1), vi.clamp(0, depth.shape[0] - 1)
            if self.cfg.get("back_ch"):
                B = back[:, vc, uc].float() * ok; hb = ok & (B[0] > 0)
                ray += [torch.where(hb, (zc - B[L]) / mu, torch.ones_like(zc)).clamp(-1, 1) for L in range(2)] + [hb.float()]
                if len(back) == 4:
                    ray += [(B[2 + L] / mu).clamp(0, 4) / 4 for L in range(2)]
            if self.cfg.get("sensor_flag"):
                ray.append(torch.full_like(zc, float(flag)))
            if self.cfg.get("nrm_ch"):
                N = nrm[:, vc, uc].float() * ok
                for L in range(3):
                    ng = torch.einsum("nji,jnp->inp", R, N[3 * L:3 * L + 3]); ray += [ng[0], ng[1], ng[2]]
        d_feat = self.cfg["d_feat"]
        if enc == "pillar":
            lin1, lin2, bn1, bn2 = self.pillar[0], self.pillar[2], None, None
        elif enc in ("vfe", "triplane", "axial"):
            lin1, lin2, bn1, bn2 = self.vfe[0], None, self.vfe[1], None
        else:
            lin1, bn1, lin2, bn2 = self.pfn1[0], self.pfn1[1], self.pfn2[0], self.pfn2[1]
        W1, b1, fmaps_folded = folded if folded is not None else self.fold(fmaps)
        H, Wd = depth.shape; Hp, Wp = math.ceil(H / VP.PATCH) * VP.PATCH, math.ceil(Wd / VP.PATCH) * VP.PATCH
        grid = torch.stack([(u + 0.5) / Wp * 2 - 1, (v + 0.5) / Hp * 2 - 1], -1).reshape(1, n, -1, 2)
        if not fmaps_folded:                                                           # geometry-only verifier (--feature none): no map term
            feat = torch.zeros((), device=depth.device)
        elif getattr(self, "sampler", "") == "triton":                                 # channel-last folded maps (Verifier), one pass
            from verifier_sample import sample2
            feat = sample2(fmaps_folded[0], fmaps_folded[1] if len(fmaps_folded) > 1 else None, grid[0], out_dtype=torch.float32)   # (n, h1, p), fp32 as grid_sample under autocast
        else:
            feat = None                                                                 # sum over maps of sampled folded maps (h1, n, p)
            for mi in fmaps_folded:
                fi = F.grid_sample(mi[None], grid, mode="bilinear", align_corners=False)[0]
                feat = fi if feat is None else feat + fi
            feat = feat.permute(1, 0, 2)
        xc = (ax[0] / 0.07)[None, :, None, None].expand(n, kx, ky, kz).reshape(n, -1)
        co = [xc]
        if self.n_coord == 3:
            co.append((ax[1] / 0.09)[None, None, :, None].expand(n, kx, ky, kz).reshape(n, -1))
            co.append((ax[2] / 0.04)[None, None, None, :].expand(n, kx, ky, kz).reshape(n, -1))
        ge = co + [tsdf, valid] + ([occ] if self.cfg.get("occ_ch") else []) + ray
        if self.n_mirror:
            ge += [t.reshape(n, kx, ky, kz).flip(2).reshape(n, -1) for t in (tsdf, valid)]
        geo = torch.stack(ge, 1).to(feat.dtype)                                         # (n, n_coord + n_geo, p): one fused 1x1 conv
        nc = self.n_coord + self.n_geo + self.n_mirror
        h1 = F.relu(F.conv1d(geo, W1[:, :nc].to(feat.dtype)[:, :, None], b1.to(feat.dtype)) + feat)
        if enc == "triplane":
            h = self.tri_planes(h1.reshape(n, -1, kx, ky, kz).to(self.planes[0][0].weight.dtype))
        elif enc == "axial":
            h = self.axial_tokens(h1.reshape(n, -1, kx, ky, kz).permute(0, 2, 1, 3, 4).to(self.pos.dtype), n, kx, ky, kz)
        elif enc == "vfe":                                                             # 3-D convs over the narrowed cells, x strided
            v = h1.reshape(n, -1, kx, ky, kz).to(self.c3[0].weight.dtype)
            h = self.c3(v).reshape(n, -1, ky, kz)                                      # NCDHW is faster for this shape (measured)
        elif enc == "pillar":
            h2 = F.relu(F.conv1d(h1, lin2.weight[:, :, 0, 0][:, :, None], lin2.bias))     # (n, 64, p)
            h = h2.reshape(n, -1, kx, ky, kz).amax(2)
        else:
            colmax = h1.reshape(n, -1, kx, ky, kz).amax(2, keepdim=True).expand(-1, -1, kx, -1, -1).reshape(n, -1, kx * ky * kz)
            W2 = lin2.weight[:, :, 0, 0]; b2 = torch.zeros(W2.shape[0], device=W2.device)
            sc = bn2.weight / torch.sqrt(bn2.running_var + bn2.eps); W2 = W2 * sc[:, None]; b2 = (b2 - bn2.running_mean) * sc + bn2.bias
            h2 = F.relu(F.conv1d(torch.cat([h1, colmax], 1), W2[:, :, None], b2))
            h = h2.reshape(n, -1, kx, ky, kz).amax(2)
        h = F.relu(self.stem(h.to(self.stem.weight.dtype)))
        if self.cfg.get("neck") == "aspp":
            br = [self.aspp_1x1(h)] + [F.conv2d(h, self.aspp_w, padding=r, dilation=r) for r in (1, 2, 4, 8)]
            h = F.relu(self.aspp_fuse(torch.cat(br, 1)))
        for b in self.blocks:
            h = h + F.relu(b(h))
        T, ls = self.predict_tsdf(h, tsdf.reshape(n, kx, ky, kz).to(h.dtype)); sig = torch.exp(ls).clamp(max=1.0) if ls is not None else None
        w = g[:, 1].clamp(0, max_w); d = g[:, 3]
        tau = getattr(self, "surf_tau", 0.0)
        surf = ((tsdf.abs() * mu <= tau) & (valid > 0)).reshape(n, kx, ky, kz) if tau > 0 else None   # cells crossed by the measured surface
        nc = torch.stack(ray[-9:], 1).reshape(n, 9, kx, ky, kz).to(T.dtype) if self.cfg.get("nrm_ch") else None
        offsets = getattr(self, "select", None)
        if offsets is not None:   # in-frame configuration choice: the same field read at the (depth, width) options
            w, d, reject, q = self.choose_config(T.clamp(-1, 1), w, d, g[:, 0], ax, max_w, offsets, surf=surf, h=h, sig=sig, nrm=nc,
                                                 alpha=getattr(self, "select_alpha", 3.0), top=getattr(self, "select_top", 0))
            return reject, q, T, torch.stack([w, d], 1)
        body, close, open_y = gripper_masks(w, d, ax)
        reject, q = self.heads(h if self.cfg["design"] == "A" else T.clamp(-1, 1), body, close, open_y, d, h=h, sig=sig, nrm=nc)
        if tau > 0:   # measured-surface test on the grid: a body cell crossed by the measured surface is a certain collision (replaces the observed-cloud check)
            cross = (surf.reshape(n, -1) & body_member(w, d, ax).reshape(n, -1)).any(1)
            reject = excluded(reject, cross)
        return reject, q, T, torch.stack([w, d], 1)

    def axial_tokens(self, v, n, kx, ky, kz):
        """v (n, kx, C, ky, kz) -> (n, C * kx, ky, kz): two pre-norm Transformer blocks over the kx cells of each column."""
        t = v.permute(0, 3, 4, 1, 2).reshape(n * ky * kz, kx, -1) + self.pos
        out = []
        for c in t.split(32768):                       # the column batch exceeds the attention kernel's grid limit in one go
            for i in range(2):
                q = self.ln[2 * i](c); c = c + self.attn[i](q, q, q, need_weights=False)[0]
                c = c + self.ffn[i](self.ln[2 * i + 1](c))
            out.append(c)
        return torch.cat(out).reshape(n, ky, kz, -1).permute(0, 3, 1, 2)

    def tri_planes(self, v):
        """v (n, C, kx, ky, kz) -> the yz-plane feature the neck runs on; the xz and xy features are kept for the cell decoder."""
        self._tri = (self.planes[1](v.mean(3)), self.planes[2](v.mean(4)))
        return self.planes[0](v.mean(2))

    def predict_tsdf(self, h, proj):
        """TSDF output; with tsdf_res the head predicts the correction to the observed projective TSDF (a skip from the
        observation, so only the occluded part has to be inferred)."""
        if self.cfg.get("encoder") == "triplane":                         # every cell reads the sum of its three plane features
            f_xz, f_xy = self._tri
            f = F.relu(self.tri_yz(h)[:, :, None] + self.tri_xz(f_xz)[:, :, :, None, :] + self.tri_xy(f_xy)[:, :, :, :, None])
            T, ls = self.tri_dec(f)[:, 0], None
        else:
            T, ls = self.split_tsdf(self.tsdf(h))
        return (T + proj if self.cfg.get("tsdf_res") else T), ls

    def split_tsdf(self, out):
        """TSDF head output -> (mean, log-scale or None)."""
        if self.cfg.get("tsdf_unc"):
            return out[:, :self.kx], out[:, self.kx:].clamp(-6, 3)
        return out, None

    def nrm_cells(self, img):
        """The normal channels of the input image as (n, 9, kx, ky, kz), or None without nrm_ch."""
        if not self.cfg.get("nrm_ch"):
            return None
        kx = self.kx; n, _, ky, kz = img.shape
        return img[:, (self.n_geo - 9) * kx:self.n_geo * kx].reshape(n, 9, kx, ky, kz)

    def forward(self, img, body, close, open_y, d):
        h = self.backbone(img)
        T, ls = self.predict_tsdf(h, img[:, :self.kx])                            # scene TSDF / mu on every cell (+ log-scale)
        sig = torch.exp(ls).clamp(max=1.0) if ls is not None else None
        reject, q = self.heads(h if self.cfg["design"] == "A" else T.clamp(-1, 1), body, close, open_y, d, h=h, sig=sig, nrm=self.nrm_cells(img))
        return reject, q, T


def prepare(split, feature, levels, need_tsdf, tsdf_levels=("decoded",), masks=False, back=False, back_src="back", nrm=False, nrm_src="nrm", depth_src="", fc_robust=False, y_and="", extra_split="", rank=False):
    """Frames of a split on the GPU (grasps, labels, depth, the selected feature maps); TSDF labels stay on the CPU.
    tsdf_levels: label files to use (decoded -> <sample>.tsdf.npy, mid -> <sample>.tsdf_mid.npy); masks: object masks
    (cache/points_mask_<split>.pt) for the depth-offset augmentation."""
    feats = [] if feature == "none" else feature.split("+")
    raw = load_frames(split, feature)
    mk = {m["name"]: m["mask"] for m in torch.load(f"{DATA}/cache/points_mask_{split}.pt", weights_only=False)} if masks else {}
    frames = []
    for r in raw:
        name = r["name"][:-3]
        tp = tsdf_path(split, name)
        if need_tsdf and not os.path.exists(tp):
            continue
        lv = r["level"].cuda(); names = r["levels"]
        c = dict(name=name, g=r["grasps"].cuda().float(), lvl=lv, y=r["y"].cuda().float(), K=r["K"].cuda().float(), depth=r["depth"].cuda(),
                 fmaps=[r[f].cuda() for f in feats], scene=r["scene_int"], max_w=MAX_W[r["dataset"]])
        if y_and:   # clearance labels: collision-free only if the deeper pose (<name>.<y_and>.npy) is free too
            c["y"] = c["y"] * torch.from_numpy(np.load(f"{DATA}/{split}/{name}.{y_and}.npy")).cuda().float()
        qp = f"{DATA}/{split}/{name}.q.npy"
        c["q"] = torch.from_numpy(np.load(qp)).cuda() if os.path.exists(qp) else None
        if fc_robust and c["q"] is not None:
            c["q_soft"] = soft_fc_targets(c["q"], lv, names, torch.tensor(MU, device="cuda"))
        c["allowed"] = torch.isin(lv, torch.tensor([names.index(l) for l in levels if l in names], device="cuda", dtype=lv.dtype)).nonzero()[:, 0]
        if rank:   # --rank_w: the jittered copies of each decoded candidate (verifier_data: contiguous blocks, J copies per parent)
            dec = (lv == 0).nonzero()[:, 0]; n0 = len(dec); ks = []
            for l in levels:
                if l in names and l != "decoded":
                    sel = (lv == names.index(l)).nonzero()[:, 0]; J = len(sel) // max(n0, 1)
                    if J:
                        ks.append(sel[:n0 * J].reshape(n0, J))
            if ks:
                c["dec"], c["kids"] = dec, torch.cat(ks, 1)
        if extra_split:   # the tail of the inference pool (verifier_data.py --rest_only): collision labels only, drawn by --extra_frac
            ep = f"{DATA}/{extra_split}/{name}.pt"
            if os.path.exists(ep) and os.path.exists(ep[:-3] + ".lbl.npy"):
                e = torch.load(ep, weights_only=False); ge = e["grasps"].cuda().float(); ne = len(ge)
                c["extra_idx"] = torch.arange(len(c["g"]), len(c["g"]) + ne, device="cuda")
                c["g"] = torch.cat([c["g"], ge]); c["y"] = torch.cat([c["y"], torch.from_numpy(np.load(ep[:-3] + ".lbl.npy")).cuda().float()])
                c["lvl"] = torch.cat([lv, torch.full((ne,), len(names), device="cuda", dtype=lv.dtype)]); lv = c["lvl"]
                if c["q"] is not None:
                    c["q"] = torch.cat([c["q"], torch.full((ne,), -2.0, device="cuda", dtype=c["q"].dtype)])
                if "q_soft" in c:
                    c["q_soft"] = torch.cat([c["q_soft"], torch.zeros(ne, c["q_soft"].shape[1], device="cuda")])
        row = torch.full((len(lv),), -1, dtype=torch.long, device="cuda"); parts = []
        for l in tsdf_levels:                                                 # rows of each labelled level, concatenated
            f = tsdf_path(split, name, l)
            if not os.path.exists(f):
                continue
            sel = (lv == names.index(l)).nonzero()[:, 0]
            t = torch.from_numpy(np.load(f)); assert len(t) == len(sel), (name, l, t.shape, len(sel))
            row[sel] = torch.arange(len(t), device="cuda") + sum(len(x) for x in parts); parts.append(t)
        c["t_row"] = row
        c["sensor_flag"] = 1.0 if "kinect" in split else 0.0                  # sensor type of a test split (used with --sensor_flag)
        if parts:
            c["tsdf"] = torch.cat(parts)                                       # (n_labelled, kx ky kz) int8 mm, CPU
        if masks:
            c["mask"] = mk[r["name"]].cuda()        # instance ids (0 = background); older caches are 0/1
        if back:   # per-pixel back-surface depth (scripts/verifier_backdepth_labels.py); frames without it are skipped
            thk = back_src == "backthk"                        # oracle: the CAD thickness hung on the SENSOR surface (no front correction)
            bp = f"{DATA}/{split}/{name}.{'back' if thk else back_src}.npy"      # back: CAD labels; backpred: scripts/backdepth_head.py dump
            if not os.path.exists(bp) or (thk and not os.path.exists(f"{DATA}/{split}/{name}.front.npy")):
                continue
            c["back"] = torch.from_numpy(np.load(bp)).cuda()
            if thk:
                Fd = torch.from_numpy(np.load(f"{DATA}/{split}/{name}.front.npy")[0]).cuda().float(); d0 = c["depth"].float()
                ok = (c["back"][0] > 0) & (Fd > 0) & (d0 > 0.1)
                c["back"] = torch.where(ok[None], d0[None] + (c["back"].float() - Fd[None]), torch.zeros_like(c["back"].float())).to(c["back"].dtype)
        if depth_src:   # oracle: the CAD visible surface (<sample>.<depth_src>.npy, e.g. front) replaces the sensor depth on object pixels
            fp = f"{DATA}/{split}/{name}.{depth_src}.npy"
            if not os.path.exists(fp):
                continue
            Fd = torch.from_numpy(np.load(fp)[0]).cuda().to(c["depth"].dtype)
            c["depth"] = torch.where(Fd > 0, Fd, c["depth"])
        if nrm:    # per-pixel CAD surface normals (verifier_backdepth_labels.py --normals); frames without them are skipped
            npth = f"{DATA}/{split}/{name}.{nrm_src}.npy"          # nrm: CAD labels; nrmpred: backdepth_head.py --target normals dump
            if not os.path.exists(npth):
                continue
            c["nrm"] = torch.from_numpy(np.load(npth)).cuda()
        frames.append(c)
    del raw
    if back:
        print(f"{split}: {len(frames)} frames with back-surface labels", flush=True)
    return frames


def scale_depth(c, a, k):
    """Depth for a similarity copy at scale k (--scale_copies): the recipe's object shift and ToF noise redrawn with amplitudes / k
    (probabilities unchanged), so that in the copy's cell units the sensor matches the recipe."""
    depth = c["depth"]
    if getattr(a, "obj_shift_max", 0) > 0 and "mask" in c and random.random() < a.depth_shift_p:
        depth = depth + random.uniform(0.0, a.obj_shift_max) / 100.0 / k * ((c["mask"] > 0) & (depth > 0.1))
    np_ = getattr(a, "noise_p", -1.0); np_ = a.depth_shift_p if np_ < 0 else np_
    if getattr(a, "noise_model", "") == "kinect" and random.random() < np_:
        s = getattr(a, "noise_scale", 1.0)
        depth = kinect_noise(depth, (c["mask"] > 0) if c.get("mask") is not None else None, sigma_a=0.004 * s / k, edge_thr=0.02 / k,
                             edge_p=min(0.5 * s, 1.0), hole_p=min(0.005 * s, 0.05), bias_max=getattr(a, "noise_bias_max", 0.008) * s / k)
    return depth


def batch(model, frames, a, ax, P, per_frame=None, idxs=None):
    """Inputs of a set of candidates across frames; returns the tensors the losses need."""
    kx, ky, kz = (len(x) for x in ax)
    out = {k: [] for k in ("img", "body", "close", "open", "d", "y", "q", "qs", "tt", "has", "t_pf", "t_q", "grp", "img_s", "tt_s")}
    goff = 0
    for i, c in enumerate(frames):
        grp = None
        if idxs is not None:
            idx = idxs[i]
        elif getattr(a, "rank_w", 0) > 0 and "kids" in c:   # groups: one decoded candidate and rank_group - 1 of its jittered copies
            kids = c["kids"]; G_ = 1 + min(a.rank_group - 1, kids.shape[1]); ng = max(per_frame // G_, 1)
            par = torch.randint(kids.shape[0], (ng,), device="cuda")
            sub = torch.argsort(torch.rand(ng, kids.shape[1], device="cuda"), 1)[:, :G_ - 1]
            idx = torch.cat([c["dec"][par][:, None], kids[par[:, None], sub]], 1).reshape(-1)
            grp = torch.arange(ng, device="cuda").repeat_interleave(G_) + goff; goff += ng
        elif "extra_idx" in c and getattr(a, "extra_frac", 0.0) > 0:      # part of the batch from the pool tail (--extra_split)
            ne = int(round(per_frame * a.extra_frac))
            idx = torch.cat([c["allowed"][torch.randint(len(c["allowed"]), (per_frame - ne,), device="cuda")],
                             c["extra_idx"][torch.randint(len(c["extra_idx"]), (ne,), device="cuda")]])
        else:
            idx = c["allowed"][torch.randint(len(c["allowed"]), (per_frame,), device="cuda")]
        g = c["g"][idx]; w = g[:, 1].clamp(0, c["max_w"]); d = g[:, 3]
        depth, fm = c["depth"], c["fmaps"]
        tof = random.random() < a.depth_shift_p if (idxs is None and getattr(a, "tof_joint", False)) else None   # one draw: shift AND noise, or neither
        if idxs is None and "pdepth" in c and random.random() < getattr(a, "pre_aug_p", 0.0):
            # consistent perturbation: depth AND detector features from the same perturbed-depth forward pass (--pre_aug)
            depth, fm = c["pdepth"], c["pfmaps"]
        if idxs is None and getattr(a, "inst_shift_max", 0) > 0 and "mask" in c and (tof if tof is not None else random.random() < a.depth_shift_p):
            # per-instance sensor-residual randomisation: each object's depth is offset independently, as the measured
            # CAD-to-sensor residual is produced per object (material, incidence angle), not per frame
            mk = c["mask"]
            k = int(mk.max().item())
            if k:
                off = torch.rand(k + 1, device=depth.device) * (a.inst_shift_max / 100.0)
                off[0] = 0.0
                depth = depth + off[mk.long()] * (depth > 0.1)
        elif idxs is None and getattr(a, "obj_shift_sd", 0) > 0 and "mask" in c and random.random() < a.depth_shift_p:
            # sensor-residual randomisation: the verifier is supervised with CAD geometry but reads sensor depth, so the
            # per-object depth is offset by the measured scale of the sensor-to-CAD surface residual (zero mean, sd fitted
            # on the training split: 12.3 mm on RealSense, 13.8 mm on Kinect)
            depth = depth + random.gauss(0.0, a.obj_shift_sd) * ((c["mask"] > 0) & (depth > 0.1))
        elif idxs is None and getattr(a, "obj_shift_max", 0) > 0 and "mask" in c and (tof if tof is not None else random.random() < a.depth_shift_p):
            depth = depth + random.uniform(0.0, a.obj_shift_max) / 100.0 * ((c["mask"] > 0) & (depth > 0.1))   # one offset for the whole frame
        if idxs is None and getattr(a, "noise_model", "") == "kinect":   # the sensor noise may fire at its own rate
            np_ = getattr(a, "noise_p", -1.0)
            np_ = a.depth_shift_p if np_ < 0 else np_
        else:
            np_ = -1.0
        if idxs is None and getattr(a, "noise_model", "") == "kinect" and (tof if tof is not None else random.random() < np_):
            k = getattr(a, "noise_scale", 1.0)
            depth = kinect_noise(depth, (c.get("mask") > 0) if c.get("mask") is not None else None, sigma_a=0.004 * k, edge_p=min(0.5 * k, 1.0), hole_p=min(0.005 * k, 0.05), bias_max=getattr(a, "noise_bias_max", 0.008) * k)
        out["img"].append(grid_inputs(g, c["K"], depth, fm, model.proj, a.mu, ax, P, feat_cells=model.cfg.get("feat_cells", False), occ_ch=model.cfg.get("occ_ch", False),
                                      back=c.get("back") if model.cfg.get("back_ch") else None, nrm=c.get("nrm") if model.cfg.get("nrm_ch") else None,
                                      flag=max(float(bool(tof)) if tof is not None else 0.0, c.get("sensor_flag", 0.0)) if model.cfg.get("sensor_flag") else None))   # ToF: simulated or the real sensor
        body, close, open_y = gripper_masks(w, d, ax)
        out["body"].append(body); out["close"].append(close); out["open"].append(open_y); out["d"].append(d)
        out["y"].append(c["y"][idx]); out["q"].append(c["q"][idx] if c["q"] is not None else torch.full((len(idx),), -2.0, device="cuda"))
        if "q_soft" in c:
            out["qs"].append(c["q_soft"][idx])
        if "t_pf" in c:    # privileged teacher's soft verdicts (--kd_teacher); NaN where the teacher has none
            out["t_pf"].append(c["t_pf"][idx]); out["t_q"].append(c["t_q"][idx])
        r = c["t_row"][idx]; has = r >= 0
        if "tsdf" in c and has.any():
            tt = c["tsdf"][r[has].cpu()].cuda().float().reshape(-1, kx, ky, kz) / 1000 / a.mu
            out["tt"].append(tt.abs().clamp(max=1) if model.cfg["udf"] else tt.clamp(-1, 1))
        else:
            has = torch.zeros_like(has)
        out["has"].append(has)
        if idxs is None and getattr(a, "scale_ks", None) and "tsdf_s" in c:
            # --scale_copies: the same candidates read on the lattice shrunk by k about the grasp centre (cells 5 / k mm), so every
            # object looks k times larger in cell units; projective TSDF truncated at mu / k and labels in the same virtual units
            for sk in a.scale_ks:
                ts = c["tsdf_s"].get(sk)
                if ts is None:
                    continue
                hs = (r >= 0) & (r < len(ts))                                   # rows with scaled labels (decoded candidates)
                if not hs.any():
                    continue
                out["img_s"].append(grid_inputs(g[hs], c["K"], scale_depth(c, a, sk), c["fmaps"], model.proj, a.mu / sk, ax, P / sk,
                                                feat_cells=model.cfg.get("feat_cells", False), occ_ch=model.cfg.get("occ_ch", False)))
                out["tt_s"].append((ts[r[hs].cpu()].cuda().float().reshape(-1, kx, ky, kz) / 1000 / (a.mu / sk)).clamp(-1, 1))
        out["grp"].append(grp if grp is not None else torch.full((len(idx),), -1, dtype=torch.long, device="cuda"))
    return {k: torch.cat(v) if v else None for k, v in out.items()}


def soft_fc_targets(q, lv, names, mu_t, use=("low", "mid")):
    """(n, 6) robust force-closure targets: for each decoded candidate, the mean of the six threshold labels over itself and its
    collision-free jittered copies of the levels in `use` (verifier_data: contiguous blocks, J copies per parent); every copy
    receives its parent's target. Falls back to the hard label where no free sample exists."""
    n = len(q); n0 = int((lv == 0).sum()); dev = q.device
    parent = torch.arange(n, device=dev)
    for li in range(1, len(names)):
        idx = (lv == li).nonzero()[:, 0]
        if len(idx):
            parent[idx] = (idx - idx[0]) // max(len(idx) // n0, 1)
    t = fc_targets(q, mu_t, "cls")
    keep = lv == 0
    for nm in use:
        if nm in names:
            keep |= lv == names.index(nm)
    w = ((q > 0) & keep).float()
    num = torch.zeros(n0, t.shape[1], device=dev).index_add_(0, parent, t * w[:, None])
    den = torch.zeros(n0, device=dev).index_add_(0, parent, w)
    soft = torch.where(den[:, None] > 0, num / den.clamp(min=1)[:, None], t[:n0])
    return soft[parent]


def fc_targets(q, mu_t, kind):
    """cls: success at each threshold (n, 6); reg: the detector's GT score 1.1 - mu_min, 0 without force closure (n, 1)."""
    if kind == "cls":
        return ((q[:, None] > 0) & (q[:, None] <= mu_t[None])).float()
    return torch.where((q > 0) & (q <= 1.2), 1.1 - q, torch.zeros_like(q))[:, None]


# positive rate of the six force-closure thresholds among collision-free candidates, measured on the training split
# (0.2 -> 38.2 %, 1.2 -> 89.7 %): without balancing, the loose thresholds carry most of the loss although they are
# nearly always positive, and the strict end -- where the benchmark separates methods -- is under-trained.
FC_POS = (0.382, 0.652, 0.780, 0.844, 0.878, 0.897)


def fc_loss(pred, tgt, kind, balance=0.0):
    if kind != "cls":
        return (pred - tgt).abs().mean()
    if balance <= 0:
        return F.binary_cross_entropy_with_logits(pred, tgt)
    r = torch.tensor(FC_POS, device=pred.device)                       # per-threshold weight ~ 1 / (entropy of the prior)
    w = (r * (1 - r)); w = (w / w.mean()) ** balance                    # balance=1: proportional to the label variance
    return F.binary_cross_entropy_with_logits(pred, tgt, weight=w.expand_as(tgt))


def fc_score(q, kind):
    """Expected benchmark score of the force-closure output: mean of the six probabilities, or the clipped regressed score."""
    return torch.sigmoid(q).mean(1) if kind == "cls" else q[:, 0].clamp(min=0)


def losses(model, b, a, mu_t):
    h = model.backbone(b["img"]); T, ls = model.predict_tsdf(h, b["img"][:, :model.kx])
    sig = torch.exp(ls).clamp(max=1.0) if ls is not None else None
    nc = model.nrm_cells(b["img"])
    Th = T.detach() if getattr(a, "heads_detach", False) else T          # --heads_detach: the field learns only from its CAD target
    reject, q = model.heads(h if model.cfg["design"] == "A" else Th.clamp(-1, 1), b["body"], b["close"], b["open"], b["d"], h=h, sig=sig, nrm=nc)
    L = {"col": F.binary_cross_entropy_with_logits(reject, 1 - b["y"])}
    free = b["q"] > 0                                                     # collision-free and labelled (9 = no force closure)
    yfc = b["qs"] if b.get("qs") is not None and model.cfg["fc_target"] == "cls" else fc_targets(b["q"], mu_t, model.cfg["fc_target"])   # --fc_robust: soft targets
    if free.any():
        L["fc"] = fc_loss(q[free], yfc[free], model.cfg["fc_target"], getattr(a, "fc_balance", 0.0))
    if getattr(a, "rank_w", 0) > 0 and b.get("grp") is not None:
        # the test-time choice compares configurations of one candidate by p^alpha q: rank each candidate's jittered copies the
        # same way (pairs whose benchmark score differs; colliding or no force closure = 0, else the share of thresholds passed)
        gid = b["grp"]; lab = (gid >= 0) & ((b["q"] > 0) | (b["y"] < 0.5))
        t = b["y"] * torch.where(b["q"] > 0, fc_targets(b["q"], mu_t, "cls").mean(1), torch.zeros_like(b["y"]))
        v = getattr(a, "rank_alpha", 3.0) * F.logsigmoid(-reject) + torch.log(fc_score(q, model.cfg["fc_target"]).clamp(min=1e-6))
        pair = (gid[:, None] == gid[None]) & lab[:, None] & lab[None] & (t[:, None] > t[None] + 0.05)
        if pair.any():
            L["rank"] = a.rank_w * F.softplus(v[None] - v[:, None])[pair].mean()
    has = b["has"]
    if has.any():
        err = (T[has] - b["tt"]).abs()
        w_i, w_o = getattr(a, "tsdf_w_info", 0.0), getattr(a, "tsdf_w_occ", 0.0)
        if w_i or w_o:      # visibility-aware weighting: saturated cells carry no shape, occluded cells carry the hard part
            kx = model.kx
            info = b["tt"].abs() < 1                                         # target inside the truncation band
            occ = (b["img"][has][:, :kx] <= -1) & (b["img"][has][:, kx:2 * kx] > 0)   # >= mu behind the measured surface
            wt = 1 + w_i * info.float() + w_o * (info & occ).float()
            wt = wt / wt.mean()                                              # keep the loss scale, change only its distribution
            err = err * wt
        wb = getattr(a, "tsdf_w_body", 0.0)
        if wb:              # contact-region weighting: the heads read only the gripper body and closing cells, so the field matters there
            wt = 1 + wb * (b["body"][has] + b["close"][has]).clamp(max=1)
            err = err * (wt / wt.mean())
        gw = getattr(a, "tsdf_grad_w", 0.0)
        if gw:   # the force-closure verdict is an angle between the contact normal and the closing axis, and the normal
            Tp, Tt = T[has], b["tt"]                                         # is the field's gradient, which an L1 on
            L["tsdf_g"] = a.tsdf_weight * gw * sum(                          # the field's value does not constrain
                (Tp.diff(dim=d) - Tt.diff(dim=d)).abs().mean() for d in (1, 2, 3)) / 3
        if ls is None:
            L["tsdf"] = a.tsdf_weight * err.mean()
        else:                                                                # Laplace NLL: |err| / scale + log scale
            L["tsdf"] = a.tsdf_weight * (err * torch.exp(-ls[has]) + ls[has]).mean()
        if model.cfg["design"] == "B":                                     # the heads also learn from the true TSDF
            r_t, q_t, = model.heads(b["tt"], b["body"][has], b["close"][has], b["open"][has], b["d"][has], h=h[has], nrm=nc[has] if nc is not None else None)
            L["col_true"] = F.binary_cross_entropy_with_logits(r_t, 1 - b["y"][has])
            fh = free[has]
            if fh.any():
                L["fc_true"] = fc_loss(q_t[fh], yfc[has][fh], model.cfg["fc_target"])
    if b.get("img_s") is not None:                                         # --scale_copies: field loss only, heads untouched
        bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        mom = [m.momentum for m in bns]
        for m in bns:                                                        # the copies do not move the running statistics,
            m.momentum = 0.0                                                 # which inference applies to real inputs only
        h_s = model.backbone(b["img_s"]); T_s, ls_s = model.predict_tsdf(h_s, b["img_s"][:, :model.kx])
        for m, v in zip(bns, mom):
            m.momentum = v
        e_s = (T_s - b["tt_s"]).abs()
        n_m = int(has.sum()); n_s = len(T_s) * getattr(a, "scale_w", 1.0)
        f_m = n_m / (n_m + n_s)                                              # pooled field loss: the recipe's total field weight
        if "tsdf" in L:                                                      # is shared by the real and the copied candidates
            L["tsdf"] = L["tsdf"] * f_m
        L["tsdf_s"] = a.tsdf_weight * (1 - f_m) * (e_s.mean() if ls_s is None else (e_s * torch.exp(-ls_s) + ls_s).mean())
    if b.get("t_pf") is not None and getattr(a, "kd_w", 0) > 0:          # learning with privileged information: match the
        ok = ~torch.isnan(b["t_pf"])                                        # teacher that saw the true back surface
        if ok.any():
            L["kd_col"] = a.kd_w * F.binary_cross_entropy_with_logits(reject[ok], 1 - b["t_pf"][ok])
            fk = ok & free
            if fk.any():
                L["kd_fc"] = a.kd_w * F.binary_cross_entropy_with_logits(q[fk], b["t_q"][fk])
    return L


def train(a):
    torch.manual_seed(a.seed); random.seed(a.seed); set_grid(a.grid)
    ax = cell_axes("cuda"); P = cell_centres("cuda")
    frames = prepare(a.train_split, a.feature, a.levels.split(","), need_tsdf=True, tsdf_levels=a.tsdf_levels.split(","), masks=a.obj_shift_max > 0 or a.obj_shift_sd > 0 or a.inst_shift_max > 0 or a.noise_model == "kinect", back=a.back_ch, back_src=a.back_src, nrm=a.nrm_ch, nrm_src=a.nrm_src, depth_src=a.depth_src, fc_robust=a.fc_robust, y_and=a.y_and, extra_split=a.extra_split, rank=a.rank_w > 0)
    if a.kd_teacher:   # the teacher's predictions on this training split (verifier_grid.py predict --tag TEACHER --split TRAIN_SPLIT)
        T = torch.load(f"{MODELS}/{a.kd_teacher}_{a.train_split}_pred.pt", weights_only=False)
        off = np.cumsum([0] + T["counts"]); at = {nm: i for i, nm in enumerate(T["names"])}
        for c in frames:
            i = at.get(c["name"])
            if i is not None and T["counts"][i] == len(c["g"]):
                c["t_pf"] = T["pred"][off[i]:off[i + 1]].float().cuda(); c["t_q"] = T["qprob"][off[i]:off[i + 1]].float().cuda()
            else:
                c["t_pf"] = torch.full((len(c["g"]),), float("nan"), device="cuda"); c["t_q"] = torch.full((len(c["g"]), len(MU)), float("nan"), device="cuda")
        print(f"teacher {a.kd_teacher}: {sum(int(not torch.isnan(c['t_pf'][0])) for c in frames)} of {len(frames)} frames", flush=True)
    if a.pre_aug:   # perturbed-depth copies with their own detector features (scratchpad/pert_tree.py dump), matched by frame name
        pert = {e["name"][:-3]: e for e in torch.load(a.pre_aug, weights_only=False)}
        feats = a.feature.split("+"); n_p = 0
        for c in frames:
            e = pert.get(c["name"])
            if e is not None:
                c["pdepth"] = e["depth"].cuda().to(c["depth"].dtype); c["pfmaps"] = [e[f].cuda() for f in feats]; n_p += 1
        print(f"pre_aug: {n_p} of {len(frames)} frames have a perturbed copy (p = {a.pre_aug_p})", flush=True)
    a.scale_ks = [float(x) for x in a.scale_copies.split(",") if x] if a.scale_copies else []
    if a.scale_ks:   # labels on the shrunk lattice for the decoded rows (the first rows of c["tsdf"])
        assert not (a.inst_shift_max or a.obj_shift_sd or a.tof_joint or a.pre_aug or a.back_ch or a.nrm_ch or a.sensor_flag or a.udf or a.occ_ch), "scale copies support the recipe's augmentation only"
        assert a.tsdf_levels.split(",")[0] == "decoded", "scaled labels cover the decoded rows, which must come first"
        n_s = 0
        for c in frames:
            c["tsdf_s"] = {}
            n_dec = int((c["lvl"] == 0).sum())
            for sk in a.scale_ks:
                f = f"{DATA}/{a.train_split}/{c['name']}.tsdf_g5s{round(sk * 100)}.npy"
                if os.path.exists(f):
                    t = torch.from_numpy(np.load(f)); assert len(t) == n_dec, (c["name"], len(t), n_dec)
                    c["tsdf_s"][sk] = t; n_s += 1
        print(f"scale copies {a.scale_ks}: {n_s} label files for {len(frames)} frames", flush=True)
    if a.keep_every > 1:   # data-scaling probe: of the 16 views per scene (every 16th annotation), keep every k-th
        frames = [c for c in frames if (int(c["name"].split("_")[-1]) // 16) % a.keep_every == 0]
        print(f"keep_every {a.keep_every}: {len(frames)} frames", flush=True)
    tr = [c for c in frames if c["scene"] < a.holdout]; ho = [c for c in frames if c["scene"] >= a.holdout]
    c_maps = [m.shape[0] for m in frames[0]["fmaps"]]
    model = GridVerifier(c_maps, design=a.design, hidden=a.hidden, mu=a.mu, fc_target=a.fc_target, udf=a.udf, fusion=a.fusion, feat_cells=a.feat_cells, heads_hT=a.heads_hT, dilation=not a.no_dilation, ordinal=a.ordinal, tsdf_unc=a.tsdf_unc, encoder=a.encoder, neck=a.neck, c_cell=a.c_cell, tsdf_res=a.tsdf_res, occ_ch=a.occ_ch, n_blocks=a.n_blocks, xyz_coord=a.xyz_coord, c3_early_stride=a.c3_early_stride, fc_flips=a.fc_flips, mirror_geo=a.mirror_geo, back_ch=a.back_ch, nrm_ch=a.nrm_ch, back_unc=a.back_unc, sensor_flag=a.sensor_flag, fc_zband=a.fc_zband, fc_no_open=a.fc_no_open, fc_field_only=a.fc_field_only).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    mu_t = torch.tensor(MU, device="cuda")
    print(f"{len(tr)} train frames, {len(ho)} holdout frames, feature {a.feature} {c_maps}, design {a.design}, "
          f"{sum(p.numel() for p in model.parameters())} params", flush=True)
    t0 = time.time(); run = {}
    swa, n_swa = None, 0                                                  # dense weight averaging over the late phase
    for step in range(a.steps):
        b = batch(model, random.sample(tr, a.frames), a, ax, P, per_frame=a.per_frame)
        L = losses(model, b, a, mu_t)
        loss = sum(L.values())
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        for k, v in L.items():
            run[k] = 0.98 * run.get(k, v.item()) + 0.02 * v.item()
        if a.swa_start and step + 1 >= int(a.swa_start * a.steps) and (step + 1) % a.swa_every == 0:
            sd = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
            swa = sd if swa is None else {k: swa[k] + sd[k] for k in swa}
            n_swa += 1
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < a.steps:   # intermediate checkpoints: <tag>_s<step>
            os.makedirs(MODELS, exist_ok=True)
            torch.save(dict(state=model.state_dict(), feature=a.feature, cfg=model.cfg, step=step + 1), f"{MODELS}/{a.tag}_s{step + 1}.pt")
        if step % 200 == 0 or step == a.steps - 1:
            print(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in run.items()) + f" {time.time() - t0:.0f}s "
                  f"mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
    if swa is not None and n_swa > 1:                                     # average, then re-estimate the BN statistics
        model.load_state_dict({k: (v / n_swa).to(model.state_dict()[k].dtype) for k, v in swa.items()})
        for m_ in model.modules():
            if isinstance(m_, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m_.reset_running_stats(); m_.momentum = None
        model.train()
        with torch.no_grad():
            for _ in range(a.swa_bn):
                b = batch(model, random.sample(tr, a.frames), a, ax, P, per_frame=a.per_frame)
                model.backbone(b["img"])
        print(f"SWA: averaged {n_swa} checkpoints, BN re-estimated over {a.swa_bn} batches", flush=True)
    os.makedirs(MODELS, exist_ok=True)
    torch.save(dict(state=model.state_dict(), feature=a.feature, cfg=model.cfg), f"{MODELS}/{a.tag}.pt")
    print("RESULT", json.dumps(dict(tag=a.tag, feature=a.feature, **model.cfg, steps=a.steps, batch=a.frames * a.per_frame, train_split=a.train_split,
                                    params=sum(p.numel() for p in model.parameters()), tsdf_w_info=a.tsdf_w_info, tsdf_w_occ=a.tsdf_w_occ, train_s=round(time.time() - t0))), flush=True)


@torch.no_grad()
def predict(a):
    from sklearn.metrics import roc_auc_score
    ck = torch.load(f"{MODELS}/{a.tag}.pt", weights_only=False); set_grid(ck["cfg"].get("grid", "g5")); ck["cfg"].pop("grid", None)
    model = GridVerifier(**ck["cfg"]).cuda(); model.load_state_dict(ck["state"]); model.eval()
    a.mu = ck["cfg"]["mu"]
    ax = cell_axes("cuda"); P = cell_centres("cuda")
    frames = prepare(a.split, ck["feature"], ["decoded"], need_tsdf=False, back=ck["cfg"].get("back_ch", False), back_src=a.back_src, nrm=ck["cfg"].get("nrm_ch", False), nrm_src=a.nrm_src, depth_src=a.depth_src)
    pf, qp, pf_t, qp_t, ys, sc, err, err_near, dl, dl_t, wp, qs, dlb = [], [], [], [], [], [], [], [], [], [], [], [], []
    err_sc, err_obs = [], []                                               # scene of each TSDF-error entry, and the same error for the raw projective TSDF
    for c in frames:
        n = len(c["g"]); o = {k: [] for k in ("pf", "qp", "pf_t", "qp_t", "dl", "dl_t", "wp", "qs", "dlb")}
        for i in range(0, n, a.chunk):
            idx = torch.arange(i, min(i + a.chunk, n), device="cuda")
            b = batch(model, [c], a, ax, P, idxs=[idx])
            reject, q, T = model(b["img"], b["body"], b["close"], b["open"], b["d"])
            o["pf"].append(torch.sigmoid(-reject)); o["qp"].append(fc_score(q, model.cfg["fc_target"]))
            o["qs"].append(torch.sigmoid(q) if model.cfg["fc_target"] == "cls" else q.expand(-1, len(MU)))   # per-threshold probabilities
            has = b["has"]
            r_t = torch.full_like(reject, float("nan")); q_t = torch.full_like(reject, float("nan")); d_t = torch.full_like(reject, float("nan"))
            o["dl"].append(surface_delta(T.clamp(-1, 1), ax))                   # predicted surface offset along the approach
            o["dlb"].append(surface_delta(T.clamp(-1, 1), ax) - surface_delta(b["img"][:, :len(ax[0])], ax))   # predicted minus measured surface: sensor bias estimate
            o["wp"].append(object_extent(T.clamp(-1, 1), ax, b["d"]))            # predicted object extent between the fingers
            if has.any():
                d_t[has] = surface_delta(b["tt"], ax)
            o["dl_t"].append(d_t)
            if has.any():
                e = (T[has].clamp(-1, 1) - b["tt"]).abs() * a.mu * 1000
                err.append(e.mean().item()); err_near.append(e[b["tt"].abs() < 0.3].mean().item()); err_sc.append(c["scene"])
                eo = (b["img"][has, :len(ax[0])].clamp(-1, 1) - b["tt"]).abs() * a.mu * 1000   # what the single view alone already gives
                err_obs.append(eo.mean().item())
                if model.cfg["design"] == "B":
                    nc = model.nrm_cells(b["img"])
                    rr, qq = model.heads(b["tt"], b["body"][has], b["close"][has], b["open"][has], b["d"][has], h=model.backbone(b["img"])[has], nrm=nc[has] if nc is not None else None)
                    r_t[has] = rr; q_t[has] = fc_score(qq, model.cfg["fc_target"])
            o["pf_t"].append(torch.sigmoid(-r_t)); o["qp_t"].append(q_t)
        pf.append(torch.cat(o["pf"])); qp.append(torch.cat(o["qp"])); pf_t.append(torch.cat(o["pf_t"])); qp_t.append(torch.cat(o["qp_t"]))
        dl.append(torch.cat(o["dl"])); dl_t.append(torch.cat(o["dl_t"])); wp.append(torch.cat(o["wp"])); qs.append(torch.cat(o["qs"])); dlb.append(torch.cat(o["dlb"]))
        ys.append(c["y"]); sc.append(torch.full((n,), c["scene"], device="cuda"))
    pf, qp, pf_t, qp_t, ys, sc, dl, dl_t, wp, qs, dlb = (torch.cat(x).cpu() for x in (pf, qp, pf_t, qp_t, ys, sc, dl, dl_t, wp, qs, dlb))
    torch.save(dict(pred=pf, qpred=qp, delta=dl, delta_true=dl_t, delta_bias=dlb, wpred=wp, qprob=qs.half(),
                    names=[c["name"] for c in frames], counts=[len(c["g"]) for c in frames]), f"{MODELS}/{a.pred_tag or a.tag}_{a.split}_pred.pt")
    if model.cfg["design"] == "B" and not torch.isnan(qp_t).all():          # the heads on the true TSDF (upper bound of B)
        torch.save(dict(pred=pf_t, qpred=qp_t), f"{MODELS}/{a.pred_tag or a.tag}T_{a.split}_pred.pt")
    res = dict(tag=a.tag, split=a.split, n=len(pf), auroc=float(roc_auc_score(ys.numpy(), pf.numpy())), delta_mm=float(dl.abs().mean() * 1000), bias_mm=float(dlb.mean() * 1000),
               delta_true_mm=float(dl_t[~torch.isnan(dl_t)].abs().mean() * 1000) if (~torch.isnan(dl_t)).any() else None,
               tsdf_err_mm=float(np.mean(err)) if err else None, tsdf_err_near_mm=float(np.mean(err_near)) if err_near else None,
               tsdf_err_observed_mm=float(np.mean(err_obs)) if err_obs else None)
    for k, (s0, s1) in SPLITS.items():
        m = ((sc >= s0) & (sc < s1)).numpy()
        if m.any() and 0 < ys.numpy()[m].mean() < 1:
            res[f"auroc_{k}"] = float(roc_auc_score(ys.numpy()[m], pf.numpy()[m]))
        me = [i for i, v in enumerate(err_sc) if s0 <= v < s1]              # does the field prediction degrade on unseen objects?
        if me:
            res[f"tsdf_err_{k}"] = float(np.mean([err[i] for i in me])); res[f"tsdf_err_near_{k}"] = float(np.mean([err_near[i] for i in me]))
            res[f"tsdf_err_obs_{k}"] = float(np.mean([err_obs[i] for i in me]))
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--design", default="A", choices=["A", "B"])
    ap.add_argument("--feature", default="dpt", help="none | vfm | dpt | geo | feat, '+' joins")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--split", default="val9")
    ap.add_argument("--levels", default="decoded,mid")
    ap.add_argument("--holdout", type=int, default=90)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--per_frame", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mu", type=float, default=0.02, help="TSDF truncation (m), inputs and targets")
    ap.add_argument("--tsdf_weight", type=float, default=5.0)
    ap.add_argument("--scale_copies", default="", help="training-only similarity copies for the field loss: comma-separated scales k (e.g. 1.41421356); labels <sample>.tsdf_g5s<100k>.npy (scratchpad/scale_labels.py)")
    ap.add_argument("--scale_w", type=float, default=1.0, help="weight of the copies' TSDF loss relative to --tsdf_weight")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--pred_tag", default="", help="predict: write <pred_tag>_<split>_pred.pt instead of <tag>_...")
    ap.add_argument("--save_every", type=int, default=0, help="also save <tag>_s<step>.pt every this many steps")
    ap.add_argument("--fc_target", default="cls", choices=["cls", "reg"], help="force closure: 6 ordinal classifiers, or regression of 1.1 - mu_min (the detector's GT score)")
    ap.add_argument("--udf", action="store_true", help="predict the unsigned distance |T| instead of the signed TSDF")
    ap.add_argument("--grid", default="g5", choices=list(GRIDS), help="gripper-frame grid (labels must exist for it)")
    ap.add_argument("--fusion", default="early", choices=["early", "mid", "late"])
    ap.add_argument("--encoder", default="stack", choices=["stack", "pillar", "pillar2", "conv3d", "vfe", "triplane", "axial"], help="stack: x cells as channels; pillar/pillar2: PointPillars-style column set encoders; conv3d: 3-D convs over per-cell attributes (x order kept); vfe: VoxelNet/SECOND linear VFE + strided 3-D convs; triplane: ConvONet/EG3D orthogonal plane projections with a per-cell sum decoder")
    ap.add_argument("--c_cell", type=int, default=0, help="width of the per-cell tensor the encoder streams (0: 64 for pillar, 16 for vfe)")
    ap.add_argument("--neck", default="serial", choices=["serial", "aspp"], help="serial dilated residual blocks or a PillarNeXt-style ASPP neck")
    ap.add_argument("--feat_cells", action="store_true", help="sample the feature map at every cell (3-D) instead of once per column")
    ap.add_argument("--heads_hT", action="store_true", help="design B heads read the backbone features next to the predicted TSDF")
    ap.add_argument("--no_dilation", action="store_true")
    ap.add_argument("--xyz_coord", action="store_true", help="give every cell all three coordinates, not only the approach axis (PointPillars uses all three offsets)")
    ap.add_argument("--swa_start", type=float, default=0.0, help="start dense weight averaging at this fraction of training (0 = off)")
    ap.add_argument("--swa_every", type=int, default=100, help="weight-averaging interval in steps")
    ap.add_argument("--swa_bn", type=int, default=50, help="batches used to re-estimate BatchNorm statistics after averaging")
    ap.add_argument("--mirror_geo", action="store_true", help="each cell also reads the tsdf/valid of its mirror across the closing axis (the observed near contact constrains the occluded far one)")
    ap.add_argument("--c3_early_stride", action="store_true", help="stride the approach axis by 4 in the first 3-D conv so the wide layer runs at quarter resolution")
    ap.add_argument("--fc_balance", type=float, default=0.0, help="re-weight the six force-closure thresholds by their label variance (1.0 = full)")
    ap.add_argument("--fc_flips", type=int, default=4, choices=[1, 2, 4], help="symmetries averaged in the force-closure head")
    ap.add_argument("--tsdf_w_body", type=float, default=0.0, help="extra TSDF-loss weight on gripper body + closing cells (0 = uniform)")
    ap.add_argument("--fc_zband", type=float, default=0.0, help="force-closure head pools only the thickness rows |z| <= this (m); 0 = whole slice")
    ap.add_argument("--extra_split", default="", help="pool-tail candidates (verifier_data.py --rest_only) appended to each training frame with collision labels only")
    ap.add_argument("--extra_frac", type=float, default=0.5, help="share of each frame's batch drawn from the pool tail")
    ap.add_argument("--y_and", default="", help="clearance labels: AND the collision label with <name>.<y_and>.npy (verifier_labels.py --shift_mm)")
    ap.add_argument("--fc_no_open", action="store_true", help="the force-closure head does not read the opening mask: a parallel jaw closes until contact, so the evaluator's force closure does not depend on the opening width (the collision head judges the width)")
    ap.add_argument("--heads_detach", action="store_true", help="the collision and force-closure heads read the predicted field without back-propagating into it: the field is trained only by its CAD TSDF target (training only; inference unchanged)")
    ap.add_argument("--fc_field_only", action="store_true", help="the force-closure head reads only the three field slices at the fingertip line: graspnetAPI force closure is a function of the object surface along that line only (neither the opening nor the absolute depth)")
    ap.add_argument("--rank_w", type=float, default=0.0, help="weight of a pairwise ranking loss among each candidate's jittered copies (the test-time configuration choice ranks configurations of one candidate)")
    ap.add_argument("--rank_group", type=int, default=8, help="--rank_w: candidates per group (one decoded + rank_group - 1 copies)")
    ap.add_argument("--rank_alpha", type=float, default=3.0, help="--rank_w: exponent of p_free in the ranked score, as VERIFIER_PF_EXP")
    ap.add_argument("--fc_robust", action="store_true", help="soft force-closure targets: mean label over each candidate's collision-free low/mid jittered copies")
    ap.add_argument("--n_blocks", type=int, default=4, help="dilated residual blocks in the 2-D neck (dilations 1,2,4,8 truncated to this many)")
    ap.add_argument("--ordinal", action="store_true", help="CORAL-style ordered thresholds for the force-closure classifier")
    ap.add_argument("--tsdf_levels", default="decoded", help="candidate levels with TSDF labels to supervise (decoded[,mid])")
    ap.add_argument("--inst_shift_max", type=float, default=0.0, help="per-instance depth offset ~ U(0, max) cm (each object independently)")
    ap.add_argument("--obj_shift_sd", type=float, default=0.0, help="sensor-residual randomisation: per-object depth offset ~ N(0, sd) in metres (measured residual sd: RS 0.0123, Kinect 0.0138)")
    ap.add_argument("--obj_shift_max", type=float, default=0.0, help="augmentation: object pixels measured farther by U(0, max) cm")
    ap.add_argument("--noise_p", type=float, default=-1.0, help="probability of applying the sensor noise per frame (default: same as --depth_shift_p)")
    ap.add_argument("--noise_scale", type=float, default=1.0, help="multiplier on the kinect_noise parameters")
    ap.add_argument("--noise_model", default="", choices=["", "kinect"], help="augmentation: parametric ToF-like depth noise (kinect_noise)")
    ap.add_argument("--tsdf_unc", action="store_true", help="heteroscedastic TSDF: predict a per-cell scale (Laplace NLL); design-B heads read it")
    ap.add_argument("--depth_shift_p", type=float, default=0.5, help="share of frames per step that receive the shift")
    ap.add_argument("--back_ch", action="store_true", help="add the per-pixel back-surface depth (two exit layers + flag) as cell channels; the oracle reads CAD labels <sample>.back.npy")
    ap.add_argument("--nrm_ch", action="store_true", help="add the per-pixel surface normals at the first hit, first exit and last exit (gripper frame) as cell channels; the oracle reads CAD labels <sample>.nrm.npy")
    ap.add_argument("--depth_src", default="", help="oracle: replace the sensor depth on object pixels by <sample>.<depth_src>.npy (e.g. front = the CAD visible surface)")
    ap.add_argument("--noise_bias_max", type=float, default=0.008, help="object-surface bias term of the ToF noise model, metres (0: noise without a surface shift)")
    ap.add_argument("--sensor_flag", action="store_true", help="a constant input channel: 1 on time-of-flight frames (the tof_joint-corrupted training frames; Kinect test splits), 0 otherwise")
    ap.add_argument("--tof_joint", action="store_true", help="apply the object depth shift and the ToF noise together in one draw (p = depth_shift_p), else leave the frame clean, so the noise texture tells the verifier when surfaces sit behind their true place")
    ap.add_argument("--pre_aug", default="", help="cache of perturbed-depth copies with re-extracted detector features (pert_tree.py dump)")
    ap.add_argument("--pre_aug_p", type=float, default=0.5, help="share of training frames drawn from the perturbed copy")
    ap.add_argument("--keep_every", type=int, default=1, help="data-scaling probe: keep every k-th of the 16 training views per scene")
    ap.add_argument("--kd_teacher", default="", help="tag of a privileged teacher (e.g. the true-back-surface verifier) whose predictions on the training split are soft targets")
    ap.add_argument("--kd_w", type=float, default=1.0, help="weight of the distillation terms")
    ap.add_argument("--back_unc", action="store_true", help="the back-surface maps carry the head's Laplace scale per layer (backdepth_head.py --unc); read it as two more channels")
    ap.add_argument("--nrm_src", default="nrm", help="suffix of the normal maps the channels read: nrm (CAD labels), nrmpred, nrmoof (out-of-fold, training split) or any dump suffix")
    ap.add_argument("--back_src", default="back", help="suffix of the back-surface maps the channels read: back (CAD labels), backpred, backoof (out-of-fold, training split) or any dump suffix")
    ap.add_argument("--occ_ch", action="store_true", help="add an unsaturated 'how far behind the measured surface' cell channel (the clamped TSDF saturates it away)")
    ap.add_argument("--tsdf_res", action="store_true", help="TSDF head predicts the correction to the observed projective TSDF (skip from the observation)")
    ap.add_argument("--tsdf_grad_w", type=float, default=0.0, help="weight on matching the field's finite-difference gradient (the contact normal the force-closure head reads), relative to the value loss")
    ap.add_argument("--tsdf_w_info", type=float, default=0.0, help="extra TSDF loss weight on cells whose target is inside the truncation band (0: uniform L1)")
    ap.add_argument("--tsdf_w_occ", type=float, default=0.0, help="further extra weight on informative cells at least mu behind the measured surface")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"train": train, "predict": predict}[a.mode](a)
