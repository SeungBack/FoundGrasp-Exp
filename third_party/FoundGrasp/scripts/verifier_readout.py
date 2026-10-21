"""Geometry-only contact readout: success at the six friction thresholds from a distance field on the contact grid.

The readout sees nothing but the distance field D (n, 3, 48, 9) around the contact line, so it cannot memorise object
appearance; its structure follows the benchmark's rule (two contacts along the closing axis, judged by the local surface
orientation there) without hard-coding it: a 1-D convolution along the closing axis over the x/z profile channels finds
the contacts and their local slope, and the output is symmetrised over finger swap (y flip) and thickness flip (z flip).

  python scripts/verifier_readout.py --split val9      # scene-disjoint 3-fold test on the TRUE fields (design check)
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verifier_points import CG  # noqa: E402

MU = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


class ContactReadout(nn.Module):
    def __init__(self, d_max=20.0, hidden=32, ctx_dim=0, obs=False):
        super().__init__()
        kx, ky, kz, _, _ = CG
        self.d_max = d_max; self.ctx_dim = ctx_dim; self.obs = obs
        c_in = kx * kz + 1 + kx * kz * obs               # x/z profile channels of the distance field + the finger-position mask (+ observed-cell flags)
        self.net = nn.Sequential(nn.Conv1d(c_in, hidden, 5, padding=2), nn.ReLU(), nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.ReLU(),
                                 nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden + ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, len(MU)))

    def _once(self, D, mask, ctx=None, obs=None):
        n = D.shape[0]
        x = torch.cat([D.permute(0, 1, 3, 2).reshape(n, -1, D.shape[2]), mask[:, None]] + ([obs.permute(0, 1, 3, 2).reshape(n, -1, D.shape[2])] if obs is not None else []), 1)   # (n, C, ky)
        h = self.net(x)
        feats = [h.max(2).values, h.mean(2)] + ([ctx] if ctx is not None else [])
        return self.head(torch.cat(feats, 1))

    def forward(self, D_mm, w, ctx=None, obs=None):
        """D_mm (n, kx, ky, kz) distance in mm; w (n,) opening width (m); obs (n, kx, ky, kz) 1 where the cell's distance is
        determined by the observation (valid pixel, cell in front of or at the measured surface) -> logits (n, 6), symmetrised."""
        kx, ky, kz, sp, mg = CG
        D = (D_mm.float().clamp(0, self.d_max) / self.d_max)
        j = torch.arange(ky, device=D.device, dtype=torch.float32)[None]
        y = -(w[:, None] / 2 + mg) + (j + 0.5) / ky * (w[:, None] + 2 * mg)
        mask = (y.abs() <= w[:, None] / 2).float()                                              # inside the opening
        o = obs if obs is not None else None
        f = lambda t, dims: t.flip(dims) if t is not None else None
        outs = [self._once(D, mask, ctx, o), self._once(D.flip(2), mask.flip(1), ctx, f(o, (2,))), self._once(D.flip(3), mask, ctx, f(o, (3,))), self._once(D.flip(2).flip(3), mask.flip(1), ctx, f(o, (2, 3)))]
        return torch.stack(outs).mean(0)


def main():
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val9"); ap.add_argument("--steps", type=int, default=3000); a = ap.parse_args()
    kx, ky, kz, _, _ = CG
    files = sorted(f for f in glob.glob(f"/data/Grasp/verifier/{a.split}/*.pt") if os.path.exists(f[:-3] + ".cg.npy"))
    D, Q, CD, SC, S, W = [], [], [], [], [], []
    for f in files:
        d = torch.load(f, weights_only=False, map_location="cpu"); g = d["grasps"].numpy(); keep = d["level"].numpy() == 0
        D.append(np.load(f[:-3] + ".cg.npy").reshape(-1, kx, ky, kz)); Q.append(np.load(f[:-3] + ".q.npy")[keep]); CD.append(d["cd_keep"].numpy().astype(bool)[keep])
        SC.append(np.full(keep.sum(), int(d["scene"].split("_")[1]))); S.append(g[keep, 0]); W.append(np.clip(g[keep, 1], 0, 0.1))
    D, Q, CD, SC, S, W = map(np.concatenate, (D, Q, CD, SC, S, W))
    m = CD & (Q > 0); D, Q, SC, S, W = D[m], Q[m], SC[m], S[m], W[m]
    Dt = torch.from_numpy(D).cuda(); Wt = torch.from_numpy(W).float().cuda(); mu = torch.tensor(MU, device="cuda")
    Y = ((torch.from_numpy(Q).cuda()[:, None] > 0) & (torch.from_numpy(Q).cuda()[:, None] <= mu[None])).float()
    print(f"{len(Q)} free rule-kept candidates")
    res = {}
    for s0, s1 in [(100, 130), (130, 160), (160, 190)]:
        te = torch.from_numpy((SC >= s0) & (SC < s1)).cuda(); tri = (~te).nonzero()[:, 0]
        torch.manual_seed(0); net = ContactReadout().cuda(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        for it in range(a.steps):
            b = tri[torch.randint(len(tri), (2048,), device="cuda")]
            loss = F.binary_cross_entropy_with_logits(net(Dt[b], Wt[b]), Y[b]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(net(Dt[te][i:i + 8192], Wt[te][i:i + 8192])) for i in range(0, int(te.sum()), 8192)]).cpu().numpy()
        yt = Y[te].cpu().numpy(); st = S[te.cpu().numpy()]
        name = {100: "seen", 130: "similar", 160: "novel"}[s0]
        res[name] = {mu_: (roc_auc_score(yt[:, k], p[:, k]), roc_auc_score(yt[:, k], st), roc_auc_score(yt[:, k], st * p[:, k])) for k, mu_ in ((1, 0.4), (3, 0.8))}
        print(f"{name:8s}", {mu_: f"readout {v[0]:.3f} | detector {v[1]:.3f} | s*readout {v[2]:.3f}" for mu_, v in res[name].items()}, flush=True)


if __name__ == "__main__":
    main()
