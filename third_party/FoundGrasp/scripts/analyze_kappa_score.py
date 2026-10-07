"""
Analyze whether approach_kappa / gripper_kappa correlate with grasp_score.
Restricted to top-50 grasps per image (post NMS) — matching evaluation set.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import minimum_filter, maximum_filter

CACHE_DIR = '/home/seung/Workspaces/ViTGrasp/logs/dav3s/pred_cache/test_mini'
OUT_DIR = '/home/seung/Workspaces/ViTGrasp/logs/dav3s/analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# Match _base.yaml inference config (kappa exponents set to 0 for ranking — analyze independently)
P = dict(
    graspness_exp=0.2,
    grasp_score_exp=1.5,
    objectness_threshold=0.5,
    graspness_threshold=0.01,
    segm_erosion_k=5,
    voxel_size=0.004,
    z_max=1.5,
    top_k=50,
)

def erode(mask, k):
    return minimum_filter(mask.astype(np.float32), size=2*k+1) > 0.5

def topk_indices(d, p):
    obj = d['objectness'][0]
    grn = d['graspness'][0]
    sc  = d['grasp_score'][0]
    xyz = d['xyz']
    total = (grn ** p['graspness_exp']) * (sc ** p['grasp_score_exp'])
    fg = (obj > p['objectness_threshold'])
    if p['segm_erosion_k'] > 0:
        fg = erode(fg, p['segm_erosion_k'])
    valid = fg & (grn > p['graspness_threshold']) & (xyz[2] > 0.1) & (xyz[2] < p['z_max']) & np.isfinite(xyz[2])
    idx = np.argwhere(valid)
    if len(idx) == 0: return None
    pts = xyz[:, idx[:,0], idx[:,1]].T
    scores = total[idx[:,0], idx[:,1]]
    # voxel NMS
    vc = np.floor(pts / p['voxel_size']).astype(np.int32)
    vc -= vc.min(axis=0)
    M = vc.max(axis=0) + 1
    keys = vc[:,0]*int(M[1])*int(M[2]) + vc[:,1]*int(M[2]) + vc[:,2]
    order = np.lexsort((-scores, keys))
    _, first = np.unique(keys[order], return_index=True)
    keep = order[first]
    keep = keep[np.argsort(-scores[keep])[:p['top_k']]]
    return idx[keep]

a_kappa, g_kappa, score, graspness = [], [], [], []
files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npz')))
print(f'Found {len(files)} cache files')

for f in files:
    d = np.load(f)
    if 'approach_kappa' not in d.files or 'gripper_kappa' not in d.files: continue
    idx = topk_indices(d, P)
    if idx is None: continue
    v, u = idx[:,0], idx[:,1]
    a_kappa.append(d['approach_kappa'][0, v, u])
    g_kappa.append(d['gripper_kappa'][0, v, u])
    score.append(d['grasp_score'][0, v, u])
    graspness.append(d['graspness'][0, v, u])

a_kappa = np.concatenate(a_kappa)
g_kappa = np.concatenate(g_kappa)
score = np.concatenate(score)
graspness = np.concatenate(graspness)
print(f'Total top-{P["top_k"]} grasps across {len(files)} images: {len(score)}')

def corrs(x, y, name):
    p, _ = stats.pearsonr(x, y)
    s, _ = stats.spearmanr(x, y)
    print(f'  {name}: Pearson={p:+.4f}  Spearman={s:+.4f}')

print('\n=== Pooled correlations (top-50 only) ===')
print(' vs grasp_score:')
corrs(a_kappa, score, 'approach_kappa')
corrs(g_kappa, score, 'gripper_kappa ')
print(' vs graspness:')
corrs(a_kappa, graspness, 'approach_kappa')
corrs(g_kappa, graspness, 'gripper_kappa ')

# Per-image correlation (more meaningful — removes between-image variance)
print('\n=== Per-image correlation (median over images) ===')
n_img = len(files)
img_corrs_a, img_corrs_g = [], []
offset = 0
for f in files:
    d = np.load(f)
    if 'approach_kappa' not in d.files: continue
    idx = topk_indices(d, P)
    if idx is None or len(idx) < 5: continue
    v, u = idx[:,0], idx[:,1]
    sc = d['grasp_score'][0, v, u]
    ak = d['approach_kappa'][0, v, u]
    gk = d['gripper_kappa'][0, v, u]
    if sc.std() < 1e-6: continue
    if ak.std() > 1e-6: img_corrs_a.append(stats.spearmanr(ak, sc).correlation)
    if gk.std() > 1e-6: img_corrs_g.append(stats.spearmanr(gk, sc).correlation)

img_corrs_a = np.array(img_corrs_a)
img_corrs_g = np.array(img_corrs_g)
print(f' approach_kappa vs grasp_score (per-image Spearman): median={np.median(img_corrs_a):+.4f}  mean={img_corrs_a.mean():+.4f}  q25={np.quantile(img_corrs_a,0.25):+.4f}  q75={np.quantile(img_corrs_a,0.75):+.4f}')
print(f' gripper_kappa  vs grasp_score (per-image Spearman): median={np.median(img_corrs_g):+.4f}  mean={img_corrs_g.mean():+.4f}  q25={np.quantile(img_corrs_g,0.25):+.4f}  q75={np.quantile(img_corrs_g,0.75):+.4f}')

print('\n=== Kappa distribution on top-50 ===')
for name, k in [('approach_kappa', a_kappa), ('gripper_kappa ', g_kappa)]:
    print(f' {name}: min={k.min():.2f}  q05={np.quantile(k,0.05):.2f}  median={np.median(k):.2f}  q95={np.quantile(k,0.95):.2f}  max={k.max():.2f}  std={k.std():.2f}')

# Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0,0]
ax.hexbin(a_kappa, score, gridsize=50, cmap='viridis', mincnt=1, bins='log')
ax.set_xlabel('approach_kappa'); ax.set_ylabel('grasp_score')
ax.set_title(f'approach_kappa vs grasp_score (top-{P["top_k"]})')

ax = axes[0,1]
ax.hexbin(g_kappa, score, gridsize=50, cmap='viridis', mincnt=1, bins='log')
ax.set_xlabel('gripper_kappa'); ax.set_ylabel('grasp_score')
ax.set_title(f'gripper_kappa vs grasp_score (top-{P["top_k"]})')

ax = axes[0,2]
ax.hist(img_corrs_a, bins=50, alpha=0.6, label=f'approach (med={np.median(img_corrs_a):.2f})', color='C0')
ax.hist(img_corrs_g, bins=50, alpha=0.6, label=f'gripper (med={np.median(img_corrs_g):.2f})', color='C1')
ax.axvline(0, color='k', ls='--', alpha=0.4)
ax.set_xlabel('per-image Spearman ρ (kappa, grasp_score)')
ax.set_ylabel('# images')
ax.set_title('Per-image rank correlation distribution')
ax.legend()

ax = axes[1,0]
ax.hist(a_kappa, bins=60, alpha=0.7, label='approach_kappa', color='C0')
ax.hist(g_kappa, bins=60, alpha=0.7, label='gripper_kappa',  color='C1')
ax.set_xlabel('kappa'); ax.set_ylabel('count'); ax.legend()
ax.set_title(f'Kappa distributions on top-{P["top_k"]}')

# binned mean
def binned_mean(k, s, nb=10):
    qs = np.linspace(0, 100, nb + 1)
    edges = np.percentile(k, qs); edges[-1] += 1e-6
    idx = np.clip(np.digitize(k, edges) - 1, 0, nb - 1)
    means = np.array([s[idx == i].mean() if (idx == i).any() else np.nan for i in range(nb)])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, means

ax = axes[1,1]
ca, ma = binned_mean(a_kappa, score)
cg, mg = binned_mean(g_kappa, score)
ax.plot(ca, ma, 'o-', label='approach_kappa', color='C0')
ax.plot(cg, mg, 'o-', label='gripper_kappa',  color='C1')
ax.set_xlabel('kappa (decile-binned)'); ax.set_ylabel('mean grasp_score')
ax.legend(); ax.set_title('Mean grasp_score vs kappa decile')

ax = axes[1,2]
ax.scatter(a_kappa[:5000], g_kappa[:5000], s=4, alpha=0.3)
ax.set_xlabel('approach_kappa'); ax.set_ylabel('gripper_kappa')
ax.set_title(f'kappa cross-correlation (Spearman={stats.spearmanr(a_kappa, g_kappa).correlation:+.3f})')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'kappa_vs_score_top50.png')
plt.savefig(out_path, dpi=120)
print(f'\nSaved plot to {out_path}')
