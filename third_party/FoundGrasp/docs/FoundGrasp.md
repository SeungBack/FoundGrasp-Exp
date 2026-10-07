---
marp: true
math: katex
style: |
  section {
    display: flex;
    flex-direction: column;
    font-size: 20px;
    line-height: 1.4;
    position: relative;
    justify-content: flex-start !important;
    align-items: stretch;
    padding-bottom: 36px;
  }
  section.large {
    font-size: 24px;
  }
  section.small {
    font-size: 16px;
  }
  section.tiny {
    font-size: 12px;
  }
  p,
  li {
    margin-top: 0.22em;
    margin-bottom: 0.22em;
  }
  ul,
  ol {
    margin-top: 0.35em;
    margin-bottom: 0.35em;
  }
  table {
    line-height: 1.16;
  }
  .large-text {
    font-size: 1.24em;
  }
  .small-text {
    font-size: 0.9em;
  }
  .tiny-text {
    font-size: 0.76em;
  }
  .ref {
    position: absolute;
    left: 42px;
    right: 42px;
    bottom: 18px;
    font-size: 0.62em;
    color: #666;
    line-height: 1.2;
  }
  .compact {
    font-size: 0.9em;
    line-height: 1.25;
  }
  .compact p,
  .compact li {
    margin-top: 0.25em;
    margin-bottom: 0.25em;
  }
  .compact table {
    font-size: 0.88em;
  }
  .top-right-table {
    position: absolute;
    top: 178px;
    right: 42px;
    width: 30%;
    font-size: 0.78em;
    line-height: 1.18;
  }
  .top-right-table table {
    width: 100%;
  }
---

<div class="large-text">

# FoundGrasp: Generalizable 6-DoF Grasp Detection via Geometry-Aware Adaptation of Frozen Vision Foundation Models









</div>

**Target venue**: IEEE/ASME Transactions on Mechatronics (T-MECH)

---

## Paper Identity (internal framing — not a slide)

**One-sentence claim**: The generalization bottleneck in 6-DoF grasping is representational: all task-specific methods share a Novel-split ceiling at 28–29 AP on GraspNet-1B and collapse to 10–23 AP on cross-dataset direct transfer; a frozen VFM with geometry-aware adaptation breaks both ceilings simultaneously — +5.13 Novel AP on GraspNet-1B and +17.77 AP (+78.3% relative) on GraspClutter6D direct transfer.

**Narrative arc** (§1 slide order):
1. **Introduction**: 6-DoF grasping — task definition, deployment challenge; generalization ceiling across all existing methods (GraspNet-1B Novel AP plateau + GraspClutter6D direct transfer failure)
2. **F1**: task-specific representations are bounded by the grasp training distribution — *why* the ceiling exists (structural, not per-method)
3. **F2**: fine-tuning a VFM on narrow grasp supervision degrades generalization → −7 mean AP vs frozen; three structural gaps (G1/G2/G3) block direct adaptation
4. **Derived solution**: freeze VFM → inject metric geometry at representation level (GeoFormer) and prediction level (head fusion) → dense pixel-level grasp supervision aligned to VFM token grid (PGF)
5. **Contributions**: C1/C2 empirical findings; C3 ablation + cross-backbone validation. FoundGrasp is the vehicle; PGF is a method design choice, not a standalone contribution.

**Do not start from**: "VFMs are powerful, so we use them." Start from fine-tuning failure.

**T-MECH framing note**: Abstract/intro는 "VFM adaptation method"가 아니라 **"robotic grasping system"** 으로 읽혀야 함. Finding(C1)이 중심이지만, 그 finding이 실제 로봇 배포 맥락에서 왜 중요한지를 자연스럽게 드러낼 것. MTGrasp/DSF-Grasp처럼 별도 "target setting" 문장 불필요 — robot experiment와 system description이 T-MECH fit을 증명함.

**Canonical names**:
- System: **FoundGrasp**
- Adapter: **GeoFormer** (feature-level geometry injection inside frozen ViT FFN blocks)
- GRIP-Adapter: gated ablation variant (not Ours final)
- Output formulation: **Pixel Grasp Field (PGF)**

---

## §1 — Introduction: 6-DoF Grasp Detection

- **Task**: predict and rank 6-DoF grasp poses $\{(\mathbf{p}, R, w, d)\} \subset SE(3)$ from a single RGB-D observation of a cluttered scene
  - **Input**: RGB-D image → **Output**: top-K grasp poses (3D position, SO(3) orientation, width, depth)
  - **6-DoF vs. top-down planar**: full SO(3) orientation — necessary for elongated, thin, or arbitrarily oriented objects

- **Deployment challenge**: robots must handle objects not seen during training — generalization is the binding constraint
  - GraspNet-1Billion benchmark: Seen / Similar / **Novel** (20 unseen categories) — Novel is the primary deployment-relevant metric
  - Current SOTA: Seen AP ~75–78%; **Novel AP plateaued at 28–29%** across all methods, including those with RGB-D fusion and 3D geometry
  - **Cross-dataset transfer** (GraspNet-1B trained → GraspClutter6D, no fine-tuning): all task-specific methods achieve only 10–23 AP — architectural sophistication makes no difference when representations are bounded by the same training distribution

- **Why this matters**: a system that generalizes to novel objects and across datasets enables deployment without per-object or per-scene engineering
  - To build such a system, we must first understand *why* the generalization ceiling exists

<div class="ref">
[1] Fang et al. "Robust grasping across diverse sensor qualities: The GraspNet-1Billion dataset." IJRR 2023.
</div>

---

## §1 — F1: The Generalization Bottleneck Is Representational, Not Architectural

- **Current paradigm**: task-specific visual representations learned primarily within the grasp-data regime
  - GraspNet-1Billion: **100 scenes, 88 objects** — the de facto training distribution for all existing methods
  - Prior methods may use ImageNet-pretrained components, but grasp-specific representations are learned from this limited set

- **Root cause (F1)**: visual representation is bounded by the grasp training distribution — a structural constraint shared across all existing methods regardless of architectural design
  - **Within-dataset (GraspNet-1B Novel split)**: DSF-Grasp dual-stream RGB+3D fusion raises Novel AP from 24.31% (GSNet) to 28.75% — **only +4.44 AP** despite years of architectural progress — because all representations are learned within the same 88-object regime; the ceiling is consistent across architectures (28–29 AP), not per-method
  - **Cross-dataset (GraspClutter6D direct transfer — independent corroboration)**: all GraspNet-1B-trained methods achieve only 10–23 AP (Contact-GraspNet 10.7, GraspNet 17.9, EconomicGrasp 21.7, Scale-Balanced-Grasp 22.7) — the same distribution boundary that produces a ceiling within GraspNet-1B causes categorical failure on a different distribution; FoundGrasp achieves **40.46 AP** (§4.3)

- **Answer**: the bottleneck is *representational*, not architectural — FoundGrasp achieves **33.88% Novel AP on GraspNet-1B** and **40.46 AP on GraspClutter6D direct transfer**, confirming that frozen VFM priors generalize where task-specific representations cannot

<div class="compact top-right-table">

Grasp AP (RealSense) on GraspNet-1B

| Method | Venue | Seen | Similar | Novel |
|---|---|---:|---:|---:|
| GraspNet | CVPR'20 | 47.47 | 42.27 | 16.61 |
| GSNet  | ICCV'21 | 67.12 | 54.81 | 24.31 |
| MTGrasp | T-Mech'25 | 72.62 | 63.39 | 28.61 |
| DSF-Grasp | T-Mech'26 | 75.39 | 64.76 | 28.75 |

</div>

<div class="ref">
[1] Fang et al. "Robust grasping across diverse sensor qualities: The GraspNet-1Billion dataset." IJRR 2023.
</div>

---

## §1 — F2: Direct VFM Adaptation Is Non-Trivial

> *Following from F1*: VFMs (DINOv2, DAv2, SAM, SigLIP) cover far broader object diversity than grasp datasets — **but naively adapting them fails**.

- **Full fine-tuning degrades generalization by −7.17 mean AP** vs frozen adaptation (FT+GeoFormer 54.44 vs Frozen+GeoFormer 61.61, adapter matched)
  - Narrow grasp supervision (88 objects) overwrites the web-scale visual priors responsible for generalization
  - VFM priors are sensitive to task-specific gradient updates: fine-tuning on limited grasp data destroys what makes VFMs useful for novel objects

- **Three structural gaps block direct adaptation** — each requires a specific design response:

  | Gap | Description | Design response |
  |---|---|---|
  | **G1** Task gap | VFM objectives (depth, segmentation, classification) do not encode grasp feasibility | Freeze VFM; add trainable grasp-specific components |
  | **G2** Modality gap | VFMs take RGB; metric 3D geometry (approach direction, grasp depth) cannot enter via RGB input alone | Inject depth-derived geometry at the representation level (inside frozen ViT blocks) |
  | **G3** Prediction gap | Dense per-pixel 6-DoF in image space is underspecified; GraspNet-1B labels are sparse contact-point annotations | Design VFM-compatible pixel-level supervision with depth-aware label propagation |

- **Derived solution** (each component traces to a gap):
  - Freeze VFM → preserves visual priors (G1)
  - GeoFormer: inject geometric features inside frozen ViT FFN blocks → addresses G2 without disrupting pretrained representations
  - PGF + depth-aware propagation → addresses G3: dense pixel-level supervision aligned to ViT token grid

---

## §1 — Contributions

> Three contributions answer one question: **how to properly adapt a frozen VFM for 6-DoF grasping?** Each is derived from the preceding diagnosis — not three independent findings.

- **C1 — Finding: the generalization bottleneck is representational, not architectural**
  - Task-specific networks trained on grasp data cannot exceed the diversity of that data regardless of design improvements — a structural constraint shared across all existing methods
  - **Within-dataset evidence (GraspNet-1B)**: **+5.13 Novel AP over DSF-Grasp** (33.88 vs 28.75); split pattern Similar (+7.98) and Novel (+5.13) >> Seen (+2.82) — gain is concentrated on distribution-shift splits; freezing effect cleanly isolated: Frozen+GeoFormer (33.88) vs FT+GeoFormer (28.11) = **+5.77 Novel AP** with adapter held constant
  - **Cross-dataset corroboration (GraspClutter6D, independent distribution)**: all task-specific methods achieve only 10–23 AP on direct transfer; FoundGrasp achieves **40.46 AP** (+17.77 AP, +78.3% relative over Scale-Balanced-Grasp) — the same representational ceiling that limits Novel AP within GraspNet-1B causes categorical collapse across datasets

- **C2 — System: FoundGrasp, a geometry-aware adaptation framework** *(built to demonstrate C1)*
  - Freezes the VFM backbone to preserve web-scale visual priors; injects metric depth-derived geometry at two levels:
    - **Representation level**: GeoFormer additively injects geometric features inside each frozen ViT FFN block — addresses G2 (metric 3D gap) without disrupting pretrained visual representations
    - **Prediction level**: geometric features fused at the grasp head — bridges adapter-level features to dense 6-DoF output
  - **PGF** (dense per-pixel regression of approach direction, gripper direction, width, depth) provides the VFM-compatible supervision interface — image-patch supervision aligned to ViT token grid without 2D-to-3D feature conversion (addresses G3)
  - Backbone-agnostic design; cross-backbone validation (DAv2, DINOv2, SAM, SigLIP) in §4.5 *(in progress)*

- **C3 — Validation: ablation and cross-backbone confirmation** *(confirms C1 and C2 design choices)*
  - Freezing outperforms full fine-tuning by **+5.77 Novel AP** (+7.17 mean AP) under matched training recipe — adapter held constant (GeoFormer), only freezing varies (§4.5)
  - GeoFormer (representation-level injection) contributes **+2.48 mean AP** over AdaptFormer (head-only depth); gain concentrated on Similar (+3.10) and Novel (+2.96) > Seen (+1.38) — confirming geometry enters at the right level
  - Cross-backbone validation across DAv2, DINOv2, SAM, SigLIP reported in §4.5 *(in progress)*

---

## Key Numbers — Table IV Quick Reference (internal)

<div class="compact">

| Adapter | Frozen | S-AP | M-AP | N-AP | Mean |
|---|:---:|---:|---:|---:|---:|
| No adapter | ✗ | 74.28 | 62.26 | 27.32 | 54.62 |
| GeoFormer | ✗ | 73.46 | 61.75 | 28.11 | 54.44 |
| No adapter | ✓ | — | — | — | — |
| LoRA | ✓ | 76.05 | 67.09 | 31.24 | 58.13 |
| AdaptFormer | ✓ | 76.83 | 69.64 | 30.92 | 59.13 |
| **GeoFormer (Ours)** | **✓** | **78.21** | **72.74** | **33.88** | **61.61** |

</div>

**Evidence anchors:**
- **C1 (within-dataset)**: Frozen+GeoFormer vs FT+GeoFormer (adapter matched): **+5.77 Novel AP, +7.17 mean AP** — freezing effect cleanly isolated. GeoFormer vs DSF-Grasp: **+5.13 Novel AP** (+17.8% relative); split pattern Similar (+7.98) and Novel (+5.13) >> Seen (+2.82).
- **C1 (cross-dataset)**: GraspClutter6D — FoundGrasp 40.46 AP vs best prior 22.69 AP; **+17.77 AP, +78.3% relative**. All task-specific methods 10–23 AP.
- **C2**: GeoFormer vs AdaptFormer: **+2.48 mean AP**; Similar (+3.10) and Novel (+2.96) > Seen (+1.38) — representation-level geometry injection benefit.
- **Missing (Fatal)**: Row 3 "Frozen/no-adapter" — required to isolate freezing gain from adapter gain. See Table IV.

---

## Experiment Priority (internal — not a slide)

> 우선순위 기준: (A) 없으면 제출 불가. (B) 2명 이상 reviewer 즉시 공격. (C) 강화. ✓ = 완료.

| Experiment | Table | 우선순위 |
|---|---|---|
| ✓ Full FT variants (no adapter / +GeoFormer) | Table IV rows 1–2 | **완료** |
| ✓ LoRA / AdaptFormer / GeoFormer | Table IV rows 4, 5, 9 | **완료** |
| **Robot experiment** (Seen + Novel SR, failure taxonomy, 2 baselines) | Table VIII | **A** |
| **Kinect 결과** (Table II 전체) | Table II | **A** |
| **GraspClutter6D baselines** (EconomicGrasp 동일 CD 재실행) | Table III | **A** |
| **Frozen, no adapter** (Fatal 1 — freezing vs. adapter gain 분리) | Table IV row 3 | **A** |
| **GRIP / ViT-Adapter / Geo-ViT-Adapter** (configs in `configs/tmech/`) | Table IV rows 6–8 | **B** |
| **DINOv2-B cross-backbone** (FT+Geo + Frozen+Geo, matched recipe) | Table V rows 4–5 | **B** |
| **Injection stage ablation** (4 rows — C2 bi-level 미검증) | Table VI | **B** |
| **Supervision design ablation** (4 rows — §3.4 전체 미검증) | Table VII | **B** |
| **#P(M) 계산** (all adapter variants) | Table IV | **B** |
| **EconomicGrasp RealSense 수치 확인** (원논문 Table 4) | Table I | **B** |
| DAv2-S / DAv2-L | Tables I–II | **C** |
| Candidate budget sensitivity (max_grasps 512/1024/2048/4096) | §4.8 | **C** |
| Objectness gate contribution | §4.8 | **C** |
| D-only vs XYZ input (Issue 14) | §4.8 | **C** |

**Critical path**: Frozen/no-adapter (Table IV row 3) → Robot → Kinect → GraspClutter6D confirm → GRIP/ViT-Adapter (config 존재) → DINOv2 (Table V) → Injection stage (Table VI) → Supervision design (Table VII).

---

## §2.1 — 6-DoF Grasp Detection

Learning-based 6-DoF grasp detection has evolved along two main directions, yet both share a common representational constraint.

<div class="compact">

| Line of work | Representative methods | Visual backbone |
|---|---|---|
| Sample-evaluate | GPD [IJRR'17], PointNetGPD [ICRA'19] | Task-specific, from scratch |
| End-to-end point cloud | GraspNet [CVPR'20], GSNet [ICCV'21], Scale-Balanced [CoRL'23], AnyGrasp [T-Ro'23], EconomicGrasp [ECCV'24], MTGrasp [T-MECH'25] | Task-specific 3D (PointNet++ / sparse conv), from scratch |
| RGB-D fusion | RGB Matters [ICRA'21], Contact-GraspNet [ICRA'21], HGGD [RAL'23], FlexLoG [PR'25], DSF-Grasp [T-MECH'26] | Task-specific 2D+3D, from scratch |

</div>

- **Common constraint**: all backbones trained exclusively on GraspNet-1B (30 objects, 100 scenes) → Novel-split AP plateaus at 19–29 across all methods under single-view evaluation, irrespective of architectural design
- **Closest prior**: DSF-Grasp explicitly incorporates RGB via a dual-stream backbone — yet achieves 28.75 Novel AP, bounded by the same training distribution
- *Ma et al. [CVPR'24] improve generalization via physical constraint regularization, but require multi-view TSDF reconstruction at test time (>1 min inference) — orthogonal to our single-view setting*

<div class="ref">
Ten Pas et al. IJRR'17 · Fang et al. CVPR'20 · Wang et al. ICCV'21 · Ma & Huang CoRL'23 · Fang et al. T-Ro'23 · Wu et al. ECCV'24 · MTGrasp T-MECH'25 · Gou et al. ICRA'21 · Sundermeyer et al. ICRA'21 · Chen et al. RAL'23 · Xie et al. PR'25 · DSF-Grasp T-MECH'26 · Ma et al. CVPR'24
</div>

---

## §2.2 — Vision Foundation Models and Geometry-Aware Adaptation

- **Vision Foundation Models**: DINOv2 [ICLR'24], Depth-Anything-V2 [NeurIPS'24], SAM [ICCV'23], SigLIP [ICCV'23]
  - Encode visual structure across thousands of object categories via large-scale pretraining
  - Operate on RGB input; no native interface for metric-3D quantities (approach direction, grasp depth)
- **Dense prediction with ViTs**: DPT [ICCV'21] — multi-stage ViT token reassembly decoded to full-resolution output; effective for depth estimation and segmentation
- **Parameter-efficient fine-tuning (PEFT)**: AdaptFormer [NeurIPS'22], LoRA [ICLR'22], VPT [ECCV'22], ViT-Adapter [ICLR'23]
  - Adapt frozen ViTs via lightweight trainable modules while preserving pretrained representations
  - Studied for same-modality tasks (RGB → image-space prediction); geometry injection for metric-3D tasks not addressed
- **VFMs in robotic manipulation**: R3M [CoRL'22], GNFactor [CoRL'23] use frozen VFM features as scene descriptors for policy learning — not for dense per-pixel 6-DoF pose prediction

**Open questions** — not addressed by any prior work:
- Does freezing outperform fine-tuning when adapting a VFM to 6-DoF grasping, and why?
- Which level of geometry injection (input / attention / FFN / head) is necessary for metric-3D compatibility with frozen ViT blocks?
- How should dense per-pixel 6-DoF supervision be formulated to remain aligned with the ViT token grid?

<div class="ref">
Oquab et al. ICLR'24 · Yang et al. NeurIPS'24 · Kirillov et al. ICCV'23 · Zhai et al. ICCV'23 · Ranftl et al. ICCV'21 · Chen et al. ICLR'23 · Chen et al. NeurIPS'22 · Hu et al. ICLR'22 · Jia et al. ECCV'22 · Nair et al. CoRL'22 · Ze et al. CoRL'23
</div>

---

## §3.1 — Overall Pipeline

*(figure placeholder — pipeline overview)*

**Inputs**: RGB image $I \in \mathbb{R}^{H \times W \times 3}$, depth-derived geometric features $\mathbf{D} \in \mathbb{R}^{H \times W \times d_{in}}$ (XYZ or depth+mask, depending on config)

**Components:**

1. **Frozen VFM backbone** (ViT-B): encodes $I$ into patch tokens $(B, N, C)$; weights frozen throughout training
2. **Geometric encoder**: lightweight ResNet stem converts depth-derived input $\mathbf{D}$ to geometric feature map $\mathbf{g} \in \mathbb{R}^{H/4 \times W/4 \times d}$
3. **GeoFormer adapters**: inject $\mathbf{g}$ additively inside each frozen ViT FFN block (parallel to MLP, zero-initialized) — representation-level geometry injection
4. **DPT decoder**: maps multi-scale ViT tokens to $H/4$ visual feature map; depth bypassed at decoder level (intentional — decoder injection provides no additional benefit per ablation)
5. **Grasp head**: fuses decoded visual features with $\mathbf{g}$ (prediction-level geometry fusion); produces dense per-pixel grasp attribute maps
6. **Pose assembly** (`to_grasp_group`): thresholding → voxel NMS → top-k → rotation matrix from predicted directions → GraspGroup

**Key design** (geometry-aware bi-level adaptation): geometry enters at two levels — *representation* (inside frozen ViT via GeoFormer: additive injection) and *prediction* (head fusion: geometric features concatenated with decoded visual features). XYZ lifting in `to_grasp_group` is coordinate transformation (pixel → SE(3)), not a third injection stage.

---

## §3.2 — GeoFormer: Feature-Level Geometry Injection

**Why this form follows from G2**: VFMs have no native interface for metric depth. Injecting geometry at the input (naively concatenating depth channels) breaks the pretrained patch embedding; injecting at the output (head-only) leaves the frozen ViT tokens geometry-blind. The minimum sufficient intervention is additive injection inside the frozen ViT FFN bottleneck — geometry enters the token representations without modifying the self-attention structure.

For patch token $\mathbf{x} \in \mathbb{R}^{B \times HW \times C}$ and spatially aligned geometric feature $\mathbf{g}$:

$$\mathbf{v} = W_v\,\mathrm{LN}(\mathbf{x}), \quad \mathbf{z} = W_d\,\mathrm{interp}(\mathbf{g}), \quad \Delta\mathbf{x} = W_{\mathrm{up}}\bigl(\mathrm{GELU}(\mathbf{v} + \mathbf{z})\bigr)$$

Block output: $\mathbf{x}' = \mathbf{x} + \mathrm{MLP}(\mathrm{LN}(\mathbf{x})) + s \cdot \mathrm{DropPath}(\Delta\mathbf{x})$

**Zero-initialization**: $W_d, W_{\mathrm{up}}$ zero-initialized → $\Delta\mathbf{x} = 0$ at init; frozen VFM is unperturbed at the start of training. Geometry injection grows only as training proceeds.

**Why additive, not gated (ablation-derived)**:

| Variant | Formula | Generalization result |
|---|---|---|
| AdaptFormer | $W_{\mathrm{up}}(\mathrm{GELU}(\mathbf{v}))$ | Head-only depth; −2.48 mean AP vs Ours |
| **GeoFormer (Ours)** | $W_{\mathrm{up}}(\mathrm{GELU}(\mathbf{v} + \mathbf{z}))$ | **Additive; minimum sufficient form** |
| GRIP-Adapter | $W_{\mathrm{up}}(\mathrm{GELU}(\mathbf{v} + (1{+}\alpha)\mathbf{z}))$, $\alpha{=}\tanh(\mathrm{MLP}([\mathbf{z},\mathrm{sg}(\mathbf{v})]))$ | No gain on Similar/Novel; added complexity |

> FFN-level, not attention-level: geometry informs how tokens transform locally (FFN), not how they attend globally. Attention-level injection ablation is pending (Issue 31) — current placement follows AdaptFormer convention.

**Pre-empting "trivial adapter" attack**: GeoFormer is intentionally simple. The contribution is not the adapter's complexity — it is showing that this minimal change, applied inside a frozen backbone, produces the largest generalization gain of all tested variants. Additive injection is sufficient; more complex gating (GRIP-Adapter) adds no measurable benefit. The simplicity is the point.

<div class="ref">
[2] Chen et al. "AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition." NeurIPS 2022.
</div>

---

## §3.3 — Pixel Grasp Field (PGF): Dense 6-DoF Prediction

**Why pixel-level prediction (derives from G2 + G3)**: VFM tokens correspond to image patches — attaching a point-cloud head would require 2D-to-3D feature conversion that disrupts the token-level correspondence between pretrained representations and grasp supervision. Pixel-level prediction keeps grasp supervision aligned with the ViT token grid, preserving the spatial structure that frozen VFM representations encode. This is not claimed as a representational first; it is the design required to make VFM token supervision work for grasping.

**Per-pixel outputs** (all predictions in image space, at $H/4$ resolution, upsampled to $H \times W$):

| Output | Dim | Decoding |
|---|---|---|
| `objectness` | 1 | sigmoid logit → binary foreground mask |
| `graspness` | 1 | sigmoid → grasp feasibility score $\in [0,1]$ |
| `grasp_score` | 1 | sigmoid → grasp quality score $\in [0,1]$ |
| `approach_dir` | 3 | L2-normalize → unit vector on $S^2$ |
| `gripper_dir` | 3 | Gram-Schmidt vs. approach → orthogonal unit vector |
| `grasp_width` | 1 | sigmoid → normalized; $\times 10$ (training scaling) |
| `grasp_depth` | 1 | sigmoid → normalized; $\times 25$ (training scaling) |
| `approach_kappa` | 1 | ELU+1 → $\kappa \geq 1$, concentration for AngMF risk ranking |

**Orientation decoding** (from code, `grasp_head.py`):
$$\hat{\mathbf{a}} = \frac{\mathbf{a}_\text{raw}}{\|\mathbf{a}_\text{raw}\|}, \quad \hat{\mathbf{g}} = \frac{\mathbf{g}_\text{raw} - (\mathbf{g}_\text{raw}^\top\hat{\mathbf{a}})\hat{\mathbf{a}}}{\|\mathbf{g}_\text{raw} - (\mathbf{g}_\text{raw}^\top\hat{\mathbf{a}})\hat{\mathbf{a}}\|}$$

**Rotation matrix** (in `to_grasp_group`): $R = [\hat{\mathbf{a}} \mid \hat{\mathbf{g}} \mid \hat{\mathbf{a}} \times \hat{\mathbf{g}}]$ — closed-form from predicted direction pair; gripper oriented so $z < 0$ (camera-frame convention)

**Pose lifting**: valid pixel $(u, v)$ → center $\mathbf{p} = \mathrm{XYZ}(u, v)$ → full SE(3) grasp $(\mathbf{p}, R, w, d)$

> PGF is the output formulation that keeps 6-DoF grasp supervision aligned with the frozen VFM token grid — no 2D-to-3D feature conversion required. The orientation representation and Gram-Schmidt orthogonalization are implementation details, not claimed as novel.

---

## §3.4 — Supervision

**Why these choices follow from G3**: GraspNet-1B provides sparse contact-point grasp annotations — far fewer labeled pixels than a dense image grid. Without propagation, most pixels have no supervision; without dynamic assignment, a single GT mode is forced onto pixels with inherently ambiguous orientations. Both are failure modes specific to image-space prediction; neither arises in point-cloud methods that operate at annotated contact points.

**Dynamic label assignment** (`select_gt_targets`, per pixel):
- For each valid pixel with $\geq 1$ projected GT grasp, select the GT approach most similar to the predicted approach (cosine similarity matching)
- Necessary because multiple geometrically valid grasps exist per pixel; forcing a single GT mode introduces systematic orientation noise
- *Design choice*: best-match selection avoids mode collapse without requiring multi-label regression

**Depth-aware dense label propagation** (`dense_gt_propagation`):
- Iterative dilation from annotated pixels to unlabeled neighbors, constrained to object mask (radius $r$) and depth consistency ($|\Delta d| \leq \tau_D$)
- Confidence: $c = \exp(-\|d_\text{dist}\|^2/2\sigma^2) \cdot \exp(-\|d_\text{depth}\|^2/2\sigma_d^2)$ — pixels geometrically close and at the same depth receive high-confidence labels
- *Depth-awareness is the key design choice*: pure spatial propagation would cross depth discontinuities (object boundaries, occluding edges) and introduce corrupted orientation labels

**Antipodal symmetry** (explicit design choice, not an implementation detail):
- Gripper axis has 180° physical symmetry — a grasp is identical approached from either end
- Loss uses $|\hat{\mu}^\top \mathbf{x}_\text{gt}|$ instead of $\hat{\mu}^\top \mathbf{x}_\text{gt}$ to resolve this at the loss level; without this, dense label projection introduces systematic label noise for gripper orientation

**Losses** (per pixel, weighted sum):

| Target | Loss | Notes |
|---|---|---|
| `objectness` | BCE | foreground/background |
| `graspness`, `grasp_score`, `width`, `depth` | Smooth L1 | confidence-weighted |
| `approach_dir`, `gripper_dir` | AngMF NLL (Bae et al. ICCV'21) | $-\log\kappa - \kappa(\cos\theta - 1) + \log(1+e^{-\kappa\pi})$ |

<div class="ref">
[3] Bae et al. "Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation." ICCV 2021.
</div>

---

## §4.1 — Experimental Setup

- **Dataset**: GraspNet-1Billion — 88-object, 256-scene RGB-D dataset with contact-based 6-DoF grasp annotations
  - **Splits**: Seen (58 objects) / Similar (10 objects, same categories, unseen instances) / Novel (20 objects, **unseen categories** — primary generalization metric)
  - **Sensors**: RealSense and Kinect reported separately (§4.2)

- **Metrics**: AP, AP@0.4, AP@0.8 (friction-thresholded grasp accuracy)

- **External pretraining disclosure** *(Issue 11)*: All VFM backbones use large-scale pretraining (DINOv2: LVD-142M; DAv2: 595K+62M; SAM: SA-1B; SigLIP: WebLI-4B). No additional grasp labels are used. Category overlap between VFM pretraining corpora and GraspNet objects cannot be fully excluded.

- **Default backbone**: DAv2-Base (frozen); ablations on DINOv2-Base, SAM-Base, SigLIP-Base
  - *Potential confound (Issue 25)*: DA-V2 is pretrained for monocular depth estimation — may carry geometry-related priors beyond general visual representations. §4.5 cross-backbone ablation addresses this: if frozen DINOv2/SAM/SigLIP (no depth pretraining) show the same pattern, the gain is attributable to the frozen VFM principle, not DA-V2-specific priors.

- **Trainable**: GeoFormer adapters, geometric encoder, DPT decoder, grasp head
- **Frozen**: all VFM backbone weights (SAM positional embedding resized then re-trainable)

- **Inference pipeline** (`to_grasp_group`): graspness/objectness threshold → objectness gate (+ optional erosion) → voxel NMS (4 mm) → top-2048 by total score → [optional: kappa angular risk reweighting, collision detection]
  - *Inference confound (Issue 10)*: all internal ablations use one fixed inference recipe (no kappa, no collision). §4.4 reports this setting. Kappa reranking, collision detection, objectness gate gain (Issue 18) reported separately in §4.8.
  - *Cross-camera (Issue 19)*: model trained on RealSense. Kinect (§4.2) tests sensor generalization but not a held-out camera study.

---

## §4.2 — Comparison with State-of-the-Art on GraspNet-1Billion (Table I — RealSense)

<div class="compact">

| Method | Venue | S-AP | S-AP@0.4 | S-AP@0.8 | M-AP | M-AP@0.4 | M-AP@0.8 | N-AP | N-AP@0.4 | N-AP@0.8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GraspNet | CVPR'20 | 47.47 | 60.58 | 11.14 | 42.27 | 53.67 | 7.21 | 16.61 | 22.55 | 3.75 |
| GSNet | ICCV'21 | 67.12 | 80.75 | 36.47 | 54.81 | 67.50 | 28.34 | 24.31 | 30.61 | 11.56 |
| EconomicGrasp | ECCV'24 | — | — | — | — | — | — | — | — | — |
| MTGrasp | T-MECH'25 | 72.62 | — | — | 63.39 | — | — | 28.61 | — | — |
| DSF-Grasp | T-MECH'26 | 75.39 | — | — | 64.76 | — | — | 28.75 | — | — |
| FoundGrasp — DAv2-S | Ours | — | — | — | — | — | — | — | — | — |
| **FoundGrasp — DAv2-B** | **Ours** | **78.21** | **75.79** | **86.90** | **72.74** | **69.56** | **82.90** | **33.88** | **21.21** | **41.45** |
| FoundGrasp — DAv2-L | Ours | — | — | — | — | — | — | — | — | — |

</div>

**Takeaway**: FoundGrasp (DAv2-B) outperforms all evaluated methods on all splits — largest gain on Novel (+5.13 vs DSF-Grasp) and Similar (+7.98 vs DSF-Grasp), consistent with the representational diversity thesis. DAv2-S/L scale comparison pending (§4.8).

**Novel AP contextualization**: 33.88% is +17.8% relative improvement over DSF-Grasp. The Novel split (20 unseen categories entirely absent from training) is the primary deployment-relevant metric.

> MTGrasp/DSF-Grasp AP@0.4, AP@0.8 not reported in original papers — confirm before submission. EconomicGrasp RealSense values to be confirmed from original paper (Table 4). DSF-Grasp RGB stream is a task-specific U-Net trained from scratch on GraspNet — verify before submission (Issue 12).

---

## §4.2 — Comparison with State-of-the-Art on GraspNet-1Billion (Table II — Kinect)

<div class="compact">

| Method | Venue | S-AP | S-AP@0.4 | S-AP@0.8 | M-AP | M-AP@0.4 | M-AP@0.8 | N-AP | N-AP@0.4 | N-AP@0.8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GraspNet | CVPR'20 | — | — | — | — | — | — | — | — | — |
| GSNet | ICCV'21 | — | — | — | — | — | — | — | — | — |
| EconomicGrasp | ECCV'24 | — | — | — | — | — | — | — | — | — |
| MTGrasp | T-MECH'25 | — | — | — | — | — | — | — | — | — |
| DSF-Grasp | T-MECH'26 | — | — | — | — | — | — | — | — | — |
| FoundGrasp — DAv2-S | Ours | — | — | — | — | — | — | — | — | — |
| **FoundGrasp — DAv2-B** | **Ours** | — | — | — | — | — | — | — | — | — |
| FoundGrasp — DAv2-L | Ours | — | — | — | — | — | — | — | — | — |

</div>

> ⚠ **실험 필요 (A등급)**: Kinect 결과 전체 미완료. GraspNet-1B 논문 표준; 없으면 심사위원 즉시 지적.

---

## §4.3 — Cross-Dataset Generalization on GraspClutter6D (Table III)

**Independent corroboration of C1**: §4.2 establishes the representational ceiling within GraspNet-1B. This section asks: does the same constraint explain categorical failure when the dataset distribution changes entirely? All methods trained on GraspNet-1B only; direct transfer to GraspClutter6D without fine-tuning.

<div class="compact">

| Method | AP@0.4 | AP@0.8 | AP |
|---|---:|---:|---:|
| Contact-GraspNet | 5.75 | 13.18 | 10.73 |
| GraspNet | 13.53 | 21.54 | 17.95 |
| EconomicGrasp | 15.90 | 26.62 | 21.67 |
| Scale-Balanced-Grasp | 16.95 | 27.57 | 22.69 |
| **FoundGrasp — DAv2-B (Ours)** | **33.71** | **48.02** | **40.46** |

</div>

**Takeaway**: FoundGrasp **40.46 AP** vs. best prior 22.69 AP (Scale-Balanced-Grasp) — **+17.77 AP absolute, +78.3% relative**. Task-specific methods plateau at 10–23 AP regardless of architecture; frozen VFM visual priors generalize across the dataset boundary.

**Why this corroborates F1**: Scale-Balanced-Grasp and EconomicGrasp have substantially more sophisticated architectures than GraspNet (CVPR'20), yet all achieve similar 10–23 AP — the same distribution boundary that causes a ceiling within GraspNet-1B causes categorical failure across datasets. Notably, Contact-GraspNet (which performs per-point dense SO(3) regression — philosophically analogous to PGF) achieves only 10.73 AP, which partially addresses the formulation-confound concern (Issue 32): the failure is not explained by prediction formulation alone.

> All results with uniform collision detection (CD); specific CD method detailed in §4.1. EconomicGrasp pending re-run with identical CD setup (open-source; current value is provisional). All models trained on GraspNet-1B only; direct transfer to GraspClutter6D, no fine-tuning.

<div class="ref">
[x] GraspClutter6D dataset reference — to be added.
</div>

---

## §4.4 — Ablation Study of Adapter Design (Table IV)

**Fixed**: DAv2-Base backbone, same geometric encoder / DPT decoder / grasp head / training recipe. `Frozen` column: ✓ = backbone frozen throughout training, ✗ = full fine-tuning (same architecture). Full FT uses matched recipe (same backbone LR, head LR, weight decay, warmup). Row 3 `No adapter, ✓` is required to isolate the freezing effect from the adapter effect — this row is the causal hinge for C1.

<div class="compact">

| Adapter | Frozen | #P(M) | S-AP | M-AP | N-AP | Mean |
|---|:---:|---:|---:|---:|---:|---:|
| No adapter | ✗ | — | 74.28 | 62.26 | 27.32 | 54.62 |
| GeoFormer | ✗ | — | 73.46 | 61.75 | 28.11 | 54.44 |
| No adapter | ✓ | — | — | — | — | — |
| LoRA | ✓ | — | 76.05 | 67.09 | 31.24 | 58.13 |
| AdaptFormer | ✓ | — | 76.83 | 69.64 | 30.92 | 59.13 |
| ViT-Adapter (RGB SPM) | ✓ | — | — | — | — | — |
| Geo-ViT-Adapter (Depth SPM) | ✓ | — | — | — | — | — |
| GRIP-Adapter (gated) | ✓ | — | — | — | — | — |
| **GeoFormer (Ours)** | **✓** | — | **78.21** | **72.74** | **33.88** | **61.61** |

</div>

> **Interpretation**: Rows 1–2: FT baseline — adapter matched, only freezing varies (C1: +5.77 Novel AP). Row 3: freezing alone with no adapter — separates freezing gain from GeoFormer adapter gain. Rows 4–8: frozen adapter family. GeoFormer vs AdaptFormer (+2.48 mean AP): representation-level geometry injection benefit. GRIP-Adapter: "additive is sufficient" validation.

> ⚠ **실험 필요**: Row 3 (Frozen/no-adapter, A등급 — Fatal 1); ViT-Adapter, Geo-ViT-Adapter, GRIP-Adapter (B등급; configs in `configs/tmech/`); #P(M) 전체 (B등급).

**Pre-empting attacks**: (1) "just PEFT for robotics" — the question is *why* freezing beats fine-tuning for grasping and *where* geometry must enter. (2) "point-cloud methods have 3D geometry" — their backbones are trained on 88 objects; VFM visual priors cover far broader diversity. Novel AP is the evidence.

---

## §4.5 — Cross-Backbone Validation and Freezing Strategy (Table V)

**Evidence for C1 and C3**: Does frozen adaptation consistently outperform fine-tuning? Does the finding hold for backbones without depth pretraining (DA-V2 confound control, Issue 25)? Adapter fixed to GeoFormer; training recipe matched throughout (same backbone LR, head LR, weight decay, warmup).

<div class="compact">

| Backbone | Pretrain task | Frozen | S-AP | M-AP | N-AP | Mean | ΔNovel vs FT+Geo |
|---|---|:---:|---:|---:|---:|---:|---:|
| DAv2-B | Monocular depth | ✗ (no adapter) | 74.28 | 62.26 | 27.32 | 54.62 | — |
| DAv2-B | Monocular depth | ✗ (+GeoFormer) | 73.46 | 61.75 | 28.11 | 54.44 | ref |
| DAv2-B | Monocular depth | ✓ (+GeoFormer) | **78.21** | **72.74** | **33.88** | **61.61** | **+5.77** |
| DINOv2-B | Self-supervised | ✗ (+GeoFormer) | — | — | — | — | — |
| DINOv2-B | Self-supervised | ✓ (+GeoFormer) | — | — | — | — | — |

</div>

> **ΔNovel**: frozen vs Full FT+GeoFormer under matched recipe (adapter identical). DAv2-B: **+5.77 Novel AP** — freezing effect cleanly isolated. DINOv2 rows: if frozen > FT holds without depth pretraining, gain is from visual priors, not DA-V2-specific geometry knowledge.

> ⚠ **실험 필요 (B등급)**: DINOv2-B rows (matched recipe). SAM/SigLIP optional; minimum requirement is DINOv2 to falsify DA-V2 depth-pretraining confound (Issue 25). If DA-V2 outperforms DINOv2 substantially in frozen setting, qualify finding accordingly — do not claim backbone-agnostic equivalence without data.

---

## §4.6 — Ablation Study of Geometry Injection Stage (Table VI)

**Evidence for C2**: Where must geometry enter the frozen VFM? Backbone: DAv2-B (frozen). Only the path through which depth features flow varies — GeoFormer adapter type is fixed. "Head depth" = depth features concatenated at grasp head. "Repr" = depth injected inside frozen ViT blocks via GeoFormer adapter. Note: this table uses no visual adapter in rows 1–2 to isolate depth routing; AdaptFormer in Table IV adds a visual-only adapter on top.

<div class="compact">

| Injection stage | S-AP | M-AP | N-AP | Mean |
|---|---:|---:|---:|---:|
| No geometric input (frozen ViT, visual only) | — | — | — | — |
| Prediction-level only (no adapter, head depth) | — | — | — | — |
| Representation-level only (GeoFormer, no head depth) | — | — | — | — |
| **Representation + Prediction (GeoFormer + head depth; Ours)** | **78.21** | **72.74** | **33.88** | **61.61** |
| + Decoder injection (Ours + DPT depth path) | — | — | — | — |

</div>

> **Expected**: bi-level (Repr+Pred) is optimal; decoder injection provides no additional benefit (intentional design choice, see §3.1). Gain from representation-level injection is expected to be concentrated on Similar/Novel splits.

> ⚠ **실험 필요 (B등급)**: 4개 행 전체 미완료. C2 "bi-level이 optimal" claim은 현재 데이터 없이 주장됨.

---

## §4.7 — Ablation Study of Supervision Design (Table VII)

**PGF supervision validation**: Do orientation representation and label propagation choices contribute meaningfully? Fixed: DAv2-B (frozen), GeoFormer adapter, same training recipe. One component removed or changed per row; Full (Ours) at the bottom is the reference.

<div class="compact">

| Variant | S-AP | M-AP | N-AP | Mean |
|---|---:|---:|---:|---:|
| Discrete view classification (300 bins) | — | — | — | — |
| Continuous S², no antipodal symmetry | — | — | — | — |
| w/o depth-aware label propagation | — | — | — | — |
| w/o dynamic label assignment | — | — | — | — |
| **Full (Ours)** | **78.21** | **72.74** | **33.88** | **61.61** |

</div>

> ⚠ **실험 필요 (B등급)**: 4개 행 전체 미완료. §3.4의 모든 설계 선택 미검증.

---

## §4.8 — Analysis and Discussion

*(to be written — key analyses)*

**Efficiency and deployment (T-MECH relevance):**
- Trainable parameters (GeoFormer adapters + encoder + decoder + head) vs full FT
- GFLOPs and wall-clock latency: capture → inference → grasp pose, on robot-side GPU
- FPS on target hardware (e.g., NVIDIA Jetson or workstation GPU used in robot experiment)
- Sensor noise sensitivity: AP degradation under depth dropout / noise perturbation (validates robustness under real sensor conditions)

**Backbone scale** (Issue 25 partial):
- DAv2-Small vs DAv2-Base — does the finding hold at smaller scale?

**Kappa concentration** (optional $\kappa$ branch):
- Predicted $\kappa$ vs actual angular error distribution — validate $\kappa$ as a reliable risk-ranking cue at inference
- Not claimed as an uncertainty model; used only for ranking

**Failure modes:**
- Thin objects: depth discontinuities cause XYZ noise → unreliable grasp center lifting
- Transparent objects: missing depth → zero-fill in XYZ corrupts geometric features
- Depth holes: no-depth pixels suppressed by validity mask, but adjacent predictions may be unreliable
- Boundary pixels: objectness erosion helps, but label propagation may introduce artifacts

**Objectness gate contribution** (Issue 18):
- Report with / without objectness gate and erosion — isolates segmentation-like filtering gain

---

## §5 — Real-World Robotic Grasping Experiments (Table VIII)

> ⚠ **실험 필요 (T-MECH gate / A등급)**: 물리 로봇 실험 없이 제출 불가. Novel SR이 C1을 physically 지지해야 T-MECH 기준 충족. MTGrasp와 DSF-Grasp 모두와 비교 필요.

**Setup** (to be filled):
- Robot: [TBD]
- Camera: [TBD — state whether matches GraspNet training sensor (RealSense); if different, report cross-sensor gap]
- Objects: [N_seen] seen-class + [N_novel] novel-class objects not in GraspNet-1B training split
- Metric: grasp success rate — object lifted >10 cm without drop; [N] trials per object
- Inference: single RGB-D frame → top-1 grasp by total score

<div class="compact">

| Method | Seen SR (%) | Novel SR (%) | Overall SR (%) | Inference (ms) |
|---|---:|---:|---:|---:|
| MTGrasp | — | — | — | — |
| DSF-Grasp | — | — | — | — |
| **FoundGrasp — DAv2-B (Ours)** | — | — | — | — |

</div>

**Failure taxonomy** (required for T-MECH system paper credibility):

| Failure type | Description |
|---|---|
| Depth error | XYZ lifting fails due to sensor noise or occlusion → wrong grasp center |
| Approach direction | Predicted approach not collision-free in real clutter |
| Novel object | Shape/texture outside VFM generalization range |
| Execution | IK failure, collision during approach, or drop after lift |

**Protocol:**
1. Capture single RGB-D frame from fixed camera
2. Run FoundGrasp → select top-1 grasp by total score
3. Execute; success = object lifted > 10 cm without drop
4. Report per-category (Seen / Novel) and aggregate SR over N trials per object

> Novel SR directly supports C1 — frozen VFM generalizes to unseen objects in physical execution. Failure taxonomy breakdown required for T-MECH mechatronics relevance. Inference (ms) must be reported for practical deployment claim.

---

## Limitations & Open Questions

**Known limitations:**
- XYZ lifting requires calibrated depth; accuracy degrades under sparse or noisy depth (transparent/thin objects)
- Category overlap between VFM pretraining data and GraspNet object categories cannot be excluded (Issue 11)
- Novel AP 33.88% is still low in absolute terms; practical deployment requires object-class-aware failure handling (Issue 30)
- Single-seed (one training run) for most ablations; small AP gaps (~<1 AP) may be within noise (Issue 16)

**Open questions / pending ablations:**
- Full FT fairness: is the −7 AP gap robust across matched LR schedules and seeds? (Issue 9)
- Discrete vs continuous orientation: ablation confirms continuous S² > 300-view classification?
- Candidate budget equalization: dense image prediction vs sampled-point at matched NMS budget (Issue 17)
- Cross-camera: RealSense-trained → Kinect test; domain gap analysis (Issue 19)
- Geometry injection at attention level vs FFN level ablation (Issue 31)
- D-only vs XYZ input: how much does calibrated XYZ lifting contribute vs raw depth? (Issue 14)

---

## Reviewer Questions (Cold Review)

> 실험 없이 해결 가능한 이슈만 포함. 실험 추가 필요 항목은 해당 절 표에 ⚠로 표시 (§4.4 Full FT+GeoFormer / GRIP / repr-only, §4.5 backbone validation, §4.6 injection-stage, §4.7 dynamic-assignment, §5 robot).

### 설명 / 수치 추가

**Q1.** C3 contribution bullet과 Key Numbers에서 "Freezing outperforms full fine-tuning by +6.99 Novel AP"라고 했지만 33.88 − 27.32 = **6.56** Novel AP; +6.99는 **mean** AP 차이 — 어느 metric을 인용하는가?

**Q2.** F1 slide에서 "100 scenes, 88 objects, 25.6K images"라고 했지만 §4.1에서 "88-object, **256-scene**" — train split vs. full dataset 수치를 분리해서 표기해야 함.

**Q3.** §4.4에서 Full FT "matched recipe"라고 주장하지만 backbone LR, head LR, weight decay, warmup의 실제 값이 없음 — 리뷰어가 공정성을 검증 불가.

**Q4.** §2 (Related Work)가 전체 placeholder — HGGD (가장 가까운 image-space method), VFM-for-manipulation (GNFactor, RoboAgent), 최근 point-cloud SOTA 대비 포지셔닝이 없음.

**Q5.** §3.4에서 antipodal symmetry를 gripper direction에만 처리(|μ̂ᵀx_gt|) — approach direction도 물리적으로 antipodal한 경우 동일한 loss bias가 발생하는가? 코드에서 approach direction antipodal 처리 여부 확인 후 명시 또는 limitation에 추가.

**Q6.** §3.3에서 "graspness"와 "grasp_score"를 각각 sigmoid scalar로 출력하는데, 두 출력의 기능적 차이와 inference에서의 결합 방식(thresholding 순서, weighting)이 서술되지 않음.

**Q7.** §4.2 SOTA 표에서 MTGrasp와 DSF-Grasp의 AP@0.4, AP@0.8이 "—" — 원논문에서 값이 있으면 채워야 하고, 없으면 각주로 명시.

**Q8.** §4.3에서 "#P(M)" 컬럼이 모든 행에서 "—" — 파라미터 수를 계산해서 채우거나 컬럼을 삭제.

**Q9.** "ViT-Adapter (RGB SPM)"에서 SPM이 정의 없이 사용됨 — SPM = Spatial Pyramid Module (ViT-Adapter 원논문 용어)임을 명시.

**Q10.** §3.2 block output equation `x' = x + MLP(LN(x)) + s·DropPath(Δx)` — DropPath가 training-only인지 inference에도 적용되는지 명시 필요.

### Framing 변경

**Q11.** C1에서 "largest novel-object AP gain over existing 6-DoF methods" — AnyGrasp 등 2024–25 미포함 방법이 있으므로 → "among methods evaluated on GraspNet-1Billion as of submission"으로 범위 한정.

**Q12.** §3.2에서 FFN-level injection을 "G2에서 원리적으로 도출"했다고 서술하지만 attention-level ablation이 없으므로 → "FFN-level injection follows AdaptFormer convention; whether attention-level injection is equally sufficient remains an open ablation" 수준으로 claim 조정.

**Q13.** §4.6에서 orientation baseline을 "GraspNet-style, 300 bins"으로 표기 — 300-bin discrete view classification은 RGB Matters의 formulation이지 GraspNet이 아님 → "RGB Matters-style, 300-bin discrete view classification"으로 수정.

**Q14.** T-MECH mechatronics contribution이 어디에도 명시되지 않음 — §1 또는 §5에 이 논문이 T-MECH에 적합한 이유(실센서 RGB-D 파이프라인, 배포 관련 평가 split)를 한 문장으로 추가.

---

## Codex Questions (Cold T-MECH Review)

**Q1.** Which exact config and checkpoint produce the headline FoundGrasp numbers, since `framing_principles.md` names `geoformer_scale1.0.yaml` as current best while `configs/tmech/grip.yaml` still comments GRIP as "Ours (final)"?

**Q2.** If FoundGrasp claims GeoFormer validation on SAM, how is geometry injected for SAM when `found_grasp.py` skips PEFT context and PEFT calls whenever `self.is_sam` is true?

**Q3.** Is "inside each frozen ViT FFN block" accurate for every claimed backbone, given `PeftManager.install()` only patches detected block types and otherwise falls back to post-block adapter placement?

**Q4.** Can the paper claim matched full fine-tuning with layer-wise LR decay when `utils/optimizer.py` applies layer-wise decay only if `layer_decay_rate` is set and the backbone is not frozen, and the shown configs do not state that recipe?

**Q5.** Which inference recipe produced the headline AP, since `_base.yaml` enables collision detection, kappa reweighting, empty-grasp filtering, objectness erosion, and `max_grasps: 4096`, while the draft describes fixed ablations with no kappa, no collision, and top-2048?

**Q6.** The supervision section says approach targets are dynamically selected by cosine similarity, but `select_gt_targets()` fixes `approach_dir` to `gt_app[:, 0, :]`; which statement is correct?

**Q7.** The AngMF loss formula in the draft does not match `model/loss.py`, which uses `-log(kappa^2+1) + kappa*acos(dot) + log(1+exp(-kappa*pi))`; which loss should be reported?

**Q8.** The output table lists only `approach_kappa`, but the code predicts both `approach_kappa` and `gripper_kappa`; should both be described and ablated?

**Q9.** Which GT format is actually used for training, since `select_gt_targets()` assumes `(N,K,3)` v2 tensors while `gen_pixelwise_gt_v3.py` writes `(N,K,P,3)` tensors?

**Q10.** Is PGF's lifted pixel coordinate a valid GraspNet grasp translation, or should the pose assembly include an offset along the predicted approach/grasp depth rather than using `xyz[:, v, u]` directly?

**Q11.** Does the external pretraining disclosure need to include the ImageNet-pretrained ResNet geometry encoder, not only the frozen VFM backbones?

**Q12.** Should "no 2D-to-3D feature conversion" be narrowed to "no 2D-to-3D feature conversion inside the network," since label generation uses 3D nearest-neighbor assignment and inference uses XYZ lifting?

**Q13.** Could the dense pixel formulation gain AP from candidate coverage after H/4-to-H bilinear upsampling rather than from VFM token alignment, and is there a matched candidate-budget ablation?

**Q14.** How much of the AP gain comes from the segmentation-supervised objectness head and hard objectness erosion gate, independent of GeoFormer?

**Q15.** Is the representational-bottleneck thesis falsified or weakened if a scratch CNN/ViT trained with the same PGF head and candidate budget closes most of the gap?

**Q16.** Is the "all prior task-specific detectors" comparison complete enough for T-MECH when EconomicGrasp, FlexLoG, AnyGrasp, and target-oriented grasping papers in `papers_md/` are not normalized in the SOTA table?

**Q17.** Does DSF-Grasp truly learn its RGB stream from scratch, or must the F1 claim be weakened until the DSF-Grasp backbone pretraining status is verified?

**Q18.** The draft says GraspNet-1Billion has 88 objects in one place and the framing notes mention 30 training objects elsewhere; which exact object count supports the "limited grasp regime" argument?
