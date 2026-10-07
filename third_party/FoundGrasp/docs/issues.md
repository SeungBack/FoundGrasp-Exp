# FoundGrasp Framing Issues

Derived from review of framing_principles.md, SOTA papers (papers_md/), and codebase.

**Writing standard**: concise, clear, logical, reasonable, academic, not over-statement.
Avoid absolute claims unless directly supported by evidence or ablation numbers. (But you can ssuming that additional experimented would be conducted and got good results)

---

## Issue 1 — F2 Diagnosis Rewrite ✓

**Status**: Resolved

**Problem**: "VFMs are designed/trained for RGB tasks" — 부정확. (1) VFM은 RGB-only input이라 metric 3D geometry를 직접 처리할 수 없다는 게 핵심. (2) FoundGrasp가 VFM-agnostic framework임을 명시하지 않음.

**Fix**: F2를 두 문장으로 압축. 문제 정의 → 왜 단순 해결이 안 되는가 + 수치. Backbone-agnostic 내용은 solution 섹션으로 이동.
> VFMs operate on RGB input and have no interface for the metric 3D quantities grasping requires — approach direction, gripper orientation, and grasp depth in camera coordinates. Fine-tuning the backbone is counterproductive: full fine-tuning degrades novel-object AP by 6.99 points relative to freezing, suggesting that visual generalization priors are sensitive to task-specific gradient updates on limited grasp data.

---

## Issue 2 — Pixel Grasp Field: Formulation Claim and Novelty Boundary

**Status**: Open — PGF repositioned as method component; finding is the primary contribution.

**Core formulation**:

> We formulate 6-DoF grasp detection as dense per-pixel regression of approach direction, gripper direction, width, and depth. Each valid pixel is lifted to a full grasp pose via aligned XYZ, with orientation assembled from the predicted continuous vectors.

**Why this matters**: PGF is not claimed as the primary contribution. It is the output formulation that makes frozen VFM adaptation compatible with 6-DoF grasping — VFM tokens correspond to image patches, so pixel-level prediction keeps grasp supervision aligned with the token grid without 2D-to-3D feature conversion. The primary contribution is the finding that representation diversity is the generalization bottleneck, and that frozen VFM + geometry injection breaks it. PGF is the method design that enables this finding to be demonstrated.

### Literature Audit for PGF

**Cold assessment**: PGF is a method design choice, not a standalone contribution. The primary contribution is the finding: task-specific visual representations hit a generalization ceiling that frozen VFM breaks. PGF is the formulation that makes VFM token supervision compatible with 6-DoF grasping. Prior image-based methods predict intermediate quantities — discrete view scores (RGB Matters: 300 bins), projected keypoints (KGN), or heatmap guidance (HGGD) — and require an additional recovery step to produce a grasp pose. PGF predicts approach direction, gripper direction, width, and depth directly per pixel. This is justified as the formulation needed for VFM adaptation, not claimed as a representational first.

| Method | Image prediction | Where final 6-DoF geometry is completed | Structural difference from PGF |
|---|---|---|---|
| [GG-CNN](https://journals.sagepub.com/doi/10.1177/0278364919859066) / [GR-ConvNet](https://arxiv.org/abs/1909.04810) | Pixel-wise planar grasp quality, angle, and width. | The image output gives a top-down planar grasp. | Pixel-wise, but not arbitrary 6-DoF; approach is effectively constrained by the camera setup. |
| [DGCAN](https://arxiv.org/abs/2302.14264) | RGB-D rectangle representation with an added grasp-depth term. | Final grasp remains a planar/rectangle grasp with depth. | Adds depth, but does not predict free 3D approach and gripper directions. |
| [GraspVDN](https://link.springer.com/article/10.1007/s40747-021-00459-x) | Heatmap plus 2D opening vector from RGB. | A tailored vector-to-pose strategy converts the 2D vector into a 6-DoF grasp under approach constraints. | Image predicts a reduced 2D vector; PGF predicts 3D approach direction, gripper direction, width, and insertion depth per valid pixel. |
| [RGB Matters / RGBD-Grasp](https://arxiv.org/abs/2103.02184) | Per-location gripper orientation, represented through angle/view prediction. | Fast Analytic Searching computes gripper width and distance from depth. | Image predicts orientation only; PGF learns orientation directions, width, and insertion depth as dense outputs. |
| [Keypoint-GraspNet](https://arxiv.org/abs/2209.08752) / [KGNv2](https://arxiv.org/abs/2303.05617) | Projected gripper keypoints, with scale/pose separation in KGNv2. | PnP or keypoint-based recovery estimates SE(3). | Image predicts recovery targets; PGF predicts grasp geometry at each candidate pixel. |
| [MonoGraspNet](https://arxiv.org/abs/2209.13036) | Keypoint heatmaps and normal map from RGB. | Grasp pose is recovered from keypoints, normal cues, and depth/direction/width/angle terms. | Strong image-based 6-DoF baseline, but still a keypoint-recovery formulation rather than per-pixel grasp-geometry prediction. |
| [HGGD](https://arxiv.org/abs/2403.18546) | Grasp confidence and attribute heatmaps on the image/grid. | Heatmaps guide local point aggregation; a local point encoder and non-uniform rotation anchors generate grasps. | Closest risk. Image output guides where/how to generate grasps; PGF predicts the grasp geometry to be assembled at the selected pixel. |
| [External-attention RGB-D fusion](https://doi.org/10.1016/j.jvcir.2024.104173) | RGB features fused with PointNet++ geometry through external attention. | Final 6-DoF estimation is tied to RGB-D/point-cloud feature processing. | RGB helps the 3D grasp estimator; PGF keeps the grasp variables on the image grid. |
| [GraspFast](https://doi.org/10.1016/j.patcog.2024.111318) / [DSF-Grasp](https://doi.org/10.1109/TMECH.2026.3675307) | Reliable point selection or RGB semantic features projected onto point-cloud geometry. | Point-cloud keypoints plus view/angle/depth parameter prediction complete the grasp. | RGB supports point-cloud grasp generation; PGF predicts grasp geometry before converting candidates to `GraspGroup`. |
| [MG-Grasp](https://arxiv.org/abs/2603.16270) | Sparse RGB observations support metric point-cloud reconstruction. | A reconstructed 3D representation is used before grasp generation. | Reconstruction-first pipeline; PGF predicts grasp geometry directly from the single RGB-D observation. |
| [Contact-GraspNet](https://arxiv.org/abs/2103.14127) | No image-space grasp output; predictions are rooted at observed 3D contact points. | Each observed 3D point predicts contact success, rotation, and width. | Closest conceptual prior. Contact-GraspNet reduces 6-DoF in point space; PGF applies the reduction on valid RGB-D pixels for image-grid prediction. |

**Contribution structure**:

PGF is a method component, not the primary contribution. The primary contribution is the finding.

- **Finding (primary contribution)**: The generalization bottleneck in 6-DoF grasping is representational, not architectural. Task-specific backbones trained on grasp data cannot exceed the diversity of that data. Frozen VFM + geometry injection breaks this limit. Evidence: Novel AP 33.88% vs. 28.75% (DSF-Grasp); frozen vs. fine-tuned +6.99 AP.
- **PGF role**: Dense per-pixel continuous regression of approach direction, gripper direction, width, and depth — the output formulation that keeps 6-DoF grasp supervision aligned with the frozen VFM token grid. Prior image-based methods predict intermediate quantities and require additional recovery steps; PGF predicts all grasp parameters directly per pixel. This is a justified design choice for VFM adaptation, not a claimed representational first.
- **Naming PGF**: Acceptable as a method name if framed as the VFM-compatible output formulation. Do not lead contribution bullets with "We introduce PGF."

**Recommended contribution wording**:

**Primary finding sentence (abstract/introduction):**
> Task-specific visual representations trained on grasp data are bounded by the diversity of that data, regardless of architectural improvements; a frozen vision foundation model, adapted with minimal geometry injection, breaks this ceiling — achieving the largest novel-object AP gain over existing 6-DoF methods.

**PGF description (method section, not contribution bullet):**
> We formulate 6-DoF grasp detection as dense per-pixel regression of approach direction, gripper direction, width, and depth. Each valid pixel is lifted to a full grasp pose via aligned XYZ, with orientation assembled from the predicted continuous vectors. This formulation keeps grasp supervision aligned with the frozen VFM token grid without 2D-to-3D feature conversion.

**Ablations required to defend the finding:**
- Frozen VFM vs. fine-tuned VFM under matched training recipe (already have: +6.99 AP).
- Frozen VFM vs. task-specific CNN/UNet trained from scratch under PGF (isolates representation from formulation).
- Continuous direction regression vs. discrete view classification under the same pixel-anchor (isolates PGF output space).
- Candidate budget equalization: PGF (dense) vs. sampled-point baseline at matched NMS/top-k budget.

---

## Issue 3 — One-Sentence Claim Rewrite

**Status**: Open

현재 claim은 component 나열 + metric claim 형태. One-Sentence Test 탈락.

> ❌ "Freezing a vision foundation model, formulate grasping as image-level pixel-wise prediction tasks, and adapting it with geometry-aware adaptation outperforms all existing 6-DoF grasp detection."

**Fix**: 인과관계 형태로.
> Task-specific backbones are bounded by the representational diversity of grasp training data; a frozen depth-aware foundation model with minimal bi-level geometry injection closes this gap, achieving the largest novel-object AP gain over existing 6-DoF methods.

---

## Issue 4 — F1-F2 Nested Structure

**Status**: Open

현재 F1, F2가 병렬 나열. 실제 논리는 nested:

```
F1: task-specific backbone → representational bottleneck → WHY use VFM
  └─ F2: VFM is RGB-only → metric 3D geometry gap → HOW to adapt VFM
       └─ Solution: freeze VFM + bi-level geometry injection
```

**Fix**: framing_principles.md의 두 진단 사이에 nesting 관계를 명시하는 문장 추가.

---

## Issue 5 — F1 Hidden Assumption Reframe

**Status**: Resolved

현재 F1: "existing grasp supervision is not enough." — 맞지만 generic함. SOTA 논문들도 이 문제를 인식하고 multiscale search, RGB fusion으로 대응함. 그럼에도 DSF-Grasp Novel AP = 28.75%에 그침 — RGB backbone도 from-scratch로 학습됐기 때문.

**Fix**: 숨은 가정을 명시적으로 이름 붙이기.
> Prior work shares a structural blind spot: task-specific feature extractors trained from scratch on grasp data are bounded by the representational diversity of that data, regardless of architectural improvements. DSF-Grasp illustrates this ceiling: dual-stream RGB+geometry fusion raises Novel AP from 24.31% (GSNet) to 28.75% — a marginal gain, because the RGB backbone is also trained on the same 88-object distribution.

---

## Issue 6 — DSF-Grasp Novel AP as Explicit Reference Point

**Status**: Resolved — handled in F1 update (Issue 5).

| Comparison | AP gap | Interpretation |
|---|---|---|
| FoundGrasp vs DSF-Grasp (Novel) | +5.13 | VFM prior beats dual-stream RGB+3D on unseen objects |
| FoundGrasp vs MTGrasp (Similar) | +9.35 | Large generalization gap |
| GeoFormer vs Full FT | +6.99 | Frozen >> Fine-tuned |
| GeoFormer vs AdaptFormer | +2.48 | Geometry injection at adapter level helps |

---

## ~~Issue 7 — GRIP Ablation~~ (Deleted — GRIP discarded)

---

## Issue 8 — Absence Answer: Strengthen with Numbers

**Status**: Resolved

> Without this work, it would remain unrecognized that visual representations bounded by grasp-specific training data are a key limiting factor for novel-object generalization, regardless of architectural design. FoundGrasp demonstrates this with frozen VFM adaptation: 33.88% vs. 28.75% Novel AP (DSF-Grasp).

---

## Additional Reviewer Attack List

Use this list before drafting `framing_principles.md` or `FoundGrasp.md`. Most dangerous: Issue 9, Issue 10, Issue 11, Issue 14, Issue 16.

---

## Issue 9 — Full Fine-Tuning Baseline Must Be Strong

**Status**: Open

**Reviewer attack**: "Full fine-tuning underperforms because the optimization recipe is weak, not because fine-tuning damages VFM priors."

**Core question**: Was Full FT tuned as carefully as RGBD-AdaptFormer?

**Fix**:
- Report backbone LR, head LR, weight decay, warmup, epochs, batch size, and layer-wise LR decay if used.
- Add a small Full FT sweep if possible.
- If no sweep exists, write "under our matched training recipe" rather than making a universal fine-tuning claim.

---

## Issue 10 — Inference and Post-Processing Confound

**Status**: Open

**Reviewer attack**: "The gain comes from kappa reranking, collision filtering, threshold tuning, voxel NMS, or objectness erosion, not from VFM adaptation."

**Core question**: Are all internal ablations and SOTA comparisons evaluated with identical inference settings?

**Fix**:
- Report raw ranking, +kappa reranking, +collision detection, and +threshold tuning separately.
- State whether each headline number uses collision detection.
- Keep internal ablations on one fixed inference recipe.

---

## Issue 11 — External Pretraining Fairness and Data Leakage Boundary

**Status**: Open

**Reviewer attack**: "FoundGrasp simply uses massive external pretraining; comparison to GraspNet-only methods is unfair."

**Core question**: Is the claim "better use of external visual priors" or "better architecture under the same data"?

**Fix**:
- Say "grasp-label-efficient," not generic "data-efficient."
- State that FoundGrasp uses external VFM pretraining but no additional grasp labels.
- Add a limitation that category/image overlap in generic VFM pretraining cannot be fully excluded.

---

## Issue 12 — "All Prior Work Trains From Scratch" May Be Overstated

**Status**: Open

**Reviewer attack**: "Some prior RGB streams may use ImageNet pretraining; your from-scratch claim is false."

**Core question**: What exactly is the structural blind spot shared by prior methods?

**Fix**:
- Replace "all prior methods train from scratch" with "prior GraspNet SOTA primarily learns task-specific grasp representations within the grasp-data regime."
- Claim only that prior work does not study frozen VFM representation preservation plus metric geometry injection for 6-DoF grasping.

---

## Issue 13 — Method Naming Is Internally Inconsistent

**Status**: Open

**Reviewer attack**: "Which method is Ours: RGBD-AdaptFormer, GRIP, Geo-Adapter, or FoundGrasp?"

**Core question**: What is the canonical paper method name?

**Fix**:
- Define: FoundGrasp = full system; GeoFormer = final adapter (renamed from RGBD-AdaptFormer); GRIP-Adapter = gated ablation variant.
- Remove or update comments that still call GRIP "Ours (final)" before release.
- Do not label both GeoFormer and GRIP-Adapter as Ours in tables.
- Code-level name `geoformer` (peft_type in configs) maps to paper name **GeoFormer** — ensure paper and code are cross-referenced.

---

## Issue 14 — Input Geometry Is XYZ, Not Just Depth

**Status**: Open

**Reviewer attack**: "You claim depth injection, but the best model uses camera-frame XYZ, so the gain may come from calibrated XYZ lifting."

**Core question**: Does the best model use raw depth, XYZ, depth mask, or a combination?

**Fix**:
- Use "depth-derived geometric features" or "XYZ geometry" for the actual best model.
- Add D-only vs XYZ ablation if possible.
- If no ablation exists, do not claim the method only needs raw depth.

---

## Issue 15 — Benchmark Table Must Normalize Camera, CD, and Metric

**Status**: Open

**Reviewer attack**: "You cherry-picked RealSense AP and mixed numbers with different cameras, collision detection settings, or metrics."

**Core question**: Are the benchmark numbers directly comparable?

**Fix**:
- Split RealSense and Kinect tables.
- Include AP, AP@0.4, AP@0.8, collision detection flag, and FPS where available.
- Say exactly "RealSense Novel AP" or "RealSense mean AP" instead of generic "SOTA."

---

## Issue 16 — One-Seed Variance and Small-Gap Claims

**Status**: Open

**Reviewer attack**: "GRIP -0.39 AP and other small gaps may be random seed noise."

**Core question**: Which conclusions are robust to training variance?

**Fix**:
- Repeat AdaptFormer vs RGBD-AdaptFormer vs GRIP if compute allows.
- If not, phrase small-gap claims as "did not improve in our run."
- Reserve strong causal language for large gaps.

---

## Issue 17 — Candidate Budget and NMS Fairness

**Status**: Open

**Reviewer attack**: "Dense image prediction generates more grasp candidates than point-cloud seed methods; AP gain may come from candidate coverage."

**Core question**: Is candidate budget comparable?

**Fix**:
- Report raw candidates before NMS and final candidates after NMS.
- Add sensitivity for `max_grasps` 512 / 1024 / 2048 / 4096.
- Describe voxel NMS and top-k evaluation clearly.

---

## Issue 18 — Objectness Mask May Be a Hidden Source of Gain

**Status**: Open

**Reviewer attack**: "Objectness gating and erosion suppress bad grasps; gains may come from segmentation-like filtering."

**Core question**: How much gain comes from objectness gating?

**Fix**:
- Report with and without objectness gate/erosion.
- Clarify objectness supervision source.
- Ensure all internal ablations use the same objectness pathway.

---

## Issue 19 — Cross-Camera and Calibration Generalization

**Status**: Open

**Reviewer attack**: "Image-space prediction may overfit RealSense intrinsics, viewpoint, and resolution."

**Core question**: Does FoundGrasp generalize beyond the training camera setting?

**Fix**:
- Report RealSense-trained -> Kinect if available.
- Report separate-camera or joint-camera training.
- State calibrated depth-to-XYZ as an assumption.

---

## Issue 20 — Dense Label Projection and Propagation Noise

**Status**: Open

**Reviewer attack**: "Pixel-wise GT projection and propagation may introduce noisy labels or collapse grasp ambiguity."

**Core question**: Are projected dense labels geometrically valid and ambiguity-aware enough?

**Fix**:
- Show qualitative projected labels: valid pixels, directions, symmetry, propagated support.
- Ablate no propagation / no depth threshold / no symmetry / no dynamic assignment where possible.
- State that continuous vectors represent one selected valid mode, while score/ranking handles candidate quality.

---

## Issue 21 — Related Work Must Include VFM-Based Robotics

**Status**: Open

**Reviewer attack**: "This is not the first robotics paper using DINO/CLIP/SAM features."

**Core question**: What makes FoundGrasp different from prior VFM-for-manipulation work?

**Fix**:
- Add related work on foundation visual representations for manipulation.
- Avoid "first use of VFM in robotics."
- Claim a narrower contribution: controlled frozen VFM geometry adaptation for dense 6-DoF grasp detection.

---

## Issue 22 — "Bi-level" Terminology Internally Inconsistent

**Status**: Open

**Problem**: Terminology table defines feature-level geometry injection (adapter, additive into ViT FFN) and prediction-level geometry injection (head fusion). But PGF uses aligned XYZ for **lifting** pixel predictions to SE(3) — not for injecting geometry into predictions. Calling this "prediction-level geometry injection" conflates two different operations.

**Reviewer attack**: "Your bi-level injection claim is inconsistent: the second level is coordinate lifting, not injection."

**Fix**: Disambiguate in method section.
- Feature-level: geometry injection (additive, into frozen ViT FFN blocks)
- Output-level: geometric lifting (aligned XYZ lifts per-pixel predictions to SE(3))
- "bi-level geometry adaptation" stays valid as umbrella if clearly defined as two distinct mechanisms.

---

## Issue 23 — PGF Lacks Ablation vs. Alternative Image-Space Formulations

**Status**: Open

**Reviewer attack**: "PGF is a design choice, not a proven-better formulation. A point-cloud head attached to the same frozen VFM features could work equally well."

**Core question**: Is the image-space pixel anchor empirically better than a sampled-point anchor under the same frozen VFM?

**Fix**:
- Ablation: PGF (dense pixel anchor) vs. sampled pixel seeds → point-cloud head, under same frozen VFM and identical candidate budget.
- If ablation is not feasible, narrow PGF claim to: "a formulation coherent with image-token supervision" — do not claim empirical superiority over point-cloud formulations.

---

## Issue 24 — "Insertion Depth" Is Non-Standard Terminology

**Status**: Open

**Problem**: GraspNet-1Billion standard term is **"grasp depth"**. Using "insertion depth" without definition risks confusing reviewers familiar with GraspNet-1B.

**Fix**: Either adopt standard term "grasp depth," or explicitly define "insertion depth" on first use and note the equivalence.

---

## Issue 25 — DA-V2 Backbone Confound: Depth Pretraining vs. VFM Principle

**Status**: Open

**Reviewer attack**: "DA-V2 is pretrained for depth estimation. Its superiority may come from depth-related pretraining, not from the frozen VFM + geometry injection principle. The backbone-agnostic claim is undermined by choosing a depth-pretrained backbone as the best model."

**Core question**: How much of the gain is from DA-V2's depth pretraining vs. the frozen VFM adaptation principle?

**Fix**:
- Report per-backbone AP: DA-V2 vs. DINOv2 vs. SAM vs. SigLIP under identical RGBD-AdaptFormer setting.
- If DA-V2 significantly outperforms others, qualify: "geometry-aware adaptation benefits most when the VFM backbone has depth-related priors."
- Do not claim backbone-agnostic equivalence if performance gaps across backbones are large.

---

## Issue 26 — Antipodal Symmetry Handling Not Explicitly Claimed

**Status**: Open

**Problem**: Approach direction has 180° antipodal symmetry. If not handled in the loss, dense label projection introduces systematic label noise for approach vectors.

**Core question**: Is antipodal symmetry explicitly handled in training loss or label generation?

**Fix**:
- If handled: state it as an explicit design choice in the method section.
- If not handled: this is a methodological gap — add it or acknowledge it as a limitation.

---

## Issue 27 — "Outperform All Existing Methods" Needs Scope Qualification

**Status**: Open

**Problem**: "Outperform all existing methods" is unqualified. New methods may appear between submission and review.

**Fix**: Qualify in paper as:
> "...outperforms all methods evaluated on the GraspNet-1Billion benchmark (RealSense, Novel AP) as of submission."

---

## Issue 28 — Contribution Coherence and Novelty: Three Contributions Must Form One Principled Thesis

**Status**: Open

**Reviewer attacks**:
> "C1 (frozen > fine-tuning), C2 (geometry injection), C3 (PGF) appear to be three independent findings. There is no single coherent research thesis."

> "Frozen backbone is standard in PEFT. Adapters are borrowed from NLP/CV. Pixel-wise regression is dense prediction applied to grasping. Where is the novel technique?"

**Core question**: Do the three contributions form one principled answer to one principled diagnosis, or are they three separate engineering choices that happened to work?

**Fix**: The paper must make the logical chain explicit in the abstract and introduction:
- PGF (C3): defines the image-space formulation that makes VFM supervision coherent
- Geometry injection (C2): fills the metric 3D gap that VFMs cannot address natively
- Frozen > fine-tuning (C1): empirically validates why freezing preserves the visual priors that drive generalization

Together, they answer one question: **how to properly adapt a frozen VFM for 6-DoF grasping**. The novelty is not any single technique but the principled diagnosis (F1 + F2) from which all three design choices are derived. Framing must make this chain explicit, or reviewers will read it as an ad hoc combination.

---

## Issue 29 — T-Mech Practical Validation: Venue Fit and Real-Robot Evidence

**Status**: Open

**Reviewer attacks**:
> "VFM adapter design and pixel-wise regression are CV/ML contributions. What is the Mechatronics contribution?"

> "GraspNet-1Billion is a simulation-heavy benchmark. Without real-robot experiments, this is insufficient for T-Mech."

**Core questions**:
- Is there real-robot validation? If not, how do we defend against this?
- What makes this a T-Mech paper vs. an IROS/CVPR paper?

**Fix**:
- If real-robot experiments exist: report grasp success rate, object set, robot platform, and compare against at least one baseline.
- If not: use DSF-Grasp (TMech'26) as precedent for benchmark-only T-Mech grasping papers. Explicitly frame the contribution as a grasping *system* with practical relevance — not a pure CV method.
- T-Mech framing: emphasize the system-level contribution (practical RGB-D grasping pipeline, real sensor input, deployment-relevant evaluation splits).

---

## Issue 30 — Practical Significance of +5.13 Novel AP

**Status**: Open

**Reviewer attack**: "Novel AP 33.88% vs. 28.75% is still a low absolute number. Is this improvement practically meaningful for robotic deployment?"

**Fix**:
- State relative improvement: +17.8% relative gain over the best prior method.
- Contextualize: Novel split = 30 previously unseen object categories; this is the hardest evaluation condition in GraspNet-1Billion.
- If available: show qualitative examples where FoundGrasp succeeds on novel objects where baselines fail.

---

## Issue 31 — Geometry Injection Level: Why FFN, Not Attention?

**Status**: Open

**Reviewer attack**: "AdaptFormer injects at FFN level by convention. Why not at attention level? There is no principled justification for this choice."

**Core question**: Is there an ablation comparing FFN-level vs. attention-level geometry injection?

**Fix**:
- If ablation exists: report it and use it to justify the choice.
- If not: state "following AdaptFormer convention" — do not claim FFN injection is principally optimal.

**Status**: Open

**Problem**: Terminology table defines:
- Feature-level geometry injection = geometry injected into ViT FFN via adapter
- Prediction-level geometry injection = geometric features fused at head

But PGF uses XYZ for **lifting** pixel predictions to SE(3), not for injecting geometry into predictions. Calling this "prediction-level geometry injection" is inaccurate — it implies a second injection, not a lifting operation.

**Reviewer attack**: "Your bi-level injection claim is inconsistent: the second level is coordinate lifting, not injection."

**Fix**: Disambiguate explicitly in method section.
- Feature-level: geometry injection (additive, into frozen ViT FFN)
- Output-level: geometric lifting (XYZ lifts per-pixel predictions to SE(3))
- Umbrella term options: "geometry-aware bi-level adaptation" stays valid if "bi-level" refers to representation + output, not two injections.

---

## Issue 23 — PGF Lacks Ablation vs. Alternative Image-Space Formulations

**Status**: Open

**Reviewer attack**: "PGF is a design choice, not a proven-better formulation. A point-cloud head attached to the same frozen VFM features could work equally well."

**Core question**: Is the image-space pixel anchor empirically better than a sampled-point anchor under the same frozen VFM?

**Fix**:
- Add ablation: PGF (dense pixel anchor) vs. sampled pixel seeds → point-cloud head, under same frozen VFM and identical candidate budget.
- If ablation is not feasible, narrow the PGF claim to: "a formulation that is coherent with image-token supervision" rather than "empirically superior to point-cloud formulations."

---

## Issue 24 — "Insertion Depth" Is Non-Standard Terminology

**Status**: Open

**Problem**: GraspNet-1Billion community standard term is **"grasp depth"** (distance from gripper center to contact surface). Using "insertion depth" without definition risks confusing reviewers familiar with GraspNet-1B.

**Fix**: Either adopt standard term "grasp depth," or define "insertion depth" explicitly on first use and note the equivalence.

---

## Issue 25 — DA-V2 Backbone Confound: Depth Pretraining vs. VFM Principle

**Status**: Open

**Reviewer attack**: "DA-V2 is pretrained for depth estimation. Its superiority may come from depth-related pretraining, not from the frozen VFM + geometry injection principle. Your backbone-agnostic claim is undermined by choosing a depth-pretrained backbone as the best model."

**Core question**: How much of the gain is from DA-V2's depth pretraining vs. the frozen VFM adaptation principle?

**Fix**:
- Report per-backbone AP: DA-V2 vs. DINOv2 vs. SAM vs. SigLIP under identical RGBD-AdaptFormer setting.
- If DA-V2 significantly outperforms others, qualify the claim: "geometry-aware adaptation benefits most when the VFM is initialized with depth-related priors."
- Do not claim backbone-agnostic equivalence if gaps are large.

---

## Issue 26 — Antipodal Symmetry Handling Not Explicitly Claimed

**Status**: Open

**Problem**: Approach direction has 180° antipodal symmetry — a grasp can be approached from two opposite directions. If the loss does not handle this, dense label projection introduces systematic label noise for approach vectors.

**Core question**: Is antipodal symmetry explicitly handled in the training loss or label generation?

**Fix**:
- If handled: state it as an explicit design choice in the method section and claim it as part of the formulation.
- If not handled: this is a real methodological gap — add it to the limitation or fix it.
- Either way, do not leave it implicit.

---

## Issue 27 — "Outperform All Existing Methods" Needs Scope Qualification

**Status**: Open

**Problem**: The one-sentence claim says "outperform all existing methods." This is unqualified — new methods may appear between submission and review/publication.

**Reviewer attack**: "Method X published after your submission outperforms FoundGrasp on Novel AP."

**Fix**: Qualify in paper to:
> "...outperforms all methods evaluated on the GraspNet-1Billion benchmark (RealSense camera, Novel AP) as of submission."

Do not use unqualified "all existing methods" in the abstract or introduction.

---

## Issue 32 — C1 Isolation: "Representational" Claim Conflates Backbone, Formulation, and Supervision

**Status**: Open

**Problem**: C1 states "the generalization bottleneck is *representational*, not architectural." However, the cross-dataset comparison (task-specific baselines vs. FoundGrasp on GraspClutter6D) changes three variables simultaneously:
1. **Backbone**: frozen VFM (ours) vs. task-specific trained backbones (theirs)
2. **Prediction formulation**: dense pixel PGF (ours) vs. sparse 3D contact points (theirs)
3. **Supervision**: depth-aware dense label propagation (ours) vs. sparse GraspNet-1B annotations (theirs)

A reviewer can legitimately argue: "The AP gain comes from PGF's denser candidate coverage, not from VFM priors." This is the most dangerous attack on C1 because there is no experiment in the current list that isolates representation from formulation.

**Mitigating factors** (why this may not be fatal):
- Contact-GraspNet is per-point continuous SO(3) in 3D — functionally analogous to PGF in spirit, yet achieves only 10.73 AP on GraspClutter6D
- The within-dataset ablation (Full FT + GeoFormer vs Frozen + GeoFormer, once run) uses identical formulation and isolates only the freezing effect
- The consistent 10–23 AP ceiling across all task-specific architectures (including sophisticated ones) is hard to explain by formulation alone

**Core question**: If a scratch ViT-B (same size, random init) + GeoFormer + same PGF head + same supervision were trained on GraspNet-1B, what would it achieve on GraspClutter6D? If similar to FoundGrasp (35+ AP), C1 must be reframed as "PGF formulation matters, not VFM priors." If much lower (25–30 AP), C1 is partially isolated.

**Resolution (decided)**: Full FT + GeoFormer가 Frozen + GeoFormer보다 훨씬 낮다고 가정. Scratch ViT + PGF 실험은 불필요.

**Evidence contract (chosen):**
1. **Within-dataset (clean isolation)**: Full FT + GeoFormer vs Frozen + GeoFormer — same formulation, same adapter, only freezing differs → X Novel AP gap isolates the freezing effect
2. **Cross-dataset (Contact-GraspNet partial defense)**: Contact-GraspNet (per-point dense SO(3) — PGF와 철학 유사)이 GraspClutter6D에서 10.73 AP. PGF formulation이 전부라는 공격을 부분적으로 방어함
3. **Claim scope**: "Frozen pre-trained visual representations substantially improve generalization over task-specific training, under the same prediction formulation and adapter design" — (1)이 완료되면 이 문장이 성립함

**Remaining risk**: C1 claim은 "representational diversity (VFM이 본 object 수) vs training stability (frozen이 VFM initialization을 보존)" 중 어느 것이 핵심인지까지 isolate하지 못함. T-MECH 수준에서는 이 수준까지 요구하지 않으나, NeurIPS/CVPR 제출 시 추가 실험 필요.

---

## Issue 33 — GraspClutter6D Self-Cite as Primary Evidence: Strategic Risk

**Status**: Open

**Problem**: The current framing positions GraspClutter6D cross-dataset results as the "primary evidence" for C1, while GraspClutter6D is the first author's own companion paper. This creates a perception risk:
- "The evaluation benchmark was designed by the same author — could it be set up to favor the proposed method?"
- "Why use your own dataset instead of an established third-party benchmark?"
- "Baselines may not be optimally configured for GraspClutter6D."

This is not a fairness problem in reality (baselines use public code, uniform CD), but perception of circular reasoning can make reviewers hostile regardless of the actual evidence quality.

**Core question**: Should GraspClutter6D be primary evidence, co-equal evidence, or supplementary validation? And is a second third-party cross-dataset evaluation feasible?

**Resolution (decided)**: "GraspNet-1B와 co-equal하되 GraspNet-1B가 first-presented."

**Chosen framing:**
- §4.2에서 GraspNet-1B Novel split ceiling을 먼저 확립 (제3자 benchmark에서 field-wide ceiling 확인)
- §4.3은 "독립 분포에서 동일 thesis 검증" — §4.2 finding의 cross-distribution corroboration
- F1 slide bullet 순서: within-dataset first → cross-dataset second (GraspNet-1B가 ceiling, GraspClutter6D가 categorical collapse로 같은 메커니즘을 더 극적으로 보여줌)
- §4.3 framing: "Primary evidence for C1" → "Independent corroboration of C1"

**Mitigation added to §4.3**: Contact-GraspNet의 10.73 AP를 formulation confound 방어 근거로 명시. GraspClutter6D가 self-cite라는 사실은 method section에서 투명하게 밝힘.

---

## Issue 34 — Factual Error: "+6.99 Novel AP" Should Be "+6.56 Novel AP"

**Status**: Resolved

**Resolution**: Full FT + GeoFormer 실험 완료. 이제 clean comparison 가능:
- Frozen+GeoFormer (33.88) vs FT+GeoFormer (28.11) = **+5.77 Novel AP** (adapter 동일, freezing만 변동)
- Mean AP delta: 61.61 − 54.44 = **+7.17 mean AP**
- C3 bullet 및 Evidence anchors 업데이트 완료.

---

## Issue 35 — Evidence Anchor Overstates Clean Freezing Effect

**Status**: Resolved

**Resolution**: Full FT + GeoFormer 실험 완료 (28.11 Novel AP).
- **Freezing effect** (adapter 동일): Frozen+GeoFormer (33.88) − FT+GeoFormer (28.11) = **+5.77 Novel AP**
- **Adapter effect** (backbone 동일, FT): FT+GeoFormer (28.11) − FT+no adapter (27.32) = **+0.79 Novel AP**
- 개선의 대부분(+5.77)이 freezing에서 오고 adapter 단독 효과(+0.79)는 미미 → C1 강화됨.

---

## Issue 36 — Unverified Claim in C2: "Backbone-Agnostic: Validated Across DAv2, DINOv2, SAM, SigLIP"

**Status**: Open

**Problem**: Contributions section C2 states: "Backbone-agnostic: validated across DAv2, DINOv2, SAM, SigLIP." However, §4.5 table has DINOv2, SAM, SigLIP rows filled with "—". This claim is currently unverified.

**Reviewer attack**: "You claim backbone-agnostic validation, but only DAv2 results are shown. This is not validation — this is assertion."

**Fix**: Remove this sentence from C2 until the cross-backbone experiments are complete. Replace with: "The design is backbone-agnostic; cross-backbone validation across DAv2, DINOv2, SAM, and SigLIP is reported in §4.5." When reporting, if DA-V2 substantially outperforms others, qualify as "frozen VFMs benefit most when initialized with depth-related priors" rather than claiming equivalence.

---

## Issue 37 — Unverified Claim in C3: "GRIP-Adapter Adds No Measurable Benefit"

**Status**: Open

**Problem**: Contributions section C3 states: "Additive injection is the minimum sufficient form: RGB-conditioned gating (GRIP-Adapter) adds no measurable benefit on generalization splits." However, the GRIP-Adapter ablation row in §4.4 is "—". This claim is currently unverified.

**Reviewer attack**: "You claim additive injection is sufficient, but there are no GRIP-Adapter results. This is a design assertion without empirical support."

**Fix**: Remove "additive injection is the minimum sufficient form: GRIP-Adapter adds no measurable benefit" from C3 until the ablation is run. If the ablation shows GRIP ≈ GeoFormer on Novel/Similar, restore the sentence. If GRIP shows improvement, the design claim changes.

---

## Issue 38 — Candidate Budget Confound Underweighted Given Cross-Dataset Primary Evidence

**Status**: Open

**Problem**: Candidate budget sensitivity (max_grasps sensitivity, raw vs. post-NMS candidate count) is classified as "있으면 좋음" in the experiment priority. However, GraspClutter6D cross-dataset results are now positioned as the primary evidence for C1. Dense pixel prediction (PGF) generates far more grasp candidates per image than point-cloud methods before NMS. If the AP gain on GraspClutter6D comes partly from candidate coverage rather than VFM representation quality, the primary evidence for C1 is weakened.

**Reviewer attack**: "Your +17.77 AP on GraspClutter6D is partly explained by the dense pixel anchor generating 100× more candidates before NMS. When you equalize the pre-NMS budget, how much of the gap remains?"

**Core question**: On GraspClutter6D specifically, how sensitive is FoundGrasp's AP to max_grasps (512/1024/2048/4096)? And what is the raw candidate count before NMS vs. the baseline methods?

---

## Issue 39 — Ma et al. CVPR 2024 Competitor Positioning

**Status**: Resolved (excluded from comparison; noted in Related Work)

**Problem**: "Generalizing 6-DoF Grasp Detection via Domain Prior Knowledge" (Ma et al., CVPR 2024) directly targets novel-object generalization on GraspNet-1B and reports 36.67 Novel AP (RealSense) — higher than FoundGrasp's 33.88. This could appear to undermine C1.

**Resolution**: Ma et al. use **multi-view TSDF-reconstructed scenes** as input (not single-view point cloud) and require >1 min inference time. This is a fundamentally different and orthogonal setting to FoundGrasp's single-view real-time deployment. Accordingly:
- Excluded from the main comparison table.
- Mentioned in §2.1 Related Work as a complementary multi-view approach.
- Framing clarification: our C1 ceiling claim ("28–29 Novel AP for all single-view task-specific methods") remains valid; Ma et al. is not a single-view method.

**In Related Work §2.1**: "Ma et al. (CVPR 2024) address novel-object generalization via physical constraint regularization (SDF-based) on multi-view TSDF reconstructed scenes with >1 min inference — orthogonal to our single-view real-time setting."

**Reviewer defense**: "Ma et al. use multi-view reconstruction at test time; their input contains substantially more geometric information than any single-view method. We follow the standard single-view evaluation protocol established by GraspNet-1B."

**Fix**:
- Reclassify candidate budget sensitivity from "있으면 좋음" to **"중요"** given the cross-dataset primary evidence positioning
- Report pre-NMS candidate count for FoundGrasp on GraspClutter6D alongside the AP numbers
- If AP degrades significantly at lower max_grasps (e.g., AP at max_grasps=512 < 30), this should be disclosed. If AP is robust (e.g., AP at max_grasps=512 ≈ 37), this directly refutes the coverage-confound attack on the primary evidence.
