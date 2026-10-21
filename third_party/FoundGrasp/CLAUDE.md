# FoundGrasp — project invariants (loaded every session; keep this file short)

FoundGrasp: 6-DoF grasp detection on a frozen vision foundation model, with grasp refinement on completed geometry. Target venue: IEEE T-RO (first submission ≤ 18 pages including references and biographies).

## Session start
1. Read docs/HANDOVER.md: one page with the current state, running work, and the next step. It is rewritten at session end, never appended.
2. Paper writing: docs/evidence.md is the verified evidence inventory (numbers with sources). The framing spec docs/paper_framing_13q.md will be rewritten from it (the pre-refiner version is archived and is not an input). Refiner recipe and constraints: docs/PLAN.md (condensed).
3. Past decisions: docs/decision_log.md (read only when a decision is questioned). Old framings, drafts, and the full session log are in docs/archive/ and papers/T_Mech/tools/*.md; they are never read or grepped unless the user asks (the user intends to move them outside the repo).

## Commands
- Python: /home/seung/.conda/envs/vit-grasp/bin/python (absolute path; never `conda run`).
- Train: `python train.py --config configs/ours_base.yaml` (anchors: configs/ours_{small,base,large}[_kinect].yaml; ablations in configs/ablation/).
- Detector dump: `python test.py --config ours_base --checkpoint logs/ours_base/checkpoints/final_model.pth --camera realsense --split test --infer --no-eval --num_workers 4 --log-suffix <sfx>`. Full system: add `--verifier G_V_dpt_fonly --skip-cd` after `source scratchpad/campaign_0930/env_cfg.sh` (Kinect in-domain uses G_K_dpt_fonly). GraspClutter6D: add `--dataset GraspClutter6D --camera realsense-d435` and `VERIFIER_MAX_W=0.10`.
- Scoring (CPU): GraspNet-1B `python scripts/score_splits_dump.py --dump <dump dir> --camera <cam> --tag <tag> --procs 16`; GraspClutter6D `python test.py ... --eval --num_workers 16 --log-suffix <sfx>`.
- Job queue: scratchpad/campaign_0930/{lib.sh,env_cfg.sh,q_gpu,q_cpu,running,done,failed} with 2 GPU and 3 CPU runners; batch status file scratchpad/campaign_0930/status_1007.txt.
- Latency: `python scripts/benchmark_inference.py --config configs/ours_base.yaml --checkpoint <ckpt> [--verifier G_V_dpt_fonly] [--postproc]` on an idle GPU and a quiet host.
- Tables: `python scripts/make_paper_tables.py` → paper_tables/generated/*.csv → `python papers/T_Mech/tools/gen_paper_tables_tex.py` → papers/T_Mech/tables/*.tex. Paper numbers come only from paper_tables/{master_results,baseline_numbers,runtime,trainable_params,refiner_ladder}.csv.
- Paper: `cd papers/T_Mech && latexmk -pdf -interaction=nonstopmode main.tex`; pages: `pdfinfo main.pdf`.

## Repository
- Single branch: main (2026-10-07). Work on main or on a short-lived branch per paper section; merge fast-forward.
- Clone without `--recursive`; the 11 submodules (ext/transformers, pytorch3d, baselines, …) are not needed for paper work. Runtime needs third_parties/graspnetAPI and, for the DA3 / ViT-Adapter backbones, ext/depth-anything-3 and ext/dcnv4.
- Never commit artifacts: reference PDFs, *.log, LaTeX by-products, checkpoints (*.pth, *.pt, *.onnx), backups of generated tables. Figure sources in paper_figures/ and papers/T_Mech/figs/ stay tracked.

## File map
model/found_grasp.py (backbone, PEFT hooks, decode) · model/peft.py · model/loss.py · train.py · test.py · grasp_detector.py (detect() with the refiner) · scripts/verifier_grid.py (grasp refiner) · scripts/verifier_points.py (Verifier wrapper, budget) · scripts/make_paper_tables.py · scripts/score_splits_dump.py · configs/ · paper_tables/ · papers/T_Mech/ (main.tex, tables/, figs/, tools/) · sim/ (simulation benchmark) · logs/ (runs, _verifier, _stats) · docs/ (HANDOVER, paper_framing_13q, PLAN, decision_log, reference_tro_traits) · ext/ (baselines).

## Terminology registry
- System: FoundGrasp (-S/-B/-L); FoundGrasp-B is the headline; rows without the refiner are "detector only".
- Detector: frozen VFM backbone · geometry encoder · geometry adapter · geometry conditioning (the principle) · DPT decoder · grasp head · SE(3) grasp map (the output) · grasp decoding.
- Refiner: section "Grasp Refinement on Completed Geometry" · module "grasp refiner" · learned core "gripper-volume completion" · inputs: projective TSDF, decoder features · output: completed TSDF T · collision head · force-closure head · measured-surface test · opening and depth refinement · refined score s'.
- Shifts: object (seen / similar / novel), sensor (RealSense→Kinect), domain (GraspClutter6D, GC6D). The ground-truth judge is "the benchmark evaluation".
- Banned: verifier, evaluator, amodal, grasp-wise, geometric scoring, bi-level, training-free, "first per-pixel", readout, gripper model, geometry-aware, collision-aware, per-grasp, "field", superlatives. Pix2Grasp / GeoFormer / GeoHead are code names only.

## Writing rules
- Literal technical verbs (is frozen, fuses, predicts, anchors, completes, refines); no metaphorical or agentive verbs (supplies, meets, enters), no slogan sentences.
- Topic-first paragraphs; one operation or claim per sentence of 15–25 words; IEEE register; hedge only with partial evidence; never over-claim.
- No seed / σ / single-run statements. Every number in chat carries its source (file:line or CSV row).
- Proposals: 세워야 할 사실 / A (현재) / B (제안) / 이유. English sentences verbatim, discussion in Korean, one item per turn. An approved sentence changes only for a stated reason.

## Operations
- No `cd` in compound commands; absolute paths; never edit a running script; kill only by explicit PID, found with `pgrep -f 'patter[n]'` so the shell does not match itself.
- GPUs 0–3 only. Count runners first (`pgrep -af 'campaign_0930/runner.s[h]'`). At most 2 inferences at once, each with `--num_workers 4`; at most 2 active scorings; estimate procs × RSS before queueing (a 16-worker test.py holds about 100 GB).
- test.py evaluates by default: pass `--no-eval` for dumps. Detector checkpoints are fixed; only post-hoc modules change.
- Session end: rewrite docs/HANDOVER.md and append dated entries to docs/decision_log.md.
