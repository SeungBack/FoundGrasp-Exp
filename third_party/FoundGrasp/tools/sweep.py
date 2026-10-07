import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import config_name_from_path, resolve_config_path


def build_hp_suffix(params):
    return (
        f"gth-{params['graspness_threshold']}"
        f"_oth-{params['objectness_threshold']}"
        f"_tth-{params['total_score_threshold']}"
        f"_ge-{params['graspness_exp']}"
        f"_gse-{params['grasp_score_exp']}"
        f"_arw-{params['approach_kappa_risk_weight']}"
        f"_grw-{params['gripper_kappa_risk_weight']}"
    )


def summarize_eval_results(eval_results):
    summary = {}
    for split_name, split_results in eval_results.items():
        if not isinstance(split_results, dict) or 'res' not in split_results:
            continue
        res = split_results['res']
        if not res:
            continue
        key = split_name
        if key.startswith('test_mini_'):
            key = key.replace('test_mini_', '')
        elif key.startswith('test_'):
            key = key.replace('test_', '')
        summary[f'{key}_ap'] = sum(res) / len(res) * 100.0
        if len(res) > 1:
            summary[f'{key}_ap@0.4'] = res[1] * 100.0
        if len(res) > 3:
            summary[f'{key}_ap@0.8'] = res[3] * 100.0
    if 'seen_ap' in summary and 'similar_ap' in summary and 'novel_ap' in summary:
        summary['mean_ap'] = (summary['seen_ap'] + summary['similar_ap'] + summary['novel_ap']) / 3.0
    return summary


def resolve_checkpoint_arg(checkpoint_arg, base_log_dir):
    if not checkpoint_arg:
        return checkpoint_arg
    if os.path.exists(checkpoint_arg):
        return checkpoint_arg

    alt_path = os.path.join(base_log_dir, 'checkpoints', os.path.basename(checkpoint_arg))
    if os.path.exists(alt_path):
        print(f"Checkpoint not found at '{checkpoint_arg}'. Using matching file under config log dir: {alt_path}")
        return alt_path

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_arg}\n"
        f"Also checked: {alt_path}"
    )


def make_test_command(args, params, log_suffix):
    cmd = [
        sys.executable, str(REPO_ROOT / 'test.py'),
        '--config', args.config,
        '--infer', '--eval',
        '--split', args.split,
        '--ann_ids_path', args.ann_ids_path,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--camera', args.camera,
        '--graspness_threshold', str(params['graspness_threshold']),
        '--objectness_threshold', str(params['objectness_threshold']),
        '--total_score_threshold', str(params['total_score_threshold']),
        '--graspness_exp', str(params['graspness_exp']),
        '--grasp_score_exp', str(params['grasp_score_exp']),
        '--approach_kappa_risk_weight', str(params['approach_kappa_risk_weight']),
        '--gripper_kappa_risk_weight', str(params['gripper_kappa_risk_weight']),
        '--log-suffix', log_suffix,
    ]
    if args.checkpoint:
        cmd.extend(['--checkpoint', args.checkpoint])
    if args.use_collision_detection:
        cmd.append('--use_collision_detection')
    if args.no_use_collision_detection:
        cmd.append('--no-use_collision_detection')
    return cmd


def collect_result(run_log_dir, params):
    candidates = [
        os.path.join(run_log_dir, 'eval_results_cd.json'),
        os.path.join(run_log_dir, 'eval_results.json'),
    ]
    eval_path = None
    for path in candidates:
        if os.path.exists(path):
            eval_path = path
            break
    if eval_path is None:
        raise FileNotFoundError(
            "Expected evaluation file not found in run directory: "
            f"{run_log_dir}"
        )

    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)

    summary = summarize_eval_results(eval_results)
    row = {**params, **summary, 'log_dir': run_log_dir}
    return row


def write_partial_results(base_log_dir, split, results):
    results_sorted = sorted(results, key=lambda x: x.get('mean_ap', -1.0), reverse=True)

    json_path = os.path.join(base_log_dir, f'{split}_sweep_results.partial.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, indent=2)

    csv_path = os.path.join(base_log_dir, f'{split}_sweep_results.partial.csv')
    fieldnames = [
        'graspness_threshold',
        'objectness_threshold',
        'total_score_threshold',
        'graspness_exp',
        'grasp_score_exp',
        'approach_kappa_risk_weight',
        'gripper_kappa_risk_weight',
        'seen_ap',
        'similar_ap',
        'novel_ap',
        'mean_ap',
        'log_dir',
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_sorted:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    return results_sorted, json_path, csv_path


def print_top_results(results, top_k=10):
    results_sorted = sorted(results, key=lambda x: x.get('mean_ap', -1.0), reverse=True)
    print(f"\n=== Current Top {min(top_k, len(results_sorted))} (mean_ap) ===")
    for row in results_sorted[:top_k]:
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--split', default='test_mini_mixed', choices=['test_mini_mixed'])
    parser.add_argument('--ann-ids-path', '--ann_ids_path', dest='ann_ids_path', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--camera', default='realsense')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Single physical GPU id to use for the entire sweep, e.g. --gpu 0')
    parser.add_argument('--use-collision-detection', dest='use_collision_detection', action='store_true', default=False)
    parser.add_argument('--no-use-collision-detection', dest='no_use_collision_detection', action='store_true', default=False)

    parser.add_argument('--graspness-thresholds', type=float, nargs='+', default=[0.01])
    parser.add_argument('--objectness-thresholds', type=float, nargs='+', default=[0.5])
    parser.add_argument('--total-score-thresholds', type=float, nargs='+', default=[0.01])
    parser.add_argument('--graspness-exps', type=float, nargs='+', default=[0.2])
    parser.add_argument('--grasp-score-exps', type=float, nargs='+', default=[1.5])
    parser.add_argument('--approach-kappa-risk-weights', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    parser.add_argument('--gripper-kappa-risk-weights', type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0])
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config_name = config_name_from_path(config_path)
    base_log_dir = os.path.join(REPO_ROOT, 'logs', config_name)
    os.makedirs(base_log_dir, exist_ok=True)
    args.checkpoint = resolve_checkpoint_arg(args.checkpoint, base_log_dir)

    keys = (
        'graspness_threshold',
        'objectness_threshold',
        'total_score_threshold',
        'graspness_exp',
        'grasp_score_exp',
        'approach_kappa_risk_weight',
        'gripper_kappa_risk_weight',
    )
    values = (
        args.graspness_thresholds,
        args.objectness_thresholds,
        args.total_score_thresholds,
        args.graspness_exps,
        args.grasp_score_exps,
        args.approach_kappa_risk_weights,
        args.gripper_kappa_risk_weights,
    )

    param_grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    results = []
    env = os.environ.copy()
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Using single GPU for sweep: {args.gpu}")

    for params in param_grid:
        hp_suffix = build_hp_suffix(params)
        run_log_dir = os.path.join(base_log_dir, hp_suffix)
        os.makedirs(run_log_dir, exist_ok=True)
        stdout_path = os.path.join(run_log_dir, 'sweep_stdout.log')
        print(f"\n=== Running: {hp_suffix} ===")
        with open(stdout_path, 'w', encoding='utf-8') as stdout_file:
            subprocess.run(
                make_test_command(args, params, hp_suffix),
                check=True,
                cwd=REPO_ROOT,
                env=env,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
            )
        row = collect_result(run_log_dir, params)
        results.append(row)
        print(row)
        print(f"Sweep log: {stdout_path}")
        _, json_path, csv_path = write_partial_results(base_log_dir, args.split, results)
        print(f"Updated partial results: {json_path}")
        print(f"Updated partial results: {csv_path}")
        print_top_results(results)

    results_sorted, _, _ = write_partial_results(base_log_dir, args.split, results)
    out_path = os.path.join(base_log_dir, f'{args.split}_sweep_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, indent=2)

    print(f"\nSaved sweep summary to: {out_path}")
    print_top_results(results_sorted)


if __name__ == '__main__':
    main()
