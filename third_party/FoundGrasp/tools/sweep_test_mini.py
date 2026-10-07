import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import config_name_from_path, resolve_config_path


def build_hp_suffix(params):
    return (
        f"gth{params['graspness_threshold']}"
        f"_oth{params['objectness_threshold']}"
        f"_tth{params['total_score_threshold']}"
        f"_ge{params['graspness_exp']}"
        f"_gse{params['grasp_score_exp']}"
        f"_arw{params['approach_kappa_risk_weight']}"
        f"_grw{params['gripper_kappa_risk_weight']}"
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


def make_test_command(args, params, log_suffix):
    cmd = [
        sys.executable, str(REPO_ROOT / 'test.py'),
        '--config', args.config,
        '--infer', '--eval',
        '--split', args.split,
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
    return cmd


def collect_result(base_log_dir, params):
    hp_suffix = build_hp_suffix(params)
    log_dir = os.path.join(base_log_dir, hp_suffix)
    candidates = [
        os.path.join(log_dir, 'eval_results_cd.json'),
        os.path.join(log_dir, 'eval_results.json'),
    ]
    eval_path = None
    for path in candidates:
        if os.path.exists(path):
            eval_path = path
            break
    if eval_path is None:
        raise FileNotFoundError(f"Expected evaluation file not found in: {log_dir}")

    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)

    summary = summarize_eval_results(eval_results)
    row = {**params, **summary, 'log_dir': log_dir}
    return row


def get_sort_key(split):
    if split == 'test_mini_novel':
        return 'novel_ap'
    if split == 'test_mini_similar':
        return 'similar_ap'
    if split == 'test_mini_seen':
        return 'seen_ap'
    return 'mean_ap'


def write_partial_results(base_log_dir, split, results):
    sort_key = get_sort_key(split)
    results_sorted = sorted(results, key=lambda x: x.get(sort_key, -1.0), reverse=True)

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


def print_top_results(split, results, top_k=10):
    sort_key = get_sort_key(split)
    results_sorted = sorted(results, key=lambda x: x.get(sort_key, -1.0), reverse=True)
    print(f"\n=== Current Top {min(top_k, len(results_sorted))} ({sort_key}) ===")
    for row in results_sorted[:top_k]:
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--split', default='test_mini',
                        choices=['test_mini', 'test_mini_seen', 'test_mini_similar', 'test_mini_novel'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--camera', default='realsense')
    parser.add_argument('--gpus', type=int, nargs='*', default=None,
                        help='Physical GPU ids for concurrent sweep, e.g. --gpus 0 1 2 3')
    parser.add_argument('--jobs-per-gpu', type=int, default=1,
                        help='How many sweep jobs to run concurrently on each GPU')
    parser.add_argument('--poll-seconds', type=float, default=5.0,
                        help='Polling interval for concurrent sweep jobs')
    parser.add_argument('--graspness-thresholds', type=float, nargs='+', default=[0.01, 0.02])
    parser.add_argument('--objectness-thresholds', type=float, nargs='+', default=[0.5])
    parser.add_argument('--total-score-thresholds', type=float, nargs='+', default=[0.01])
    parser.add_argument('--graspness-exps', type=float, nargs='+', default=[0.1, 0.25])
    parser.add_argument('--grasp-score-exps', type=float, nargs='+', default=[1.5, 2.0])
    parser.add_argument('--approach-kappa-risk-weights', type=float, nargs='+', default=[0.5])
    parser.add_argument('--gripper-kappa-risk-weights', type=float, nargs='+', default=[1.0])
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config_name = config_name_from_path(config_path)
    base_log_dir = os.path.join(REPO_ROOT, 'logs', config_name)

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

    param_grid = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        param_grid.append(params)

    results = []
    os.makedirs(base_log_dir, exist_ok=True)
    if not args.gpus:
        for params in param_grid:
            hp_suffix = build_hp_suffix(params)
            print(f"\n=== Running: {hp_suffix} ===")
            subprocess.run(make_test_command(args, params, hp_suffix), check=True, cwd=REPO_ROOT)
            row = collect_result(base_log_dir, params)
            results.append(row)
            print(row)
            _, json_path, csv_path = write_partial_results(base_log_dir, args.split, results)
            print(f"Updated partial results: {json_path}")
            print(f"Updated partial results: {csv_path}")
            print_top_results(args.split, results)
    else:
        gpu_ids = [str(gpu) for gpu in args.gpus]
        slots = []
        for gpu_id in gpu_ids:
            for slot_idx in range(max(1, args.jobs_per_gpu)):
                slots.append((gpu_id, slot_idx))

        pending = list(param_grid)
        running = []
        failed = []

        while pending or running:
            active_slots = {(job['gpu_id'], job['slot_idx']) for job in running}
            free_slots = [slot for slot in slots if slot not in active_slots]
            while pending and free_slots:
                params = pending.pop(0)
                hp_suffix = build_hp_suffix(params)
                gpu_id, slot_idx = free_slots.pop(0)
                log_dir = os.path.join(base_log_dir, hp_suffix)
                os.makedirs(log_dir, exist_ok=True)
                stdout_path = os.path.join(log_dir, 'sweep_stdout.log')
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = gpu_id
                print(f"\n=== Launching: {hp_suffix} on GPU {gpu_id} (slot {slot_idx}) ===")
                stdout_file = open(stdout_path, 'w', encoding='utf-8')
                proc = subprocess.Popen(
                    make_test_command(args, params, hp_suffix),
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                )
                running.append({
                    'proc': proc,
                    'params': params,
                    'gpu_id': gpu_id,
                    'slot_idx': slot_idx,
                    'stdout_file': stdout_file,
                    'stdout_path': stdout_path,
                })

            next_running = []
            for job in running:
                ret = job['proc'].poll()
                if ret is None:
                    next_running.append(job)
                    continue

                job['stdout_file'].close()
                hp_suffix = build_hp_suffix(job['params'])
                if ret != 0:
                    failed.append((hp_suffix, job['gpu_id'], job['stdout_path']))
                    print(f"[FAILED] {hp_suffix} on GPU {job['gpu_id']} -> {job['stdout_path']}")
                    continue

                row = collect_result(base_log_dir, job['params'])
                results.append(row)
                print(f"[DONE] {hp_suffix} on GPU {job['gpu_id']}")
                print(row)
                _, json_path, csv_path = write_partial_results(base_log_dir, args.split, results)
                print(f"Updated partial results: {json_path}")
                print(f"Updated partial results: {csv_path}")
                print_top_results(args.split, results)
            running = next_running

            if running:
                time.sleep(args.poll_seconds)

        if failed:
            print("\n=== Failed Jobs ===")
            for hp_suffix, gpu_id, stdout_path in failed:
                print({'hp_suffix': hp_suffix, 'gpu_id': gpu_id, 'log': stdout_path})
            raise RuntimeError(f"{len(failed)} sweep jobs failed.")

    sort_key = get_sort_key(args.split)
    results.sort(key=lambda x: x.get(sort_key, -1.0), reverse=True)
    out_path = os.path.join(base_log_dir, f'{args.split}_sweep_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("\n=== Top Results ===")
    for row in results[:10]:
        print(row)
    print(f"\nSorted by: {sort_key}")
    print(f"Saved sweep summary to: {out_path}")


if __name__ == '__main__':
    main()
