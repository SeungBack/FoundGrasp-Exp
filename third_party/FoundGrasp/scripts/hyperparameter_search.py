#!/usr/bin/env python3
"""
Sequential hyperparameter search script for grasp NMS parameters.
Finds optimal value for each parameter one at a time.
"""

import subprocess
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Search space
NMS_KERNEL_SIZES = [0, 3, 5, 7]
GRASPNESS_EXPS = [0.0, 0.1, 0.2, 0.5, 1.0]
GRASP_SCORE_EXPS = [0.5, 1.0, 1.5, 2.0, 3.0]

# Default values
DEFAULT_NMS = 3
DEFAULT_GRASPNESS_EXP = 0.1
DEFAULT_GRASP_SCORE_EXP = 2.0

# Default settings
DEFAULT_CHECKPOINT = "logs/lr/parallel_ch64_bs32_lr2e-4/checkpoints/checkpoint_epoch_300.pth"
DEFAULT_CONFIG = "lr/parallel_ch64_bs32_lr2e-4.yaml"
DEFAULT_SPLIT = "test_mini"
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 32


def run_experiment(args):
    """Run a single experiment with given parameters."""
    gpu_id, nms_kernel, graspness_exp, grasp_score_exp, checkpoint, config, split, batch_size, num_workers = args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python", "test.py",
        "--checkpoint", checkpoint,
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--eval", "--infer",
        "--split", split,
        "--config", config,
        "--nms_kernel_size", str(nms_kernel),
        "--graspness_exp", str(graspness_exp),
        "--grasp_score_exp", str(grasp_score_exp),
    ]

    param_str = f"nms={nms_kernel}, ge={graspness_exp}, gse={grasp_score_exp}"
    print(f"[GPU {gpu_id}] Starting: {param_str}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )

        output = result.stdout + result.stderr
        metrics = parse_metrics(output)

        return {
            "nms_kernel_size": nms_kernel,
            "graspness_exp": graspness_exp,
            "grasp_score_exp": grasp_score_exp,
            "gpu_id": gpu_id,
            "metrics": metrics,
            "success": result.returncode == 0,
            "output": output[-2000:] if len(output) > 2000 else output,
        }
    except subprocess.TimeoutExpired:
        return {
            "nms_kernel_size": nms_kernel,
            "graspness_exp": graspness_exp,
            "grasp_score_exp": grasp_score_exp,
            "gpu_id": gpu_id,
            "metrics": {},
            "success": False,
            "output": "TIMEOUT",
        }
    except Exception as e:
        return {
            "nms_kernel_size": nms_kernel,
            "graspness_exp": graspness_exp,
            "grasp_score_exp": grasp_score_exp,
            "gpu_id": gpu_id,
            "metrics": {},
            "success": False,
            "output": str(e),
        }


def parse_metrics(output):
    """Parse AP metrics from test.py output."""
    metrics = {}
    lines = output.split('\n')
    current_split = None

    for line in lines:
        # Track current split (test_seen, test_similar, test_novel)
        if 'Split:' in line:
            if 'seen' in line.lower():
                current_split = 'seen'
            elif 'similar' in line.lower():
                current_split = 'similar'
            elif 'novel' in line.lower():
                current_split = 'novel'

        # Parse AP lines like "AP@0.4: 67.97%" or "AP    : 71.05%"
        if 'AP' in line and ':' in line:
            try:
                parts = line.split(':')
                key = parts[0].strip()
                value_str = parts[1].strip().replace('%', '')
                value = float(value_str)

                # Add split prefix if available
                if current_split:
                    key = f"{current_split}_{key}"
                metrics[key] = value
            except (IndexError, ValueError):
                continue

    # Calculate overall average AP if we have all splits
    ap_values = [v for k, v in metrics.items() if k.endswith('_AP')]
    if ap_values:
        metrics['mean_AP'] = sum(ap_values) / len(ap_values)

    return metrics


def get_score(result):
    """Extract score from result for comparison."""
    if not result["metrics"]:
        return -1
    # Priority: mean_AP (average of seen/similar/novel)
    for key in ["mean_AP", "seen_AP", "AP"]:
        if key in result["metrics"]:
            return result["metrics"][key]
    return list(result["metrics"].values())[0] if result["metrics"] else -1


def run_parallel_search(job_args, gpu_ids, desc):
    """Run experiments in parallel across GPUs."""
    num_gpus = len(gpu_ids)
    results = []

    print(f"\n{'='*60}")
    print(f"Stage: {desc}")
    print(f"{'='*60}")

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(run_experiment, job): job for job in job_args}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            status = "✓" if result["success"] else "✗"
            metrics_str = json.dumps(result["metrics"]) if result["metrics"] else "N/A"
            print(f"  {status} nms={result['nms_kernel_size']}, "
                  f"ge={result['graspness_exp']}, gse={result['grasp_score_exp']} "
                  f"-> {metrics_str}")

    # Find best
    results_sorted = sorted(results, key=get_score, reverse=True)
    best = results_sorted[0] if results_sorted else None

    if best:
        print(f"  >> Best: nms={best['nms_kernel_size']}, ge={best['graspness_exp']}, "
              f"gse={best['grasp_score_exp']} -> {best['metrics']}")

    return results, best


def main():
    parser = argparse.ArgumentParser(description="Sequential hyperparameter search")
    parser.add_argument("--gpus", type=str, default="0,2,3,4,5",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--output", type=str, default="hyperparam_search_results.json")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    common_args = (args.checkpoint, args.config, args.split, args.batch_size, args.num_workers)

    print(f"Using GPUs: {gpu_ids}")
    print(f"Sequential search: nms_kernel -> graspness_exp -> grasp_score_exp")
    print(f"Total experiments: {len(NMS_KERNEL_SIZES) + len(GRASPNESS_EXPS) + len(GRASP_SCORE_EXPS)}")

    start_time = time.time()
    all_results = {}

    # Stage 1: Find best nms_kernel_size
    job_args = [
        (gpu_ids[i % len(gpu_ids)], nms, DEFAULT_GRASPNESS_EXP, DEFAULT_GRASP_SCORE_EXP, *common_args)
        for i, nms in enumerate(NMS_KERNEL_SIZES)
    ]
    results, best = run_parallel_search(job_args, gpu_ids, "nms_kernel_size")
    all_results["nms_kernel_size"] = results
    best_nms = best["nms_kernel_size"] if best else DEFAULT_NMS

    # Stage 2: Find best graspness_exp (with best nms)
    job_args = [
        (gpu_ids[i % len(gpu_ids)], best_nms, ge, DEFAULT_GRASP_SCORE_EXP, *common_args)
        for i, ge in enumerate(GRASPNESS_EXPS)
    ]
    results, best = run_parallel_search(job_args, gpu_ids, "graspness_exp")
    all_results["graspness_exp"] = results
    best_ge = best["graspness_exp"] if best else DEFAULT_GRASPNESS_EXP

    # Stage 3: Find best grasp_score_exp (with best nms and ge)
    job_args = [
        (gpu_ids[i % len(gpu_ids)], best_nms, best_ge, gse, *common_args)
        for i, gse in enumerate(GRASP_SCORE_EXPS)
    ]
    results, best = run_parallel_search(job_args, gpu_ids, "grasp_score_exp")
    all_results["grasp_score_exp"] = results
    best_gse = best["grasp_score_exp"] if best else DEFAULT_GRASP_SCORE_EXP

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULT:")
    print(f"{'='*60}")
    print(f"  nms_kernel_size: {best_nms}")
    print(f"  graspness_exp: {best_ge}")
    print(f"  grasp_score_exp: {best_gse}")
    print(f"\nTotal time: {time.time() - start_time:.1f}s")

    # Save results
    output_data = {
        "best": {
            "nms_kernel_size": best_nms,
            "graspness_exp": best_ge,
            "grasp_score_exp": best_gse,
        },
        "search_space": {
            "nms_kernel_size": NMS_KERNEL_SIZES,
            "graspness_exp": GRASPNESS_EXPS,
            "grasp_score_exp": GRASP_SCORE_EXPS,
        },
        "all_results": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
