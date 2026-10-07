#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from summarize_hard_collision_sweep import collect_rows


PARAM_ORDER = [
    ("approach_dist", [0.03, 0.05], "hard_approach_dist"),
    ("min_inner_points", [2, 5, 10], "hard_min_inner_points"),
    ("finger_length", [0.04, 0.06], "hard_finger_length"),
    ("finger_width", [0.01, 0.02], "hard_finger_width"),
    ("height", [0.01, 0.02], "hard_height"),
]

DEFAULT_PARAMS = {
    "approach_dist": 0.05,
    "min_inner_points": 2,
    "finger_length": 0.04,
    "finger_width": 0.01,
    "height": 0.02,
}


def _param_key(params):
    return (
        float(params["approach_dist"]),
        int(params["min_inner_points"]),
        float(params["finger_length"]),
        float(params["finger_width"]),
        float(params["height"]),
    )


def _row_matches(row, params):
    return _param_key(row) == _param_key(params)


def _latest_matching_row(root, params):
    rows = [row for row in collect_rows(root) if row["ap_mean"] is not None and _row_matches(row, params)]
    if not rows:
        return None
    return max(rows, key=lambda row: Path(row["run_dir"], "eval_results.json").stat().st_mtime)


def _run_test(args, params):
    cmd = [
        args.python,
        "test.py",
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--infer",
        "--eval",
        "--split", args.split,
        "--ann_ids_path", args.ann_ids_path,
        "--config", args.config,
        "--use_collision_detection",
        "--hard_approach_dist", str(params["approach_dist"]),
        "--hard_min_inner_points", str(params["min_inner_points"]),
        "--hard_finger_length", str(params["finger_length"]),
        "--hard_finger_width", str(params["finger_width"]),
        "--hard_height", str(params["height"]),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=args.repo_root, env=env, check=True)


def _write_history(root, trials, final_params):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "hard_collision_greedy_summary.csv"
    json_path = root / "hard_collision_greedy_summary.json"

    fieldnames = [
        "phase", "variable", "value", "selected",
        "ap_mean", "ap_seen", "ap_similar", "ap_novel",
        "approach_dist", "min_inner_points", "finger_length", "finger_width", "height",
        "run_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)

    payload = {
        "final_params": final_params,
        "trials": trials,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return csv_path, json_path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--config", default="configs/paper/dav3b.yaml")
    parser.add_argument("--split", default="test_mini_mixed")
    parser.add_argument("--ann-ids-path", default="mini_mixed_ann_ids.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--root", default="logs/paper/dav3b")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Reuse an existing eval_results.json for a parameter set instead of rerunning it.")
    parser.add_argument("--start-approach-dist", type=float, default=DEFAULT_PARAMS["approach_dist"])
    parser.add_argument("--start-min-inner-points", type=int, default=DEFAULT_PARAMS["min_inner_points"])
    parser.add_argument("--start-finger-length", type=float, default=DEFAULT_PARAMS["finger_length"])
    parser.add_argument("--start-finger-width", type=float, default=DEFAULT_PARAMS["finger_width"])
    parser.add_argument("--start-height", type=float, default=DEFAULT_PARAMS["height"])
    args = parser.parse_args()
    args.repo_root = str(repo_root)

    current = {
        "approach_dist": args.start_approach_dist,
        "min_inner_points": args.start_min_inner_points,
        "finger_length": args.start_finger_length,
        "finger_width": args.start_finger_width,
        "height": args.start_height,
    }

    print("Greedy hard-collision sweep")
    print(f"Initial params: {current}")

    cache = {}
    trials = []
    phase = 0

    for variable, values, _cli_name in PARAM_ORDER:
        phase += 1
        print(f"\n[Phase {phase}/{len(PARAM_ORDER)}] Sweep {variable}; fixed params: {current}")
        phase_rows = []

        for value in values:
            params = dict(current)
            params[variable] = value
            key = _param_key(params)

            print(f"\nTrying {variable}={value}: {params}")
            if key in cache:
                row = cache[key]
                print("Reusing result from this greedy sweep.")
            elif args.reuse_existing and (row := _latest_matching_row(args.root, params)) is not None:
                print(f"Reusing existing result: {row['run_dir']}")
            else:
                _run_test(args, params)
                row = _latest_matching_row(args.root, params)
                if row is None:
                    raise RuntimeError(f"No eval_results.json found for params: {params}")

            cache[key] = row
            trial = {
                "phase": phase,
                "variable": variable,
                "value": value,
                "selected": False,
                "ap_mean": row["ap_mean"],
                "ap_seen": row["ap_seen"],
                "ap_similar": row["ap_similar"],
                "ap_novel": row["ap_novel"],
                "approach_dist": params["approach_dist"],
                "min_inner_points": params["min_inner_points"],
                "finger_length": params["finger_length"],
                "finger_width": params["finger_width"],
                "height": params["height"],
                "run_dir": row["run_dir"],
            }
            phase_rows.append(trial)
            trials.append(trial)
            print(f"AP mean: {row['ap_mean'] * 100:.2f}%")

        best = max(phase_rows, key=lambda row: row["ap_mean"])
        best["selected"] = True
        current[variable] = best[variable]
        print(f"\nSelected {variable}={best[variable]} with AP mean {best['ap_mean'] * 100:.2f}%")

    csv_path, json_path = _write_history(args.root, trials, current)
    print("\nFinal greedy params:")
    print(json.dumps(current, indent=2))
    print(f"Saved greedy CSV:  {csv_path}")
    print(f"Saved greedy JSON: {json_path}")


if __name__ == "__main__":
    main()
