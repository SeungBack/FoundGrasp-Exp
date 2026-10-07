#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


PARAM_RE = re.compile(
    r"_(?:cd|hc)_ad(?P<approach_dist>[^_]+)"
    r"_mi(?P<min_inner_points>[^_]+)"
    r"_fl(?P<finger_length>[^_]+)"
    r"_fw(?P<finger_width>[^_]+)"
    r"_h(?P<height>[^_]+)"
)


def _mean_split_ap(result):
    split_aps = []
    for split_result in result.values():
        if isinstance(split_result, dict) and "ap" in split_result:
            split_aps.append(float(split_result["ap"]))
    return sum(split_aps) / len(split_aps) if split_aps else None


def _split_ap(result, split):
    split_result = result.get(split)
    if isinstance(split_result, dict) and "ap" in split_result:
        return float(split_result["ap"])
    return None


def collect_rows(root):
    rows = []
    for eval_path in sorted(Path(root).glob("**/eval_results.json")):
        run_dir = eval_path.parent
        match = PARAM_RE.search(run_dir.name)
        if match is None:
            continue

        with eval_path.open("r", encoding="utf-8") as f:
            result = json.load(f)

        params = match.groupdict()
        row = {
            "run_dir": str(run_dir),
            "approach_dist": float(params["approach_dist"]),
            "min_inner_points": int(float(params["min_inner_points"])),
            "finger_length": float(params["finger_length"]),
            "finger_width": float(params["finger_width"]),
            "height": float(params["height"]),
            "ap_mean": _mean_split_ap(result),
            "ap_seen": _split_ap(result, "test_seen"),
            "ap_similar": _split_ap(result, "test_similar"),
            "ap_novel": _split_ap(result, "test_novel"),
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs/paper/dav3b",
                        help="Root directory to search for hard collision sweep eval_results.json files.")
    parser.add_argument("--out", default=None,
                        help="CSV output path. Defaults to <root>/hard_collision_sweep_summary.csv.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rows = collect_rows(args.root)
    rows = [row for row in rows if row["ap_mean"] is not None]
    rows.sort(key=lambda row: row["ap_mean"], reverse=True)

    out_path = Path(args.out) if args.out else Path(args.root) / "hard_collision_sweep_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ap_mean", "ap_seen", "ap_similar", "ap_novel",
        "approach_dist", "min_inner_points", "finger_length", "finger_width", "height",
        "run_dir",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Found {len(rows)} completed hard-collision runs.")
    print(f"Saved CSV:  {out_path}")
    print(f"Saved JSON: {json_path}")
    for rank, row in enumerate(rows[:args.top_k], start=1):
        print(
            f"#{rank:02d} AP={row['ap_mean'] * 100:.2f} "
            f"seen={row['ap_seen'] * 100:.2f} "
            f"similar={row['ap_similar'] * 100:.2f} "
            f"novel={row['ap_novel'] * 100:.2f} "
            f"ad={row['approach_dist']} mi={row['min_inner_points']} "
            f"fl={row['finger_length']} fw={row['finger_width']} h={row['height']}"
        )


if __name__ == "__main__":
    main()
