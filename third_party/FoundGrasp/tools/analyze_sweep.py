import argparse
import csv
import html
import json
import math
from pathlib import Path


DEFAULT_METRICS = [
    "mean_ap",
    "seen_ap",
    "similar_ap",
    "novel_ap",
    "seen_ap@0.4",
    "similar_ap@0.4",
    "novel_ap@0.4",
    "seen_ap@0.8",
    "similar_ap@0.8",
    "novel_ap@0.8",
]

COMPACT_AP_METRICS = ["seen_ap", "similar_ap", "novel_ap"]

PARAM_KEYS = [
    "approach_kappa_risk_weight",
    "gripper_kappa_risk_weight",
    "graspness_threshold",
    "objectness_threshold",
    "total_score_threshold",
    "graspness_exp",
    "grasp_score_exp",
]


def _fmt(value, digits=3):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("graspnet_metrics"), dict):
        metrics = data["graspnet_metrics"]
        row = {
            "config": data.get("config", path.parent.name),
            "split": data.get("split", ""),
            "epoch": data.get("epoch", ""),
            "mean_ap": metrics.get("test/graspnet/all_ap"),
            "seen_ap": metrics.get("test/graspnet/seen_ap"),
            "similar_ap": metrics.get("test/graspnet/similar_ap"),
            "novel_ap": metrics.get("test/graspnet/novel_ap"),
            "seen_ap@0.4": metrics.get("test/graspnet/seen_ap@0.4"),
            "similar_ap@0.4": metrics.get("test/graspnet/similar_ap@0.4"),
            "novel_ap@0.4": metrics.get("test/graspnet/novel_ap@0.4"),
            "seen_ap@0.8": metrics.get("test/graspnet/seen_ap@0.8"),
            "similar_ap@0.8": metrics.get("test/graspnet/similar_ap@0.8"),
            "novel_ap@0.8": metrics.get("test/graspnet/novel_ap@0.8"),
            "log_dir": str(path.parent),
        }
        return [row]
    raise ValueError(f"Unsupported JSON format: {path}")


def label_for_path(path):
    path = Path(path)
    parts = path.parts
    if "paper" in parts:
        idx = parts.index("paper")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def infer_metrics(rows):
    present = {key for row in rows for key in row}
    return [metric for metric in DEFAULT_METRICS if metric in present]


def sort_rows(rows, metric):
    return sorted(rows, key=lambda row: _as_float(row.get(metric)) or -math.inf, reverse=True)


def baseline_row(rows):
    for row in rows:
        if (
            _as_float(row.get("approach_kappa_risk_weight")) == 0.0
            and _as_float(row.get("gripper_kappa_risk_weight")) == 0.0
        ):
            return row
    return None


def add_deltas(rows, metrics):
    base = baseline_row(rows)
    if base is None:
        return rows
    for row in rows:
        for metric in metrics:
            val = _as_float(row.get(metric))
            base_val = _as_float(base.get(metric))
            if val is not None and base_val is not None:
                row[f"delta_{metric}"] = val - base_val
    return rows


def write_csv(rows, metrics, out_path):
    keys = [
        "approach_kappa_risk_weight",
        "gripper_kappa_risk_weight",
        "graspness_threshold",
        "objectness_threshold",
        "total_score_threshold",
        "graspness_exp",
        "grasp_score_exp",
    ]
    fieldnames = [key for key in keys if any(key in row for row in rows)]
    fieldnames += metrics
    fieldnames += [f"delta_{metric}" for metric in metrics if any(f"delta_{metric}" in row for row in rows)]
    fieldnames += ["config", "split", "epoch", "log_dir"]
    fieldnames = [key for key in fieldnames if any(key in row for row in rows)]

    with Path(out_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def color_for(value, vmin, vmax):
    if value is None:
        return "#f3f4f6"
    if vmax <= vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # Red -> pale -> green.
    if t < 0.5:
        a = t / 0.5
        r, g, b = 248, int(113 + 120 * a), int(113 + 120 * a)
    else:
        a = (t - 0.5) / 0.5
        r, g, b = int(232 - 158 * a), int(240 - 68 * a), int(232 - 110 * a)
    return f"rgb({r},{g},{b})"


def heatmap_html(rows, metric):
    x_key = "gripper_kappa_risk_weight"
    y_key = "approach_kappa_risk_weight"
    if not all(x_key in row and y_key in row for row in rows):
        return ""

    xs = sorted({row[x_key] for row in rows}, key=float)
    ys = sorted({row[y_key] for row in rows}, key=float)
    cell = {}
    values = []
    for row in rows:
        value = _as_float(row.get(metric))
        cell[(row[y_key], row[x_key])] = value
        if value is not None:
            values.append(value)
    if not values:
        return ""
    vmin, vmax = min(values), max(values)

    parts = [f"<h3>{html.escape(metric)} heatmap</h3>", "<table class='heatmap'>"]
    parts.append("<tr><th>app \\ grip</th>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in xs) + "</tr>")
    for y in ys:
        row_parts = [f"<th>{html.escape(str(y))}</th>"]
        for x in xs:
            value = cell.get((y, x))
            bg = color_for(value, vmin, vmax)
            row_parts.append(f"<td style='background:{bg}'>{_fmt(value)}</td>")
        parts.append("<tr>" + "".join(row_parts) + "</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def heatmap_matrix(rows, metric):
    x_key = "gripper_kappa_risk_weight"
    y_key = "approach_kappa_risk_weight"
    if not all(x_key in row and y_key in row for row in rows):
        return None

    xs = sorted({row[x_key] for row in rows}, key=float)
    ys = sorted({row[y_key] for row in rows}, key=float)
    matrix = []
    for y in ys:
        matrix_row = []
        for x in xs:
            matched = [
                _as_float(row.get(metric))
                for row in rows
                if row.get(y_key) == y and row.get(x_key) == x
            ]
            matrix_row.append(matched[0] if matched and matched[0] is not None else float("nan"))
        matrix.append(matrix_row)
    return xs, ys, matrix


def write_png_heatmaps(rows, metrics, out_dir, stem):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"PNG skipped: matplotlib is not available ({exc})")
        return []

    out_paths = []
    for metric in metrics:
        heatmap = heatmap_matrix(rows, metric)
        if heatmap is None:
            continue
        xs, ys, matrix = heatmap
        if not matrix or not matrix[0]:
            continue

        width = max(5.0, 1.0 + 1.0 * len(xs))
        height = max(3.8, 1.0 + 0.75 * len(ys))
        fig, ax = plt.subplots(figsize=(width, height), dpi=180)
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(f"{metric} sweep")
        ax.set_xlabel("gripper_kappa_risk_weight")
        ax.set_ylabel("approach_kappa_risk_weight")
        ax.set_xticks(range(len(xs)), [str(x) for x in xs])
        ax.set_yticks(range(len(ys)), [str(y) for y in ys])

        for y_idx, row in enumerate(matrix):
            for x_idx, value in enumerate(row):
                if value == value:
                    ax.text(x_idx, y_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        safe_metric = metric.replace("@", "at").replace("/", "_")
        out_path = Path(out_dir) / f"{stem}_{safe_metric}_heatmap.png"
        fig.savefig(out_path)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def write_compare_csv(summary_rows, out_path):
    fieldnames = [
        "label",
        "source",
        "rank_metric",
        "rank_value",
        "approach_kappa_risk_weight",
        "gripper_kappa_risk_weight",
        "seen_ap",
        "similar_ap",
        "novel_ap",
        "mean_ap",
        "delta_seen_ap",
        "delta_similar_ap",
        "delta_novel_ap",
        "delta_mean_ap",
    ]
    with Path(out_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parameter_key(row):
    return tuple(str(row.get(key, "")) for key in PARAM_KEYS)


def parameter_dict_from_key(key):
    return dict(zip(PARAM_KEYS, key))


def write_shared_param_csv(rows, out_path):
    fieldnames = [
        "rank",
        "n_models",
        "avg_mean_ap",
        "avg_seen_ap",
        "avg_similar_ap",
        "avg_novel_ap",
        "min_mean_ap",
        "max_mean_ap",
        "std_mean_ap",
    ] + PARAM_KEYS
    with Path(out_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            out = {"rank": idx, **row}
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def shared_parameter_rows(paths, metric):
    per_json = []
    for json_path in paths:
        path = Path(json_path)
        rows = load_rows(path)
        label = label_for_path(path)
        by_param = {}
        for row in rows:
            key = parameter_key(row)
            # If duplicate parameter rows exist, keep the stronger one for the selected metric.
            old = by_param.get(key)
            if old is None or (_as_float(row.get(metric)) or -math.inf) > (_as_float(old.get(metric)) or -math.inf):
                by_param[key] = row
        per_json.append((label, by_param))

    common_keys = set(per_json[0][1])
    for _, by_param in per_json[1:]:
        common_keys &= set(by_param)

    shared_rows = []
    for key in common_keys:
        rows = [by_param[key] for _, by_param in per_json]
        mean_values = [_as_float(row.get("mean_ap")) for row in rows]
        seen_values = [_as_float(row.get("seen_ap")) for row in rows]
        similar_values = [_as_float(row.get("similar_ap")) for row in rows]
        novel_values = [_as_float(row.get("novel_ap")) for row in rows]
        rank_values = [_as_float(row.get(metric)) for row in rows]
        param_values = parameter_dict_from_key(key)
        shared = {
            **param_values,
            "n_models": len(rows),
            "avg_rank_metric": mean(rank_values),
            "avg_mean_ap": mean(mean_values),
            "avg_seen_ap": mean(seen_values),
            "avg_similar_ap": mean(similar_values),
            "avg_novel_ap": mean(novel_values),
            "min_mean_ap": min(v for v in mean_values if v is not None),
            "max_mean_ap": max(v for v in mean_values if v is not None),
            "std_mean_ap": std(mean_values),
        }
        for label, row in zip([item[0] for item in per_json], rows):
            safe_label = label.replace("-", "_").replace("/", "_")
            shared[f"{safe_label}_mean_ap"] = row.get("mean_ap")
            shared[f"{safe_label}_seen_ap"] = row.get("seen_ap")
            shared[f"{safe_label}_similar_ap"] = row.get("similar_ap")
            shared[f"{safe_label}_novel_ap"] = row.get("novel_ap")
        shared_rows.append(shared)

    return sorted(shared_rows, key=lambda row: _as_float(row.get("avg_rank_metric")) or -math.inf, reverse=True)


def write_shared_param_png(rows, out_path, top_k=8):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Shared-param PNG skipped: matplotlib is not available ({exc})")
        return None

    rows = rows[:top_k]
    labels = [
        f"a={row.get('approach_kappa_risk_weight')}, g={row.get('gripper_kappa_risk_weight')}"
        for row in rows
    ]
    metrics = ["avg_seen_ap", "avg_similar_ap", "avg_novel_ap"]
    colors = ["#2563eb", "#16a34a", "#f97316"]
    x = list(range(len(rows)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8.0, 1.3 * len(rows)), 4.8), dpi=180)

    for metric, color, offset in zip(metrics, colors, [-width, 0.0, width]):
        values = [_as_float(row.get(metric)) or 0.0 for row in rows]
        ax.bar([i + offset for i in x], values, width=width, color=color, label=metric)
        for i, value in enumerate(values):
            ax.text(i + offset, value + 0.2, f"{value:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_ylabel("Average AP across JSONs")
    ax.set_title("Shared parameter candidates")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend(ncols=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def write_compare_png(summary_rows, metrics, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Compare PNG skipped: matplotlib is not available ({exc})")
        return None

    labels = [row["label"] for row in summary_rows]
    x = list(range(len(labels)))
    width = 0.24
    fig_width = max(7.0, 1.3 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), dpi=180)

    offsets = [(-width), 0.0, width]
    colors = ["#2563eb", "#16a34a", "#f97316"]
    for metric, offset, color in zip(metrics, offsets, colors):
        values = [_as_float(row.get(metric)) or 0.0 for row in summary_rows]
        ax.bar([i + offset for i in x], values, width=width, label=metric, color=color)
        for i, value in enumerate(values):
            ax.text(i + offset, value + 0.25, f"{value:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_ylabel("AP")
    ax.set_title("Best sweep row per JSON")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend(ncols=len(metrics), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def write_compare_delta_png(summary_rows, metrics, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Compare delta PNG skipped: matplotlib is not available ({exc})")
        return None

    delta_metrics = [f"delta_{metric}" for metric in metrics]
    if not any(any(key in row for key in delta_metrics) for row in summary_rows):
        return None

    labels = [row["label"] for row in summary_rows]
    x = list(range(len(labels)))
    width = 0.24
    fig_width = max(7.0, 1.3 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.3), dpi=180)

    offsets = [(-width), 0.0, width]
    colors = ["#1d4ed8", "#15803d", "#c2410c"]
    for metric, offset, color in zip(metrics, offsets, colors):
        key = f"delta_{metric}"
        values = [_as_float(row.get(key)) or 0.0 for row in summary_rows]
        ax.bar([i + offset for i in x], values, width=width, label=key, color=color)
        for i, value in enumerate(values):
            ax.text(
                i + offset,
                value + (0.035 if value >= 0 else -0.035),
                f"{value:+.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
                rotation=90,
            )

    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_ylabel("AP delta vs app=0, grip=0")
    ax.set_title("Best-row delta per JSON")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend(ncols=len(metrics), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def compare_jsons(paths, metric, out_dir, top_k):
    summary_rows = []
    for json_path in paths:
        path = Path(json_path)
        rows = load_rows(path)
        metrics = infer_metrics(rows)
        rows = add_deltas(rows, metrics)
        rank_metric = metric if metric in metrics else ("mean_ap" if "mean_ap" in metrics else metrics[0])
        best = sort_rows(rows, rank_metric)[0]
        summary = {
            "label": label_for_path(path),
            "source": str(path),
            "rank_metric": rank_metric,
            "rank_value": best.get(rank_metric),
        }
        for key, value in best.items():
            summary[key] = value
        summary_rows.append(summary)

    summary_rows = sorted(summary_rows, key=lambda row: _as_float(row.get("rank_value")) or -math.inf, reverse=True)
    out_dir = Path(out_dir) if out_dir else Path(paths[0]).parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "compare_seen_similar_novel.csv"
    png_path = out_dir / "compare_seen_similar_novel.png"
    delta_png_path = out_dir / "compare_seen_similar_novel_delta.png"

    write_compare_csv(summary_rows, csv_path)
    made_png = write_compare_png(summary_rows, COMPACT_AP_METRICS, png_path)
    made_delta_png = write_compare_delta_png(summary_rows, COMPACT_AP_METRICS, delta_png_path)

    print("\n=== Compare JSONs ===")
    for idx, row in enumerate(summary_rows[:top_k], start=1):
        print(
            f"{idx:02d}. {row['label']} "
            f"seen={_fmt(row.get('seen_ap'))} "
            f"similar={_fmt(row.get('similar_ap'))} "
            f"novel={_fmt(row.get('novel_ap'))} "
            f"mean={_fmt(row.get('mean_ap'))} "
            f"app={row.get('approach_kappa_risk_weight', '')} "
            f"grip={row.get('gripper_kappa_risk_weight', '')}"
        )
    print(f"csv: {csv_path}")
    if made_png:
        print(f"png: {made_png}")
    if made_delta_png:
        print(f"png: {made_delta_png}")


def matches_fixed_params(row, fixed_params):
    for key, expected in fixed_params.items():
        actual = row.get(key)
        actual_float = _as_float(actual)
        expected_float = _as_float(expected)
        if actual_float is not None and expected_float is not None:
            if abs(actual_float - expected_float) > 1e-9:
                return False
        elif str(actual) != str(expected):
            return False
    return True


def compare_fixed_params(paths, fixed_params, out_dir, top_k):
    summary_rows = []
    missing = []
    for json_path in paths:
        path = Path(json_path)
        rows = load_rows(path)
        rows = add_deltas(rows, infer_metrics(rows))
        matched = [row for row in rows if matches_fixed_params(row, fixed_params)]
        if not matched:
            missing.append(str(path))
            continue
        row = matched[0]
        summary = {
            "label": label_for_path(path),
            "source": str(path),
            "rank_metric": "fixed_params",
            "rank_value": row.get("mean_ap"),
        }
        summary.update(row)
        summary_rows.append(summary)

    summary_rows = sorted(summary_rows, key=lambda row: _as_float(row.get("mean_ap")) or -math.inf, reverse=True)
    out_dir = Path(out_dir) if out_dir else Path(paths[0]).parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_".join(f"{key}-{value}" for key, value in fixed_params.items())
    suffix = suffix.replace(".", "p")
    csv_path = out_dir / f"compare_fixed_{suffix}.csv"
    png_path = out_dir / f"compare_fixed_{suffix}.png"
    delta_png_path = out_dir / f"compare_fixed_{suffix}_delta.png"

    write_compare_csv(summary_rows, csv_path)
    made_png = write_compare_png(summary_rows, COMPACT_AP_METRICS, png_path)
    made_delta_png = write_compare_delta_png(summary_rows, COMPACT_AP_METRICS, delta_png_path)

    print("\n=== Fixed-Parameter Method Comparison ===")
    print("fixed params:", fixed_params)
    if missing:
        print("missing fixed row:")
        for path in missing:
            print(f"  {path}")
    for idx, row in enumerate(summary_rows[:top_k], start=1):
        print(
            f"{idx:02d}. {row['label']} "
            f"seen={_fmt(row.get('seen_ap'))} "
            f"similar={_fmt(row.get('similar_ap'))} "
            f"novel={_fmt(row.get('novel_ap'))} "
            f"mean={_fmt(row.get('mean_ap'))} "
            f"dmean={_fmt(row.get('delta_mean_ap'))}"
        )
    print(f"csv: {csv_path}")
    if made_png:
        print(f"png: {made_png}")
    if made_delta_png:
        print(f"png: {made_delta_png}")


def select_shared_params(paths, metric, out_dir, top_k):
    rows = shared_parameter_rows(paths, metric)
    out_dir = Path(out_dir) if out_dir else Path(paths[0]).parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "shared_parameter_selection.csv"
    png_path = out_dir / "shared_parameter_selection.png"
    write_shared_param_csv(rows, csv_path)
    made_png = write_shared_param_png(rows, png_path, top_k=top_k)

    print("\n=== Shared Parameter Selection ===")
    print(f"common parameter rows: {len(rows)}")
    for idx, row in enumerate(rows[:top_k], start=1):
        print(
            f"{idx:02d}. avg_mean={_fmt(row.get('avg_mean_ap'))} "
            f"avg_seen={_fmt(row.get('avg_seen_ap'))} "
            f"avg_similar={_fmt(row.get('avg_similar_ap'))} "
            f"avg_novel={_fmt(row.get('avg_novel_ap'))} "
            f"std_mean={_fmt(row.get('std_mean_ap'))} "
            f"app={row.get('approach_kappa_risk_weight')} "
            f"grip={row.get('gripper_kappa_risk_weight')}"
        )
    print(f"csv: {csv_path}")
    if made_png:
        print(f"png: {made_png}")


def rows_table_html(rows, columns, limit=None):
    shown = rows[:limit] if limit else rows
    parts = ["<table>", "<tr>" + "".join(f"<th>{html.escape(col)}</th>" for col in columns) + "</tr>"]
    for row in shown:
        parts.append(
            "<tr>"
            + "".join(f"<td>{html.escape(_fmt(row.get(col)))}</td>" for col in columns)
            + "</tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)


def write_html(rows, metrics, metric, out_path, title):
    sorted_rows = sort_rows(rows, metric)
    top_cols = [
        key for key in [
            "approach_kappa_risk_weight",
            "gripper_kappa_risk_weight",
            metric,
            f"delta_{metric}",
            "seen_ap",
            "similar_ap",
            "novel_ap",
            "log_dir",
        ]
        if any(key in row for row in rows)
    ]
    best = sorted_rows[0] if sorted_rows else {}
    heatmaps = "\n".join(heatmap_html(rows, m) for m in metrics if m in rows[0])

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #111827; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    h3 {{ font-size: 15px; margin-top: 22px; }}
    .summary {{ padding: 12px 14px; border: 1px solid #d1d5db; border-radius: 8px; background: #f9fafb; width: fit-content; }}
    table {{ border-collapse: collapse; margin: 10px 0 18px; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; white-space: nowrap; }}
    th {{ background: #f3f4f6; font-weight: 650; }}
    td:last-child {{ text-align: left; }}
    .heatmap td {{ min-width: 78px; font-weight: 650; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="summary">
    <div><b>Rows:</b> {len(rows)}</div>
    <div><b>Primary metric:</b> <code>{html.escape(metric)}</code></div>
    <div><b>Best:</b> {_fmt(best.get(metric))}</div>
    <div><b>Best app/grip risk:</b> {_fmt(best.get("approach_kappa_risk_weight"))} / {_fmt(best.get("gripper_kappa_risk_weight"))}</div>
  </div>

  <h2>Top Rows</h2>
  {rows_table_html(sorted_rows, top_cols, limit=20)}

  <h2>Heatmaps</h2>
  {heatmaps}
</body>
</html>
"""
    Path(out_path).write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+", help="Sweep result JSON(s) or final_test result JSON(s)")
    parser.add_argument("--metric", default="mean_ap")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--compare", action="store_true",
                        help="Compare the best row from each JSON using only seen/similar/novel AP.")
    parser.add_argument("--select-shared-params", action="store_true",
                        help="Choose one shared parameter row by averaging AP across all JSONs.")
    parser.add_argument("--fixed-approach-risk", type=float, default=None,
                        help="Compare methods at this exact approach_kappa_risk_weight.")
    parser.add_argument("--fixed-gripper-risk", type=float, default=None,
                        help="Compare methods at this exact gripper_kappa_risk_weight.")
    args = parser.parse_args()

    if args.select_shared_params:
        select_shared_params(args.json, args.metric, args.out_dir, args.top_k)
        return

    fixed_params = {}
    if args.fixed_approach_risk is not None:
        fixed_params["approach_kappa_risk_weight"] = args.fixed_approach_risk
    if args.fixed_gripper_risk is not None:
        fixed_params["gripper_kappa_risk_weight"] = args.fixed_gripper_risk
    if fixed_params:
        compare_fixed_params(args.json, fixed_params, args.out_dir, args.top_k)
        return

    if args.compare:
        compare_jsons(args.json, args.metric, args.out_dir, args.top_k)
        return

    for json_path in args.json:
        path = Path(json_path)
        rows = load_rows(path)
        metrics = infer_metrics(rows)
        metrics = [metric for metric in COMPACT_AP_METRICS + ["mean_ap"] if metric in metrics]
        metric = args.metric if args.metric in metrics else (metrics[0] if metrics else args.metric)
        rows = add_deltas(rows, metrics)
        sorted_rows = sort_rows(rows, metric)

        out_dir = Path(args.out_dir) if args.out_dir else path.parent / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem
        csv_path = out_dir / f"{stem}_analysis.csv"
        html_path = out_dir / f"{stem}_analysis.html"

        write_csv(sorted_rows, metrics, csv_path)
        write_html(rows, metrics, metric, html_path, title=str(path))
        png_paths = write_png_heatmaps(rows, metrics, out_dir, stem)

        print(f"\n=== {path} ===")
        print(f"metric: {metric}")
        print(f"rows: {len(rows)}")
        for idx, row in enumerate(sorted_rows[: args.top_k], start=1):
            app = row.get("approach_kappa_risk_weight", "")
            grip = row.get("gripper_kappa_risk_weight", "")
            delta = row.get(f"delta_{metric}", "")
            print(
                f"{idx:02d}. {metric}={_fmt(row.get(metric))} "
                f"delta={_fmt(delta)} app={app} grip={grip}"
            )
        print(f"csv:  {csv_path}")
        print(f"html: {html_path}")
        for png_path in png_paths:
            print(f"png:  {png_path}")


if __name__ == "__main__":
    main()
