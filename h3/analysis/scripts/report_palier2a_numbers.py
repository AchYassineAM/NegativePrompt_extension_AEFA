import csv
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
H3_ROOT = SCRIPT_DIR.parent.parent
REPO_ROOT = H3_ROOT.parent

ANALYSIS_DIR = H3_ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
H3_TABLES_DIR = TABLES_DIR / "h3"

AGG_PATH = H3_TABLES_DIR / "palier2a_aggregated_conditions.csv"
NEG_SUMMARY_PATH = H3_TABLES_DIR / "palier2a_negative_original_summary.csv"
H3_VS_NEG_PATH = H3_TABLES_DIR / "palier2a_h3_vs_negative_original.csv"

OUT_TOP_H3 = H3_TABLES_DIR / "palier2a_top_h3_vs_baseline.csv"
OUT_TOP_NEG = H3_TABLES_DIR / "palier2a_top_negative_original_vs_baseline.csv"
OUT_TOP_CONTRAST = H3_TABLES_DIR / "palier2a_top_h3_vs_negative_original.csv"
OUT_FAMILY_SUMMARY = H3_TABLES_DIR / "palier2a_family_summary.csv"
OUT_TASK_SUMMARY = H3_TABLES_DIR / "palier2a_task_summary.csv"


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sort_key_desc_abs(row, field):
    v = safe_float(row[field])
    return -abs(v if v is not None else -9999)


def main():
    agg = read_csv(AGG_PATH)
    neg = read_csv(NEG_SUMMARY_PATH)
    contrast = read_csv(H3_VS_NEG_PATH)

    # --------------------------------------------------------
    # 1) TOP TASKES / CATEGORIES H3 VS BASELINE
    # --------------------------------------------------------
    h3_rows = []
    for r in agg:
        if r["stimulus_group"] in {"baseline", "negative_original"}:
            continue
        delta = safe_float(r["delta_vs_baseline"])
        if delta is None:
            continue
        h3_rows.append({
            "task": r["task"],
            "task_family": r["task_family"],
            "stimulus_group": r["stimulus_group"],
            "mean_score": r["mean_score"],
            "baseline_score": r["baseline_score"],
            "delta_vs_baseline": r["delta_vs_baseline"],
            "std_score": r["std_score"],
            "direction": "better_than_baseline" if delta > 0 else ("worse_than_baseline" if delta < 0 else "equal_to_baseline")
        })

    h3_sorted = sorted(h3_rows, key=lambda r: abs(float(r["delta_vs_baseline"])), reverse=True)
    write_csv(
        OUT_TOP_H3,
        fieldnames=[
            "task", "task_family", "stimulus_group",
            "mean_score", "baseline_score", "delta_vs_baseline",
            "std_score", "direction"
        ],
        rows=h3_sorted
    )

    # --------------------------------------------------------
    # 2) TOP NEGATIVE_ORIGINAL VS BASELINE
    # --------------------------------------------------------
    neg_rows = []
    for r in neg:
        delta = safe_float(r["delta_negative_original_vs_baseline"])
        if delta is None:
            continue
        neg_rows.append({
            "task": r["task"],
            "task_family": r["task_family"],
            "mean_negative_original_score": r["mean_negative_original_score"],
            "baseline_score": r["baseline_score"],
            "delta_negative_original_vs_baseline": r["delta_negative_original_vs_baseline"],
            "direction": "better_than_baseline" if delta > 0 else ("worse_than_baseline" if delta < 0 else "equal_to_baseline")
        })

    neg_sorted = sorted(
        neg_rows,
        key=lambda r: abs(float(r["delta_negative_original_vs_baseline"])),
        reverse=True
    )
    write_csv(
        OUT_TOP_NEG,
        fieldnames=[
            "task", "task_family",
            "mean_negative_original_score", "baseline_score",
            "delta_negative_original_vs_baseline", "direction"
        ],
        rows=neg_sorted
    )

    # --------------------------------------------------------
    # 3) TOP H3 VS NEGATIVE_ORIGINAL
    # --------------------------------------------------------
    contrast_rows = []
    for r in contrast:
        delta = safe_float(r["delta_h3_minus_negative_original"])
        if delta is None:
            continue
        contrast_rows.append({
            "task": r["task"],
            "task_family": r["task_family"],
            "stimulus_group": r["stimulus_group"],
            "delta_vs_baseline": r["delta_vs_baseline"],
            "negative_original_delta_vs_baseline": r["negative_original_delta_vs_baseline"],
            "delta_h3_minus_negative_original": r["delta_h3_minus_negative_original"],
            "direction": "H3_better_than_negative_original" if delta > 0 else ("H3_worse_than_negative_original" if delta < 0 else "equal")
        })

    contrast_sorted = sorted(
        contrast_rows,
        key=lambda r: abs(float(r["delta_h3_minus_negative_original"])),
        reverse=True
    )
    write_csv(
        OUT_TOP_CONTRAST,
        fieldnames=[
            "task", "task_family", "stimulus_group",
            "delta_vs_baseline", "negative_original_delta_vs_baseline",
            "delta_h3_minus_negative_original", "direction"
        ],
        rows=contrast_sorted
    )

    # --------------------------------------------------------
    # 4) RESUME PAR FAMILLE DE TACHES
    # --------------------------------------------------------
    family_h3 = defaultdict(list)
    family_neg = defaultdict(list)
    family_contrast = defaultdict(list)

    for r in h3_rows:
        family_h3[(r["task_family"], r["stimulus_group"])].append(float(r["delta_vs_baseline"]))

    for r in neg_rows:
        family_neg[r["task_family"]].append(float(r["delta_negative_original_vs_baseline"]))

    for r in contrast_rows:
        family_contrast[(r["task_family"], r["stimulus_group"])].append(float(r["delta_h3_minus_negative_original"]))

    family_summary_rows = []

    all_families = sorted({r["task_family"] for r in agg})

    for fam in all_families:
        neg_mean = mean(family_neg.get(fam, []))
        family_summary_rows.append({
            "task_family": fam,
            "stimulus_group": "negative_original",
            "mean_delta_vs_baseline": round(neg_mean, 6) if neg_mean is not None else "",
            "mean_delta_h3_minus_negative_original": "",
            "n_tasks": len(family_neg.get(fam, []))
        })

        for stim in sorted({r["stimulus_group"] for r in h3_rows}):
            h3_mean = mean(family_h3.get((fam, stim), []))
            contrast_mean = mean(family_contrast.get((fam, stim), []))
            family_summary_rows.append({
                "task_family": fam,
                "stimulus_group": stim,
                "mean_delta_vs_baseline": round(h3_mean, 6) if h3_mean is not None else "",
                "mean_delta_h3_minus_negative_original": round(contrast_mean, 6) if contrast_mean is not None else "",
                "n_tasks": len(family_h3.get((fam, stim), []))
            })

    write_csv(
        OUT_FAMILY_SUMMARY,
        fieldnames=[
            "task_family", "stimulus_group",
            "mean_delta_vs_baseline",
            "mean_delta_h3_minus_negative_original",
            "n_tasks"
        ],
        rows=family_summary_rows
    )

    # --------------------------------------------------------
    # 5) RESUME PAR TACHE
    # --------------------------------------------------------
    task_summary_rows = []
    baseline_lookup = {}
    neg_lookup = {}

    for r in agg:
        if r["stimulus_group"] == "baseline":
            baseline_lookup[r["task"]] = float(r["mean_score"])

    for r in neg_rows:
        neg_lookup[r["task"]] = float(r["delta_negative_original_vs_baseline"])

    h3_by_task = defaultdict(list)
    contrast_by_task = defaultdict(list)

    for r in h3_rows:
        h3_by_task[r["task"]].append(float(r["delta_vs_baseline"]))

    for r in contrast_rows:
        contrast_by_task[r["task"]].append(float(r["delta_h3_minus_negative_original"]))

    for task in sorted({r["task"] for r in agg}):
        task_summary_rows.append({
            "task": task,
            "task_family": next((r["task_family"] for r in agg if r["task"] == task), ""),
            "baseline_score": baseline_lookup.get(task, ""),
            "negative_original_delta_vs_baseline": neg_lookup.get(task, ""),
            "best_h3_delta_vs_baseline": round(max(h3_by_task.get(task, [0])), 6) if h3_by_task.get(task) else "",
            "worst_h3_delta_vs_baseline": round(min(h3_by_task.get(task, [0])), 6) if h3_by_task.get(task) else "",
            "best_h3_minus_negative_original": round(max(contrast_by_task.get(task, [0])), 6) if contrast_by_task.get(task) else "",
            "worst_h3_minus_negative_original": round(min(contrast_by_task.get(task, [0])), 6) if contrast_by_task.get(task) else "",
        })

    write_csv(
        OUT_TASK_SUMMARY,
        fieldnames=[
            "task", "task_family", "baseline_score",
            "negative_original_delta_vs_baseline",
            "best_h3_delta_vs_baseline", "worst_h3_delta_vs_baseline",
            "best_h3_minus_negative_original", "worst_h3_minus_negative_original"
        ],
        rows=task_summary_rows
    )

    print(f"Wrote: {OUT_TOP_H3}")
    print(f"Wrote: {OUT_TOP_NEG}")
    print(f"Wrote: {OUT_TOP_CONTRAST}")
    print(f"Wrote: {OUT_FAMILY_SUMMARY}")
    print(f"Wrote: {OUT_TASK_SUMMARY}")


if __name__ == "__main__":
    main()