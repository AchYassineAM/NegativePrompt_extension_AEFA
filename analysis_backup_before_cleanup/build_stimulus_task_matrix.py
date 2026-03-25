import csv
from pathlib import Path
from collections import OrderedDict

# ============================================================
# BUILD STIMULUS × TASK MATRICES
# ============================================================
#
# Objectif :
# transformer les agrégations déjà produites en matrices
# directement exploitables pour des heatmaps.
#
# Entrées :
# - analysis/aggregated_by_strata.csv
# - analysis/aggregated_secondary_metrics_numeric.csv
#
# Sorties :
# - analysis/stimulus_task_long.csv
# - analysis/stimulus_task_delta_matrix.csv
# - analysis/stimulus_task_mean_score_matrix.csv
# - analysis/stimulus_task_std_matrix.csv
# - analysis/stimulus_task_numeric_long.csv
# - analysis/stimulus_task_numeric_delta_abs_error_matrix.csv
#
# Idée :
# 1) métrique principale = delta_vs_baseline
# 2) métrique secondaire numérique = delta_mean_abs_error_vs_baseline
#
# Interprétation :
# - Pour delta_vs_baseline :
#     positif  = meilleur que la baseline
#     négatif  = moins bon que la baseline
# - Pour delta_mean_abs_error_vs_baseline :
#     positif  = erreur plus grande, donc pire
#     négatif  = erreur plus faible, donc meilleur
# ============================================================

PRIMARY_INPUT = Path("analysis") / "aggregated_by_strata.csv"
SECONDARY_INPUT = Path("analysis") / "aggregated_secondary_metrics_numeric.csv"

PRIMARY_LONG = Path("analysis") / "stimulus_task_long.csv"
PRIMARY_DELTA_MATRIX = Path("analysis") / "stimulus_task_delta_matrix.csv"
PRIMARY_MEAN_MATRIX = Path("analysis") / "stimulus_task_mean_score_matrix.csv"
PRIMARY_STD_MATRIX = Path("analysis") / "stimulus_task_std_matrix.csv"

SECONDARY_LONG = Path("analysis") / "stimulus_task_numeric_long.csv"
SECONDARY_DELTA_MATRIX = Path("analysis") / "stimulus_task_numeric_delta_abs_error_matrix.csv"

ROW_ORDER = [
    "negative_original",
    "competence_threat",
    "inconsistency",
    "social_comparison",
    "urgency",
    "regret",
]

TASK_ORDER = [
    "sentiment",
    "sentence_similarity",
    "word_in_context",
    "cause_and_effect",
    "sum",
    "diff",
]


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def ordered_unique(values, preferred_order=None):
    values = list(OrderedDict.fromkeys(values))
    if not preferred_order:
        return values

    preferred = [v for v in preferred_order if v in values]
    others = [v for v in values if v not in preferred]
    return preferred + others


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_primary():
    rows = read_csv(PRIMARY_INPUT)

    # --------------------------------------------------------
    # On ne garde que la vue principale de base :
    # - prompt_source = base
    # - benchmark = instruction_induction
    # - on enlève baseline (delta=0 partout)
    # --------------------------------------------------------
    filtered = []
    for r in rows:
        if r.get("prompt_source") != "base":
            continue
        if r.get("benchmark") != "instruction_induction":
            continue
        if r.get("stimulus_group") == "baseline":
            continue

        task_name = r.get("tasks", "").strip()
        filtered.append({
            "model": r.get("model", "").strip(),
            "stimulus_group": r.get("stimulus_group", "").strip(),
            "task": task_name,
            "task_family": r.get("task_family", "").strip(),
            "mean_score": r.get("mean_score", "").strip(),
            "std_score": r.get("std_score", "").strip(),
            "delta_vs_baseline": r.get("delta_vs_baseline", "").strip(),
            "n_seeds": r.get("n_seeds", "").strip(),
            "n_stimuli": r.get("n_stimuli", "").strip(),
        })

    write_csv(
        PRIMARY_LONG,
        fieldnames=[
            "model",
            "stimulus_group",
            "task",
            "task_family",
            "mean_score",
            "std_score",
            "delta_vs_baseline",
            "n_seeds",
            "n_stimuli",
        ],
        rows=filtered,
    )

    task_values = ordered_unique([r["task"] for r in filtered], TASK_ORDER)
    row_values = ordered_unique([r["stimulus_group"] for r in filtered], ROW_ORDER)

    delta_rows = []
    mean_rows = []
    std_rows = []

    for stim in row_values:
        row_delta = {"stimulus_group": stim}
        row_mean = {"stimulus_group": stim}
        row_std = {"stimulus_group": stim}

        for task in task_values:
            match = next(
                (r for r in filtered if r["stimulus_group"] == stim and r["task"] == task),
                None
            )

            row_delta[task] = match["delta_vs_baseline"] if match else ""
            row_mean[task] = match["mean_score"] if match else ""
            row_std[task] = match["std_score"] if match else ""

        delta_rows.append(row_delta)
        mean_rows.append(row_mean)
        std_rows.append(row_std)

    fieldnames = ["stimulus_group"] + task_values
    write_csv(PRIMARY_DELTA_MATRIX, fieldnames, delta_rows)
    write_csv(PRIMARY_MEAN_MATRIX, fieldnames, mean_rows)
    write_csv(PRIMARY_STD_MATRIX, fieldnames, std_rows)

    return len(filtered), len(delta_rows)


def build_secondary():
    if not SECONDARY_INPUT.exists():
        print(f"Fichier absent, matrice secondaire ignorée : {SECONDARY_INPUT}")
        return 0, 0

    rows = read_csv(SECONDARY_INPUT)

    filtered = []
    for r in rows:
        if r.get("stimulus_group") == "baseline":
            continue

        task_name = r.get("task", "").strip()
        filtered.append({
            "model": r.get("model", "").strip(),
            "stimulus_group": r.get("stimulus_group", "").strip(),
            "task": task_name,
            "mean_parseable_rate": r.get("mean_parseable_rate", "").strip(),
            "mean_mean_abs_error": r.get("mean_mean_abs_error", "").strip(),
            "mean_median_abs_error": r.get("mean_median_abs_error", "").strip(),
            "delta_mean_abs_error_vs_baseline": r.get("delta_mean_abs_error_vs_baseline", "").strip(),
            "delta_median_abs_error_vs_baseline": r.get("delta_median_abs_error_vs_baseline", "").strip(),
            "n_seeds": r.get("n_seeds", "").strip(),
            "n_stimuli": r.get("n_stimuli", "").strip(),
        })

    write_csv(
        SECONDARY_LONG,
        fieldnames=[
            "model",
            "stimulus_group",
            "task",
            "mean_parseable_rate",
            "mean_mean_abs_error",
            "mean_median_abs_error",
            "delta_mean_abs_error_vs_baseline",
            "delta_median_abs_error_vs_baseline",
            "n_seeds",
            "n_stimuli",
        ],
        rows=filtered,
    )

    task_values = ordered_unique([r["task"] for r in filtered], TASK_ORDER)
    row_values = ordered_unique([r["stimulus_group"] for r in filtered], ROW_ORDER)

    delta_rows = []

    for stim in row_values:
        row = {"stimulus_group": stim}
        for task in task_values:
            match = next(
                (r for r in filtered if r["stimulus_group"] == stim and r["task"] == task),
                None
            )
            row[task] = match["delta_mean_abs_error_vs_baseline"] if match else ""
        delta_rows.append(row)

    fieldnames = ["stimulus_group"] + task_values
    write_csv(SECONDARY_DELTA_MATRIX, fieldnames, delta_rows)

    return len(filtered), len(delta_rows)


def main():
    n_primary_long, n_primary_rows = build_primary()
    n_secondary_long, n_secondary_rows = build_secondary()

    print(f"{n_primary_long} lignes écrites dans : {PRIMARY_LONG}")
    print(f"{n_primary_rows} lignes écrites dans : {PRIMARY_DELTA_MATRIX}")
    print(f"{n_secondary_long} lignes écrites dans : {SECONDARY_LONG}")
    print(f"{n_secondary_rows} lignes écrites dans : {SECONDARY_DELTA_MATRIX}")


if __name__ == "__main__":
    main()