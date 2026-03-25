import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ANALYSIS_DIR = Path("analysis")
PLOT_DIR = ANALYSIS_DIR / "plots_palier2a_family_compact"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_SUMMARY_PATH = ANALYSIS_DIR / "palier2a_family_summary.csv"

ROW_ORDER_ALL = [
    "negative_original",
    "competence_threat",
    "inconsistency",
    "social_comparison",
    "urgency",
    "regret",
]

ROW_ORDER_H3 = [
    "competence_threat",
    "inconsistency",
    "social_comparison",
    "urgency",
    "regret",
]

FAMILY_ORDER = [
    "causal_reasoning",
    "comparative_reasoning",
    "lexical_generation",
    "lexical_semantics",
    "morphology",
    "numerical_reasoning",
    "numerical_verbalization",
    "rewriting_transformation",
    "semantic_abstraction",
    "semantic_filtering",
    "semantic_similarity",
    "sentiment_classification",
    "string_extraction",
    "string_filtering",
    "string_transformation",
    "translation",
]


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def build_matrix(rows, value_field, row_order):
    lookup = {}
    for r in rows:
        lookup[(r["stimulus_group"], r["task_family"])] = to_float(r.get(value_field, ""))

    data = []
    for stim in row_order:
        row = []
        for family in FAMILY_ORDER:
            row.append(lookup.get((stim, family), np.nan))
        data.append(row)

    return np.array(data, dtype=float)


def compute_symmetric_vmax(data):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    return max(np.max(np.abs(finite)), 1e-6)


def annotate_heatmap(ax, data, fmt="{:.2f}", fontsize=8):
    n_rows, n_cols = data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=fontsize)


def plot_heatmap(data, row_labels, col_labels, title, out_path, cbar_label):
    fig, ax = plt.subplots(figsize=(2.2 + 0.8 * len(col_labels), 2.0 + 0.9 * len(row_labels)))

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="lightgray")

    vmax = compute_symmetric_vmax(data)
    vmin = -vmax

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title(title, pad=14)
    ax.set_xlabel("Familles de tâches")
    ax.set_ylabel("Conditions / stimuli")

    annotate_heatmap(ax, data, fmt="{:.2f}", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = read_csv(FAMILY_SUMMARY_PATH)

    # Heatmap 1 : toutes les conditions vs baseline
    data_all = build_matrix(rows, "mean_delta_vs_baseline", ROW_ORDER_ALL)
    plot_heatmap(
        data_all,
        ROW_ORDER_ALL,
        FAMILY_ORDER,
        title="Toutes les conditions vs baseline — task families",
        out_path=PLOT_DIR / "family_all_conditions_vs_baseline_heatmap.png",
        cbar_label="Delta score vs baseline",
    )

    # Heatmap 2 : H3 vs negative_original
    rows_h3 = [r for r in rows if r["stimulus_group"] != "negative_original"]
    data_contrast = build_matrix(rows_h3, "mean_delta_h3_minus_negative_original", ROW_ORDER_H3)
    plot_heatmap(
        data_contrast,
        ROW_ORDER_H3,
        FAMILY_ORDER,
        title="H3 vs negative_original — task families",
        out_path=PLOT_DIR / "family_h3_vs_negative_original_heatmap.png",
        cbar_label="Delta(H3) - Delta(negative_original)",
    )

    print(f"Heatmaps compactes écrites dans : {PLOT_DIR}")


if __name__ == "__main__":
    main()