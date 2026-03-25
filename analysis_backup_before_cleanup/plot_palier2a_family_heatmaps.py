import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ANALYSIS_DIR = Path("analysis")
PLOT_DIR = ANALYSIS_DIR / "plots_palier2a_family"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

NEG_MATRIX = ANALYSIS_DIR / "palier2a_family_negative_original_delta_matrix.csv"
H3_MATRIX = ANALYSIS_DIR / "palier2a_family_h3_delta_matrix.csv"
CONTRAST_MATRIX = ANALYSIS_DIR / "palier2a_family_h3_minus_negative_original_matrix.csv"

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


def read_matrix_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Aucune ligne dans {path}")

    y_labels = [row["stimulus_group"] for row in rows]
    x_labels = [c for c in rows[0].keys() if c != "stimulus_group"]

    values = []
    for row in rows:
        current = []
        for x in x_labels:
            raw = str(row.get(x, "")).strip()
            if raw == "":
                current.append(np.nan)
            else:
                try:
                    current.append(float(raw))
                except Exception:
                    current.append(np.nan)
        values.append(current)

    return x_labels, y_labels, np.array(values, dtype=float)


def align_matrix(x_labels, data, desired_order):
    x_to_idx = {x: i for i, x in enumerate(x_labels)}
    aligned_cols = []

    for family in desired_order:
        if family in x_to_idx:
            aligned_cols.append(data[:, x_to_idx[family]])
        else:
            aligned_cols.append(np.full((data.shape[0],), np.nan))

    return desired_order, np.column_stack(aligned_cols)


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


def plot_heatmap(x_labels, y_labels, data, title, output_path, cbar_label):
    fig, ax = plt.subplots(figsize=(1.8 + 0.8 * len(x_labels), 1.8 + 0.9 * len(y_labels)))

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="lightgray")

    vmax = compute_symmetric_vmax(data)
    vmin = -vmax

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)

    ax.set_title(title, pad=14)
    ax.set_xlabel("Familles de tâches")
    ax.set_ylabel("Stimuli / conditions")

    annotate_heatmap(ax, data, fmt="{:.2f}", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    x_neg, y_neg, d_neg = read_matrix_csv(NEG_MATRIX)
    x_neg, d_neg = align_matrix(x_neg, d_neg, FAMILY_ORDER)
    plot_heatmap(
        x_neg,
        y_neg,
        d_neg,
        title="Negative original vs baseline — task families",
        output_path=PLOT_DIR / "family_negative_original_vs_baseline_heatmap.png",
        cbar_label="Delta score vs baseline",
    )

    x_h3, y_h3, d_h3 = read_matrix_csv(H3_MATRIX)
    x_h3, d_h3 = align_matrix(x_h3, d_h3, FAMILY_ORDER)
    plot_heatmap(
        x_h3,
        y_h3,
        d_h3,
        title="H3 vs baseline — task families",
        output_path=PLOT_DIR / "family_h3_vs_baseline_heatmap.png",
        cbar_label="Delta score vs baseline",
    )

    x_contrast, y_contrast, d_contrast = read_matrix_csv(CONTRAST_MATRIX)
    x_contrast, d_contrast = align_matrix(x_contrast, d_contrast, FAMILY_ORDER)
    plot_heatmap(
        x_contrast,
        y_contrast,
        d_contrast,
        title="H3 vs negative original — task families",
        output_path=PLOT_DIR / "family_h3_minus_negative_original_heatmap.png",
        cbar_label="Delta(H3) - Delta(negative original)",
    )

    print(f"Heatmaps écrites dans : {PLOT_DIR}")


if __name__ == "__main__":
    main()