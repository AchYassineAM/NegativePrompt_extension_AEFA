import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PLOT STIMULUS × TASK HEATMAPS
# ============================================================
#
# Objectif :
# produire des visualisations directes des interactions
# stimulus × tâche à partir des matrices CSV déjà construites.
#
# Entrées :
# - analysis/stimulus_task_delta_matrix.csv
# - analysis/stimulus_task_numeric_delta_abs_error_matrix.csv
#
# Sorties :
# - analysis/plots/stimulus_task_delta_heatmap.png
# - analysis/plots/stimulus_task_numeric_abs_error_heatmap.png
#
# Interprétation :
# 1) Heatmap principale : delta_vs_baseline
#    positif = meilleur que baseline
#    négatif = moins bon que baseline
#
# 2) Heatmap numérique : delta_mean_abs_error_vs_baseline
#    positif = erreur plus grande, donc pire
#    négatif = erreur plus faible, donc meilleur
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

ANALYSIS_DIR = REPO_ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
STIMULUS_TASK_DIR = TABLES_DIR / "stimulus_task"

PRIMARY_MATRIX = STIMULUS_TASK_DIR / "stimulus_task_delta_matrix.csv"
NUMERIC_MATRIX = STIMULUS_TASK_DIR / "stimulus_task_numeric_delta_abs_error_matrix.csv"

PLOT_DIR = ANALYSIS_DIR / "plots" / "stimulus_task"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


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


def compute_symmetric_vmax(data: np.ndarray):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    vmax = np.max(np.abs(finite))
    return max(vmax, 1e-6)


def annotate_heatmap(ax, data, fmt="{:.3f}"):
    n_rows, n_cols = data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j, i, fmt.format(val),
                ha="center",
                va="center",
                fontsize=9
            )


def plot_heatmap(x_labels, y_labels, data, title, output_path, cbar_label):
    fig, ax = plt.subplots(figsize=(1.5 + 1.6 * len(x_labels), 1.5 + 0.75 * len(y_labels)))

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="lightgray")

    vmax = compute_symmetric_vmax(data)
    vmin = -vmax

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticklabels(y_labels)

    ax.set_title(title, pad=16)
    ax.set_xlabel("Tâches")
    ax.set_ylabel("Stimuli")

    annotate_heatmap(ax, data)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    x1, y1, d1 = read_matrix_csv(PRIMARY_MATRIX)
    plot_heatmap(
        x1, y1, d1,
        title="Stimulus × tâche — delta vs baseline",
        output_path=PLOT_DIR / "stimulus_task_delta_heatmap.png",
        cbar_label="Delta score vs baseline"
    )

    x2, y2, d2 = read_matrix_csv(NUMERIC_MATRIX)
    plot_heatmap(
        x2, y2, d2,
        title="Stimulus × tâche numérique — delta erreur absolue",
        output_path=PLOT_DIR / "stimulus_task_numeric_abs_error_heatmap.png",
        cbar_label="Delta mean absolute error vs baseline"
    )

    print(f"Heatmaps écrites dans : {PLOT_DIR}")


if __name__ == "__main__":
    main()