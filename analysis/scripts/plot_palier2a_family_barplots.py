import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt



SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

ANALYSIS_DIR = REPO_ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
H3_TABLES_DIR = TABLES_DIR / "h3"

PLOT_DIR = ANALYSIS_DIR / "plots" / "family"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

AGG_PATH = H3_TABLES_DIR / "palier2a_aggregated_conditions.csv"
CONTRAST_PATH = H3_TABLES_DIR / "palier2a_h3_vs_negative_original.csv"

H3_GROUPS = [
    "competence_threat",
    "inconsistency",
    "social_comparison",
    "urgency",
    "regret",
]


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def build_family_delta_vs_baseline():
    rows = read_csv(AGG_PATH)

    neg_by_family = defaultdict(list)
    h3_by_family_group = defaultdict(list)

    for r in rows:
        fam = r["task_family"]
        group = r["stimulus_group"]
        delta = safe_float(r["delta_vs_baseline"])
        if delta is None:
            continue

        if group == "negative_original":
            neg_by_family[fam].append(delta)
        elif group in H3_GROUPS:
            h3_by_family_group[(fam, group)].append(delta)

    families = sorted({r["task_family"] for r in rows})

    neg_means = {fam: mean(neg_by_family.get(fam, [])) for fam in families}
    h3_means = {
        (fam, group): mean(h3_by_family_group.get((fam, group), []))
        for fam in families for group in H3_GROUPS
    }

    return families, neg_means, h3_means


def build_family_h3_minus_neg():
    rows = read_csv(CONTRAST_PATH)

    contrast_by_family_group = defaultdict(list)
    for r in rows:
        fam = r["task_family"]
        group = r["stimulus_group"]
        delta = safe_float(r["delta_h3_minus_negative_original"])
        if delta is None:
            continue
        contrast_by_family_group[(fam, group)].append(delta)

    families = sorted({r["task_family"] for r in rows})
    contrast_means = {
        (fam, group): mean(contrast_by_family_group.get((fam, group), []))
        for fam in families for group in H3_GROUPS
    }

    return families, contrast_means


def plot_family_delta_vs_baseline():
    families, neg_means, h3_means = build_family_delta_vs_baseline()

    x = np.arange(len(families))
    width = 0.13

    fig, ax = plt.subplots(figsize=(1.8 + 0.95 * len(families), 7))

    offsets = {
        "negative_original": -2.5 * width,
        "competence_threat": -1.5 * width,
        "inconsistency": -0.5 * width,
        "social_comparison": 0.5 * width,
        "urgency": 1.5 * width,
        "regret": 2.5 * width,
    }

    neg_vals = [neg_means.get(fam, np.nan) for fam in families]
    ax.bar(x + offsets["negative_original"], neg_vals, width=width, label="negative_original")

    for group in H3_GROUPS:
        vals = [h3_means.get((fam, group), np.nan) for fam in families]
        ax.bar(x + offsets[group], vals, width=width, label=group)

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.set_ylabel("Delta score vs baseline")
    ax.set_title("Delta vs baseline par famille de tâches")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "family_delta_vs_baseline_barplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_family_h3_minus_neg():
    families, contrast_means = build_family_h3_minus_neg()

    x = np.arange(len(families))
    width = 0.16

    fig, ax = plt.subplots(figsize=(1.8 + 0.95 * len(families), 7))

    offsets = {
        "competence_threat": -2 * width,
        "inconsistency": -1 * width,
        "social_comparison": 0,
        "urgency": 1 * width,
        "regret": 2 * width,
    }

    for group in H3_GROUPS:
        vals = [contrast_means.get((fam, group), np.nan) for fam in families]
        ax.bar(x + offsets[group], vals, width=width, label=group)

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.set_ylabel("Delta(H3) - Delta(negative_original)")
    ax.set_title("Contraste H3 vs negative_original par famille de tâches")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "family_h3_minus_negative_original_barplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_family_delta_vs_baseline()
    plot_family_h3_minus_neg()
    print(f"Barplots écrits dans : {PLOT_DIR}")


if __name__ == "__main__":
    main()