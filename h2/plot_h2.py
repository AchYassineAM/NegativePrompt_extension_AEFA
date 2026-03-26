"""
plot_h2.py
Génère le barplot des résultats H2 (cause_and_effect, seed 42).
Lance avec : python plot_h2.py
Produit : h2_results_barplot.png dans le même dossier.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Données ──────────────────────────────────────────────────────────
data = {
    "Flan-T5-Large": {
        "NP04": {"None": 0.52, "Neutral": 0.60, "Positive": 0.56},
        "NP05": {"None": 0.64, "Neutral": 0.64, "Positive": 0.60},
        "NP08": {"None": 0.48, "Neutral": 0.52, "Positive": 0.52},
    },
    "Vicuna-7b": {
        "NP04": {"None": 0.56, "Neutral": 0.64, "Positive": 0.48},
        "NP05": {"None": 0.32, "Neutral": 0.80, "Positive": 0.48},
        "NP08": {"None": 0.64, "Neutral": 0.80, "Positive": 0.52},
    },
    "Llama-2-7b": {
        "NP04": {"None": 0.32, "Neutral": 0.56, "Positive": 0.28},
        "NP05": {"None": 0.28, "Neutral": 0.32, "Positive": 0.32},
        "NP08": {"None": 0.56, "Neutral": 0.28, "Positive": 0.08},
    },
}

models   = list(data.keys())
stimuli  = ["NP04", "NP05", "NP08"]
primes   = ["None", "Neutral", "Positive"]

# ── Couleurs & style ─────────────────────────────────────────────────
COLORS = {
    "None":     "#6C757D",   # gris neutre
    "Neutral":  "#2B6CB0",   # bleu foncé
    "Positive": "#276749",   # vert foncé
}

plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "axes.axisbelow":   True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "axes.edgecolor":   "#333333",
    "axes.linewidth":   0.8,
})

# ── Figure : 1 ligne × 3 colonnes (un subplot par modèle) ────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
fig.suptitle(
    "Figure H2.1 — Test score sur cause_and_effect par condition de prime et par modèle\n"
    "(stimuli NP04, NP05, NP08 — seed 42 — ordre prime→task→stimulus)",
    fontsize=10.5, fontweight="bold", y=1.01
)

n_stim   = len(stimuli)
n_prime  = len(primes)
bar_w    = 0.22
group_gap = 0.10
x_pos    = np.arange(n_stim)

for ax, model in zip(axes, models):
    for j, prime in enumerate(primes):
        offset = (j - 1) * (bar_w + 0.02)
        scores = [data[model][s][prime] for s in stimuli]
        bars = ax.bar(
            x_pos + offset, scores,
            width=bar_w,
            color=COLORS[prime],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            label=prime,
            zorder=3,
        )
        # valeur au-dessus de chaque barre
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{score:.2f}",
                ha="center", va="bottom",
                fontsize=7.5, color="#333333"
            )

    ax.set_title(model, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stimuli, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.20, 0.40, 0.60, 0.80, 1.0])
    ax.yaxis.set_tick_params(labelsize=9)
    if ax == axes[0]:
        ax.set_ylabel("Test score (proportion réponses correctes)", fontsize=9.5)
    ax.set_xlabel("Stimulus négatif", fontsize=9.5)

# Légende commune en bas
patches = [mpatches.Patch(color=COLORS[p], label=f"Prime : {p}") for p in primes]
fig.legend(
    handles=patches,
    loc="lower center",
    ncol=3,
    fontsize=9.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.08),
)

plt.tight_layout()
import pathlib
out = str(pathlib.Path(__file__).resolve().parent / "h2_results_barplot.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Sauvegardé : {out}")
plt.close()
