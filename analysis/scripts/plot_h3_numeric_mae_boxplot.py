from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# BOXPLOT 10 BOITES - PRECISION NUMERIQUE PAR STIMULUS
# ============================================================
#
# 5 colonnes = 5 catégories H3
# Dans chaque colonne :
#   - baseline
#   - catégorie H3 correspondante
#
# Cela donne 10 boxplots au total.
#
# Restriction :
#   - uniquement les tâches numériques : sum et diff
#
# Les boîtes contiennent les valeurs de mean_abs_error
# au niveau le plus fin disponible dans les fichiers.
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

ANALYSIS_DIR = REPO_ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
SECONDARY_DIR = TABLES_DIR / "secondary_metrics"
SOURCES_FOR_PLOTS_DIR = TABLES_DIR / "sources_for_plots"

PLOT_DIR = ANALYSIS_DIR / "plots" / "h3"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_FILE = SECONDARY_DIR / "h3_secondary_metrics_numeric.csv"
FALLBACK_FILE = SECONDARY_DIR / "aggregated_secondary_metrics_numeric.csv"

OUTPUT_PNG = PLOT_DIR / "h3_numeric_mae_10_boxplots_sum_diff.png"
OUTPUT_CSV = SOURCES_FOR_PLOTS_DIR / "h3_numeric_mae_10_boxplots_sum_diff_source.csv"

H3_GROUPS = [
    "competence_threat",
    "inconsistency",
    "social_comparison",
    "urgency",
    "regret",
]

NUMERIC_TASKS = {"sum", "diff"}

COLOR_BASELINE = "#4C78A8"
COLOR_H3 = "#54A24B"


def load_numeric_data() -> pd.DataFrame:
    """
    Charge et fusionne les fichiers numériques disponibles.
    """
    frames = []

    if PRIMARY_FILE.exists():
        df_primary = pd.read_csv(PRIMARY_FILE)
        print(f"Loaded primary: {PRIMARY_FILE}")
        frames.append(df_primary)

    if FALLBACK_FILE.exists():
        df_fallback = pd.read_csv(FALLBACK_FILE)
        print(f"Loaded fallback: {FALLBACK_FILE}")
        frames.append(df_fallback)

    if not frames:
        raise FileNotFoundError(
            "Aucun fichier de métriques numériques trouvé.\n"
            f"Attendu : {PRIMARY_FILE} ou {FALLBACK_FILE}"
        )

    df = pd.concat(frames, ignore_index=True, sort=False)

    candidate_subset = [
        c for c in ["task", "stimulus_group", "seed", "model", "mean_abs_error"]
        if c in df.columns
    ]
    if candidate_subset:
        df = df.drop_duplicates(subset=candidate_subset)

    print(f"Combined rows after dedup: {len(df)}")
    return df


def detect_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Impossible de trouver une colonne pour '{label}'. "
        f"Candidats testés : {candidates}\n"
        f"Colonnes disponibles : {list(df.columns)}"
    )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les noms de colonnes utiles.
    """
    stimulus_col = detect_column(
        df,
        ["stimulus_group", "group", "condition"],
        "stimulus group"
    )

    mae_col = detect_column(
        df,
        ["mean_abs_error", "mae", "abs_error_mean"],
        "mean absolute error"
    )

    task_col = detect_column(
        df,
        ["task", "task_name"],
        "task"
    )

    seed_col = None
    for c in ["seed", "run_seed"]:
        if c in df.columns:
            seed_col = c
            break

    model_col = None
    for c in ["model", "llm_model"]:
        if c in df.columns:
            model_col = c
            break

    out = pd.DataFrame({
        "task": df[task_col].astype(str).str.strip(),
        "stimulus_group": df[stimulus_col].astype(str).str.strip(),
        "mean_abs_error": pd.to_numeric(df[mae_col], errors="coerce"),
    })

    out["seed"] = df[seed_col] if seed_col is not None else "NA"
    out["model"] = df[model_col] if model_col is not None else "NA"

    out = out.dropna(subset=["mean_abs_error"]).copy()
    return out


def keep_relevant_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garde baseline + catégories H3, uniquement pour sum et diff.
    """
    allowed_groups = {"baseline", *H3_GROUPS}
    out = df[
        df["stimulus_group"].isin(allowed_groups) &
        df["task"].isin(NUMERIC_TASKS)
    ].copy()

    if out.empty:
        raise ValueError(
            "Aucune ligne correspondante à baseline / H3 "
            "sur les tâches sum/diff n'a été trouvée."
        )

    return out


def build_boxplot_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la table plate source pour les 10 boîtes.
    Dans chaque colonne H3, on répète baseline,
    puis on ajoute la catégorie H3 propre à la colonne.
    """
    rows = []

    base_df = df[df["stimulus_group"] == "baseline"].copy()

    if base_df.empty:
        raise ValueError(
            "Aucune ligne baseline trouvée pour les tâches numériques sum/diff."
        )

    for group in H3_GROUPS:
        h3_df = df[df["stimulus_group"] == group].copy()
        if h3_df.empty:
            raise ValueError(
                f"Aucune ligne trouvée pour la catégorie H3 '{group}' "
                "sur les tâches numériques sum/diff."
            )

        tmp = base_df.copy()
        tmp["column_group"] = group
        tmp["box_condition"] = "baseline"
        rows.append(tmp)

        tmp = h3_df.copy()
        tmp["column_group"] = group
        tmp["box_condition"] = group
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out["box_label"] = out["box_condition"]
    return out


def plot_boxplots(source_df: pd.DataFrame) -> None:
    """
    Produit le graphique à 10 boxplots.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    positions = []
    data = []
    colors = []
    xtick_positions = []
    xtick_labels = []

    current_x = 1
    gap_between_groups = 1.2

    for group in H3_GROUPS:
        group_center_positions = []

        order = ["baseline", group]

        for cond in order:
            subset = source_df[
                (source_df["column_group"] == group) &
                (source_df["box_condition"] == cond)
            ]["mean_abs_error"].tolist()

            positions.append(current_x)
            data.append(subset)

            if cond == "baseline":
                colors.append(COLOR_BASELINE)
            else:
                colors.append(COLOR_H3)

            group_center_positions.append(current_x)
            current_x += 1

        xtick_positions.append(sum(group_center_positions) / len(group_center_positions))
        xtick_labels.append(group)
        current_x += gap_between_groups

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showfliers=True
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    for median in bp["medians"]:
        median.set_linewidth(1.5)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=20, ha="right")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlabel("Catégorie H3")
    ax.set_title(
        "Précision numérique selon le type de stimulus\n"
        "(tâches : sum et diff ; 2 boxplots par colonne : baseline / H3)"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_handles = [
        Patch(facecolor=COLOR_BASELINE, label="baseline"),
        Patch(facecolor=COLOR_H3, label="H3 category"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {OUTPUT_PNG}")


def main() -> None:
    df_raw = load_numeric_data()
    df = prepare_dataframe(df_raw)
    df = keep_relevant_rows(df)
    source_df = build_boxplot_source(df)

    source_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote: {OUTPUT_CSV}")

    print("\nCounts per box:")
    counts = (
        source_df.groupby(["column_group", "box_condition"])
        .size()
        .reset_index(name="n_points")
    )
    print(counts.to_string(index=False))

    print("\nTasks included:")
    print(sorted(source_df["task"].unique().tolist()))

    plot_boxplots(source_df)


if __name__ == "__main__":
    main()