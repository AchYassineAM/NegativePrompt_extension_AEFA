import csv
from collections import defaultdict
from pathlib import Path

# ============================================================
# AGGREGATION DES METRIQUES SECONDAIRES
# ============================================================
#
# Objectif :
# agréger les métriques secondaires calculées dans
# h3_secondary_metrics_numeric.csv pour obtenir une lecture
# propre par catégorie de stimulus.
#
# Pourquoi ce script :
# - éviter une lecture manuelle ligne par ligne ;
# - résumer les performances numériques par catégorie H3 ;
# - comparer ces catégories à la baseline ;
# - documenter non seulement l’exactitude binaire, mais aussi
#   la précision numérique effective.
#
# Ce que fait le script :
# 1. lit le CSV des métriques secondaires par run ;
# 2. déduplique les lignes identiques si besoin ;
# 3. agrège par :
#       task, model, stimulus_group
# 4. calcule des moyennes sur :
#       parseable_rate
#       mean_abs_error
#       median_abs_error
# 5. ajoute des deltas par rapport à la baseline.
#
# Convention importante :
# - pour l’erreur absolue, plus petit = meilleur ;
# - donc un delta positif vs baseline signifie :
#       "pire que la baseline"
# - un delta négatif signifie :
#       "meilleur que la baseline"
# ============================================================


# ============================================================
# CHEMINS
# ============================================================

INPUT_PATH = Path("analysis") / "h3_secondary_metrics_numeric.csv"
OUTPUT_PATH = Path("analysis") / "aggregated_secondary_metrics_numeric.csv"


# ============================================================
# OUTILS
# ============================================================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def deduplicate_rows(rows):
    """
    Déduplication prudente.
    On garde une seule ligne par combinaison :
        task, model, stimulus_group, stimulus_index, seed
    Cela évite qu’un run déjà lancé manuellement soit compté
    deux fois dans l’agrégation finale.
    """
    seen = set()
    deduped = []

    for row in rows:
        key = (
            row.get("task", "").strip(),
            row.get("model", "").strip(),
            row.get("stimulus_group", "").strip(),
            row.get("stimulus_index", "").strip(),
            row.get("seed", "").strip(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


# ============================================================
# LECTURE DES DONNEES
# ============================================================

def read_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================
# AGREGATION
# ============================================================

def aggregate_rows(rows):
    """
    Agrégation par :
        task, model, stimulus_group

    On résume donc une catégorie H3 entière, potentiellement
    sur plusieurs prompts (stimulus_index) et plusieurs seeds.
    """
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row.get("task", "").strip(),
            row.get("model", "").strip(),
            row.get("stimulus_group", "").strip(),
        )
        grouped[key].append(row)

    aggregated = []
    baseline_lookup = {}

    # --------------------------------------------------------
    # 1) agrégation brute
    # --------------------------------------------------------
    for key, group_rows in grouped.items():
        task, model, stimulus_group = key

        parseable_rates = [safe_float(r.get("parseable_rate")) for r in group_rows]
        mean_abs_errors = [safe_float(r.get("mean_abs_error")) for r in group_rows]
        median_abs_errors = [safe_float(r.get("median_abs_error")) for r in group_rows]

        seeds = sorted(set(r.get("seed", "").strip() for r in group_rows if r.get("seed", "").strip()))
        stimulus_indices = sorted(set(r.get("stimulus_index", "").strip() for r in group_rows if r.get("stimulus_index", "").strip()))

        row = {
            "task": task,
            "model": model,
            "stimulus_group": stimulus_group,
            "n_runs": len(group_rows),
            "n_seeds": len(seeds),
            "seeds": " | ".join(seeds),
            "n_stimuli": len(stimulus_indices),
            "stimulus_indices": " | ".join(stimulus_indices),
            "mean_parseable_rate": round(mean(parseable_rates), 6) if mean(parseable_rates) is not None else "",
            "mean_mean_abs_error": round(mean(mean_abs_errors), 6) if mean(mean_abs_errors) is not None else "",
            "mean_median_abs_error": round(mean(median_abs_errors), 6) if mean(median_abs_errors) is not None else "",
            "delta_mean_abs_error_vs_baseline": "",
            "delta_median_abs_error_vs_baseline": "",
            "delta_parseable_rate_vs_baseline": "",
        }

        aggregated.append(row)

        if stimulus_group == "baseline":
            baseline_lookup[(task, model)] = {
                "mean_parseable_rate": mean(parseable_rates),
                "mean_mean_abs_error": mean(mean_abs_errors),
                "mean_median_abs_error": mean(median_abs_errors),
            }

    # --------------------------------------------------------
    # 2) deltas par rapport à la baseline
    # --------------------------------------------------------
    for row in aggregated:
        baseline = baseline_lookup.get((row["task"], row["model"]))
        if baseline is None:
            continue

        current_parseable = safe_float(row["mean_parseable_rate"])
        current_mean_abs = safe_float(row["mean_mean_abs_error"])
        current_median_abs = safe_float(row["mean_median_abs_error"])

        baseline_parseable = baseline["mean_parseable_rate"]
        baseline_mean_abs = baseline["mean_mean_abs_error"]
        baseline_median_abs = baseline["mean_median_abs_error"]

        if current_mean_abs is not None and baseline_mean_abs is not None:
            row["delta_mean_abs_error_vs_baseline"] = round(current_mean_abs - baseline_mean_abs, 6)

        if current_median_abs is not None and baseline_median_abs is not None:
            row["delta_median_abs_error_vs_baseline"] = round(current_median_abs - baseline_median_abs, 6)

        if current_parseable is not None and baseline_parseable is not None:
            row["delta_parseable_rate_vs_baseline"] = round(current_parseable - baseline_parseable, 6)

    aggregated.sort(key=lambda x: (x["task"], x["model"], x["stimulus_group"]))
    return aggregated


# ============================================================
# ECRITURE CSV
# ============================================================

def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task",
        "model",
        "stimulus_group",
        "n_runs",
        "n_seeds",
        "seeds",
        "n_stimuli",
        "stimulus_indices",
        "mean_parseable_rate",
        "mean_mean_abs_error",
        "mean_median_abs_error",
        "delta_mean_abs_error_vs_baseline",
        "delta_median_abs_error_vs_baseline",
        "delta_parseable_rate_vs_baseline",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    rows = read_rows(INPUT_PATH)
    rows = deduplicate_rows(rows)
    aggregated = aggregate_rows(rows)
    write_csv(aggregated, OUTPUT_PATH)

    print(f"{len(rows)} lignes dédupliquées lues")
    print(f"{len(aggregated)} lignes agrégées écrites dans : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()