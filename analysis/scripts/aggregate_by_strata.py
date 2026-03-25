import csv
import json
import math
from pathlib import Path
from collections import defaultdict

# ============================================================
# AGGREGATION PAR STRATES - METRIQUES PRINCIPALES
# ============================================================
#
# Objectif :
# agréger les scores principaux issus des results.csv produits
# par main.py / main_ape.py, puis les organiser par familles
# de tâches ("strates") définies dans configs/task_strata.json.
#
# Pourquoi ce script :
# - lire les résultats benchmark-compatibles de façon structurée ;
# - comparer les groupes de stimuli par familles de tâches ;
# - préparer les analyses H3 (interaction catégorie × type de tâche) ;
# - servir aussi plus tard à H4 (sélection stratifiée).
#
# Différence avec aggregate_secondary_metrics.py :
# - ici on agrège les scores principaux du benchmark ;
# - là-bas on agrège des métriques secondaires plus fines
#   (erreur absolue, parseability, etc.).
#
# Ce script ajoute :
# - déduplication explicite des runs
# - n_seeds
# - std_score
# - seeds list
# - n_stimuli
# - delta_vs_baseline
# ============================================================

ROOT_RESULTS = Path("results")
STRATA_PATH = Path("configs") / "task_strata.json"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

ANALYSIS_DIR = REPO_ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

ANALYSIS_DIR = REPO_ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
STIMULUS_TASK_DIR = TABLES_DIR / "stimulus_task"

OUTPUT_PATH = STIMULUS_TASK_DIR / "aggregated_by_strata.csv"

# ============================================================
# CHARGEMENT DES STRATES
# ============================================================

def load_task_strata(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_task_info(task: str, strata: dict):
    if task in strata["instruction_induction"]:
        return "instruction_induction", strata["instruction_induction"][task]
    if task in strata["bigbench"]:
        return "bigbench", strata["bigbench"][task]
    return "unknown", "unknown"


# ============================================================
# DETECTION DE LA SOURCE DE PROMPT
# ============================================================
#
# results/ape/... -> prompt_source = "ape"
# sinon            -> prompt_source = "base"
# ============================================================

def detect_prompt_source(csv_path: Path):
    parts = [p.lower() for p in csv_path.parts]
    if "ape" in parts:
        return "ape"
    return "base"


# ============================================================
# LECTURE DES CSV DE RESULTATS
# ============================================================

def read_all_result_rows(root_results: Path):
    rows = []
    for csv_path in root_results.rglob("results.csv"):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            prompt_source = detect_prompt_source(csv_path)

            for row in reader:
                row["prompt_source"] = prompt_source
                row["__csv_path__"] = str(csv_path)
                rows.append(row)
    return rows


# ============================================================
# OUTILS NUMERIQUES
# ============================================================

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def std(values):
    vals = [v for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0 if vals else None
    m = mean(vals)
    variance = sum((x - m) ** 2 for x in vals) / len(vals)
    return math.sqrt(variance)


# ============================================================
# DEDUPLICATION
# ============================================================
#
# But :
# éviter qu’un run déjà lancé manuellement puis relancé en batch
# soit compté deux fois.
#
# Clé de déduplication :
# - prompt_source
# - task
# - model
# - stimulus_group
# - stimulus_index
# - few_shot
# - seed
# - temperature
# - do_sample
#
# On suppose ici qu’une telle combinaison représente un run unique.
# ============================================================

def deduplicate_rows(rows):
    seen = set()
    deduped = []

    for row in rows:
        key = (
            row.get("prompt_source", "").strip(),
            row.get("task", "").strip(),
            row.get("model", "").strip(),
            row.get("stimulus_group", "").strip(),
            row.get("stimulus_index", "").strip(),
            str(row.get("few_shot", "")).strip(),
            str(row.get("seed", "")).strip(),
            str(row.get("temperature", "")).strip(),
            str(row.get("do_sample", "")).strip(),
        )

        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped


# ============================================================
# AGREGATION PAR STRATE
# ============================================================

def aggregate_rows(rows, strata):
    grouped = defaultdict(list)

    for row in rows:
        task = row.get("task", "").strip()
        model = row.get("model", "").strip()
        stimulus_group = row.get("stimulus_group", "").strip()
        prompt_source = row.get("prompt_source", "").strip()
        score = safe_float(row.get("score"))

        if score is None:
            continue

        benchmark, task_family = find_task_info(task, strata)

        key = (
            prompt_source,
            benchmark,
            model,
            stimulus_group,
            task_family,
        )
        grouped[key].append(row)

    aggregated = []
    baseline_lookup = {}

    # --------------------------------------------------------
    # 1) agrégation brute
    # --------------------------------------------------------
    for key, group_rows in grouped.items():
        prompt_source, benchmark, model, stimulus_group, task_family = key

        scores = [safe_float(r.get("score")) for r in group_rows]
        scores = [s for s in scores if s is not None]

        unique_tasks = sorted(set(r.get("task", "").strip() for r in group_rows if r.get("task", "").strip()))
        unique_seeds = sorted(set(str(r.get("seed", "")).strip() for r in group_rows if str(r.get("seed", "")).strip()))
        unique_stimuli = sorted(set(str(r.get("stimulus_index", "")).strip() for r in group_rows if str(r.get("stimulus_index", "")).strip()))

        mean_score = mean(scores)
        std_score = std(scores)

        row = {
            "prompt_source": prompt_source,
            "benchmark": benchmark,
            "model": model,
            "stimulus_group": stimulus_group,
            "task_family": task_family,
            "n_rows": len(group_rows),
            "n_unique_tasks": len(unique_tasks),
            "n_seeds": len(unique_seeds),
            "seeds": " | ".join(unique_seeds),
            "n_stimuli": len(unique_stimuli),
            "stimulus_indices": " | ".join(unique_stimuli),
            "mean_score": round(mean_score, 6) if mean_score is not None else "",
            "std_score": round(std_score, 6) if std_score is not None else "",
            "delta_vs_baseline": "",
            "tasks": " | ".join(unique_tasks),
        }
        aggregated.append(row)

        if stimulus_group == "baseline" and mean_score is not None:
            baseline_lookup[(prompt_source, benchmark, model, task_family)] = mean_score

    # --------------------------------------------------------
    # 2) delta vs baseline
    # --------------------------------------------------------
    for row in aggregated:
        baseline_score = baseline_lookup.get(
            (row["prompt_source"], row["benchmark"], row["model"], row["task_family"])
        )
        current_score = safe_float(row["mean_score"])

        if baseline_score is not None and current_score is not None:
            row["delta_vs_baseline"] = round(current_score - baseline_score, 6)

    aggregated.sort(
        key=lambda x: (
            x["prompt_source"],
            x["benchmark"],
            x["model"],
            x["stimulus_group"],
            x["task_family"],
        )
    )
    return aggregated


# ============================================================
# ECRITURE DU CSV
# ============================================================

def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prompt_source",
        "benchmark",
        "model",
        "stimulus_group",
        "task_family",
        "n_rows",
        "n_unique_tasks",
        "n_seeds",
        "seeds",
        "n_stimuli",
        "stimulus_indices",
        "mean_score",
        "std_score",
        "delta_vs_baseline",
        "tasks",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    if not ROOT_RESULTS.exists():
        raise FileNotFoundError(f"Dossier introuvable: {ROOT_RESULTS}")

    if not STRATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable: {STRATA_PATH}")

    strata = load_task_strata(STRATA_PATH)
    rows = read_all_result_rows(ROOT_RESULTS)

    if not rows:
        print("Aucun results.csv trouvé dans le dossier results/")
        return

    rows = deduplicate_rows(rows)
    aggregated = aggregate_rows(rows, strata)
    write_csv(aggregated, OUTPUT_PATH)

    print(f"{len(rows)} lignes dédupliquées lues")
    print(f"{len(aggregated)} lignes agrégées écrites dans : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()