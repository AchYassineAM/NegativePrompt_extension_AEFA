import csv
from collections import defaultdict
from pathlib import Path

# ============================================================
# AGREGATION DEDIEE AUX NEGATIVE PROMPTS ORIGINAUX
# ============================================================
#
# Objectif :
# produire une vue explicitement dédiée à la comparaison avec
# le papier NegativePrompt, en isolant les 10 prompts négatifs
# historiques (Negative_SET, via pnum=1..10).
#
# Pourquoi un script séparé :
# - garder une section du rapport clairement orientée "papier" ;
# - distinguer l’analyse globale de l’extension H3 de l’analyse
#   spécifique des stimuli originaux ;
# - produire deux niveaux de lecture :
#     1) détail par pnum
#     2) moyenne globale sur negative_original
#
# Ce script lit :
# - results/negative_original/.../results.csv
# - results/baseline/.../results.csv
#
# Et produit :
# - un CSV détaillé par pnum
# - un CSV résumé (moyenne des prompts originaux) avec delta
#   par rapport à la baseline.
# ============================================================


# ============================================================
# CHEMINS
# ============================================================

BASELINE_RESULTS = Path("results") / "baseline"
NEGATIVE_RESULTS = Path("results") / "negative_original"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

ANALYSIS_DIR = REPO_ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
H3_DIR = TABLES_DIR / "h3"

OUTPUT_DETAILED = H3_DIR / "negative_original_by_pnum.csv"
OUTPUT_SUMMARY = H3_DIR / "negative_original_summary.csv"


# ============================================================
# OUTILS
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


def detect_model_from_csv_path(csv_path: Path):
    # attendu : results/<group>/<model>/results.csv
    parts = csv_path.parts
    if len(parts) >= 3:
        return parts[-2]
    return "unknown"


# ============================================================
# LECTURE DES CSV
# ============================================================

def read_csv_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["__csv_path__"] = str(csv_path)
            rows.append(row)
    return rows


def read_result_tree(root_dir: Path):
    rows = []
    if not root_dir.exists():
        return rows

    for csv_path in root_dir.rglob("results.csv"):
        rows.extend(read_csv_rows(csv_path))
    return rows


# ============================================================
# DEDUPLICATION
# ============================================================
#
# Baseline :
#   clé = task, model, seed, few_shot, temperature, do_sample
#
# Negative original :
#   clé = task, model, legacy_pnum, seed, few_shot, temperature, do_sample
# ============================================================

def deduplicate_baseline_rows(rows):
    seen = set()
    deduped = []

    for row in rows:
        key = (
            row.get("task", "").strip(),
            row.get("model", "").strip(),
            str(row.get("seed", "")).strip(),
            str(row.get("few_shot", "")).strip(),
            str(row.get("temperature", "")).strip(),
            str(row.get("do_sample", "")).strip(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped


def deduplicate_negative_rows(rows):
    seen = set()
    deduped = []

    for row in rows:
        key = (
            row.get("task", "").strip(),
            row.get("model", "").strip(),
            str(row.get("legacy_pnum", "")).strip(),
            str(row.get("seed", "")).strip(),
            str(row.get("few_shot", "")).strip(),
            str(row.get("temperature", "")).strip(),
            str(row.get("do_sample", "")).strip(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped


# ============================================================
# BASELINE LOOKUP
# ============================================================

def build_baseline_lookup(rows):
    grouped = defaultdict(list)

    for row in rows:
        task = row.get("task", "").strip()
        model = row.get("model", "").strip()
        score = safe_float(row.get("score"))
        if score is None:
            continue

        key = (task, model)
        grouped[key].append(score)

    baseline_lookup = {}
    for key, scores in grouped.items():
        baseline_lookup[key] = mean(scores)

    return baseline_lookup


# ============================================================
# DETAIL PAR PNUM
# ============================================================

def aggregate_negative_by_pnum(rows, baseline_lookup):
    grouped = defaultdict(list)

    for row in rows:
        task = row.get("task", "").strip()
        model = row.get("model", "").strip()
        pnum = str(row.get("legacy_pnum", "")).strip()
        score = safe_float(row.get("score"))

        if score is None or pnum == "":
            continue

        key = (task, model, pnum)
        grouped[key].append(row)

    detailed_rows = []

    for key, group_rows in grouped.items():
        task, model, pnum = key
        scores = [safe_float(r.get("score")) for r in group_rows]
        scores = [s for s in scores if s is not None]
        seeds = sorted(set(str(r.get("seed", "")).strip() for r in group_rows if str(r.get("seed", "")).strip()))

        mean_score = mean(scores)
        baseline_score = baseline_lookup.get((task, model))
        delta_vs_baseline = None

        if mean_score is not None and baseline_score is not None:
            delta_vs_baseline = mean_score - baseline_score

        detailed_rows.append({
            "task": task,
            "model": model,
            "legacy_pnum": pnum,
            "n_runs": len(group_rows),
            "n_seeds": len(seeds),
            "seeds": " | ".join(seeds),
            "mean_score": round(mean_score, 6) if mean_score is not None else "",
            "baseline_score": round(baseline_score, 6) if baseline_score is not None else "",
            "delta_vs_baseline": round(delta_vs_baseline, 6) if delta_vs_baseline is not None else "",
        })

    detailed_rows.sort(key=lambda x: (x["task"], x["model"], int(x["legacy_pnum"])))
    return detailed_rows


# ============================================================
# RESUME GLOBAL NEGATIVE ORIGINAL
# ============================================================

def aggregate_negative_summary(detailed_rows):
    grouped = defaultdict(list)

    for row in detailed_rows:
        task = row["task"]
        model = row["model"]
        grouped[(task, model)].append(row)

    summary_rows = []

    for key, rows in grouped.items():
        task, model = key

        scores = [safe_float(r["mean_score"]) for r in rows]
        scores = [s for s in scores if s is not None]

        baseline_scores = [safe_float(r["baseline_score"]) for r in rows]
        baseline_scores = [s for s in baseline_scores if s is not None]

        mean_negative_original = mean(scores)
        baseline_score = baseline_scores[0] if baseline_scores else None
        delta_vs_baseline = None

        if mean_negative_original is not None and baseline_score is not None:
            delta_vs_baseline = mean_negative_original - baseline_score

        summary_rows.append({
            "task": task,
            "model": model,
            "n_negative_prompts": len(rows),
            "mean_negative_original_score": round(mean_negative_original, 6) if mean_negative_original is not None else "",
            "baseline_score": round(baseline_score, 6) if baseline_score is not None else "",
            "delta_vs_baseline": round(delta_vs_baseline, 6) if delta_vs_baseline is not None else "",
        })

    summary_rows.sort(key=lambda x: (x["task"], x["model"]))
    return summary_rows


# ============================================================
# ECRITURE CSV
# ============================================================

def write_csv(rows, output_path: Path, fieldnames):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    baseline_rows = read_result_tree(BASELINE_RESULTS)
    negative_rows = read_result_tree(NEGATIVE_RESULTS)

    if not baseline_rows:
        raise FileNotFoundError("Aucun results.csv trouvé dans results/baseline/")
    if not negative_rows:
        raise FileNotFoundError("Aucun results.csv trouvé dans results/negative_original/")

    baseline_rows = deduplicate_baseline_rows(baseline_rows)
    negative_rows = deduplicate_negative_rows(negative_rows)

    baseline_lookup = build_baseline_lookup(baseline_rows)
    detailed_rows = aggregate_negative_by_pnum(negative_rows, baseline_lookup)
    summary_rows = aggregate_negative_summary(detailed_rows)

    write_csv(
        detailed_rows,
        OUTPUT_DETAILED,
        fieldnames=[
            "task",
            "model",
            "legacy_pnum",
            "n_runs",
            "n_seeds",
            "seeds",
            "mean_score",
            "baseline_score",
            "delta_vs_baseline",
        ],
    )

    write_csv(
        summary_rows,
        OUTPUT_SUMMARY,
        fieldnames=[
            "task",
            "model",
            "n_negative_prompts",
            "mean_negative_original_score",
            "baseline_score",
            "delta_vs_baseline",
        ],
    )

    print(f"{len(detailed_rows)} lignes écrites dans : {OUTPUT_DETAILED}")
    print(f"{len(summary_rows)} lignes écrites dans : {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()