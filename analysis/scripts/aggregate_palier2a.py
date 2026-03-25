import csv
from pathlib import Path
from statistics import mean, pstdev

# ============================================================
# AGREGATION COMPLETE - PALIER 2A
# ============================================================
#
# Objectif :
# réunir baseline, negative_original et H3 dans un seul jeu
# d'analyse, puis produire :
# - un format long consolidé
# - les matrices par tâche
# - la matrice de contraste H3 vs negative_original
#
# Portée :
# - Instruction Induction
# - flan-t5-large
# - seeds fixes
#
# Remarque :
# ce script travaille uniquement à partir des CSV déjà produits.
# Il ne relance aucune expérience.
# ============================================================

ROOT = Path("results")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

ANALYSIS_DIR = REPO_ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
H3_DIR = TABLES_DIR / "h3"

OUT = H3_DIR
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "flan-t5-large"

TASKS = [
    "active_to_passive",
    "antonyms",
    "cause_and_effect",
    "common_concept",
    "diff",
    "first_word_letter",
    "informal_to_formal",
    "larger_animal",
    "letters_list",
    "negation",
    "num_to_verbal",
    "orthography_starts_with",
    "rhymes",
    "second_word_letter",
    "sentence_similarity",
    "sentiment",
    "singular_to_plural",
    "sum",
    "synonyms",
    "taxonomy_animal",
    "translation_en-de",
    "translation_en-es",
    "translation_en-fr",
    "word_in_context",
]

TASK_FAMILY = {
    "active_to_passive": "rewriting_transformation",
    "antonyms": "lexical_semantics",
    "cause_and_effect": "causal_reasoning",
    "common_concept": "semantic_abstraction",
    "diff": "numerical_reasoning",
    "first_word_letter": "string_extraction",
    "informal_to_formal": "rewriting_transformation",
    "larger_animal": "comparative_reasoning",
    "letters_list": "string_transformation",
    "negation": "rewriting_transformation",
    "num_to_verbal": "numerical_verbalization",
    "orthography_starts_with": "string_filtering",
    "rhymes": "lexical_generation",
    "second_word_letter": "string_extraction",
    "sentence_similarity": "semantic_similarity",
    "sentiment": "sentiment_classification",
    "singular_to_plural": "morphology",
    "sum": "numerical_reasoning",
    "synonyms": "lexical_semantics",
    "taxonomy_animal": "semantic_filtering",
    "translation_en-de": "translation",
    "translation_en-es": "translation",
    "translation_en-fr": "translation",
    "word_in_context": "lexical_semantics",
}

H3_GROUPS = [
    "competence_threat",
    "inconsistency",
    "social_comparison",
    "urgency",
    "regret",
]

ALL_GROUPS = ["baseline", "negative_original"] + H3_GROUPS


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def dedupe_rows(rows, keys):
    seen = set()
    out = []
    for r in rows:
        k = tuple(r.get(x, "") for x in keys)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def collect_all_rows():
    all_rows = []

    # baseline
    baseline_path = ROOT / "baseline" / MODEL / "results.csv"
    if baseline_path.exists():
        for r in read_csv(baseline_path):
            if r.get("model") != MODEL or r.get("stimulus_group") != "baseline":
                continue
            all_rows.append(r)

    # negative_original
    neg_path = ROOT / "negative_original" / MODEL / "results.csv"
    if neg_path.exists():
        for r in read_csv(neg_path):
            if r.get("model") != MODEL or r.get("stimulus_group") != "negative_original":
                continue
            all_rows.append(r)

    # H3 groups
    for group in H3_GROUPS:
        path = ROOT / group / MODEL / "results.csv"
        if not path.exists():
            continue
        for r in read_csv(path):
            if r.get("model") != MODEL or r.get("stimulus_group") != group:
                continue
            all_rows.append(r)

    # Déduplication explicite
    all_rows = dedupe_rows(
        all_rows,
        keys=[
            "task",
            "model",
            "stimulus_group",
            "stimulus_index",
            "legacy_pnum",
            "few_shot",
            "seed",
            "temperature",
            "do_sample",
        ],
    )
    return all_rows


def aggregate_condition(rows):
    """
    Agrège au niveau :
    - task
    - stimulus_group
    """
    grouped = {}
    for r in rows:
        task = r["task"]
        group = r["stimulus_group"]
        score = safe_float(r["score"])
        if score is None:
            continue

        key = (task, group)
        grouped.setdefault(key, []).append(score)

    out = []
    for (task, group), scores in sorted(grouped.items()):
        out.append({
            "task": task,
            "task_family": TASK_FAMILY.get(task, "other"),
            "stimulus_group": group,
            "model": MODEL,
            "n_runs": len(scores),
            "mean_score": round(mean(scores), 6),
            "std_score": round(pstdev(scores), 6) if len(scores) > 1 else 0.0,
        })
    return out


def compute_baseline_lookup(agg_rows):
    lookup = {}
    for r in agg_rows:
        if r["stimulus_group"] == "baseline":
            lookup[r["task"]] = r["mean_score"]
    return lookup


def enrich_with_deltas(agg_rows):
    baseline_lookup = compute_baseline_lookup(agg_rows)
    out = []

    for r in agg_rows:
        task = r["task"]
        baseline = baseline_lookup.get(task)
        delta = None
        if baseline is not None:
            delta = round(r["mean_score"] - baseline, 6)

        row = dict(r)
        row["baseline_score"] = baseline
        row["delta_vs_baseline"] = delta
        out.append(row)

    return out


def build_negative_original_summary(rows_with_delta):
    grouped = {}
    for r in rows_with_delta:
        if r["stimulus_group"] != "negative_original":
            continue
        grouped.setdefault(r["task"], []).append(r["mean_score"])

    baseline_lookup = compute_baseline_lookup(rows_with_delta)
    out = []
    for task, vals in sorted(grouped.items()):
        mean_neg = round(mean(vals), 6)
        baseline = baseline_lookup.get(task)
        delta = round(mean_neg - baseline, 6) if baseline is not None else None
        out.append({
            "task": task,
            "task_family": TASK_FAMILY.get(task, "other"),
            "model": MODEL,
            "mean_negative_original_score": mean_neg,
            "baseline_score": baseline,
            "delta_negative_original_vs_baseline": delta,
        })
    return out


def build_h3_vs_negative_original(rows_with_delta):
    """
    Pour chaque tâche et groupe H3 :
    delta(H3) - delta(negative_original moyen)
    """
    neg_summary = {}
    for r in build_negative_original_summary(rows_with_delta):
        neg_summary[r["task"]] = r["delta_negative_original_vs_baseline"]

    out = []
    for r in rows_with_delta:
        if r["stimulus_group"] not in H3_GROUPS:
            continue
        task = r["task"]
        neg_delta = neg_summary.get(task)
        h3_delta = r["delta_vs_baseline"]
        contrast = None
        if neg_delta is not None and h3_delta is not None:
            contrast = round(h3_delta - neg_delta, 6)

        out.append({
            "task": task,
            "task_family": r["task_family"],
            "model": MODEL,
            "stimulus_group": r["stimulus_group"],
            "mean_score": r["mean_score"],
            "delta_vs_baseline": h3_delta,
            "negative_original_delta_vs_baseline": neg_delta,
            "delta_h3_minus_negative_original": contrast,
        })
    return out


def build_matrix(rows, value_field, groups, tasks):
    matrix = []
    for group in groups:
        row = {"stimulus_group": group}
        for task in tasks:
            match = next(
                (r for r in rows if r["stimulus_group"] == group and r["task"] == task),
                None
            )
            row[task] = "" if match is None or match.get(value_field) is None else match[value_field]
        matrix.append(row)
    return matrix


def main():
    raw_rows = collect_all_rows()
    print(f"Raw deduplicated rows: {len(raw_rows)}")

    consolidated_path = OUT / "palier2a_consolidated_long.csv"
    write_csv(
        consolidated_path,
        fieldnames=[
            "task", "model", "stimulus_group", "stimulus_index", "legacy_pnum",
            "few_shot", "seed", "temperature", "do_sample", "score"
        ],
        rows=raw_rows,
    )

    agg_rows = aggregate_condition(raw_rows)
    agg_rows = enrich_with_deltas(agg_rows)

    agg_path = OUT / "palier2a_aggregated_conditions.csv"
    write_csv(
        agg_path,
        fieldnames=[
            "task", "task_family", "stimulus_group", "model",
            "n_runs", "mean_score", "std_score",
            "baseline_score", "delta_vs_baseline"
        ],
        rows=agg_rows,
    )

    neg_summary = build_negative_original_summary(agg_rows)
    neg_summary_path = OUT / "palier2a_negative_original_summary.csv"
    write_csv(
        neg_summary_path,
        fieldnames=[
            "task", "task_family", "model",
            "mean_negative_original_score",
            "baseline_score",
            "delta_negative_original_vs_baseline"
        ],
        rows=neg_summary,
    )

    h3_contrast = build_h3_vs_negative_original(agg_rows)
    h3_contrast_path = OUT / "palier2a_h3_vs_negative_original.csv"
    write_csv(
        h3_contrast_path,
        fieldnames=[
            "task", "task_family", "model", "stimulus_group",
            "mean_score", "delta_vs_baseline",
            "negative_original_delta_vs_baseline",
            "delta_h3_minus_negative_original"
        ],
        rows=h3_contrast,
    )

    # Matrices
    h3_rows = [r for r in agg_rows if r["stimulus_group"] in H3_GROUPS]
    h3_delta_matrix = build_matrix(h3_rows, "delta_vs_baseline", H3_GROUPS, TASKS)
    write_csv(
        OUT / "palier2a_h3_delta_matrix.csv",
        fieldnames=["stimulus_group"] + TASKS,
        rows=h3_delta_matrix,
    )

    contrast_matrix = build_matrix(h3_contrast, "delta_h3_minus_negative_original", H3_GROUPS, TASKS)
    write_csv(
        OUT / "palier2a_h3_minus_negative_original_matrix.csv",
        fieldnames=["stimulus_group"] + TASKS,
        rows=contrast_matrix,
    )

    neg_matrix_rows = []
    neg_lookup = {r["task"]: r["delta_negative_original_vs_baseline"] for r in neg_summary}
    neg_row = {"stimulus_group": "negative_original"}
    for task in TASKS:
        neg_row[task] = neg_lookup.get(task, "")
    neg_matrix_rows.append(neg_row)
    write_csv(
        OUT / "palier2a_negative_original_delta_matrix.csv",
        fieldnames=["stimulus_group"] + TASKS,
        rows=neg_matrix_rows,
    )

    print(f"Wrote: {consolidated_path}")
    print(f"Wrote: {agg_path}")
    print(f"Wrote: {neg_summary_path}")
    print(f"Wrote: {h3_contrast_path}")
    print(f"Wrote: {OUT / 'palier2a_h3_delta_matrix.csv'}")
    print(f"Wrote: {OUT / 'palier2a_h3_minus_negative_original_matrix.csv'}")
    print(f"Wrote: {OUT / 'palier2a_negative_original_delta_matrix.csv'}")


if __name__ == "__main__":
    main()