import csv
import json
from pathlib import Path
from collections import defaultdict


ROOT_RESULTS = Path("results")
STRATA_PATH = Path("configs") / "task_strata.json"
OUTPUT_PATH = Path("analysis") / "aggregated_by_strata.csv"


def load_task_strata(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_task_info(task: str, strata: dict):
    if task in strata["instruction_induction"]:
        return "instruction_induction", strata["instruction_induction"][task]
    if task in strata["bigbench"]:
        return "bigbench", strata["bigbench"][task]
    return "unknown", "unknown"


def detect_prompt_source(csv_path: Path):
    parts = [p.lower() for p in csv_path.parts]
    # results/ape/<group>/<model>/results.csv
    if "ape" in parts:
        return "ape"
    return "base"


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


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


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
        grouped[key].append((task, score))

    aggregated = []
    baseline_lookup = {}

    # 1) calcul des moyennes agrÃ©gÃ©es
    for key, values in grouped.items():
        prompt_source, benchmark, model, stimulus_group, task_family = key

        scores = [score for _, score in values]
        unique_tasks = sorted(set(task for task, _ in values))
        mean_score = sum(scores) / len(scores) if scores else 0.0

        row = {
            "prompt_source": prompt_source,
            "benchmark": benchmark,
            "model": model,
            "stimulus_group": stimulus_group,
            "task_family": task_family,
            "n_rows": len(values),
            "n_unique_tasks": len(unique_tasks),
            "mean_score": round(mean_score, 6),
            "delta_vs_baseline": "",
            "tasks": " | ".join(unique_tasks),
        }
        aggregated.append(row)

        if stimulus_group == "baseline":
            baseline_lookup[(prompt_source, benchmark, model, task_family)] = mean_score

    # 2) ajout du delta par rapport Ã  baseline
    for row in aggregated:
        key = (
            row["prompt_source"],
            row["benchmark"],
            row["model"],
            row["task_family"],
        )
        baseline_score = baseline_lookup.get(key)

        if baseline_score is not None:
            delta = row["mean_score"] - baseline_score
            row["delta_vs_baseline"] = round(delta, 6)

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
    "mean_score",
    "delta_vs_baseline",
    "tasks",
]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not ROOT_RESULTS.exists():
        raise FileNotFoundError(f"Dossier introuvable: {ROOT_RESULTS}")

    if not STRATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable: {STRATA_PATH}")

    strata = load_task_strata(STRATA_PATH)
    rows = read_all_result_rows(ROOT_RESULTS)

    if not rows:
        print("Aucun results.csv trouvÃ© dans le dossier results/")
        return

    aggregated = aggregate_rows(rows, strata)
    write_csv(aggregated, OUTPUT_PATH)

    print(f"{len(rows)} lignes brutes lues")
    print(f"{len(aggregated)} lignes agrÃ©gÃ©es Ã©crites dans: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
