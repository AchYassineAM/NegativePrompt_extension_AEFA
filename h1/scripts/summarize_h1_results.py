import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

MASTER_CSV = ROOT / "h1_master.csv"
PAIRWISE_CSV = ROOT / "h1_neutral_vs_negative_original.csv"
GROUP_MEANS_CSV = ROOT / "h1_group_means.csv"


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def main():
    # 1) Collect all results.csv
    all_rows = []
    for csv_path in RESULTS_DIR.rglob("results.csv"):
        rows = read_csv(csv_path)
        for r in rows:
            r["_source"] = str(csv_path)
            all_rows.append(r)

    # 2) Deduplicate rows
    # Key = task, model, group, index, pnum, seed
    dedup = {}
    for r in all_rows:
        key = (
            r.get("task"),
            r.get("model"),
            r.get("stimulus_group"),
            r.get("stimulus_index"),
            r.get("legacy_pnum"),
            r.get("seed"),
        )
        dedup[key] = r  # keep last occurrence

    rows = list(dedup.values())

    # 3) Sort master rows
    rows.sort(key=lambda r: (
        r.get("task", ""),
        r.get("model", ""),
        r.get("stimulus_group", ""),
        to_int(r.get("stimulus_index", ""), 0) or 0,
        to_int(r.get("legacy_pnum", ""), 0) or 0,
        to_int(r.get("seed", ""), 0) or 0,
    ))

    # 4) Write master CSV
    master_fields = [
        "task",
        "model",
        "stimulus_group",
        "stimulus_index",
        "legacy_pnum",
        "few_shot",
        "seed",
        "temperature",
        "do_sample",
        "score",
        "run_time_seconds",
        "avg_time_per_sample_seconds",
        "_source",
    ]
    with open(MASTER_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=master_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in master_fields})

    # 5) Build neutral vs negative_original pair table
    neutral_map = {}
    negative_map = {}

    for r in rows:
        task = r.get("task")
        model = r.get("model")
        seed = r.get("seed")
        group = r.get("stimulus_group")
        idx = r.get("stimulus_index")
        pnum = r.get("legacy_pnum")
        score = to_float(r.get("score"))

        if model != "flan-t5-large":
            continue

        if group == "neutral" and idx not in ("", None):
            neutral_map[(task, seed, idx)] = score

        if group == "negative_original" and pnum not in ("", None):
            negative_map[(task, seed, pnum)] = score

    pair_rows = []
    for (task, seed, pair), neutral_score in neutral_map.items():
        negative_score = negative_map.get((task, seed, pair))
        if negative_score is None:
            continue
        pair_rows.append({
            "task": task,
            "pair": pair,
            "seed": seed,
            "neutral_score": neutral_score,
            "negative_score": negative_score,
            "delta_negative_minus_neutral": negative_score - neutral_score,
        })

    pair_rows.sort(key=lambda r: (
        r["task"],
        to_int(r["pair"], 0),
        to_int(r["seed"], 0),
    ))

    with open(PAIRWISE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task",
            "pair",
            "seed",
            "neutral_score",
            "negative_score",
            "delta_negative_minus_neutral",
        ])
        writer.writeheader()
        writer.writerows(pair_rows)

    # 6) Group means
    grouped = defaultdict(list)
    for r in rows:
        model = r.get("model")
        task = r.get("task")
        group = r.get("stimulus_group")
        score = to_float(r.get("score"))
        if score is None:
            continue

        if group == "negative_original":
            # split original negatives by pnum
            label = f"negative_original_p{r.get('legacy_pnum')}"
        else:
            idx = r.get("stimulus_index")
            label = f"{group}_i{idx}" if idx not in ("", None) else group

        grouped[(model, task, label)].append(score)

    mean_rows = []
    for (model, task, label), vals in grouped.items():
        mean_rows.append({
            "model": model,
            "task": task,
            "group_label": label,
            "n": len(vals),
            "mean_score": sum(vals) / len(vals),
            "min_score": min(vals),
            "max_score": max(vals),
        })

    mean_rows.sort(key=lambda r: (r["model"], r["task"], r["group_label"]))

    with open(GROUP_MEANS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model",
            "task",
            "group_label",
            "n",
            "mean_score",
            "min_score",
            "max_score",
        ])
        writer.writeheader()
        writer.writerows(mean_rows)

    print("Wrote:", MASTER_CSV)
    print("Wrote:", PAIRWISE_CSV)
    print("Wrote:", GROUP_MEANS_CSV)
    print(f"Master rows: {len(rows)}")
    print(f"Pairwise rows: {len(pair_rows)}")
    print(f"Group-mean rows: {len(mean_rows)}")


if __name__ == "__main__":
    main()