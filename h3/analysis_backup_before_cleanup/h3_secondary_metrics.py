import ast
import csv
import re
from pathlib import Path
from statistics import median

# ============================================================
# H3 SECONDARY METRICS
# ============================================================
#
# Objectif :
# calculer des métriques secondaires pour certaines tâches où
# la métrique principale du benchmark (souvent exact match)
# est trop binaire.
#
# Cette version parse à la fois :
# - les logs H3 :
#     TASK=... | MODEL=... | GROUP=... | INDEX=... | SEED=...
# - les logs baseline :
#     TASK=... | MODEL=... | BASELINE | SEED=...
# ============================================================

LOG_DIR = Path("results") / "batch_logs"
OUTPUT_PATH = Path("analysis") / "h3_secondary_metrics_numeric.csv"

TARGET_TASKS = {"sum", "diff"}

RUN_HEADER_H3_RE = re.compile(
    r"TASK=(?P<task>.*?)\s+\|\s+MODEL=(?P<model>.*?)\s+\|\s+GROUP=(?P<group>.*?)\s+\|\s+INDEX=(?P<index>.*?)\s+\|\s+SEED=(?P<seed>.*)"
)

RUN_HEADER_BASELINE_RE = re.compile(
    r"TASK=(?P<task>.*?)\s+\|\s+MODEL=(?P<model>.*?)\s+\|\s+BASELINE\s+\|\s+SEED=(?P<seed>.*)"
)

INPUT_RE = re.compile(r"^Model Input:\s*(.*)$")
PROCESSED_RE = re.compile(r"^Processed Output:\s*(.*)$")
ANS_RE = re.compile(r"^Ans:\s*(.*)$")


def safe_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        return None


def parse_answer_list(ans_text):
    try:
        parsed = ast.literal_eval(ans_text.strip())
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return None
    except Exception:
        return None


def parse_log_file(log_path: Path):
    runs = []
    current_run = None
    current_input = None
    current_processed = None
    current_ans = None

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    for line in lines:
        header_h3 = RUN_HEADER_H3_RE.search(line)
        header_baseline = RUN_HEADER_BASELINE_RE.search(line)

        if header_h3:
            if current_run is not None:
                runs.append(current_run)

            current_run = {
                "task": header_h3.group("task").strip(),
                "model": header_h3.group("model").strip(),
                "stimulus_group": header_h3.group("group").strip(),
                "stimulus_index": header_h3.group("index").strip(),
                "seed": header_h3.group("seed").strip(),
                "log_file": str(log_path),
                "examples": []
            }
            current_input = None
            current_processed = None
            current_ans = None
            continue

        if header_baseline:
            if current_run is not None:
                runs.append(current_run)

            current_run = {
                "task": header_baseline.group("task").strip(),
                "model": header_baseline.group("model").strip(),
                "stimulus_group": "baseline",
                "stimulus_index": "",
                "seed": header_baseline.group("seed").strip(),
                "log_file": str(log_path),
                "examples": []
            }
            current_input = None
            current_processed = None
            current_ans = None
            continue

        if current_run is None:
            continue

        m_in = INPUT_RE.match(line)
        if m_in:
            current_input = m_in.group(1).strip()
            continue

        m_proc = PROCESSED_RE.match(line)
        if m_proc:
            current_processed = m_proc.group(1).strip()
            continue

        m_ans = ANS_RE.match(line)
        if m_ans:
            current_ans = m_ans.group(1).strip()

            current_run["examples"].append({
                "input": current_input,
                "processed_output": current_processed,
                "ans_text": current_ans,
            })

            current_input = None
            current_processed = None
            current_ans = None
            continue

    if current_run is not None:
        runs.append(current_run)

    return runs


def compute_numeric_metrics_for_run(run):
    task = run["task"]
    if task not in TARGET_TASKS:
        return None

    absolute_errors = []
    n_total = 0
    n_parseable = 0

    for ex in run["examples"]:
        n_total += 1

        pred_raw = ex["processed_output"]
        ans_raw = parse_answer_list(ex["ans_text"])

        pred_val = safe_int(pred_raw)
        ans_val = safe_int(ans_raw)

        if pred_val is not None:
            n_parseable += 1

        if pred_val is not None and ans_val is not None:
            absolute_errors.append(abs(pred_val - ans_val))

    parseable_rate = (n_parseable / n_total) if n_total > 0 else 0.0
    mean_abs_error = (sum(absolute_errors) / len(absolute_errors)) if absolute_errors else None
    median_abs_error = median(absolute_errors) if absolute_errors else None

    return {
        "task": run["task"],
        "model": run["model"],
        "stimulus_group": run["stimulus_group"],
        "stimulus_index": run["stimulus_index"],
        "seed": run["seed"],
        "n_examples": n_total,
        "n_parseable": n_parseable,
        "parseable_rate": round(parseable_rate, 6),
        "mean_abs_error": round(mean_abs_error, 6) if mean_abs_error is not None else "",
        "median_abs_error": round(median_abs_error, 6) if median_abs_error is not None else "",
        "log_file": run["log_file"],
    }


def collect_all_numeric_metrics():
    rows = []

    if not LOG_DIR.exists():
        print(f"Dossier introuvable: {LOG_DIR}")
        return rows

    for log_path in LOG_DIR.glob("*.log"):
        runs = parse_log_file(log_path)
        for run in runs:
            row = compute_numeric_metrics_for_run(run)
            if row is not None:
                rows.append(row)

    return rows


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task",
        "model",
        "stimulus_group",
        "stimulus_index",
        "seed",
        "n_examples",
        "n_parseable",
        "parseable_rate",
        "mean_abs_error",
        "median_abs_error",
        "log_file",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = collect_all_numeric_metrics()

    if not rows:
        print("Aucune métrique secondaire numérique extraite.")
        return

    write_csv(rows, OUTPUT_PATH)
    print(f"{len(rows)} runs numériques écrits dans : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()