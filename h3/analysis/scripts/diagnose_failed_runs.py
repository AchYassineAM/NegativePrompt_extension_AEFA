import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# DIAGNOSTIC DES RUNS EN ERREUR
# ============================================================
#
# Objectif :
# analyser les logs batch existants pour identifier :
# - quels runs ont échoué
# - sur quelles tâches
# - dans quels scripts batch
# - avec quels stimuli / pnum
# - avec quelle cause probable
#
# Pourquoi ce script :
# avant de lancer la campagne complète, il faut savoir si les
# échecs sont :
# - aléatoires
# - liés à certaines tâches
# - liés à certains formats de sortie
# - liés à certains modèles
#
# Principe :
# - chaque run est un bloc séparé par une ligne de "===="
# - un run est considéré comme réussi si son STDOUT contient
#   "Test score:"
# - sinon, il est considéré comme échoué
# - on extrait ensuite une cause probable depuis STDERR / STDOUT
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
H3_ROOT = SCRIPT_DIR.parent.parent
REPO_ROOT = H3_ROOT.parent

LOG_DIR = H3_ROOT / "results" / "batch_logs"

ANALYSIS_DIR = H3_ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
DEBUG_DIR = TABLES_DIR / "debug"

OUTPUT_CSV = DEBUG_DIR / "failed_runs_diagnostic.csv"
OUTPUT_SUMMARY = DEBUG_DIR / "failed_runs_summary.txt"

BLOCK_SEPARATOR = "=" * 100

# ------------------------------------------------------------
# REGEX DES HEADERS DE RUN
# ------------------------------------------------------------

H3_HEADER_RE = re.compile(
    r"TASK=(?P<task>.*?)\s+\|\s+MODEL=(?P<model>.*?)\s+\|\s+GROUP=(?P<group>.*?)\s+\|\s+INDEX=(?P<index>.*?)\s+\|\s+SEED=(?P<seed>.*)"
)

BASELINE_HEADER_RE = re.compile(
    r"TASK=(?P<task>.*?)\s+\|\s+MODEL=(?P<model>.*?)\s+\|\s+BASELINE\s+\|\s+SEED=(?P<seed>.*)"
)

NEGATIVE_HEADER_RE = re.compile(
    r"TASK=(?P<task>.*?)\s+\|\s+MODEL=(?P<model>.*?)\s+\|\s+NEGATIVE_ORIGINAL\s+\|\s+PNUM=(?P<pnum>.*?)\s+\|\s+SEED=(?P<seed>.*)"
)


# ------------------------------------------------------------
# EXTRACTION DES BLOCS
# ------------------------------------------------------------

def split_log_into_blocks(text: str):
    parts = text.split(BLOCK_SEPARATOR)
    blocks = []
    for part in parts:
        block = part.strip()
        if block:
            blocks.append(block)
    return blocks


# ------------------------------------------------------------
# PARSE DU HEADER DE RUN
# ------------------------------------------------------------

def parse_header(first_line: str):
    first_line = first_line.strip()

    m = H3_HEADER_RE.match(first_line)
    if m:
        return {
            "run_type": "h3",
            "task": m.group("task").strip(),
            "model": m.group("model").strip(),
            "stimulus_group": m.group("group").strip(),
            "stimulus_index": m.group("index").strip(),
            "legacy_pnum": "",
            "seed": m.group("seed").strip(),
        }

    m = BASELINE_HEADER_RE.match(first_line)
    if m:
        return {
            "run_type": "baseline",
            "task": m.group("task").strip(),
            "model": m.group("model").strip(),
            "stimulus_group": "baseline",
            "stimulus_index": "",
            "legacy_pnum": "",
            "seed": m.group("seed").strip(),
        }

    m = NEGATIVE_HEADER_RE.match(first_line)
    if m:
        return {
            "run_type": "negative_original",
            "task": m.group("task").strip(),
            "model": m.group("model").strip(),
            "stimulus_group": "negative_original",
            "stimulus_index": "",
            "legacy_pnum": m.group("pnum").strip(),
            "seed": m.group("seed").strip(),
        }

    return None


# ------------------------------------------------------------
# SEPARATION STDOUT / STDERR
# ------------------------------------------------------------

def extract_stdout_stderr(block: str):
    stdout = ""
    stderr = ""

    if "--- STDOUT ---" in block:
        after_stdout = block.split("--- STDOUT ---", 1)[1]
        if "--- STDERR ---" in after_stdout:
            stdout, stderr = after_stdout.split("--- STDERR ---", 1)
        else:
            stdout = after_stdout
    elif "--- STDERR ---" in block:
        stderr = block.split("--- STDERR ---", 1)[1]

    return stdout.strip(), stderr.strip()


# ------------------------------------------------------------
# DETERMINER SI LE RUN A REUSSI
# ------------------------------------------------------------

def classify_run(stdout: str, stderr: str):
    """
    Heuristique :
    - succès si "Test score:" apparaît dans stdout
    - sinon échec
    """
    success = "Test score:" in stdout
    return "success" if success else "failed"


# ------------------------------------------------------------
# EXTRAIRE UNE CAUSE PROBABLE
# ------------------------------------------------------------

def extract_probable_cause(stdout: str, stderr: str):
    haystack = stderr if stderr.strip() else stdout

    priority_patterns = [
        r"ValueError:.*",
        r"RuntimeError:.*",
        r"AssertionError:.*",
        r"KeyError:.*",
        r"TypeError:.*",
        r"IndexError:.*",
        r"FileNotFoundError:.*",
        r"ModuleNotFoundError:.*",
        r"Traceback.*",
        r"CUDA out of memory.*",
        r"OutOfMemoryError.*",
        r"Modèle non reconnu.*",
        r"Nombre de scores inattendu.*",
        r"Mismatch entre le nombre de requêtes.*",
        r"Le modèle a renvoyé une liste vide.*",
    ]

    lines = [line.strip() for line in haystack.splitlines() if line.strip()]

    for pattern in priority_patterns:
        for line in lines:
            if re.search(pattern, line):
                return line[:300]

    # fallback : première ligne non vide significative
    for line in lines:
        if line and not line.startswith("Loading weights:"):
            return line[:300]

    return "Unknown / no clear message"


# ------------------------------------------------------------
# ANALYSE D'UN FICHIER DE LOG
# ------------------------------------------------------------

def parse_log_file(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    blocks = split_log_into_blocks(text)

    rows = []

    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        header = parse_header(lines[0].strip())
        if header is None:
            continue

        stdout, stderr = extract_stdout_stderr(block)
        status = classify_run(stdout, stderr)
        probable_cause = extract_probable_cause(stdout, stderr) if status == "failed" else ""

        row = {
            "log_file": str(log_path),
            "run_type": header["run_type"],
            "task": header["task"],
            "model": header["model"],
            "stimulus_group": header["stimulus_group"],
            "stimulus_index": header["stimulus_index"],
            "legacy_pnum": header["legacy_pnum"],
            "seed": header["seed"],
            "status": status,
            "probable_cause": probable_cause,
        }
        rows.append(row)

    return rows


# ------------------------------------------------------------
# ECRITURE CSV
# ------------------------------------------------------------

def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "log_file",
        "run_type",
        "task",
        "model",
        "stimulus_group",
        "stimulus_index",
        "legacy_pnum",
        "seed",
        "status",
        "probable_cause",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# ECRITURE RESUME TEXTE
# ------------------------------------------------------------

def write_summary(rows, output_path: Path):
    total = len(rows)
    failed = [r for r in rows if r["status"] == "failed"]
    success = [r for r in rows if r["status"] == "success"]

    by_task = Counter(r["task"] for r in failed)
    by_run_type = Counter(r["run_type"] for r in failed)
    by_group = Counter(r["stimulus_group"] for r in failed if r["stimulus_group"])
    by_model = Counter(r["model"] for r in failed)
    by_cause = Counter(r["probable_cause"] for r in failed)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("===== FAILED RUNS SUMMARY =====\n")
        f.write(f"Total runs found: {total}\n")
        f.write(f"Successful runs : {len(success)}\n")
        f.write(f"Failed runs     : {len(failed)}\n\n")

        f.write("===== FAILED BY TASK =====\n")
        for k, v in by_task.most_common():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("===== FAILED BY RUN TYPE =====\n")
        for k, v in by_run_type.most_common():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("===== FAILED BY STIMULUS GROUP =====\n")
        for k, v in by_group.most_common():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("===== FAILED BY MODEL =====\n")
        for k, v in by_model.most_common():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("===== MOST COMMON PROBABLE CAUSES =====\n")
        for k, v in by_cause.most_common(20):
            f.write(f"{v} | {k}\n")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    if not LOG_DIR.exists():
        raise FileNotFoundError(f"Log directory not found: {LOG_DIR}")

    all_rows = []
    for log_path in LOG_DIR.glob("*.log"):
        all_rows.extend(parse_log_file(log_path))

    if not all_rows:
        print("Aucun run trouvé dans les logs.")
        return

    write_csv(all_rows, OUTPUT_CSV)
    write_summary(all_rows, OUTPUT_SUMMARY)

    failed = sum(1 for r in all_rows if r["status"] == "failed")
    print(f"{len(all_rows)} runs diagnostiqués")
    print(f"{failed} runs en échec")
    print(f"CSV écrit dans : {OUTPUT_CSV}")
    print(f"Résumé écrit dans : {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()