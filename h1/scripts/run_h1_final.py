import subprocess
import re
import csv
import time
from pathlib import Path

TASKS = [
    "sentiment",
    "sentence_similarity",
    "larger_animal",
    "sum",
    "orthography_starts_with",
    "word_in_context",
    "cause_and_effect",
    "second_word_letter",
    "first_word_letter",
    "letters_list",
    "singular_to_plural",
    "active_to_passive",
    "negation",
    "taxonomy_animal",
    "rhymes",
    "diff",
    "num_to_verbal",
    "translation_en-de",
    "translation_en-es",
    "translation_en-fr",
    "antonyms",
    "synonyms",
    "common_concept",
    "informal_to_formal",
]

SEEDS = [42, 43, 44]
MODEL = "flan-t5-large"

OUTPUT_CSV = "h1_results.csv"
LOG_FILE = "run_log.txt"


def log(msg, logfile):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    logfile.write(line + "\n")
    logfile.flush()


def run_and_extract_score(cmd, logfile):
    log(f"RUN: {' '.join(cmd)}", logfile)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # on écrit aussi la sortie brute du sous-processus dans le log
    logfile.write(result.stdout + "\n")
    logfile.flush()

    if result.returncode != 0:
        log(f"ERROR: return code {result.returncode}", logfile)
        return None

    match = re.search(r"Test score:\s*([0-9.]+)", result.stdout)
    if match:
        score = float(match.group(1))
        log(f"SCORE: {score}", logfile)
        return score
    else:
        log("ERROR: Test score not found in output", logfile)
        return None


def main():
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as logfile, \
         open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow([
            "task",
            "pair",
            "seed",
            "neutral_score",
            "negative_score",
            "delta"
        ])
        csvfile.flush()

        total = len(TASKS) * len(SEEDS) * 4
        current = 0

        log("=== START H1 RUN ===", logfile)
        log(f"Tasks: {TASKS}", logfile)
        log(f"Seeds: {SEEDS}", logfile)
        log(f"Model: {MODEL}", logfile)
        log(f"Total pair-comparisons: {total}", logfile)

        for task in TASKS:
            for seed in SEEDS:
                for i in [1, 2, 3, 4]:
                    current += 1
                    log(f"=== COMPARISON {current}/{total} | task={task} | pair={i} | seed={seed} ===", logfile)

                    neutral_cmd = [
                        "python", "main.py",
                        "--task", task,
                        "--model", MODEL,
                        "--few_shot", "False",
                        "--stimulus_group", "neutral",
                        "--stimulus_index", str(i),
                        "--seed", str(seed),
                        "--temperature", "0.0",
                        "--do_sample", "False",
                    ]

                    neutral_score = run_and_extract_score(neutral_cmd, logfile)

                    negative_cmd = [
                        "python", "main.py",
                        "--task", task,
                        "--model", MODEL,
                        "--few_shot", "False",
                        "--pnum", str(i),
                        "--seed", str(seed),
                        "--temperature", "0.0",
                        "--do_sample", "False",
                    ]

                    negative_score = run_and_extract_score(negative_cmd, logfile)

                    delta = None
                    if neutral_score is not None and negative_score is not None:
                        delta = negative_score - neutral_score

                    writer.writerow([
                        task,
                        i,
                        seed,
                        neutral_score,
                        negative_score,
                        delta
                    ])
                    csvfile.flush()

                    log(
                        f"WROTE CSV | task={task} | pair={i} | seed={seed} | "
                        f"neutral={neutral_score} | negative={negative_score} | delta={delta}",
                        logfile
                    )

        log("=== END H1 RUN ===", logfile)


if __name__ == "__main__":
    main()