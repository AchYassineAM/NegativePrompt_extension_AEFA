import subprocess
import sys
import time
import os
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.instruction_induction.load_data import tasks

PYTHON = sys.executable

SEEDS = [42, 43, 44]
MODEL = "flan-t5-large"
MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 10

STIMULUS_GROUPS_H1 = {
    "negative_control": [1, 2, 3, 4],
    "positive": [1, 2, 3, 4],
    "neutral": [1, 2, 3, 4],
    "ambiguous": [1, 2, 3, 4],
}

NEGATIVE_PNUMS = [1, 2, 3, 4]

STATE_DIR = ROOT / ".h1_runner_state"
STATE_DIR.mkdir(exist_ok=True)

COMPLETED_FILE = STATE_DIR / "completed.txt"
FAILED_FILE = STATE_DIR / "failed.csv"
RUNNER_LOG = STATE_DIR / "runner.log"


def build_commands():
    commands = []

    for seed in SEEDS:
        for task in tasks:
            for pnum in NEGATIVE_PNUMS:
                cmd = [
                    PYTHON, "main.py",
                    "--task", task,
                    "--model", MODEL,
                    "--pnum", str(pnum),
                    "--seed", str(seed),
                    "--temperature", "0.0",
                    "--do_sample", "False",
                ]
                commands.append(cmd)

            for group, indices in STIMULUS_GROUPS_H1.items():
                for idx in indices:
                    cmd = [
                        PYTHON, "main.py",
                        "--task", task,
                        "--model", MODEL,
                        "--stimulus_group", group,
                        "--stimulus_index", str(idx),
                        "--seed", str(seed),
                        "--temperature", "0.0",
                        "--do_sample", "False",
                    ]
                    commands.append(cmd)

    return commands


def command_key(cmd):
    raw = " ".join(cmd)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_completed():
    if not COMPLETED_FILE.exists():
        return set()
    with open(COMPLETED_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def append_completed(key):
    with open(COMPLETED_FILE, "a", encoding="utf-8") as f:
        f.write(key + "\n")


def ensure_failed_header():
    if not FAILED_FILE.exists():
        with open(FAILED_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["command_key", "return_code", "attempt", "command"])


def append_failed(key, return_code, attempt, cmd):
    ensure_failed_header()
    with open(FAILED_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([key, return_code, attempt, " ".join(cmd)])


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(RUNNER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_command(cmd):
    key = command_key(cmd)
    log_file = STATE_DIR / f"{key}.log"

    with open(log_file, "a", encoding="utf-8") as lf:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=lf,
            stderr=lf,
            text=True
        )
    return result.returncode, log_file


def main():
    commands = build_commands()
    completed = load_completed()

    total = len(commands)
    pending = [cmd for cmd in commands if command_key(cmd) not in completed]

    log(f"Total commands defined: {total}")
    log(f"Already completed: {len(completed)}")
    log(f"Pending commands: {len(pending)}")
    log(f"State dir: {STATE_DIR}")

    if not pending:
        log("Nothing to do. All commands are already completed.")
        return

    global_start = time.perf_counter()
    run_times = []
    success_count = 0
    fail_count = 0

    for idx, cmd in enumerate(pending, start=1):
        key = command_key(cmd)
        label = " ".join(cmd)
        attempt = 0
        done = False

        while attempt <= MAX_RETRIES and not done:
            attempt += 1
            start = time.perf_counter()

            log("=" * 100)
            log(f"RUN {idx}/{len(pending)} | ATTEMPT {attempt}/{MAX_RETRIES + 1}")
            log(f"CMD: {label}")

            return_code, log_file = run_command(cmd)

            elapsed = time.perf_counter() - start
            run_times.append(elapsed)

            avg_time = sum(run_times) / len(run_times)
            remaining = avg_time * (len(pending) - idx)

            if return_code == 0:
                append_completed(key)
                success_count += 1
                done = True
                log(
                    f"SUCCESS | time={elapsed:.2f}s | avg={avg_time:.2f}s | "
                    f"eta_remaining={remaining/60:.2f} min | log={log_file.name}"
                )
            else:
                log(
                    f"FAIL | return_code={return_code} | time={elapsed:.2f}s | "
                    f"log={log_file.name}"
                )
                append_failed(key, return_code, attempt, cmd)

                if attempt <= MAX_RETRIES:
                    log(f"Retrying in {RETRY_SLEEP_SECONDS}s...")
                    time.sleep(RETRY_SLEEP_SECONDS)
                else:
                    fail_count += 1
                    log("Giving up on this command and continuing to the next one.")

    total_elapsed = time.perf_counter() - global_start
    avg_time = (sum(run_times) / len(run_times)) if run_times else 0.0

    log("=" * 100)
    log(
        f"FINISHED | success={success_count} | failed={fail_count} | "
        f"total_time={total_elapsed/60:.2f} min | avg_command_time={avg_time:.2f}s"
    )
    log(f"Completed file: {COMPLETED_FILE}")
    log(f"Failed file: {FAILED_FILE}")
    log(f"Runner log: {RUNNER_LOG}")


if __name__ == "__main__":
    main()