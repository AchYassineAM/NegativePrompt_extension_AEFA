import subprocess
import re
import csv
from pathlib import Path
from statistics import mean, pstdev

# =========================
# CONFIGURATION
# =========================
TASK = "first_word_letter"
MODEL = "flan-t5-large"
PNUM = 6
FEW_SHOT = "False"
TEMPERATURE = "0.0"
DO_SAMPLE = "False"

# Seeds à tester
SEEDS = [42, 43, 44]

# Fichier de sortie
output_file = Path("replicates_results.csv")

rows = []

print("=== LANCEMENT DES RÉPLICATIONS ===")

for seed in SEEDS:
    cmd = [
        "python", "main.py",
        "--task", TASK,
        "--model", MODEL,
        "--pnum", str(PNUM),
        "--few_shot", FEW_SHOT,
        "--seed", str(seed),
        "--temperature", TEMPERATURE,
        "--do_sample", DO_SAMPLE
    ]

    print("\nRUNNING:", " ".join(cmd))

    process = subprocess.run(cmd, capture_output=True, text=True)

    stdout = process.stdout
    stderr = process.stderr

    match = re.search(r"Test score:\s*([0-9.]+)", stdout)
    score = float(match.group(1)) if match else None

    rows.append({
        "task": TASK,
        "model": MODEL,
        "pnum": PNUM,
        "few_shot": FEW_SHOT,
        "seed": seed,
        "temperature": TEMPERATURE,
        "do_sample": DO_SAMPLE,
        "score": score,
        "returncode": process.returncode,
        "stderr": stderr.strip()
    })

# Sauvegarde CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "task", "model", "pnum", "few_shot",
            "seed", "temperature", "do_sample",
            "score", "returncode", "stderr"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Résultats écrits dans {output_file}")

# Petit résumé
valid_scores = [r["score"] for r in rows if r["score"] is not None]

if len(valid_scores) > 0:
    print("\n=== RÉSUMÉ ===")
    print("Scores :", valid_scores)
    print("Moyenne :", mean(valid_scores))
    if len(valid_scores) > 1:
        print("Écart-type (population) :", pstdev(valid_scores))
else:
    print("\n⚠️ Aucun score valide récupéré.")