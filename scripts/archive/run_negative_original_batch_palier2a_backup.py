import subprocess
import sys
from pathlib import Path

# ============================================================
# SCRIPT BATCH NEGATIVE ORIGINAL
# ============================================================
#
# Objectif :
# lancer automatiquement les prompts négatifs historiques du
# papier NegativePrompt, c’est-à-dire les stimuli originaux
# indexés par pnum = 1..10 dans Negative_SET.
#
# Pourquoi ce script :
# - compléter l’infrastructure entre baseline et H3 ;
# - disposer d’un point de comparaison direct avec le papier ;
# - permettre ensuite une lecture à trois niveaux :
#       baseline
#       negative_original
#       catégories H3
#
# Rappel méthodologique :
# - baseline = prompt seul
# - negative_original = prompts négatifs du papier
# - H3 = taxonomie raffinée de la négativité
#
# Ce script ne remplace pas H1 :
# c’est surtout un ancrage expérimental pour situer H3
# par rapport à l’article.
# ============================================================


# ============================================================
# PARAMÈTRES GLOBAUX
# ============================================================

USE_APE = False
FEW_SHOT = False
TEMPERATURE = 0.0
DO_SAMPLE = False


# ============================================================
# MODÈLES
# ============================================================

MODELS = [
    "flan-t5-large",
    #"llama2",
    #"vicuna",
]


# ============================================================
# TÂCHES
# ============================================================
#
# Pour rester comparable aux autres scripts batch, on commence
# par les mêmes tâches pilotes. Ensuite, tu pourras élargir.
# ============================================================

TASKS = [
    "sentiment",
    "sum",
]


# ============================================================
# PROMPTS NEGATIFS ORIGINAUX
# ============================================================
#
# Dans le repo, Negative_SET contient 10 prompts historiques.
# Ici, on les parcourt via pnum = 1..10.
# ============================================================

NEGATIVE_PNUMS = list(range(1, 11))


# ============================================================
# SEEDS
# ============================================================

SEEDS = [42, 43, 44]


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
MAIN_SCRIPT = ROOT / ("main_ape.py" if USE_APE else "main.py")

LOG_DIR = ROOT / "results" / "batch_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / ("negative_original_batch_ape.log" if USE_APE else "negative_original_batch_base.log")


# ============================================================
# FONCTION run_one
# ============================================================
#
# Ici, contrairement à H3 :
# - on ne passe pas stimulus_group
# - on passe pnum = 1..10
#
# Grâce à la logique déjà ajoutée dans main.py / main_ape.py,
# ces runs seront écrits dans result_group = negative_original.
# ============================================================

def run_one(task, model, pnum, seed):
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--task", task,
        "--model", model,
        "--pnum", str(pnum),
        "--few_shot", str(FEW_SHOT),
        "--seed", str(seed),
        "--temperature", str(TEMPERATURE),
        "--do_sample", str(DO_SAMPLE),
    ]

    print("RUN:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True
    )

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"TASK={task} | MODEL={model} | NEGATIVE_ORIGINAL | PNUM={pnum} | SEED={seed}\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.write("\n--- STDOUT ---\n")
        f.write(result.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr)
        f.write("\n")

    return result.returncode


# ============================================================
# FONCTION main
# ============================================================
#
# Boucle expérimentale :
#   modèle
#   -> tâche
#      -> pnum (1..10)
#         -> seed
#
# Cela permet de relire ensuite les 10 prompts historiques
# séparément, avant d’éventuellement les moyenner.
# ============================================================

def main():
    total = 0
    failed = 0

    print(f"Using script: {MAIN_SCRIPT.name}")
    print(f"Logging to: {LOG_FILE}")

    for model in MODELS:
        for task in TASKS:
            for pnum in NEGATIVE_PNUMS:
                for seed in SEEDS:
                    total += 1
                    code = run_one(task, model, pnum, seed)

                    if code != 0:
                        failed += 1
                        print(f"[FAIL] task={task} model={model} pnum={pnum} seed={seed}")
                    else:
                        print(f"[OK]   task={task} model={model} pnum={pnum} seed={seed}")

    print("\n" + "=" * 60)
    print(f"TOTAL RUNS: {total}")
    print(f"FAILED RUNS: {failed}")
    print(f"LOG FILE: {LOG_FILE}")
    print("=" * 60)


# ============================================================
# POINT D’ENTRÉE
# ============================================================
#
# Lancement :
#   python .\scripts\run_negative_original_batch.py
#
# Smoke test conseillé :
# - un seul modèle
# - une ou deux tâches
# - une seule seed
# ============================================================

if __name__ == "__main__":
    main()