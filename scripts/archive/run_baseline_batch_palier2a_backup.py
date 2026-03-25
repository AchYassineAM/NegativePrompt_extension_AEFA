import subprocess
import sys
from pathlib import Path

# ============================================================
# SCRIPT BATCH BASELINE
# ============================================================
#
# Objectif :
# lancer automatiquement les conditions "baseline", c’est-à-dire
# les tâches sans aucun stimulus ajouté.
#
# Pourquoi ce script :
# - produire une référence propre pour H3 ;
# - utiliser exactement la même infrastructure que pour les
#   catégories de négativité ;
# - éviter de lancer les baselines manuellement ;
# - garantir que la comparaison baseline vs H3 repose sur :
#     * les mêmes tâches
#     * les mêmes modèles
#     * les mêmes seeds
#     * les mêmes paramètres de génération
#
# Important :
# cette baseline n’est PAS la réplication du papier.
# Elle correspond simplement au prompt de tâche seul.
#
# Exemple :
#   "Determine whether a movie review is positive or negative."
# sans aucun ajout de type NegativePrompt.
# ============================================================


# ============================================================
# PARAMÈTRES GLOBAUX
# ============================================================
#
# USE_APE :
#   False -> baseline sur prompts du repo (main.py)
#   True  -> baseline sur prompts APE (main_ape.py)
#
# FEW_SHOT :
#   False -> zero-shot
#   True  -> few-shot
#
# TEMPERATURE / DO_SAMPLE :
#   on garde les mêmes paramètres que pour H3 afin que la baseline
#   soit comparable directement aux conditions avec stimuli.
# ============================================================

USE_APE = False
FEW_SHOT = False
TEMPERATURE = 0.0
DO_SAMPLE = False


# ============================================================
# MODÈLES
# ============================================================
#
# Commencer par un smoke test réduit, puis élargir.
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
# Ici on garde les mêmes tâches que dans le batch H3, pour que la
# comparaison baseline vs H3 soit parfaitement alignée.
#
# Ensuite, si le pipeline est stable, tu pourras élargir cette
# liste à l’ensemble des tâches qui t’intéressent.
# ============================================================

TASKS = [
    "sentiment",
    "sentence_similarity",
    "word_in_context",
    "cause_and_effect",
    "sum",
]


# ============================================================
# SEEDS
# ============================================================
#
# Même logique que pour H3 :
# plusieurs seeds permettent de mesurer la stabilité.
#
# Pour un test rapide :
#   SEEDS = [42]
# Pour une vraie campagne :
#   SEEDS = [42, 43, 44]
# ============================================================

SEEDS = [42, 43, 44]


# ============================================================
# CHEMINS
# ============================================================
#
# ROOT :
#   racine du dépôt
#
# MAIN_SCRIPT :
#   script cible à exécuter
#
# LOG_FILE :
#   log d’exécution pour tracer précisément les runs baseline
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
MAIN_SCRIPT = ROOT / ("main_ape.py" if USE_APE else "main.py")

LOG_DIR = ROOT / "results" / "batch_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / ("baseline_batch_ape.log" if USE_APE else "baseline_batch_base.log")


# ============================================================
# FONCTION run_one
# ============================================================
#
# Ici, contrairement au batch H3 :
# - on ne passe PAS de stimulus_group
# - on fixe pnum=0
#
# Cela force le pipeline à écrire les résultats dans le groupe
# "baseline", conformément à la logique que tu as ajoutée dans
# main.py / main_ape.py.
# ============================================================

def run_one(task, model, seed):
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--task", task,
        "--model", model,
        "--pnum", "0",
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
        f.write(f"TASK={task} | MODEL={model} | BASELINE | SEED={seed}\n")
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
# Boucle expérimentale baseline :
#   modèle
#   -> tâche
#      -> seed
#
# Cette structure est volontairement parallèle à celle du batch H3,
# mais sans catégorie / stimulus_index.
# ============================================================

def main():
    total = 0
    failed = 0

    print(f"Using script: {MAIN_SCRIPT.name}")
    print(f"Logging to: {LOG_FILE}")

    for model in MODELS:
        for task in TASKS:
            for seed in SEEDS:
                total += 1
                code = run_one(task, model, seed)

                if code != 0:
                    failed += 1
                    print(f"[FAIL] task={task} model={model} baseline seed={seed}")
                else:
                    print(f"[OK]   task={task} model={model} baseline seed={seed}")

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
#   python .\scripts\run_baseline_batch.py
#
# Conseillé :
# faire d’abord un smoke test avec :
#   MODELS = ["flan-t5-large"]
#   TASKS = ["sentiment"]
#   SEEDS = [42]
# ============================================================

if __name__ == "__main__":
    main()