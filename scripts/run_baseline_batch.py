import subprocess
import sys
from pathlib import Path

# ============================================================
# SCRIPT BATCH BASELINE
# ============================================================
#
# Objectif :
# lancer automatiquement les conditions "baseline", câ€™est-Ã -dire
# les tÃ¢ches sans aucun stimulus ajoutÃ©.
#
# Pourquoi ce script :
# - produire une rÃ©fÃ©rence propre pour H3 ;
# - utiliser exactement la mÃªme infrastructure que pour les
#   catÃ©gories de nÃ©gativitÃ© ;
# - Ã©viter de lancer les baselines manuellement ;
# - garantir que la comparaison baseline vs H3 repose sur :
#     * les mÃªmes tÃ¢ches
#     * les mÃªmes modÃ¨les
#     * les mÃªmes seeds
#     * les mÃªmes paramÃ¨tres de gÃ©nÃ©ration
#
# Important :
# cette baseline nâ€™est PAS la rÃ©plication du papier.
# Elle correspond simplement au prompt de tÃ¢che seul.
#
# Exemple :
#   "Determine whether a movie review is positive or negative."
# sans aucun ajout de type NegativePrompt.
# ============================================================


# ============================================================
# PARAMÃˆTRES GLOBAUX
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
#   on garde les mÃªmes paramÃ¨tres que pour H3 afin que la baseline
#   soit comparable directement aux conditions avec stimuli.
# ============================================================

USE_APE = False
FEW_SHOT = False
TEMPERATURE = 0.0
DO_SAMPLE = False


# ============================================================
# MODÃˆLES
# ============================================================
#
# Commencer par un smoke test rÃ©duit, puis Ã©largir.
# ============================================================

MODELS = [
    "flan-t5-large",
    #"llama2",
    #"vicuna",
]


# ============================================================
# TÃ‚CHES
# ============================================================
#
# Ici on garde les mÃªmes tÃ¢ches que dans le batch H3, pour que la
# comparaison baseline vs H3 soit parfaitement alignÃ©e.
#
# Ensuite, si le pipeline est stable, tu pourras Ã©largir cette
# liste Ã  lâ€™ensemble des tÃ¢ches qui tâ€™intÃ©ressent.
# ============================================================

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


# ============================================================
# SEEDS
# ============================================================
#
# MÃªme logique que pour H3 :
# plusieurs seeds permettent de mesurer la stabilitÃ©.
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
#   racine du dÃ©pÃ´t
#
# MAIN_SCRIPT :
#   script cible Ã  exÃ©cuter
#
# LOG_FILE :
#   log dâ€™exÃ©cution pour tracer prÃ©cisÃ©ment les runs baseline
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
# Cela force le pipeline Ã  Ã©crire les rÃ©sultats dans le groupe
# "baseline", conformÃ©ment Ã  la logique que tu as ajoutÃ©e dans
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
# Boucle expÃ©rimentale baseline :
#   modÃ¨le
#   -> tÃ¢che
#      -> seed
#
# Cette structure est volontairement parallÃ¨le Ã  celle du batch H3,
# mais sans catÃ©gorie / stimulus_index.
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
# POINT Dâ€™ENTRÃ‰E
# ============================================================
#
# Lancement :
#   python .\scripts\run_baseline_batch.py
#
# ConseillÃ© :
# faire dâ€™abord un smoke test avec :
#   MODELS = ["flan-t5-large"]
#   TASKS = ["sentiment"]
#   SEEDS = [42]
# ============================================================

if __name__ == "__main__":
    main()
