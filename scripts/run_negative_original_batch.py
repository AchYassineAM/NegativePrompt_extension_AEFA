import subprocess
import sys
from pathlib import Path

# ============================================================
# SCRIPT BATCH NEGATIVE ORIGINAL
# ============================================================
#
# Objectif :
# lancer automatiquement les prompts nÃ©gatifs historiques du
# papier NegativePrompt, câ€™est-Ã -dire les stimuli originaux
# indexÃ©s par pnum = 1..10 dans Negative_SET.
#
# Pourquoi ce script :
# - complÃ©ter lâ€™infrastructure entre baseline et H3 ;
# - disposer dâ€™un point de comparaison direct avec le papier ;
# - permettre ensuite une lecture Ã  trois niveaux :
#       baseline
#       negative_original
#       catÃ©gories H3
#
# Rappel mÃ©thodologique :
# - baseline = prompt seul
# - negative_original = prompts nÃ©gatifs du papier
# - H3 = taxonomie raffinÃ©e de la nÃ©gativitÃ©
#
# Ce script ne remplace pas H1 :
# câ€™est surtout un ancrage expÃ©rimental pour situer H3
# par rapport Ã  lâ€™article.
# ============================================================


# ============================================================
# PARAMÃˆTRES GLOBAUX
# ============================================================

USE_APE = False
FEW_SHOT = False
TEMPERATURE = 0.0
DO_SAMPLE = False


# ============================================================
# MODÃˆLES
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
# Pour rester comparable aux autres scripts batch, on commence
# par les mÃªmes tÃ¢ches pilotes. Ensuite, tu pourras Ã©largir.
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
# Ici, contrairement Ã  H3 :
# - on ne passe pas stimulus_group
# - on passe pnum = 1..10
#
# GrÃ¢ce Ã  la logique dÃ©jÃ  ajoutÃ©e dans main.py / main_ape.py,
# ces runs seront Ã©crits dans result_group = negative_original.
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
# Boucle expÃ©rimentale :
#   modÃ¨le
#   -> tÃ¢che
#      -> pnum (1..10)
#         -> seed
#
# Cela permet de relire ensuite les 10 prompts historiques
# sÃ©parÃ©ment, avant dâ€™Ã©ventuellement les moyenner.
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
# POINT Dâ€™ENTRÃ‰E
# ============================================================
#
# Lancement :
#   python .\scripts\run_negative_original_batch.py
#
# Smoke test conseillÃ© :
# - un seul modÃ¨le
# - une ou deux tÃ¢ches
# - une seule seed
# ============================================================

if __name__ == "__main__":
    main()
