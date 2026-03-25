import subprocess
import sys
from pathlib import Path

# ============================================================
# SCRIPT BATCH H3
# ============================================================
#
# Objectif :
# lancer automatiquement les expÃ©riences liÃ©es Ã  lâ€™hypothÃ¨se H3
# (â€œdiffÃ©rents types de nÃ©gativitÃ©â€) sans exÃ©cution manuelle
# rÃ©pÃ©titive.
#
# Pourquoi ce script :
# - figer le protocole expÃ©rimental avant la vraie campagne ;
# - Ã©viter les erreurs manuelles (oubli de seed, mauvais stimulus,
#   mauvaise tÃ¢che, mauvais modÃ¨le) ;
# - produire une collecte homogÃ¨ne des rÃ©sultats pour lâ€™analyse ;
# - sÃ©parer clairement la phase "infrastructure" de la phase
#   "interprÃ©tation / rÃ©daction".
#
# DiffÃ©rence par rapport au papier original :
# le papier teste 10 NegativePrompts historiques. Ici, on passe Ã 
# une taxonomie plus structurÃ©e, organisÃ©e par catÃ©gories H3 :
#   - competence_threat
#   - inconsistency
#   - social_comparison
#   - urgency
#   - regret
#
# Le but nâ€™est plus seulement de reproduire le papier, mais de
# comparer des sous-types de nÃ©gativitÃ© de maniÃ¨re systÃ©matique.
# ============================================================


# ============================================================
# PARAMÃˆTRES GLOBAUX DU PROTOCOLE
# ============================================================
#
# USE_APE :
#   False -> on utilise main.py (prompts de base du repo)
#   True  -> on utilise main_ape.py (baseline APE)
#
# FEW_SHOT :
#   False -> protocole zero-shot
#   True  -> few-shot avec dÃ©monstrations
#
# TEMPERATURE / DO_SAMPLE :
#   on les fixe explicitement pour Ã©viter que la variance de
#   gÃ©nÃ©ration soit confondue avec lâ€™effet des stimuli.
#
# Remarque mÃ©thodologique :
# dans le papier original, certains paramÃ¨tres diffÃ¨rent entre
# modÃ¨les. Ici, on cherche au contraire Ã  harmoniser les settings
# pour rendre les comparaisons plus propres.
# ============================================================

USE_APE = False
FEW_SHOT = False
TEMPERATURE = 0.0
DO_SAMPLE = False


# ============================================================
# MODÃˆLES Ã€ TESTER
# ============================================================
#
# Commencer petit pour un "smoke test", puis Ã©largir.
# Exemple :
#   - dâ€™abord flan-t5-large seul
#   - puis ajouter llama2
#   - puis vicuna
#
# Conseil :
# ne lancer les trois modÃ¨les que lorsque le pipeline est validÃ©
# sur un sous-ensemble rÃ©duit de tÃ¢ches et de seeds.
# ============================================================

MODELS = [
    "flan-t5-large",
    #"llama2",
    #"vicuna",
]


# ============================================================
# TÃ‚CHES Ã€ TESTER
# ============================================================
#
# Pour lâ€™instant, on peut commencer par un petit sous-ensemble
# pilote. Ensuite, on pourra Ã©tendre aux autres tÃ¢ches.
#
# Pourquoi commencer petit :
# - valider le bon fonctionnement du batch ;
# - vÃ©rifier la structure des CSV ;
# - confirmer que les catÃ©gories H3 tournent bien ;
# - Ã©viter de lancer trop tÃ´t une grosse campagne coÃ»teuse.
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
# CATÃ‰GORIES H3 ET NOMBRE DE PROMPTS PAR CATÃ‰GORIE
# ============================================================
#
# Chaque entrÃ©e indique :
#   nom_de_categorie -> nombre de prompts disponibles
#
# Ici, chaque catÃ©gorie H3 contient 2 formulations candidates.
#
# Rappel thÃ©orique :
# lâ€™objectif est de comparer des types de nÃ©gativitÃ©, pas seulement
# une valence nÃ©gative globale. On cherche donc Ã  voir si certaines
# catÃ©gories performent mieux selon les familles de tÃ¢ches.
# ============================================================

H3_GROUPS = {
    "competence_threat": 2,
    "inconsistency": 2,
    "social_comparison": 2,
    "urgency": 2,
    "regret": 2,
}


# ============================================================
# SEEDS
# ============================================================
#
# Pourquoi plusieurs seeds :
# - mÃªme avec tempÃ©rature nulle, il peut subsister de lâ€™alÃ©atoire
#   dans le tirage des items ou des dÃ©monstrations few-shot ;
# - plusieurs seeds permettent dâ€™estimer la stabilitÃ© des effets.
#
# Pour un premier test :
#   SEEDS = [42]
# Pour une vraie campagne :
#   SEEDS = [42, 43, 44] ou davantage
# ============================================================

SEEDS = [42, 43, 44]

# ============================================================
# CHEMINS DE TRAVAIL
# ============================================================
#
# ROOT :
#   racine du dÃ©pÃ´t
#
# MAIN_SCRIPT :
#   script cible Ã  appeler automatiquement
#
# LOG_DIR / LOG_FILE :
#   journal dâ€™exÃ©cution, utile pour :
#   - retracer les runs,
#   - repÃ©rer les erreurs,
#   - documenter prÃ©cisÃ©ment ce qui a Ã©tÃ© lancÃ©.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
MAIN_SCRIPT = ROOT / ("main_ape.py" if USE_APE else "main.py")
LOG_DIR = ROOT / "results" / "batch_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / ("h3_batch_ape.log" if USE_APE else "h3_batch_base.log")


# ============================================================
# FONCTION run_one
# ============================================================
#
# RÃ´le :
# exÃ©cuter UNE combinaison expÃ©rimentale complÃ¨te :
#   (task, model, stimulus_group, stimulus_index, seed)
#
# Ce que fait cette fonction :
# 1. construit la ligne de commande Python ;
# 2. appelle le script principal (main.py ou main_ape.py) ;
# 3. capture stdout/stderr ;
# 4. Ã©crit un log dÃ©taillÃ© ;
# 5. renvoie un code de succÃ¨s/Ã©chec.
#
# Pourquoi capturer stdout/stderr :
# cela permet de conserver une trace exploitable si un run Ã©choue,
# sans perdre les messages de debug produits par le pipeline.
# ============================================================

def run_one(task, model, stimulus_group, stimulus_index, seed):
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--task", task,
        "--model", model,
        "--few_shot", str(FEW_SHOT),
        "--stimulus_group", stimulus_group,
        "--stimulus_index", str(stimulus_index),
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
        f.write(
            f"TASK={task} | MODEL={model} | "
            f"GROUP={stimulus_group} | INDEX={stimulus_index} | SEED={seed}\n"
        )
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
# RÃ´le :
# parcourir toutes les combinaisons dÃ©finies plus haut.
#
# Boucle expÃ©rimentale :
#   modÃ¨le
#   -> tÃ¢che
#      -> catÃ©gorie H3
#         -> prompt de la catÃ©gorie
#            -> seed
#
# Pourquoi cet ordre :
# il est lisible, et correspond bien Ã  la logique analytique :
# on compare dâ€™abord les catÃ©gories Ã  lâ€™intÃ©rieur dâ€™une tÃ¢che,
# puis les tÃ¢ches Ã  lâ€™intÃ©rieur dâ€™un modÃ¨le.
#
# Sortie finale :
# un rÃ©sumÃ© simple :
#   - nombre total de runs
#   - nombre dâ€™Ã©checs
#   - chemin du log
# ============================================================

def main():
    total = 0
    failed = 0

    print(f"Using script: {MAIN_SCRIPT.name}")
    print(f"Logging to: {LOG_FILE}")

    for model in MODELS:
        for task in TASKS:
            for group, n_prompts in H3_GROUPS.items():
                for stimulus_index in range(1, n_prompts + 1):
                    for seed in SEEDS:
                        total += 1
                        code = run_one(task, model, group, stimulus_index, seed)

                        if code != 0:
                            failed += 1
                            print(
                                f"[FAIL] task={task} model={model} "
                                f"group={group} idx={stimulus_index} seed={seed}"
                            )
                        else:
                            print(
                                f"[OK]   task={task} model={model} "
                                f"group={group} idx={stimulus_index} seed={seed}"
                            )

    print("\n" + "=" * 60)
    print(f"TOTAL RUNS: {total}")
    print(f"FAILED RUNS: {failed}")
    print(f"LOG FILE: {LOG_FILE}")
    print("=" * 60)


# ============================================================
# POINT Dâ€™ENTRÃ‰E
# ============================================================
#
# Le script est pensÃ© pour Ãªtre lancÃ© directement :
#
#   python .\scripts\run_h3_batch.py
#
# Avant une campagne complÃ¨te :
# rÃ©duire provisoirement MODELS, TASKS et SEEDS pour un smoke test.
# ============================================================

if __name__ == "__main__":
    main()
