import subprocess
import sys
from pathlib import Path

# ============================================================
# SCRIPT BATCH H3
# ============================================================
#
# Objectif :
# lancer automatiquement les expériences liées à l’hypothèse H3
# (“différents types de négativité”) sans exécution manuelle
# répétitive.
#
# Pourquoi ce script :
# - figer le protocole expérimental avant la vraie campagne ;
# - éviter les erreurs manuelles (oubli de seed, mauvais stimulus,
#   mauvaise tâche, mauvais modèle) ;
# - produire une collecte homogène des résultats pour l’analyse ;
# - séparer clairement la phase "infrastructure" de la phase
#   "interprétation / rédaction".
#
# Différence par rapport au papier original :
# le papier teste 10 NegativePrompts historiques. Ici, on passe à
# une taxonomie plus structurée, organisée par catégories H3 :
#   - competence_threat
#   - inconsistency
#   - social_comparison
#   - urgency
#   - regret
#
# Le but n’est plus seulement de reproduire le papier, mais de
# comparer des sous-types de négativité de manière systématique.
# ============================================================


# ============================================================
# PARAMÈTRES GLOBAUX DU PROTOCOLE
# ============================================================
#
# USE_APE :
#   False -> on utilise main.py (prompts de base du repo)
#   True  -> on utilise main_ape.py (baseline APE)
#
# FEW_SHOT :
#   False -> protocole zero-shot
#   True  -> few-shot avec démonstrations
#
# TEMPERATURE / DO_SAMPLE :
#   on les fixe explicitement pour éviter que la variance de
#   génération soit confondue avec l’effet des stimuli.
#
# Remarque méthodologique :
# dans le papier original, certains paramètres diffèrent entre
# modèles. Ici, on cherche au contraire à harmoniser les settings
# pour rendre les comparaisons plus propres.
# ============================================================

USE_APE = False
FEW_SHOT = False
TEMPERATURE = 0.0
DO_SAMPLE = False


# ============================================================
# MODÈLES À TESTER
# ============================================================
#
# Commencer petit pour un "smoke test", puis élargir.
# Exemple :
#   - d’abord flan-t5-large seul
#   - puis ajouter llama2
#   - puis vicuna
#
# Conseil :
# ne lancer les trois modèles que lorsque le pipeline est validé
# sur un sous-ensemble réduit de tâches et de seeds.
# ============================================================

MODELS = [
    "flan-t5-large",
    #"llama2",
    #"vicuna",
]


# ============================================================
# TÂCHES À TESTER
# ============================================================
#
# Pour l’instant, on peut commencer par un petit sous-ensemble
# pilote. Ensuite, on pourra étendre aux autres tâches.
#
# Pourquoi commencer petit :
# - valider le bon fonctionnement du batch ;
# - vérifier la structure des CSV ;
# - confirmer que les catégories H3 tournent bien ;
# - éviter de lancer trop tôt une grosse campagne coûteuse.
# ============================================================

TASKS = [
    "sentiment",
    "sentence_similarity",
    "word_in_context",
    "cause_and_effect",
    "sum",
]


# ============================================================
# CATÉGORIES H3 ET NOMBRE DE PROMPTS PAR CATÉGORIE
# ============================================================
#
# Chaque entrée indique :
#   nom_de_categorie -> nombre de prompts disponibles
#
# Ici, chaque catégorie H3 contient 2 formulations candidates.
#
# Rappel théorique :
# l’objectif est de comparer des types de négativité, pas seulement
# une valence négative globale. On cherche donc à voir si certaines
# catégories performent mieux selon les familles de tâches.
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
# - même avec température nulle, il peut subsister de l’aléatoire
#   dans le tirage des items ou des démonstrations few-shot ;
# - plusieurs seeds permettent d’estimer la stabilité des effets.
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
#   racine du dépôt
#
# MAIN_SCRIPT :
#   script cible à appeler automatiquement
#
# LOG_DIR / LOG_FILE :
#   journal d’exécution, utile pour :
#   - retracer les runs,
#   - repérer les erreurs,
#   - documenter précisément ce qui a été lancé.
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
# Rôle :
# exécuter UNE combinaison expérimentale complète :
#   (task, model, stimulus_group, stimulus_index, seed)
#
# Ce que fait cette fonction :
# 1. construit la ligne de commande Python ;
# 2. appelle le script principal (main.py ou main_ape.py) ;
# 3. capture stdout/stderr ;
# 4. écrit un log détaillé ;
# 5. renvoie un code de succès/échec.
#
# Pourquoi capturer stdout/stderr :
# cela permet de conserver une trace exploitable si un run échoue,
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
# Rôle :
# parcourir toutes les combinaisons définies plus haut.
#
# Boucle expérimentale :
#   modèle
#   -> tâche
#      -> catégorie H3
#         -> prompt de la catégorie
#            -> seed
#
# Pourquoi cet ordre :
# il est lisible, et correspond bien à la logique analytique :
# on compare d’abord les catégories à l’intérieur d’une tâche,
# puis les tâches à l’intérieur d’un modèle.
#
# Sortie finale :
# un résumé simple :
#   - nombre total de runs
#   - nombre d’échecs
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
# POINT D’ENTRÉE
# ============================================================
#
# Le script est pensé pour être lancé directement :
#
#   python .\scripts\run_h3_batch.py
#
# Avant une campagne complète :
# réduire provisoirement MODELS, TASKS et SEEDS pour un smoke test.
# ============================================================

if __name__ == "__main__":
    main()