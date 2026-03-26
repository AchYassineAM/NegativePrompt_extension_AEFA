# CHANGES.md — Audit des modifications par fichier

Ce document liste, pour chaque fichier du depot, son origine (baseline Wang et al. 2024
ou creation projet AEFA) et la nature exacte des modifications.

---

## Fichiers a la racine

### `config.py` — MODIFIE

| Lignes | Contenu | Origine |
|--------|---------|---------|
| 1-26 | `PROMPT_SET` | Baseline (inchange) |
| 29-40 | `Negative_SET` | Baseline (inchange, 10 stimuli NP01-NP10) |
| 43-77 | `APE_PROMPT_SET`, `APE_PROMPTs` | Baseline (inchange) |
| 85-112 | `Positive_SET`, `Neutral_SET`, `Ambiguous_SET`, `Negative_Control_SET` | **AJOUT H1** (Brahim, Ammar) |
| 115-138 | `CompetenceThreat_SET`, `Inconsistency_SET`, `SocialComparison_SET`, `Urgency_SET`, `Regret_SET` | **AJOUT H3** (Yassine) |
| 141-191 | `STIMULUS_REGISTRY`, `STIMULUS_GROUPS` | **AJOUT commun** — registre unifie pour adresser tout stimulus par groupe + index |

**Resume** : les 77 premieres lignes sont strictement identiques au papier original.
Tout ce qui suit est une extension pour le projet AEFA.

---

### `llm_response.py` — REECRIT

Le fichier original du papier utilisait des appels API (OpenAI / ChatGPT).
Cette version est une reecriture complete pour l'inference locale avec HuggingFace.

| Section | Contenu | Type de modification |
|---------|---------|----------------------|
| L1-19 | Header de documentation | **AJOUT** |
| L20-77 | Branche Flan-T5-Large | **REECRIT** — chargement local, device detection, greedy decoding |
| L78-135 | Branche Vicuna-7b-v1.5 | **REECRIT** — decoder-only, post-processing regex "Answer:" |
| L136-200 | Branche Llama-2-7b-chat | **REECRIT** — idem Vicuna + auth token HuggingFace |
| Toutes | Gestion memoire GPU (del model + empty_cache) | **AJOUT** |

**Impact** : ce fichier est utilise par `main.py` (H1) et `main_h2.py` (H2).
C'est la modification partagee la plus importante du projet.

---

### `exec_accuracy.py` — MODIFIE

| Section | Modification |
|---------|--------------|
| `subsample_data()` | Ajout du parametre `rng` pour un sous-echantillonnage reproductible par seed |
| `exec_accuracy_evaluator()` | Ajout du parametre `seed`, creation d'un RNG local |
| Post-processing par tache | **INCHANGE** — logique identique au papier original |
| Gardes defensives (None check, type check, length mismatch) | **AJOUT** |
| Traces DEBUG | **AJOUT** |

---

### `main.py` — MODIFIE

| Section | Modification |
|---------|--------------|
| `getPrompt()` | Ajout du mode `stimulus_group + stimulus_index` (en plus du mode legacy `pnum`) |
| `run()` | Ajout des parametres `stimulus_group`, `stimulus_index`, `seed`, `temperature`, `do_sample` |
| Seed management | **AJOUT** — random, numpy, torch |
| Ecriture CSV | **AJOUT** — log structure des resultats |
| Chemin de sortie | **MODIFIE** — ecrit dans `h1/results/` au lieu de `results/` |
| Logique de base (template, eval) | **INCHANGE** |

---

### `h2/main_h2.py` — CREE (specifique H2, dans le sous-dossier h2/)

Fichier entierement nouveau. Non present dans le code original.
Entry point pour les experiences de priming contextuel (H2).
Se lance depuis la racine : `python h2/main_h2.py ...`
Importe `h2.h2_primes` pour les definitions de primes.
Utilise `sys.path` pour remonter a la racine et trouver les modules partages.
Contient son propre loader UTF-8 (`load_data_utf8`) pour eviter de modifier `load_data.py`.
Ecrit ses resultats dans `h2/results/`.

---

### `main_ape.py` — MODIFIE (mineur)

Memes ajouts que `main.py` : `stimulus_group`, seed management, CSV logging.
Utilise `APE_PROMPTs` au lieu de `PROMPT_SET`.

---

### `template.py` — INCHANGE ou quasi

Classes `DemosTemplate` et `EvalTemplate` — logique de substitution de templates.
La gestion de `demo_data is None` (retourne "") pourrait etre un ajout mineur
pour supporter le mode no-few-shot, mais la structure est identique au papier.

---

### `utility.py` — INCHANGE

Metriques d'evaluation (exact match, F1, contains, exact set).
Aucune modification par rapport au code original.

---

### `data/instruction_induction/load_data.py` — INCHANGE

Loader original du papier. NON modifie.
Le fix UTF-8 pour Windows est dans `main_h2.py` directement (`load_data_utf8`),
ce qui evite de toucher a ce fichier partage.

---

## Dossier `h2/` — TOUT CREE (specifique H2)

| Fichier | Role |
|---------|------|
| `h2_primes.py` | Definition des primes (none, neutral, positive) + `build_h2_prompt()` |
| `plot_h2.py` | Barplot matplotlib des resultats H2 |
| `stimuli_mapping.csv` | Table de correspondance NP01-NP10 <-> pnum (score de similarite = 1.000) |
| `results/` | Tous les CSV et TXT de resultats des experiences H2 |
| `scripts/check_mapping.py` | Verification d'unicite du mapping NP-pnum |
| `scripts/dump_negative_set.py` | Export du Negative_SET en CSV |
| `scripts/map_np_to_pnum.py` | Generation automatique du mapping NP-pnum par similarite textuelle |
| `scripts/run_replicates.py` | Runner multi-seeds automatise |
| `scripts/stimuli_paper.py` | Stimuli NP01-NP10 tels que cites dans le papier (reference) |

---

## Dossier `h1/` — PLACEHOLDER

Structure prevue pour les resultats et scripts H1 (Brahim, Ammar).
A remplir avec : `results/`, `scripts/` (run_h1_*, analyze_h1_*, summarize_h1_*).

---

## Dossier `h3/` — PLACEHOLDER

Structure prevue pour les resultats et scripts H3 (Yassine).
