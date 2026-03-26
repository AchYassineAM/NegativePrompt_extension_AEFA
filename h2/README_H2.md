# H2 — Effet de priming contextuel

**Responsable** : Ryad Fikri

## Description

H2 teste si un prime contextuel (neutre ou positif) injecte avant le stimulus
negatif modifie la performance du modele. L'effet recherche n'est pas
necessairement un gain : il peut s'agir d'une attenuation, d'une stabilisation,
ou d'une perturbation de l'effet du stimulus negatif.

## Conditions experimentales

### Primes (definis dans h2_primes.py)

| Groupe | Texte (prime_index=1) |
|--------|----------------------|
| none | *(vide)* |
| neutral | "Take a calm and methodical approach to this task. Focus on accuracy and follow the instruction carefully." |
| positive | "You have handled similar tasks successfully before. Stay confident, focused, and verify your final answer carefully." |

### Ordres d'injection

- `prime_then_task_then_stimulus` : prime -> consigne -> stimulus
- `task_then_prime_then_stimulus` : consigne -> prime -> stimulus

### Stimuli utilises (mapping verifie dans stimuli_mapping.csv)

- NP04 (pnum=4) : "Everyone else managed to do it; why can't you?"
- NP05 (pnum=5) : "I've seen others do this much more efficiently and effectively."
- NP08 (pnum=8) : "When I hear how seamlessly others finished their tasks, jealousy stirs within me."

### Modeles

- Flan-T5-Large (encoder-decoder) : seeds 42, 43, 44
- Vicuna-7b-v1.5 (decoder-only) : seed 42 uniquement
- Llama-2-7b-chat (decoder-only) : seed 42 uniquement

### Parametres d'inference

- temperature = 0.0, do_sample = False (greedy decoding)

## Entry point

```bash
python h2/main_h2.py --task cause_and_effect --model flan-t5-large \
  --pnum 4 --prime_group neutral --prime_index 1 \
  --seed 42 --temperature 0.0 --do_sample False
```

## Structure des resultats

```
h2/results/
+-- negative_original/
    |-- none/
    |   |-- flan-t5-large/results.csv
    |   |-- vicuna/results.csv
    |   +-- llama2/results.csv
    |-- neutral/
    |   |-- flan-t5-large/results.csv
    |   |-- vicuna/results.csv
    |   +-- llama2/results.csv
    +-- positive/
        |-- flan-t5-large/results.csv
        |-- vicuna/results.csv
        +-- llama2/results.csv
```

## Resultats principaux (cause_and_effect, seed 42)

| Modele | Stimulus | None | Neutral | Positive |
|--------|----------|------|---------|----------|
| Flan-T5 | NP04 | 0.52 | 0.60 | 0.56 |
| Flan-T5 | NP05 | 0.64 | 0.64 | 0.60 |
| Flan-T5 | NP08 | 0.48 | 0.52 | 0.52 |
| Vicuna | NP04 | 0.56 | 0.64 | 0.48 |
| Vicuna | NP05 | 0.32 | 0.80 | 0.48 |
| Vicuna | NP08 | 0.64 | 0.80 | 0.52 |
| Llama-2 | NP04 | 0.32 | 0.56 | 0.28 |
| Llama-2 | NP05 | 0.28 | 0.32 | 0.32 |
| Llama-2 | NP08 | 0.56 | 0.28 | 0.08 |

## Barplot

Generer avec : `python h2/plot_h2.py`
Fichier produit : `h2/h2_results_barplot.png`
