# H1 — Effet de valence / controle de structure syntaxique

**Responsables** : Brahim El-Farh, Ammar Azerradj

## Description

H1 teste si les prompts negatifs du papier original (Wang et al., 2024)
produisent un effet qui depend de leur valence emotionnelle ou de leur
structure syntaxique. Pour cela, H1 compare les stimuli negatifs originaux
a des stimuli neutres construits avec une structure syntaxique equivalente.

## Structure attendue

```
h1/
|-- README_H1.md
|-- scripts/
|   |-- run_h1_windows_gpu.py
|   |-- run_h1_final.py
|   |-- run_h1_grid.py
|   |-- analyze_h1_ab.py
|   +-- summarize_h1_results.py
+-- results/
    |-- h1_master.csv
    |-- h1a_raw.csv
    |-- h1a_summary.csv
    |-- h1b_raw.csv
    |-- h1b_summary.csv
    |-- h1_group_means.csv
    +-- h1_negative_vs_neutral_summary.csv
```

## Entry point

```bash
python main.py --task <task> --model flan-t5-large --stimulus_group neutral --stimulus_index 1 --seed 42 --temperature 0.0 --do_sample False
```

## Stimulus groups utilises (definis dans config.py)

- `negative_original` : stimuli du papier (Negative_SET)
- `neutral` : stimuli neutres a structure equivalente (Neutral_SET)
- `positive` : stimuli positifs (Positive_SET)
- `ambiguous` : stimuli ambigus (Ambiguous_SET)
- `negative_control` : stimuli negatifs simplifies (Negative_Control_SET)
