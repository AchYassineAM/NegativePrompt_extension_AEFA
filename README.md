# NegativePrompt — Reproduction and Extension Study

This repository contains the full experimental pipeline and final deliverables of our Master's thesis project. 

It presents a **reproduction and extension study** of the paper:

**NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli**  
Xu Wang, Cheng Li, Yi Chang, Jindong Wang, Yuan Wu  
(IJCAI 2024)

---

## Deliverables

- [Download report](./)
- [Download report](./)
- [Download report](./)

---
## Authors

**Reproduction and extension study conducted by:**

- Yassine Achargui  
- Brahim El-Farh  
- Ryad Fikri  
- Amar Azerradj  

**Academic supervision:**  
Séverine Affeldt

---

## Institutional Context

This work was carried out within the framework of the:

**Master 2 Informatique — Parcours Machine Learning et Science des Données**  
Université Paris Cité  
UFR Sciences Fondamentales et Biomédicales  
Centre Borelli

---

## Abstract

This project investigates the effect of negative prompts on large language model performance through a controlled experimental framework.

While the original paper reports substantial average gains  
(+12.89% on Instruction Induction and +46.25% on BIG-Bench),  
our results reveal a more heterogeneous landscape.

Three hypotheses are evaluated:

- **H1 — Instruction salience vs valence**
- **H2 — Contextual priming effects**
- **H3 — Taxonomy of negative stimuli (extension)**

The results show that the effect of negative prompts depends on both the **type of stimulus** and the **task family**, and is often better captured through **precision-based metrics** rather than exact match.

---

## Methodology

The experimental pipeline was restructured to ensure:

- Controlled prompt structure (NEG / NEU / POS)
- Deterministic decoding (temperature = 0.0)
- Multi-seed evaluation
- Clear separation of conditions:
  - `baseline`
  - `negative_original`
  - `H3`

### H3 Stimulus Taxonomy

- `competence_threat`
- `inconsistency`
- `social_comparison`
- `urgency`
- `regret`

Tasks are grouped into **families** to enable structured analysis.

---

## Key Results

- No systematic advantage of negative prompts under control (H1)
- Limited and task-dependent priming effects (H2)
- Strong interaction **stimulus × task family** (H3)

For numerical tasks, effects appear primarily on:

- `mean_abs_error`
- `median_abs_error`

rather than exact match.

---

## Repository Structure
. ├── analysis/            # Aggregation, 
metrics, visualization ├── data/                # 
Benchmark datasets ├── results/             # 
Raw model outputs ├── scripts/             # 
Execution pipelines ├── configs/             # 
Experimental configuration ├── artifacts/
# Intermediate outputs └── main.py              
# Entry poin

---

## Installation

```sh
git clone <your-repo-url>
cd NegativePrompt
```
conda create --name negativeprompt python=3.9
conda activate negativeprompt
pip install -r requirements.txt

# USAGE
Run Baseline :
python main.py --task task_name --model model_name

Run H3 Experiments :
python scripts/run_h3_batch.py

# CITATION
Original paper

@misc{wang2024negativeprompt,
  title={NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli},
  author={Xu Wang and Cheng Li and Yi Chang and Jindong Wang and Yuan Wu},
  year={2024},
  eprint={2405.02814},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}


# Acknowledgements
We acknowledge the authors of the original work for providing the foundation of this study.
We thank Séverine Affeldt for academic supervision, as well as the Centre Borelli and Université Paris Cité for their support

---

## Résultat

Dans VSCode / GitHub, tu auras :

- titres bien hiérarchisés
- blocs de code propres
- sections lisibles
- rendu “papier + repo sérieux”

---
