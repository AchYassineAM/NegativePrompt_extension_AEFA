# scripts/map_np_to_pnum.py
import sys
from pathlib import Path
from difflib import SequenceMatcher
import csv

# Ajoute la racine du repo (dossier où se trouve config.py) au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Negative_SET
from stimuli_paper import PAPER_NP

def norm(x: str) -> str:
    return " ".join(x.lower().strip().split())

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()

rows = []

for np_id, paper_text in PAPER_NP.items():
    scored = []
    for i, code_text in enumerate(Negative_SET, start=1):
        scored.append((i, sim(paper_text, code_text), code_text))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_pnum, best_score, best_code_text = scored[0]

    if best_score < 0.85:
        raise ValueError(
            f"Match faible pour {np_id} (score={best_score:.3f}).\n"
            f"paper: {paper_text}\n"
            f"code:  {best_code_text}\n"
            f"-> Vérifie la copie du stimulus papier (typo ?) ou baisse le seuil."
        )

    rows.append({
        "np_id": np_id,
        "pnum": best_pnum,
        "score": f"{best_score:.3f}",
        "paper_text": paper_text,
        "code_text": best_code_text,
    })

output_path = ROOT / "stimuli_mapping.csv"

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["np_id", "pnum", "score", "paper_text", "code_text"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Mapping écrit dans : {output_path}")