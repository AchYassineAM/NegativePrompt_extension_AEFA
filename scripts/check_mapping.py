# scripts/check_mapping.py
import csv
from collections import Counter

with open("stimuli_mapping.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

np_ids = [row["np_id"] for row in rows]
pnums = [row["pnum"] for row in rows]

np_counter = Counter(np_ids)
pnum_counter = Counter(pnums)

print("=== Vérification NP ===")
for k, v in np_counter.items():
    if v > 1:
        print(f"Doublon NP : {k} apparaît {v} fois")

print("=== Vérification pnum ===")
for k, v in pnum_counter.items():
    if v > 1:
        print(f"Doublon pnum : {k} est utilisé {v} fois")

if len(np_counter) == len(rows) and len(pnum_counter) == len(rows):
    print("✅ Mapping 1-1 correct : aucun doublon")
else:
    print("⚠️ Problème de correspondance : vérifie le fichier")