import csv
from pathlib import Path

ANALYSIS_DIR = Path("analysis")

FAMILY_SUMMARY_PATH = ANALYSIS_DIR / "palier2a_family_summary.csv"

NEG_OUT = ANALYSIS_DIR / "palier2a_family_negative_original_delta_matrix.csv"
H3_OUT = ANALYSIS_DIR / "palier2a_family_h3_delta_matrix.csv"
CONTRAST_OUT = ANALYSIS_DIR / "palier2a_family_h3_minus_negative_original_matrix.csv"

H3_GROUPS = [
    "competence_threat",
    "inconsistency",
    "social_comparison",
    "urgency",
    "regret",
]

FAMILY_ORDER = [
    "causal_reasoning",
    "comparative_reasoning",
    "lexical_generation",
    "lexical_semantics",
    "morphology",
    "numerical_reasoning",
    "numerical_verbalization",
    "rewriting_transformation",
    "semantic_abstraction",
    "semantic_filtering",
    "semantic_similarity",
    "sentiment_classification",
    "string_extraction",
    "string_filtering",
    "string_transformation",
    "translation",
]


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = read_csv(FAMILY_SUMMARY_PATH)

    # dictionnaires d'accès
    delta_lookup = {}
    contrast_lookup = {}

    for r in rows:
        family = r["task_family"]
        group = r["stimulus_group"]

        delta_lookup[(family, group)] = r["mean_delta_vs_baseline"]
        contrast_lookup[(family, group)] = r["mean_delta_h3_minus_negative_original"]

    # --------------------------------------------------------
    # 1) negative_original vs baseline
    # --------------------------------------------------------
    neg_row = {"stimulus_group": "negative_original"}
    for family in FAMILY_ORDER:
        neg_row[family] = delta_lookup.get((family, "negative_original"), "")

    write_csv(
        NEG_OUT,
        fieldnames=["stimulus_group"] + FAMILY_ORDER,
        rows=[neg_row],
    )

    # --------------------------------------------------------
    # 2) H3 vs baseline
    # --------------------------------------------------------
    h3_rows = []
    for group in H3_GROUPS:
        row = {"stimulus_group": group}
        for family in FAMILY_ORDER:
            row[family] = delta_lookup.get((family, group), "")
        h3_rows.append(row)

    write_csv(
        H3_OUT,
        fieldnames=["stimulus_group"] + FAMILY_ORDER,
        rows=h3_rows,
    )

    # --------------------------------------------------------
    # 3) H3 vs negative_original
    # --------------------------------------------------------
    contrast_rows = []
    for group in H3_GROUPS:
        row = {"stimulus_group": group}
        for family in FAMILY_ORDER:
            row[family] = contrast_lookup.get((family, group), "")
        contrast_rows.append(row)

    write_csv(
        CONTRAST_OUT,
        fieldnames=["stimulus_group"] + FAMILY_ORDER,
        rows=contrast_rows,
    )

    print(f"Wrote: {NEG_OUT}")
    print(f"Wrote: {H3_OUT}")
    print(f"Wrote: {CONTRAST_OUT}")


if __name__ == "__main__":
    main()