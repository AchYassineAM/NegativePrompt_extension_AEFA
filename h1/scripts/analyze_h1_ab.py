import csv
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
MASTER_CSV = ROOT / "h1_master.csv"

H1A_RAW = ROOT / "h1a_raw.csv"
H1A_SUMMARY = ROOT / "h1a_summary.csv"

H1B_RAW = ROOT / "h1b_raw.csv"
H1B_SUMMARY = ROOT / "h1b_summary.csv"

MODEL_FILTER = "flan-t5-large"


def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def mean(vals):
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def std(vals):
    if not vals:
        return 0.0
    if len(vals) == 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def read_master():
    with open(MASTER_CSV, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Déduplication robuste
    dedup = {}
    for r in rows:
        key = (
            r.get("task"),
            r.get("model"),
            r.get("stimulus_group"),
            r.get("stimulus_index"),
            r.get("legacy_pnum"),
            r.get("seed"),
        )
        dedup[key] = r

    rows = list(dedup.values())

    # Filtre modèle
    rows = [r for r in rows if r.get("model") == MODEL_FILTER]

    return rows


def build_maps(rows):
    neg_orig = {}
    neg_ctrl = {}
    neutral = {}
    positive = {}
    ambiguous = {}

    for r in rows:
        task = r.get("task")
        seed = r.get("seed")
        group = r.get("stimulus_group")
        idx = r.get("stimulus_index")
        pnum = r.get("legacy_pnum")
        score = to_float(r.get("score"))

        if score is None:
            continue

        if group == "negative_original" and pnum not in ("", None):
            neg_orig[(task, seed, str(pnum))] = score
        elif group == "negative_control" and idx not in ("", None):
            neg_ctrl[(task, seed, str(idx))] = score
        elif group == "neutral" and idx not in ("", None):
            neutral[(task, seed, str(idx))] = score
        elif group == "positive" and idx not in ("", None):
            positive[(task, seed, str(idx))] = score
        elif group == "ambiguous" and idx not in ("", None):
            ambiguous[(task, seed, str(idx))] = score

    return neg_orig, neg_ctrl, neutral, positive, ambiguous


def analyze_h1a(rows):
    neg_orig, neg_ctrl, _, _, _ = build_maps(rows)

    raw_rows = []
    for (task, seed, pair), ctrl_score in neg_ctrl.items():
        neg_score = neg_orig.get((task, seed, pair))
        if neg_score is None:
            continue

        raw_rows.append({
            "task": task,
            "pair": pair,
            "seed": seed,
            "negative_control_score": ctrl_score,
            "negative_original_score": neg_score,
            "delta_negative_original_minus_control": neg_score - ctrl_score
        })

    raw_rows.sort(key=lambda r: (r["task"], to_int(r["pair"], 0), to_int(r["seed"], 0)))

    with open(H1A_RAW, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "pair", "seed",
            "negative_control_score",
            "negative_original_score",
            "delta_negative_original_minus_control"
        ])
        writer.writeheader()
        writer.writerows(raw_rows)

    grouped = defaultdict(list)
    for r in raw_rows:
        grouped[(r["task"], r["pair"])].append(r["delta_negative_original_minus_control"])

    summary_rows = []
    for (task, pair), vals in grouped.items():
        m = mean(vals)
        s = std(vals)

        summary_rows.append({
            "task": task,
            "pair": pair,
            "n_seeds": len(vals),
            "mean_delta_negative_original_minus_control": round(m, 6),
            "std_delta_negative_original_minus_control": round(s, 6),
            "direction": (
                "negative_original_better" if m > 0
                else "control_better" if m < 0
                else "equal"
            )
        })

    summary_rows.sort(key=lambda r: (r["task"], to_int(r["pair"], 0)))

    with open(H1A_SUMMARY, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "pair", "n_seeds",
            "mean_delta_negative_original_minus_control",
            "std_delta_negative_original_minus_control",
            "direction"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    return raw_rows, summary_rows


def analyze_h1b(rows):
    _, neg_ctrl, neutral, positive, ambiguous = build_maps(rows)

    raw_rows = []
    for (task, seed, pair), neg_score in neg_ctrl.items():
        neu = neutral.get((task, seed, pair))
        pos = positive.get((task, seed, pair))
        amb = ambiguous.get((task, seed, pair))

        if neu is None or pos is None or amb is None:
            continue

        raw_rows.append({
            "task": task,
            "pair": pair,
            "seed": seed,
            "negative_control_score": neg_score,
            "neutral_score": neu,
            "positive_score": pos,
            "ambiguous_score": amb,
            "delta_neg_minus_neutral": neg_score - neu,
            "delta_neg_minus_positive": neg_score - pos,
            "delta_neg_minus_ambiguous": neg_score - amb,
        })

    raw_rows.sort(key=lambda r: (r["task"], to_int(r["pair"], 0), to_int(r["seed"], 0)))

    with open(H1B_RAW, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "pair", "seed",
            "negative_control_score",
            "neutral_score",
            "positive_score",
            "ambiguous_score",
            "delta_neg_minus_neutral",
            "delta_neg_minus_positive",
            "delta_neg_minus_ambiguous",
        ])
        writer.writeheader()
        writer.writerows(raw_rows)

    grouped = defaultdict(list)
    for r in raw_rows:
        grouped[(r["task"], r["pair"])].append(r)

    summary_rows = []
    for (task, pair), vals in grouped.items():
        d_nn = [v["delta_neg_minus_neutral"] for v in vals]
        d_np = [v["delta_neg_minus_positive"] for v in vals]
        d_na = [v["delta_neg_minus_ambiguous"] for v in vals]

        m_nn = mean(d_nn)
        s_nn = std(d_nn)
        m_np = mean(d_np)
        s_np = std(d_np)
        m_na = mean(d_na)
        s_na = std(d_na)

        summary_rows.append({
            "task": task,
            "pair": pair,
            "n_seeds": len(vals),
            "mean_neg_minus_neutral": round(m_nn, 6),
            "std_neg_minus_neutral": round(s_nn, 6),
            "mean_neg_minus_positive": round(m_np, 6),
            "std_neg_minus_positive": round(s_np, 6),
            "mean_neg_minus_ambiguous": round(m_na, 6),
            "std_neg_minus_ambiguous": round(s_na, 6),
        })

    summary_rows.sort(key=lambda r: (r["task"], to_int(r["pair"], 0)))

    with open(H1B_SUMMARY, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "pair", "n_seeds",
            "mean_neg_minus_neutral", "std_neg_minus_neutral",
            "mean_neg_minus_positive", "std_neg_minus_positive",
            "mean_neg_minus_ambiguous", "std_neg_minus_ambiguous",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    return raw_rows, summary_rows


def quick_text_summary(h1a_summary, h1b_summary):
    h1a_pos = sum(1 for r in h1a_summary if r["mean_delta_negative_original_minus_control"] > 0)
    h1a_neg = sum(1 for r in h1a_summary if r["mean_delta_negative_original_minus_control"] < 0)

    h1b_vs_neu = sum(1 for r in h1b_summary if r["mean_neg_minus_neutral"] > 0)
    h1b_vs_pos = sum(1 for r in h1b_summary if r["mean_neg_minus_positive"] > 0)
    h1b_vs_amb = sum(1 for r in h1b_summary if r["mean_neg_minus_ambiguous"] > 0)

    print("\n===== QUICK SUMMARY =====")
    print(f"H1A rows: {len(h1a_summary)}")
    print(f"H1A: negative_original > negative_control in {h1a_pos} cases")
    print(f"H1A: negative_original < negative_control in {h1a_neg} cases")
    print()
    print(f"H1B rows: {len(h1b_summary)}")
    print(f"H1B: negative_control > neutral in {h1b_vs_neu} cases")
    print(f"H1B: negative_control > positive in {h1b_vs_pos} cases")
    print(f"H1B: negative_control > ambiguous in {h1b_vs_amb} cases")
    print("=========================\n")


def main():
    rows = read_master()
    h1a_raw, h1a_summary = analyze_h1a(rows)
    h1b_raw, h1b_summary = analyze_h1b(rows)

    print("Wrote:", H1A_RAW)
    print("Wrote:", H1A_SUMMARY)
    print("Wrote:", H1B_RAW)
    print("Wrote:", H1B_SUMMARY)
    print(f"H1A raw rows: {len(h1a_raw)}")
    print(f"H1A summary rows: {len(h1a_summary)}")
    print(f"H1B raw rows: {len(h1b_raw)}")
    print(f"H1B summary rows: {len(h1b_summary)}")

    quick_text_summary(h1a_summary, h1b_summary)


if __name__ == "__main__":
    main()