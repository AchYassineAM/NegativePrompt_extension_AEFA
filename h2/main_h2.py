# =============================================================================
# h2/main_h2.py — Pipeline H2 (priming contextuel)
# =============================================================================
# SPECIFIQUE A H2 — cree par Ryad pour tester l'effet de priming.
#
# Lancement depuis la RACINE du projet :
#   python -m h2.main_h2 --task cause_and_effect --model flan-t5-large ...
#
# OU directement :
#   cd NegativePrompt_clean
#   python h2/main_h2.py --task cause_and_effect --model flan-t5-large ...
# =============================================================================

import sys
import os

# Ajoute la racine du projet au PYTHONPATH pour trouver config, exec_accuracy, etc.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import fire
import csv
import json
import random
import numpy as np

from data.instruction_induction.load_data import tasks
from exec_accuracy import exec_accuracy_evaluator
from config import PROMPT_SET, Negative_SET, STIMULUS_GROUPS
from h2.h2_primes import PRIME_GROUPS, build_h2_prompt
import template


def get_stimulus_text(pnum=0, stimulus_group=None, stimulus_index=1):
    """
    Recover the target stimulus text without altering the original codebase.
    """
    if stimulus_group is not None:
        assert stimulus_group in STIMULUS_GROUPS, f"Unknown stimulus_group: {stimulus_group}"
        group = STIMULUS_GROUPS[stimulus_group]
        assert 1 <= stimulus_index <= len(group), (
            f"stimulus_index must be between 1 and {len(group)} for group {stimulus_group}"
        )
        return group[stimulus_index - 1].strip()

    if pnum > 0:
        assert 1 <= pnum <= len(Negative_SET), f"pnum must be between 1 and {len(Negative_SET)}"
        return Negative_SET[pnum - 1].strip()

    return ""


def load_data_utf8(type, task):
    """
    Local UTF-8-safe loader for H2 only.
    Keeps the original load_data.py untouched while fixing Windows decoding issues.
    """
    base_dir = os.path.join(
        ROOT,
        "data",
        "instruction_induction",
        "raw",
        "induce" if type == "induce" else "execute"
    )
    path = os.path.join(base_dir, f"{task}.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data["examples"]
    num_examples = len(examples)

    inputs, outputs = [], []
    for i in range(num_examples):
        ex = examples[str(i + 1)]

        if task == "cause_and_effect":
            cause, effect = ex["cause"], ex["effect"]
            if random.random() < 0.5:
                input_ = f"Sentence 1: {cause} Sentence 2: {effect}"
            else:
                input_ = f"Sentence 1: {effect} Sentence 2: {cause}"
            output_ = [cause]
        elif task == "common_concept":
            items = ex["items"]
            input_ = ", ".join(items[:-1])
            output_ = ex["all_common_concepts"]
        elif task == "rhymes":
            input_, output_ = ex["input"], ex["other_rhymes"]
        elif "translation" in task:
            input_, output_ = ex["input"], ex["possible_translations"]
        else:
            input_, output_ = ex["input"], [ex["output"]]

        inputs.append(input_)
        outputs.append(output_)

    return inputs, outputs


def run(
    task,
    model,
    pnum=0,
    few_shot=False,
    stimulus_group=None,
    stimulus_index=1,
    prime_group="none",
    prime_index=1,
    sequence="prime_then_task_then_stimulus",
    seed=42,
    temperature=0.0,
    do_sample=False
):
    try:
        import torch
    except ImportError:
        torch = None

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    assert task in tasks, "Task not found!"
    assert prime_group in PRIME_GROUPS, f"Unknown prime_group: {prime_group}"
    assert 1 <= prime_index <= len(PRIME_GROUPS[prime_group]), (
        f"prime_index must be between 1 and {len(PRIME_GROUPS[prime_group])} for group {prime_group}"
    )

    test_data = load_data_utf8("eval", task)
    origin_prompt = PROMPT_SET[task]

    induce_data = load_data_utf8("induce", task)
    num_demos = 5

    demos_template_str = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template_str = "Instruction: [PROMPT]\n\n[full_DEMO]\nInput: [INPUT]\nAnswer: [OUTPUT]"

    demos_template = template.DemosTemplate(demos_template_str)
    eval_template = template.EvalTemplate(eval_template_str)

    stimulus_text = get_stimulus_text(
        pnum=pnum,
        stimulus_group=stimulus_group,
        stimulus_index=stimulus_index
    )

    prime_text = PRIME_GROUPS[prime_group][prime_index - 1]
    new_prompt = build_h2_prompt(
        task_prompt=origin_prompt,
        stimulus_text=stimulus_text,
        prime_text=prime_text,
        sequence=sequence
    )

    print("LLM:", model)
    print("Task:", task)
    print("Few_shot:", few_shot)
    print("Prime_group:", prime_group)
    print("Prime_index:", prime_index)
    print("Sequence:", sequence)
    print("Stimulus_group:", stimulus_group)
    print("Stimulus_index:", stimulus_index)
    print("Legacy_pnum:", pnum)
    print("Prompt:", new_prompt)

    test_num = min(100, len(test_data[0]))

    test_res = exec_accuracy_evaluator(
        prompts=[new_prompt],
        eval_template=eval_template,
        eval_data=test_data,
        llm_model=model,
        pnum=pnum if stimulus_group is None else stimulus_index,
        task=task,
        num_samples=test_num,
        few_shot=few_shot,
        demos_template=demos_template,
        few_shot_data=induce_data,
        num_demos=num_demos,
        temperature=temperature,
        do_sample=do_sample,
        seed=seed
    )

    test_score = test_res.sorted()[1][0]
    print(f"Test score: {test_score}")

    # Resultats ecrits dans h2/results/ (relatif a la racine du projet)
    result_group = stimulus_group if stimulus_group is not None else "negative_original"
    dir_path = os.path.join(ROOT, "h2", "results", result_group, prime_group, model)
    os.makedirs(dir_path, exist_ok=True)

    txt_path = os.path.join(dir_path, f"{task}.txt")
    with open(txt_path, "a+", encoding="utf-8") as f:
        f.write(f"Test score: {test_score}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Few_shot: {few_shot}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Do_sample: {do_sample}\n")
        f.write(f"Prime_group: {prime_group}\n")
        f.write(f"Prime_index: {prime_index}\n")
        f.write(f"Sequence: {sequence}\n")
        f.write(f"Stimulus_group: {stimulus_group}\n")
        f.write(f"Stimulus_index: {stimulus_index}\n")
        f.write(f"Legacy_pnum: {pnum}\n")
        f.write(f"Stimulus_text: {stimulus_text}\n")
        f.write(f"Prime_text: {prime_text}\n")
        f.write(f"Prompt: {new_prompt}\n")
        f.write("-" * 80 + "\n")

    csv_path = os.path.join(dir_path, "results.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a+", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow([
                "task", "model", "stimulus_group", "stimulus_index",
                "legacy_pnum", "prime_group", "prime_index", "sequence",
                "few_shot", "seed", "temperature", "do_sample", "score"
            ])

        writer.writerow([
            task, model, result_group,
            stimulus_index if stimulus_group is not None else "",
            pnum, prime_group, prime_index, sequence,
            few_shot, seed, temperature, do_sample, test_score
        ])


if __name__ == "__main__":
    fire.Fire(run)
