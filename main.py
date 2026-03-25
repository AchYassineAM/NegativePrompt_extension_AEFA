import fire
from data.instruction_induction.load_data import load_data, tasks
from exec_accuracy import exec_accuracy_evaluator
from config import PROMPT_SET, APE_PROMPT_SET, APE_PROMPTs, Negative_SET, STIMULUS_GROUPS
import template
import os
import random
import numpy as np
import csv

'''
def getPrompt(ori_prompt, num_str):
    new_prompt = ori_prompt
    if num_str > 0:
        new_prompt = ori_prompt + Negative_SET[num_str - 1]
    return new_prompt
'''

def getPrompt(ori_prompt, pnum=0, stimulus_group=None, stimulus_index=1):
    """
    Backward-compatible prompt builder.

    Modes:
    - legacy mode: use pnum with Negative_SET
    - new mode: use stimulus_group + stimulus_index with STIMULUS_GROUPS
    """
    new_prompt = ori_prompt

    # Nouveau mode avec groupe explicite
    if stimulus_group is not None:
        assert stimulus_group in STIMULUS_GROUPS, f"Unknown stimulus_group: {stimulus_group}"
        group = STIMULUS_GROUPS[stimulus_group]
        assert 1 <= stimulus_index <= len(group), (
            f"stimulus_index must be between 1 and {len(group)} for group {stimulus_group}"
        )
        return ori_prompt.strip() + " " + group[stimulus_index - 1].strip()

    # Ancien mode : pnum sur Negative_SET
    if pnum > 0:
        assert 1 <= pnum <= len(Negative_SET), f"pnum must be between 1 and {len(Negative_SET)}"
        return ori_prompt.strip() + " " + Negative_SET[pnum - 1].strip()

    return new_prompt


def run(
    task,
    model,
    pnum=0,
    few_shot=False,
    stimulus_group=None,
    stimulus_index=1,
    seed=42,
    temperature=0.0, #modèle déterministe max, greedy
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
    
    assert task in tasks, 'Task not found!'

    test_data = load_data('eval', task)
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\nAnswer: [OUTPUT]"
    origin_prompt = PROMPT_SET[task]
    # origin_prompt = APE_PROMPTs[task]


    # few-shot setting
    num_demos = 5
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "Instruction: [PROMPT]\n\n[full_DEMO]\nInput: [INPUT]\nAnswer: [OUTPUT]"
    demos_template = template.DemosTemplate(demos_template)

    if few_shot:
        induce_data = load_data('induce', task)
        few_shot_data = (
            induce_data[0],
            [random.sample(output, 1)[0] for output in induce_data[1]]
        )
    else:
        induce_data = None
        few_shot_data = None



    # Evaluate on test data
    print('LLM: ', model)
    print('Evaluating on test data...')

    new_prompt = getPrompt(
    origin_prompt,
    pnum=pnum,
    stimulus_group=stimulus_group,
    stimulus_index=stimulus_index
    )
    print('Prompt: ', new_prompt)
    print('Few_shot: ', few_shot)

    test_num = min(100, len(test_data[0]))

    # p_list = APE_PROMPT_SET[task]
    eval_template = template.EvalTemplate(eval_template)
    # for p in p_list:
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

    print(f'Test score: {test_score}')

    if stimulus_group is not None:
        result_group = stimulus_group
    elif pnum > 0:
        result_group = 'negative_original'
    else:
        result_group = 'baseline'
    dir_path = f'results/{result_group}/{model}'
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

    with open(f'results/{result_group}/{model}/{task}.txt', 'a+', encoding='utf-8') as f:
        f.write(f'Test score: {test_score}\n')
        f.write(f'Task: {task}\n')
        f.write(f'Model: {model}\n')
        f.write(f'Few_shot: {few_shot}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Temperature: {temperature}\n')
        f.write(f'Do_sample: {do_sample}\n')
        f.write(f'Stimulus_group: {stimulus_group}\n')
        f.write(f'Stimulus_index: {stimulus_index}\n')
        f.write(f'Legacy_pnum: {pnum}\n')
        f.write(f'Prompt: {new_prompt}\n')
        f.write('-' * 80 + '\n')

    
    csv_path = f'results/{result_group}/{model}/results.csv'
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a+', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow([
                'task',
                'model',
                'stimulus_group',
                'stimulus_index',
                'legacy_pnum',
                'few_shot',
                'seed',
                'temperature',
                'do_sample',
                'score'
            ])

        writer.writerow([
            task,
            model,
            result_group,
            stimulus_index if stimulus_group is not None else '',
            pnum,
            few_shot,
            seed,
            temperature,
            do_sample,
            test_score
        ])


if __name__ == '__main__':
    fire.Fire(run)
