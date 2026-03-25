import numpy as np
import random
import utility
import re
import string
from llm_response import get_response_from_llm
import json


def get_query(prompt, eval_template, input_, few_shot, demos_template, demo_data):
    if few_shot == True:
        demos = demos_template.fill(demo_data)
        query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output='',
                               full_demo=demos)
    else:
        query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output='')
    # print('DEMOS:', demos)
    return query


def subsample_data(data, subsample_size, rng=None):
    """
    Subsample data. Data is in the form of a tuple of lists.
    Uses a local RNG if provided for reproducibility.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)

    if rng is None:
        rng = random

    indices = rng.sample(range(len(inputs)), subsample_size)
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    return inputs, outputs


def exec_accuracy_evaluator(
    prompts,
    eval_template,
    eval_data,
    llm_model,
    pnum,
    task,
    num_samples,
    few_shot,
    demos_template,
    few_shot_data,
    num_demos,
    temperature=0.0,
    do_sample=False,
    seed=None
):
    queries = []
    answers = []
    my_inputs = []
    rng = random.Random(seed) if seed is not None else random

    print("=== DEBUG BUILD QUERIES ===")
    print("Task =", task)
    print("LLM =", llm_model)
    print("Few-shot =", few_shot)
    print("Num prompts =", len(prompts))
    print("Num samples =", num_samples)

    subsampled_data = subsample_data(eval_data, num_samples, rng=rng)
    inputs, outputs = subsampled_data

    print("DEBUG sampled inputs =", len(inputs))
    print("DEBUG sampled outputs =", len(outputs))

    # Pré-échantillonne aussi les démos une seule fois par item
    demo_data_list = []
    for _ in range(len(inputs)):
        if few_shot:
            demo_data = subsample_data(few_shot_data, num_demos, rng=rng)
        else:
            demo_data = None
        demo_data_list.append(demo_data)

    for prompt in prompts:
        for input_, output_, demo_data in zip(inputs, outputs, demo_data_list):
            query = get_query(prompt, eval_template, input_, few_shot, demos_template, demo_data)

            queries.append(query)
            answers.append(output_)
            my_inputs.append(input_)

    print("=== DEBUG AFTER QUERY BUILD ===")
    print("len(queries) =", len(queries))
    print("len(answers) =", len(answers))
    print("len(my_inputs) =", len(my_inputs))

    if len(queries) == 0:
        raise RuntimeError("Aucune requête n'a été générée. Vérifie eval_data / num_samples / task.")

    print("\n=== DEBUG FIRST QUERY ===")
    print(queries[0])

    # get response from LLM
    print("\n=== DEBUG CALL MODEL ===")
    model_outputs = get_response_from_llm(
    llm_model=llm_model,
    queries=queries,
    task=task,
    few_shot=few_shot,
    temperature=temperature,
    do_sample=do_sample
    )

    print("\n=== DEBUG MODEL OUTPUT RAW ===")
    print("type(model_outputs) =", type(model_outputs))

    if model_outputs is None:
        raise RuntimeError("get_response_from_llm a renvoyé None.")

    if isinstance(model_outputs, str):
        print("⚠️ model_outputs est une string, pas une liste.")
        print("Contenu :", model_outputs[:500])
        raise RuntimeError(
            "Le modèle a renvoyé une string unique au lieu d'une liste de réponses. "
            "Vérifie llm_response.py."
        )

    try:
        print("len(model_outputs) =", len(model_outputs))
    except Exception as e:
        raise RuntimeError(
            f"Impossible de calculer len(model_outputs). "
            f"Type reçu: {type(model_outputs)} ; erreur: {repr(e)}"
        )

    if len(model_outputs) == 0:
        raise RuntimeError(
            "Le modèle a renvoyé une liste vide. "
            "Le problème est probablement dans llm_response.py "
            "(requêtes non traitées, exception silencieuse, ou sortie vide)."
        )

    if len(model_outputs) != len(queries):
        print("⚠️ Nombre de sorties différent du nombre de requêtes")
        print("len(queries) =", len(queries))
        print("len(model_outputs) =", len(model_outputs))
        print("Exemple de sortie brute :", model_outputs[0] if len(model_outputs) > 0 else "VIDE")
        raise RuntimeError(
            "Mismatch entre le nombre de requêtes et le nombre de sorties du modèle. "
            "Vérifie llm_response.py."
        )

    print("\n=== DEBUG FIRST MODEL OUTPUT ===")
    print(model_outputs[0])

    metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)
    print(f'\nUsing metric "{metric}" for task "{task}"...')

    if metric == 'es':
        score_fn = utility.get_multi_answer_exact_set
    elif metric == 'em':
        score_fn = utility.get_multi_answer_em
    elif metric == 'f1':
        score_fn = utility.get_multi_answer_f1
    elif metric == 'contains':
        score_fn = utility.get_multi_answer_contains
    else:
        raise ValueError(f"Métrique inconnue : {metric}")

    # postprocess the answers for some tasks
    if task == 'cause_and_effect':
        new_ans_ = []
        for my_input, ans_ in zip(my_inputs, answers):
            sentences = my_input.split('.')
            for i in range(len(sentences)):
                if ans_[0].lower() in sentences[i].lower() + '.':
                    new_a = f'Sentence {i+1}: ' + ans_[0]
                    new_ans_.append([new_a])
                    break
        answers = new_ans_

    elif task == 'larger_animal':
        new_ans_ = []
        for my_input, ans_ in zip(my_inputs, answers):
            animals = my_input.split(',')
            for i in range(len(animals)):
                if ans_[0].lower() in animals[i].lower():
                    new_a = f'Animal {i}: ' + ans_[0]
                    new_ans_.append([new_a])
                    break
        answers = new_ans_

    # IMPORTANT :
    # on stocke les prédictions post-traitées dans une nouvelle liste,
    # sinon la boucle suivante rescoring utilisera les sorties brutes.
    processed_outputs = []

    for my_input, prediction, ans_ in zip(my_inputs, model_outputs, answers):
        original_prediction = prediction

        for a in ans_:
            if task == 'cause_and_effect':
                ans_parts = a.split(':')
                for p in ans_parts:
                    p = p.strip().lower()
                    p = p.replace('.', '')
                    if p in prediction.lower():
                        prediction = a
                        break

            elif task == 'rhymes':
                for p in prediction.split():
                    p = p.replace('-', ' ')
                    p = p.translate(str.maketrans('', '', string.punctuation))
                    p = p.strip().lower()
                    if p == a.lower():
                        prediction = a
                        break

            elif task == 'orthography_starts_with':
                prediction = prediction.lower()
                prediction = prediction.replace('confidence score:', '')
                prediction = prediction.replace(',', ' ')
                prediction = prediction.replace('.', ' ')
                prediction = prediction.replace('-', ' ')
                prediction = prediction.translate(str.maketrans('', '', string.punctuation))
                prediction = re.sub(r'\d+', '', prediction)
                preds = prediction.split()
                preds_set = set([pred.strip() for pred in preds])

                a_items = a.split()
                a_set = set([x.strip() for x in a_items])

                if a_set == preds_set:
                    prediction = a

            elif task == 'taxonomy_animal':
                prediction = prediction.lower()
                prediction = prediction.replace('confidence score:', '')
                prediction = prediction.replace(',', ' ')
                prediction = prediction.replace('.', ' ')
                prediction = prediction.replace('-', ' ')
                prediction = prediction.translate(str.maketrans('', '', string.punctuation))
                prediction = re.sub(r'\d+', '', prediction)
                preds = prediction.split()
                preds_set = set([pred.strip() for pred in preds])

                a_items = a.split(',')
                a_set = set([x.strip() for x in a_items])

                if a_set == preds_set:
                    prediction = a

            elif task == 'letters_list':
                prediction = prediction.lower()
                prediction = prediction.replace('confidence score:', '')
                prediction = prediction.replace(',', ' ')
                prediction = prediction.replace('.', ' ')
                prediction = prediction.replace('-', ' ')
                prediction = prediction.translate(str.maketrans('', '', string.punctuation))
                prediction = re.sub(r'\d+', '', prediction)
                preds = prediction.split()
                preds = [pred.strip() for pred in preds]
                a_items = [x.strip() for x in a.split()]
                if preds == a_items:
                    prediction = a_items

            elif task == 'sentiment':
                prediction = prediction.replace('-', ' ')
                prediction = prediction.translate(str.maketrans('', '', string.punctuation))
                prediction = prediction.strip().lower()
                if 'does not mention any negative' in prediction or 'a positive review than a negative one' in prediction:
                    prediction = 'positive'
                    break
                elif 'does not mention any positive' in prediction or 'a negative review than a positive one' in prediction:
                    prediction = 'negative'
                    break
                if 'positive' in prediction and 'negative' in prediction:
                    prediction = ''
                    break
                elif 'positive' in prediction or 'positiv' in prediction:
                    prediction = 'positive'
                    break
                elif 'negative' in prediction or 'negativ' in prediction:
                    prediction = 'negative'
                    break
                if len(prediction.split()) == 1:
                    prediction = postprocess_prediction_4sentiment(prediction)
                elif len(prediction.split()) > 1:
                    items = prediction.split()
                    new_res = postprocess_prediction_4sentiment(items[0].strip())
                    if new_res == 'positive' or new_res == 'negative':
                        prediction = new_res
                if a in prediction:
                    prediction = a
                    break

            elif task == 'sentence_similarity':
                a_score = a.split()[0]
                prediction = prediction.replace('-', ' ')
                prediction = prediction.translate(str.maketrans('', '', string.punctuation))
                prediction = prediction.strip().lower()
                prediction_list = prediction.split()
                for item in prediction_list:
                    if item.isdigit():
                        p = item
                        p_score = p[0]
                        if p_score == a_score:
                            prediction = a_score
                        break

            elif task == 'word_in_context':
                prediction = prediction.strip().lower()
                if len(prediction.split()) > 0:
                    p = prediction.split()[0]
                    p = p.replace('-', ' ')
                    p = p.translate(str.maketrans('', '', string.punctuation))
                    p = p.strip()
                    if p in ['true', 'yes', '1', '10', 'same', 'match', 'similar'] or 'same' in p:
                        prediction = 'same'
                    elif p in ['false', 'no', '0', '00', 'different', 'not', 'opposite'] or 'different' in p:
                        prediction = 'not the same'
                    elif 'different' in prediction and 'not' not in prediction:
                        prediction = 'not the same'
                    elif 'different' in prediction and 'not' in prediction:
                        prediction = 'same'
                    elif 'same' in prediction and 'not' not in prediction:
                        prediction = 'same'
                    elif 'same' in prediction and 'not' in prediction:
                        prediction = 'not the same'

            elif task == 'larger_animal':
                prediction = prediction.lower()
                if 'larger' in prediction and 'than' in prediction:
                    index = prediction.find('larger')
                    prediction = prediction[:index]
                if 'between' in prediction and 'and' in prediction and 'is' in prediction:
                    index = prediction.find('is')
                    prediction = prediction[index:]
                if 'confidence' in prediction:
                    index = prediction.find('confidence')
                    prediction = prediction[:index].strip()
                if llm_model.lower() in ['t5', 'bloom']:
                    pred_list = prediction.split()
                    if len(pred_list) > 0:
                        ans_part = pred_list[0]
                        if ',' in ans_part and llm_model.lower() != 'chatgpt':
                            prediction = ''
                if llm_model.lower() == 'bard' and ',' in prediction:
                    pred_list = prediction.split(',')
                    prediction = pred_list[-1]
                a = a.strip().lower()
                a_items = a.split()
                if a in prediction.lower():
                    prediction = a
                    break
                elif len(a_items) > 1:
                    a_2 = a_items[-1].strip()
                    if a_2 in prediction.lower():
                        prediction = a
                        break
                if prediction == '0' and '0' in a:
                    prediction = a
                elif prediction == '1' and '1' in a:
                    prediction = a
                elif '1' in a and ('1.0' in prediction or '1' in prediction or '2' in prediction):
                    prediction = a
                elif '0' in a and ('0.0' in prediction or '0' in prediction):
                    prediction = a

            else:
                a = a.strip().lower()
                if a in prediction.lower():
                    prediction = a
                    break

        processed_outputs.append(prediction)

        print('Model Input: ', my_input)
        print('Raw Model Output: ', original_prediction)
        print('Processed Output: ', prediction)
        print('Ans: ', ans_)
        print('---')

    scores = []
    for prediction, ans_ in zip(processed_outputs, answers):
        try:
            score = score_fn(prediction, ans_, task, llm_model.lower())
            scores.append(score)
        except Exception as e:
            print("DEBUG scoring exception =", repr(e))
            print("prediction =", prediction)
            print("ans_ =", ans_)
            raise

    print("\n=== DEBUG FINAL ===")
    print("len(scores) =", len(scores))
    print("len(prompts) =", len(prompts))
    print("num_samples =", num_samples)
    print("scores =", scores)

    expected_size = len(prompts) * num_samples
    if len(scores) != expected_size:
        raise RuntimeError(
            f"Nombre de scores inattendu : len(scores)={len(scores)} "
            f"alors qu'on attendait {expected_size} "
            f"(len(prompts)={len(prompts)}, num_samples={num_samples}). "
            f"Le problème est probablement dans llm_response.py "
            f"ou dans le format des sorties modèle."
        )

    scores = np.array(scores).reshape(len(prompts), num_samples)

    res = ExecAccuracyEvaluationResult(prompts, scores)
    return res


def postprocess_prediction_4sentiment(prediction):
    if prediction == 'neg':
        prediction = 'negative'
    elif prediction == 'pos':
        prediction = 'positive'
    elif prediction.isdigit() or (prediction[0] == '-' and prediction[1:].isdigit()):
        p_digit = int(prediction)
        if p_digit > 0:
            prediction = 'positive'
        else:
            prediction = 'negative'
    return prediction


class ExecAccuracyEvaluationResult():

    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def _agg_scores(self, method):
        """For each prompt, compute a statistic of the scores (e.g., mean, median)"""
        if method == 'mean':
            return [np.mean(s) for s in self.scores]
        elif method == 'median':
            return [np.median(s) for s in self.scores]
        elif method == 'std':
            return [np.std(s) for s in self.scores]
        elif method == 'max':
            return [np.max(s) for s in self.scores]
        elif method == 'min':
            return [np.min(s) for s in self.scores]
        elif method == 'iqm':
            return [np.mean(np.percentile(lps, [25, 75])) for lps in self.scores]
        else:
            raise ValueError('Invalid method: {}'.format(method))

    def sorted(self):
        scores = [np.mean(s) for s in self.scores]
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted(scores)
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores
