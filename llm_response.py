import os
import re
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
)


def _safe_text(x):
    """
    Convertit n'importe quel objet en texte affichable mÃªme sous
    console Windows cp1252. Les caractÃ¨res non imprimables sont
    Ã©chappÃ©s plutÃ´t que de faire planter le run.
    """
    text = str(x)
    try:
        text.encode(sys.stdout.encoding or "utf-8", errors="strict")
        return text
    except Exception:
        return text.encode("ascii", errors="backslashreplace").decode("ascii")


def safe_debug_print(*parts):
    """
    Print robuste : ne plante pas si la console n'accepte pas
    certains caractÃ¨res Unicode.
    """
    text = " ".join(_safe_text(p) for p in parts)
    try:
        print(text)
    except Exception:
        # dernier filet de sÃ©curitÃ©
        print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def get_response_from_llm(
    llm_model,
    queries,
    task,
    few_shot,
    api_num=4,
    temperature=0.0,  # greedy, dÃ©terministe max
    do_sample=False,
):
    safe_debug_print("=== DEBUG llm_response ===")
    safe_debug_print("llm_model =", llm_model)
    safe_debug_print("temperature =", temperature)
    safe_debug_print("do_sample =", do_sample)
    safe_debug_print("len(queries) =", len(queries))

    model_outputs = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    safe_debug_print("=== DEBUG llm_response ===")
    safe_debug_print("llm_model =", llm_model)
    safe_debug_print("device =", device)
    safe_debug_print("len(queries) =", len(queries))

    # -------------------------
    # FLAN-T5 / T5
    # -------------------------
    if llm_model.lower() in ["t5", "flan-t5-large", "flan_t5_large"]:
        safe_debug_print("DEBUG branch = FLAN-T5")

        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-large",
            device_map="auto" if device == "cuda" else None,
        )

        if device == "cpu":
            model = model.to(device)

        for idx, q in enumerate(queries):
            try:
                safe_debug_print(f"\n--- QUERY {idx+1}/{len(queries)} ---")
                safe_debug_print(q)

                inputs = tokenizer(q, return_tensors="pt")
                input_ids = inputs.input_ids.to(device)

                gen_kwargs = {
                    "max_new_tokens": 20,
                    "do_sample": do_sample,
                }

                if do_sample:
                    gen_kwargs["temperature"] = temperature

                outputs = model.generate(input_ids, **gen_kwargs)

                out_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                safe_debug_print("DEBUG raw output =", out_text)
                model_outputs.append(out_text)

            except Exception as e:
                safe_debug_print("DEBUG exception in FLAN-T5 branch =", repr(e))
                raise

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        safe_debug_print("DEBUG returning", len(model_outputs), "outputs")
        return model_outputs

    # -------------------------
    # VICUNA
    # -------------------------
    elif llm_model.lower() == "vicuna":
        safe_debug_print("DEBUG branch = VICUNA")

        model_id = "lmsys/vicuna-7b-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            load_in_4bit=True if device == "cuda" else False,
        )

        if device == "cpu":
            model = model.to(device)

        for idx, q in enumerate(queries):
            try:
                safe_debug_print(f"\n--- QUERY {idx+1}/{len(queries)} ---")
                safe_debug_print(q)

                inputs = tokenizer(q, return_tensors="pt")
                input_ids = inputs.input_ids.to(device)

                gen_kwargs = {
                    "max_new_tokens": 50,
                    "do_sample": do_sample,
                }

                if do_sample:
                    gen_kwargs["temperature"] = temperature

                outputs = model.generate(input_ids, **gen_kwargs)

                out_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                if out_text.startswith(q):
                    out_text = out_text[len(q):].strip()

                ans = re.search(r"Answer:\s*(.*)", out_text)
                final_text = ans.group(1).strip() if ans else out_text.split("\n")[0].strip()

                safe_debug_print("DEBUG raw output =", out_text)
                safe_debug_print("DEBUG final output =", final_text)

                model_outputs.append(final_text)

            except Exception as e:
                safe_debug_print("DEBUG exception in VICUNA branch =", repr(e))
                raise

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        safe_debug_print("DEBUG returning", len(model_outputs), "outputs")
        return model_outputs

    # -------------------------
    # LLAMA2
    # -------------------------
    elif llm_model.lower() == "llama2":
        safe_debug_print("DEBUG branch = LLAMA2")

        model_id = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            load_in_4bit=True if device == "cuda" else False,
        )

        if device == "cpu":
            model = model.to(device)

        for idx, q in enumerate(queries):
            try:
                safe_debug_print(f"\n--- QUERY {idx+1}/{len(queries)} ---")
                safe_debug_print(q)

                inputs = tokenizer(q, return_tensors="pt")
                input_ids = inputs.input_ids.to(device)

                gen_kwargs = {
                    "max_new_tokens": 50,
                    "do_sample": do_sample,
                }

                if do_sample:
                    gen_kwargs["temperature"] = temperature

                outputs = model.generate(input_ids, **gen_kwargs)

                out_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                if out_text.startswith(q):
                    out_text = out_text[len(q):].strip()

                ans = re.search(r"Answer:\s*(.*)", out_text)
                final_text = ans.group(1).strip() if ans else out_text.split("\n")[0].strip()

                safe_debug_print("DEBUG raw output =", out_text)
                safe_debug_print("DEBUG final output =", final_text)

                model_outputs.append(final_text)

            except Exception as e:
                safe_debug_print("DEBUG exception in LLAMA2 branch =", repr(e))
                raise

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        safe_debug_print("DEBUG returning", len(model_outputs), "outputs")
        return model_outputs

    else:
        raise ValueError(
            f"ModÃ¨le non reconnu dans llm_response.py : {llm_model}. "
            f"Utilise par exemple 't5', 'flan-t5-large', 'vicuna' ou 'llama2'."
        )


