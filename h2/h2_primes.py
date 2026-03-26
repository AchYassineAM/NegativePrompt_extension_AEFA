# h2_primes.py

PRIME_GROUPS = {
    "none": [
        ""
    ],
    "neutral": [
        "Take a calm and methodical approach to this task. Focus on accuracy and follow the instruction carefully.",
        "Read the instruction carefully and proceed step by step. Aim for a precise and well-checked answer."
    ],
    "positive": [
        "You have handled similar tasks successfully before. Stay confident, focused, and verify your final answer carefully.",
        "You are capable of solving this correctly. Work with confidence, stay attentive, and check your final answer."
    ]
}


def build_h2_prompt(task_prompt, stimulus_text, prime_text="", sequence="prime_then_task_then_stimulus"):
    """
    Build a sequential prompt for H2 without changing the downstream scoring logic.

    Supported sequences:
    - prime_then_task_then_stimulus
    - task_then_prime_then_stimulus
    - task_then_stimulus
    """
    task_prompt = (task_prompt or "").strip()
    stimulus_text = (stimulus_text or "").strip()
    prime_text = (prime_text or "").strip()

    if sequence == "task_then_stimulus":
        parts = [task_prompt, stimulus_text]

    elif sequence == "prime_then_task_then_stimulus":
        parts = [prime_text, task_prompt, stimulus_text] if prime_text else [task_prompt, stimulus_text]

    elif sequence == "task_then_prime_then_stimulus":
        parts = [task_prompt, prime_text, stimulus_text] if prime_text else [task_prompt, stimulus_text]

    else:
        raise ValueError(f"Unknown sequence: {sequence}")

    return " ".join([p for p in parts if p]).strip()