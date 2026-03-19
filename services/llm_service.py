def format_phi3(prompt: str) -> str:
    return f"<|endoftext|><|user|>\n{prompt}<|end|>\n<|assistant|>"

def classify_query(question: str, state: dict) -> str:
    q = (question or "").lower()

    variables = [
        str(v).lower()
        for v in state.get("variables", [])
    ]

    system_keywords = [
        "shock",
        "shocks",
        "driver",
        "drivers",
        "path",
        "paths",
        "affect",
        "affected",
        "affecting",
        "impact",
        "move",
        "moved",
        "moving",
        "contribution",
        "contributed",
        "trajectory",
        "simulate",
        "simulation",
        "model",
    ]

    if any(v and v in q for v in variables):
        return "system"

    if any(k in q for k in system_keywords):
        return "system"

    return "open"


def build_messages(ctx, question):
    messages = []

    if ctx:
        context_text = "\n".join(ctx)

        messages.append({
            "role": "assistant",
            "content": f"Context from the system:\n{context_text}"
        })

    messages.append({
        "role": "user",
        "content": question
    })

    return messages


def build_structural_prompt(question, analysis):
    """
    STRICT mode:
    - No hallucination
    - No external knowledge
    - Only rewrite + clarify
    """
    
    return f"""
You are explaining the output of a causal economic system.

Only use the analysis provided below.
Do not add outside knowledge.
Do not generalize beyond the data.

Rewrite it clearly and concisely.

Analysis:
{analysis}

Question:
{question}
"""