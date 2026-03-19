def format_phi3(prompt: str) -> str:
    return f"<|endoftext|><|user|>\n{prompt}<|end|>\n<|assistant|>"

def classify_query(question: str, state: dict) -> str:
    q = (question or "").lower().strip()

    # short queries → usually variable lookup
    if len(q) < 12:
        return "system"

    variables = [
        str(v).lower()
        for v in state.get("variables", [])
    ]

    system_keywords = [
        "shock", "driver", "path",
        "affect", "impact", "move",
        "contribution", "trajectory",
        "simulate", "model"
    ]

    if any(v and v in q for v in variables):
        return "system"

    if any(k in q for k in system_keywords):
        return "system"

    return "open"


def build_messages(ctx, question, mode="open", history=None):
    """
    Full conversational message builder with:
    - mode routing
    - short-term memory
    - RAG injection (system mode only)
    """

    history = history or []
    messages = []

    # ----------------------------
    # SYSTEM ROLE
    # ----------------------------
    if mode == "open":
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant. Respond naturally."
        })
    else:
        messages.append({
            "role": "system",
            "content": "You analyze a causal economic system. Use provided data when available."
        })

    # ----------------------------
    # CHAT HISTORY (last 2 turns)
    # ----------------------------
    for h in history[-2:]:
        q = h.get("question")
        a = h.get("answer")

        if q:
            messages.append({"role": "user", "content": q})
        if a:
            messages.append({"role": "assistant", "content": a})

    # ----------------------------
    # SYSTEM CONTEXT (ONLY in system mode)
    # ----------------------------
    if mode == "system" and ctx:
        messages.append({
            "role": "assistant",
            "content": "Relevant system data:\n" + "\n".join(ctx)
        })

    # ----------------------------
    # CURRENT QUESTION
    # ----------------------------
    messages.append({
        "role": "user",
        "content": question
    })

    return messages