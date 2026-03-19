def classify_query(question: str, state: dict) -> str:
    q = (question or "").lower().strip()

    # Greetings and social openers → always open
    greetings = ["hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "cool", "great"]
    if any(q == g or q.startswith(g + " ") for g in greetings):
        return "open"

    # Short queries that look like ticker codes (all caps or known patterns) → system
    if len(q) < 12 and (question.strip().isupper() or question.strip().replace("_","").isalnum()):
        return "system"

    variables = [str(v).lower() for v in state.get("variables", [])]

    system_keywords = [
        "shock", "driver", "path", "affect", "impact", "move",
        "contribution", "trajectory", "simulate", "model",
        "memory", "remember", "history", "cause", "predict",
        "explain", "why", "how does", "what happens"
    ]

    if any(v and v in q for v in variables):
        return "system"

    if any(k in q for k in system_keywords):
        return "system"

    return "open"


def build_messages(ctx, question, mode="open", history=None):
    history = history or []
    messages = []

    if mode == "open":
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant. Respond naturally and concisely."
        })
    else:
        # Separate memory vs system context for clearer prompting
        system_ctx = [c for c in ctx if c.startswith("[SYSTEM]")]
        memory_ctx = [c for c in ctx if c.startswith("[MEMORY]")]

        system_prompt = """You analyze a causal economic system. Answer using the system data and memory below.

Rules:
- Synthesize across BOTH system data and memory when answering
- If memory contains relevant prior reasoning, use it explicitly
- Be specific: name variables, strengths, and directions
- Do not pad or repeat; be concise
"""
        if memory_ctx:
            system_prompt += "\nPrior memory (use this for cross-variable reasoning):\n" + "\n".join(memory_ctx)

        messages.append({"role": "system", "content": system_prompt})

    for h in history[-3:]:  # bumped from 2 to 3
        q = h.get("question")
        a = h.get("answer")
        if q:
            messages.append({"role": "user", "content": q})
        if a:
            messages.append({"role": "assistant", "content": a})

    # System context injected just before the question
    system_ctx = [c for c in ctx if c.startswith("[SYSTEM]")]
    if mode == "system" and system_ctx:
        messages.append({
            "role": "system",
            "content": "Current system data:\n" + "\n".join(system_ctx)
        })

    messages.append({"role": "user", "content": question})
    return messages