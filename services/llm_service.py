def build_reasoning_prompt(ctx, question):
    return f"""
You are reasoning about a simulated economic system.

Relevant system + past reasoning:
{chr(10).join(ctx)}

Use system data first, but build on prior reasoning if helpful.

Question:
{question}
"""


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