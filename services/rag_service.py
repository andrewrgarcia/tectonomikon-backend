from cognition.encoder import encode_state
from cognition.vectorstore import retrieve
from cognition.memory import get_memory, add_memory

def run_rag(state, question, embed_model):
    state_docs = encode_state(state)
    memory_docs = get_memory()

    retrieved_state = retrieve(state_docs, question, embed_model)
    retrieved_memory = retrieve(memory_docs, question, embed_model, k=4)  # was 2

    ctx = (
        ["[SYSTEM] " + d["text"] for d in retrieved_state] +
        ["[MEMORY] " + d["text"] for d in retrieved_memory]
    )
    return ctx


def store_memory(question, answer, state, embed_model):
    drivers = state.get("drivers", []) if state else []
    paths = state.get("paths", []) if state else []

    top_codes = [d.get("code", "") for d in drivers[:3] if isinstance(d, dict)]
    
    # Include dominant path info — this is the key causal insight worth storing
    path_summaries = []
    for p in paths[:2]:
        codes = p.get("codes", [])
        strength = p.get("strength", 0)
        path_summaries.append(f"{' → '.join(codes)} (strength: {strength:.2f})")

    memory_text = f"""Question: {question}
Key variables: {", ".join(top_codes)}
Causal paths: {"; ".join(path_summaries) if path_summaries else "none"}
Summary: {answer[:400]}
"""  # bumped from 150 to 400

    memory_doc = {
        "type": "memory",
        "text": memory_text,
        "embedding": embed_model.encode(memory_text),
        "metadata": {}
    }
    add_memory(memory_doc)