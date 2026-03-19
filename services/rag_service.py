from cognition.encoder import encode_state
from cognition.vectorstore import retrieve
from cognition.memory import get_memory, add_memory


def run_rag(state, question, embed_model):
    # ----------------------------
    # ENCODE
    # ----------------------------
    state_docs = encode_state(state)
    memory_docs = get_memory()

    # ----------------------------
    # RETRIEVE (FIXED)
    # ----------------------------
    retrieved_state = retrieve(state_docs, question, embed_model)
    retrieved_memory = retrieve(memory_docs, question, embed_model)

    # ----------------------------
    # BUILD CONTEXT
    # ----------------------------
    ctx = (
        ["[SYSTEM] " + d["text"] for d in retrieved_state] +
        ["[MEMORY] " + d["text"] for d in retrieved_memory[:2]]
    )

    return ctx


def store_memory(question, answer, state, embed_model):
    # ----------------------------
    # SAFE DRIVER EXTRACTION
    # ----------------------------
    drivers = state.get("drivers", []) if state else []

    top_codes = [
        d.get("code", "")
        for d in drivers[:3]
        if isinstance(d, dict)
    ]

    # ----------------------------
    # BUILD MEMORY TEXT
    # ----------------------------
    memory_text = f"""Question: {question}

Key variables: {", ".join(top_codes)}

Summary: {answer[:150]}
"""

    # ----------------------------
    # CREATE MEMORY DOC
    # ----------------------------
    memory_doc = {
        "type": "memory",
        "text": memory_text,
        "embedding": embed_model.encode(memory_text),
        "metadata": {}
    }

    add_memory(memory_doc)