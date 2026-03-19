from cognition.encoder import encode_state
from cognition.vectorstore import retrieve
from cognition.memory import get_memory, add_memory


def run_rag(state, question, embed_model):
    # ----------------------------
    # ENCODE
    # ----------------------------
    state_docs = encode_state(state)
    memory_docs = get_memory()

    docs = state_docs + memory_docs

    # ----------------------------
    # RETRIEVE
    # ----------------------------
    retrieved = retrieve(docs, question, embed_model)
    ctx = [d["text"] for d in retrieved]

    return ctx


def store_memory(question, answer, embed_model):
    memory_text = f"{question} → {answer[:200]}"

    memory_doc = {
        "type": "memory",
        "text": memory_text,
        "embedding": embed_model.encode(memory_text),
        "metadata": {}
    }

    add_memory(memory_doc)