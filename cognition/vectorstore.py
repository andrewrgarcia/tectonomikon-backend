import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cosine(a, b):
    return np.dot(a, b)


def retrieve(docs, query, model, k=5):
    if not docs:
        return []

    # ----------------------------
    # QUERY EMBEDDING (normalized)
    # ----------------------------
    q_emb = normalize(model.encode(query))

    scored = []

    for d in docs:
        emb = d.get("embedding")

        # skip bad docs
        if emb is None:
            continue

        emb = normalize(np.array(emb))

        score = cosine(q_emb, emb)

        scored.append((score, d))

    if not scored:
        return []

    # ----------------------------
    # SORT (high → low)
    # ----------------------------
    scored.sort(key=lambda x: x[0], reverse=True)

    return [d for _, d in scored[:k]]