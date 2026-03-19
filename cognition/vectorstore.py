import numpy as np


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def retrieve(docs, query, model, k=5):
    q_emb = model.encode(query)

    scored = []
    for d in docs:
        score = cosine(q_emb, d["embedding"])
        scored.append((score, d))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [d for _, d in scored[:k]]