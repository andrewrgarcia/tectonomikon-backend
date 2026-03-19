def build_docs(state):
    docs = []

    # ----------------------------
    # SHOCKS
    # ----------------------------
    for s in state.get("shocks", []):
        docs.append(
            f"Shock: {s['code']} changed by {s['value']:.2f}"
        )

    # ----------------------------
    # DRIVERS
    # ----------------------------
    for d in state.get("drivers", []):
        docs.append(
            f"Driver: {d['title']} contributed {d['contribution']:.3f}"
        )

    # ----------------------------
    # PATHS
    # ----------------------------
    for p in state.get("paths", []):
        chain = " → ".join(p["titles"])
        docs.append(
            f"Path: {chain} (strength {p['strength']:.2f})"
        )

    return docs


def retrieve(docs, question, k=5):
    q = question.lower()

    scored = []
    for d in docs:
        score = sum(word in d.lower() for word in q.split())
        scored.append((score, d))

    scored.sort(reverse=True)

    return [d for s, d in scored[:k] if s > 0]