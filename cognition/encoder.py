from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def encode_state(state):
    docs = []

    # ----------------------------
    # SHOCKS
    # ----------------------------
    for s in state.get("shocks", []):
        text = f"{s['code']} shock of {s['value']:.2f}"

        docs.append({
            "type": "shock",
            "text": text,
            "embedding": model.encode(text),
            "metadata": {
                "code": s["code"],
                "value": s["value"]
            }
        })

    # ----------------------------
    # DRIVERS
    # ----------------------------
    for d in state.get("drivers", []):
        text = f"{d['title']} drives system with contribution {d['contribution']:.3f}"

        docs.append({
            "type": "driver",
            "text": text,
            "embedding": model.encode(text),
            "metadata": {
                "code": d["code"],
                "contribution": d["contribution"]
            }
        })

    # ----------------------------
    # PATHS
    # ----------------------------
    for p in state.get("paths", []):
        chain = " causes ".join(p["titles"])
        text = f"{chain} with strength {p['strength']:.2f}"

        docs.append({
            "type": "path",
            "text": text,
            "embedding": model.encode(text),
            "metadata": {
                "codes": p["codes"],
                "strength": p["strength"]
            }
        })

    return docs