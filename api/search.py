import json
from fastapi import APIRouter, Query

router = APIRouter()

# ----------------------------
# LOAD MAPPING (ONCE)
# ----------------------------
with open("data/fred_id_to_title.json", "r") as f:
    ID_TO_TITLE = json.load(f)

# precompute lowercase for speed
SEARCH_INDEX = [
    (k, k.lower(), v.lower(), v)
    for k, v in ID_TO_TITLE.items()
]


# ----------------------------
# SEARCH
# ----------------------------
import re

def tokenize(text):
    return set(re.findall(r'\b\w+\b', text.lower()))

@router.get("/search")
def search(q: str, k: int = 50):
    ql = q.lower()
    q_words = tokenize(q)

    exact_code_matches = []
    strict_matches = []
    partial_matches = []

    for id_, id_l, title_l, title in SEARCH_INDEX:
        title_words = tokenize(title)

        # ----------------------------
        # 1. CODE MATCH (HIGHEST PRIORITY)
        # ----------------------------
        if ql in id_l:
            exact_code_matches.append((id_, title))
            continue

        # ----------------------------
        # 2. STRICT MATCH (ALL WORDS PRESENT)
        # ----------------------------
        if q_words.issubset(title_words):
            strict_matches.append((id_, title))
            continue

        # ----------------------------
        # 3. PARTIAL MATCH (OVERLAP)
        # ----------------------------
        overlap = len(q_words & title_words)

        if overlap > 0:
            partial_matches.append((overlap, id_, title))

    # sort partial by overlap strength
    partial_matches.sort(reverse=True)

    # flatten
    results = (
        exact_code_matches +
        strict_matches +
        [(id_, title) for _, id_, title in partial_matches]
    )

    # remove duplicates (important)
    seen = set()
    final = []
    for id_, title in results:
        if id_ not in seen:
            final.append({"id": id_, "title": title})
            seen.add(id_)
        if len(final) >= k:
            break

    return final