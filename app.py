from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer

from core.preprocess import preprocess
from core.millipede_selector import select_variables, HAS_MILLIPEDE
from api.search import router as search_router
from llm import load_llm
from cognition.encoder import encode_state
from cognition.vectorstore import retrieve
from cognition.memory import get_memory, add_memory



print("STARTING APP...")

DATA_DIR = "data"
PARQUET_PATH = os.path.join(DATA_DIR, "fred_monthly_master_1994.parquet")
DF = None
EMBED_MODEL = None

def get_embed_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        print("[EMBED] loading model...")
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return EMBED_MODEL

def get_df():
    global DF
    if DF is None:
        print("Loading dataset into memory...")
        DF = pd.read_parquet(PARQUET_PATH)
    return DF

def download_if_needed():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(PARQUET_PATH):
        print("Dataset already exists.")
        return

    print("Downloading dataset...")

    url = "https://drive.google.com/uc?export=download&id=1g5FvsF_b6w6bdMRBzxfL0HpNAVRB4iAU"

    r = requests.get(url, stream=True)

    print("Status code:", r.status_code)

    total = 0
    with open(PARQUET_PATH, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)
                print(f"Downloaded {total / 1e6:.2f} MB")

    print("Download complete.")

app = FastAPI()

LLM = None

@app.on_event("startup")
def startup_event():
    global LLM
    download_if_needed()

    try:
        LLM = load_llm()
        print("[LLM] Loaded:", type(LLM).__name__)
    except Exception as e:
        print("[LLM] Disabled:", e)
        LLM = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)

# ----------------------------
# Request schema
# ----------------------------
class BuildModelRequest(BaseModel):
    target: str
    k: int = 5
    preprocess_mode: str = "clean"
    selector: str = "auto"   # NEW


class SimulateRequest(BaseModel):
    A: list
    initial_state: list
    steps: int = 24


# ----------------------------
# Core builder
# ----------------------------
def build_system(df, target, selected_vars):
    vars_all = selected_vars + [target]

    df = df[vars_all].dropna()

    # ----------------------------
    # Normalize
    # ----------------------------
    mean = df.mean()
    std = df.std().replace(0, 1)

    df_norm = (df - mean) / std

    # ----------------------------
    # Learn dynamics
    # ----------------------------
    X = df_norm.shift(1).dropna()
    Y = df_norm.loc[X.index]

    model = LinearRegression(fit_intercept=False)
    model.fit(X, Y)

    A = model.coef_

    # ----------------------------
    # Spectral normalization
    # ----------------------------
    eigvals = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigvals))

    if max_eig > 0:
        A = A / max_eig * 0.95

    return {
        "variables": vars_all,
        "A": A.tolist(),
        "mean": mean[vars_all].tolist(),
        "std": std[vars_all].tolist()
    }


# ----------------------------
# Simulation function
# ----------------------------
def simulate_system(A, x0, steps):
    A = np.array(A)
    x = np.array(x0)

    trajectory = [x.tolist()]

    for _ in range(steps):
        x = A @ x
        trajectory.append(x.tolist())

    return trajectory


@app.post("/narrate")
def narrate(body: dict):
    """
    Deterministic explanation mode.
    Takes structural output and rewrites it clearly.
    """
    if LLM is None:
        return {"answer": body.get("analysis")}

    question = body.get("question", "")
    analysis = body.get("analysis", "")

    prompt = build_structural_prompt(question, analysis)

    try:
        return {"answer": LLM.generate(prompt)}
    except Exception as e:
        print("[LLM ERROR]", e)
        return {"answer": analysis}


def build_structural_prompt(question: str, analysis: str) -> str:
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


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/capabilities")
def capabilities():
    return {
        "llm": LLM is not None,
        "llm_name": type(LLM).__name__ if LLM else None,
        "selector": "millipede" if HAS_MILLIPEDE else "correlation"
    }

@app.get("/search")
def search(q: str):

    # ⚠️ SAME DATA YOU USE FOR BUILD MODEL
    df = get_df()

    cols = df.columns.tolist()

    q = q.lower()

    matches = [c for c in cols if q in c.lower()]

    return matches[:20]

@app.post("/ask")
def ask(body: dict):
    if LLM is None:
        return {"answer": "LLM not available"}

    state = body.get("state", {})
    question = body.get("question", "")

    # ----------------------------
    # 1. ENCODE STATE + MEMORY
    # ----------------------------
    state_docs = encode_state(state)

    memory_docs = get_memory()

    docs = state_docs + memory_docs

    # ----------------------------
    # 2. RETRIEVE
    # ----------------------------
    model = get_embed_model()
    retrieved = retrieve(docs, question, model)

    ctx = [d["text"] for d in retrieved]

    # ----------------------------
    # 3. PROMPT
    # ----------------------------
    prompt = f"""
You are reasoning about a simulated economic system.

Relevant system + past reasoning:
{chr(10).join(ctx)}

Use system data first, but build on prior reasoning if helpful.

Question:
{question}
"""

    try:
        answer = LLM.generate(prompt)

        # ----------------------------
        # 4. STORE MEMORY
        # ----------------------------
        memory_text = f"{question} → {answer[:200]}"

        memory_doc = {
            "type": "memory",
            "text": memory_text,
            "embedding": model.encode(memory_text),
            "metadata": {}
        }

        add_memory(memory_doc)

        return {"answer": answer}

    except Exception as e:
        print("[LLM ERROR]", e)
        return {"answer": "LLM failed"}


@app.post("/build-model")
def build_model(req: BuildModelRequest):

    # ----------------------------
    # 1. Load data
    # ----------------------------
    df = get_df()

    if req.target not in df.columns:
        return {"error": f"Target '{req.target}' not found"}

    # ----------------------------
    # 2. Preprocess
    # ----------------------------
    if req.preprocess_mode == "clean":
        df = preprocess(df)
    else:
        df = df.fillna(df.median())

    # ----------------------------
    # 3. Select variables
    # ----------------------------
    result = select_variables(
        df=df,
        target=req.target,
        top_k=req.k,
        method=req.selector
    )

    selected = result["selected"]

    # ----------------------------
    # 4. Build system (A matrix)
    # ----------------------------
    system = build_system(df, req.target, selected)

    # ----------------------------
    # 5. Return full model
    # ----------------------------
    return {
        "target": req.target,
        "k": req.k,
        "selected": selected,
        "pip": result["pip"],
        **system
    }


@app.post("/simulate")
def simulate(req: SimulateRequest):

    trajectory = simulate_system(
        req.A,
        req.initial_state,
        req.steps
    )

    return {
        "trajectory": trajectory
    }