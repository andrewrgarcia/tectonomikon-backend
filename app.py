import os

import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from api.search import router as search_router
from core.millipede_selector import HAS_MILLIPEDE, select_variables
from core.preprocess import preprocess
from llm import load_llm
from services.llm_service import build_messages, classify_query
from services.model_service import build_system, simulate_system
from services.rag_service import run_rag, store_memory

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

@app.post("/narrate")
def narrate(body: dict):
    """
    Deterministic explanation mode.
    Rewrites structural output clearly.
    """
    if LLM is None:
        return {"answer": body.get("analysis")}

    question = body.get("question", "")
    analysis = body.get("analysis", "")

    # ----------------------------
    # BUILD MESSAGES (chat-native)
    # ----------------------------
    messages = [
        {
            "role": "user",
            "content": f"""
You are explaining the output of a causal economic system.

Only use the analysis below.
Do not add external knowledge.
Do not generalize.

Rewrite clearly and concisely.

Analysis:
{analysis}

Question:
{question}
"""
        }
    ]

    try:
        answer = LLM.generate(messages, mode="system")
        return {"answer": answer}

    except Exception as e:
        print("[LLM ERROR]", e)
        return {"answer": analysis}

@app.post("/ask")
def ask(body: dict):
    # ----------------------------
    # 0. LLM CHECK
    # ----------------------------
    if LLM is None:
        return {"answer": "LLM not available"}

    # ----------------------------
    # 1. INPUTS
    # ----------------------------
    state = body.get("state", {}) or {}
    question = (body.get("question") or "").strip()
    history = body.get("history", []) or []

    if not question:
        return {"answer": "Ask a question."}

    # ----------------------------
    # 2. EMBEDDING MODEL
    # ----------------------------
    model = get_embed_model()

    # ----------------------------
    # 3. RAG
    # ----------------------------
    try:
        ctx = run_rag(state, question, model)
    except Exception as e:
        print("[RAG ERROR]", e)
        ctx = []

    # ----------------------------
    # 4. MODE ROUTING
    # ----------------------------
    try:
        mode = classify_query(question, state)
    except Exception:
        mode = "open"

    # ----------------------------
    # 5. BUILD MESSAGES
    # ----------------------------
    try:
        messages = build_messages(ctx, question, mode, history)
    except Exception as e:
        print("[MESSAGE ERROR]", e)
        messages = [{"role": "user", "content": question}]

    # ----------------------------
    # 6. GENERATE
    # ----------------------------
    try:
        answer = LLM.generate(messages, mode=mode)
    except Exception as e:
        print("[LLM ERROR]", e)
        return {"answer": "LLM failed"}

    # ----------------------------
    # 7. STORE MEMORY
    # ----------------------------
    try:
        store_memory(question, answer, state, model)
    except Exception as e:
        print("[MEMORY ERROR]", e)

    # ----------------------------
    # 8. RETURN
    # ----------------------------
    return {"answer": answer}

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