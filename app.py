from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from core.preprocess import preprocess
from core.millipede_selector import select_variables
from api.search import router as search_router

print("STARTING APP...")

DATA_DIR = "data"
PARQUET_PATH = os.path.join(DATA_DIR, "fred_monthly_master_1994.parquet")
DF = None

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

@app.on_event("startup")
def startup_event():
    print("Running startup tasks...")
    download_if_needed()
    print("Startup complete.")

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


# ----------------------------
# OPTIONAL LLM IMPORT (LIKE MILLIPEDE)
# ----------------------------
try:
    from llm.local_llm import generate as llm_generate

    # force a lightweight sanity check so it doesn't fail later
    _ = llm_generate  # just reference, no execution

    HAS_LLM = True
    print("[LLM] AVAILABLE")

except Exception as e:
    HAS_LLM = False
    print("[LLM] NOT AVAILABLE:", e)



@app.post("/narrate")
def narrate(body: dict):
    question = body.get("question", "")
    answer = body.get("answer", "")

    if not HAS_LLM:
        return {"answer": answer}

    try:
        prompt = f"""
User asked about: {question}

Here is a system-generated economic analysis:

{answer}

Rewrite this as a clear explanation answering the user's query.
Keep it concise. Do not repeat instructions.
"""

        rewritten = llm_generate(prompt)

        return {"answer": rewritten}

    except Exception as e:
        print("[LLM RUNTIME FAILED]", e)
        return {"answer": answer}
    

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search")
def search(q: str):

    # ⚠️ SAME DATA YOU USE FOR BUILD MODEL
    df = get_df()

    cols = df.columns.tolist()

    q = q.lower()

    matches = [c for c in cols if q in c.lower()]

    return matches[:20]

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