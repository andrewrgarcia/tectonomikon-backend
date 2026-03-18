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

DATA_DIR = "data"
PARQUET_PATH = os.path.join(DATA_DIR, "fred_monthly_master_1994.parquet")

def download_if_needed():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(PARQUET_PATH):
        print("Downloading dataset...")

        url = "https://YOUR_LINK_HERE/fred_monthly_master_1994.parquet"

        r = requests.get(url)
        with open(PARQUET_PATH, "wb") as f:
            f.write(r.content)

        print("Download complete.")

download_if_needed()

app = FastAPI()

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
# Endpoints
# ----------------------------
@app.get("/search")
def search(q: str):

    # ⚠️ SAME DATA YOU USE FOR BUILD MODEL
    df = pd.read_parquet("data/fred_monthly_master_1994.parquet")

    cols = df.columns.tolist()

    q = q.lower()

    matches = [c for c in cols if q in c.lower()]

    return matches[:20]

@app.post("/build-model")
def build_model(req: BuildModelRequest):

    # ----------------------------
    # 1. Load data
    # ----------------------------
    df = pd.read_parquet("data/fred_monthly_master_1994.parquet")

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
        top_k=req.k
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