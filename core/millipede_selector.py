import numpy as np
import pandas as pd
import warnings

# -----------------------------------
# OPTIONAL IMPORT
# -----------------------------------
try:
    from millipede import NormalLikelihoodVariableSelector
    HAS_MILLIPEDE = True
except ImportError:
    HAS_MILLIPEDE = False


# ----------------------------
# CONFIG
# ----------------------------
T_BURNIN = 1000
T_SAMPLES = 2000
S_PRIOR = 4
SEED = 42

PRESCREEN_K = 3000
SUBSET_SIZE_FALLBACK = 1000


# ----------------------------
# HELPERS
# ----------------------------
def correlation_prescreen(df, predictors, target, top_k):
    """
    Select top_k predictors by |corr| with target.
    """
    y = df[target].values
    scores = {}

    for col in predictors:
        x = df[col].values
        mask = np.isfinite(x) & np.isfinite(y)

        if mask.sum() < 10:
            scores[col] = 0.0
        else:
            scores[col] = abs(np.corrcoef(x[mask], y[mask])[0, 1])

    return sorted(scores, key=scores.get, reverse=True)[:top_k]


def run_millipede(df, target, subset_size=None):
    """
    Run MCMC and return PIP series.
    """
    if not HAS_MILLIPEDE:
        raise RuntimeError("millipede not installed")

    selector = NormalLikelihoodVariableSelector(
        df,
        target,
        S=S_PRIOR,
        prior="isotropic"
    )

    kwargs = dict(
        T=T_SAMPLES,
        T_burnin=T_BURNIN,
        seed=SEED
    )

    if subset_size is not None:
        kwargs["subset_size"] = subset_size

    selector.run(**kwargs)

    return selector.pip


# ----------------------------
# FAST FALLBACK
# ----------------------------
def correlation_selector(df, target, predictors, top_k):
    """
    Simple fallback if millipede unavailable.
    """
    y = df[target].values
    scores = {}

    for col in predictors:
        x = df[col].values
        mask = np.isfinite(x) & np.isfinite(y)

        if mask.sum() < 10:
            scores[col] = 0.0
        else:
            scores[col] = abs(np.corrcoef(x[mask], y[mask])[0, 1])

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    selected = [k for k, _ in ranked[:top_k]]

    return {
        "selected": selected,
        "pip": {k: float(v) for k, v in ranked[:top_k]},
        "all_pip": dict(ranked),
        "method": "correlation"
    }


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def select_variables(
    df: pd.DataFrame,
    target: str,
    top_k: int = 5,
    prescreen_k: int = PRESCREEN_K,
    method: str = "auto"   # NEW
):
    """
    Variable selection system.

    method:
        - "auto"       → millipede if available else correlation
        - "millipede"  → force millipede
        - "corr"       → force correlation
    """

    assert target in df.columns, f"{target} not in dataframe"

    df = df.copy()

    predictors = [c for c in df.columns if c != target]

    print(f"\n[Selector] Target: {target}")
    print(f"Initial predictors: {len(predictors)}")

    # ----------------------------
    # PRESCREEN
    # ----------------------------
    if len(predictors) > prescreen_k:
        print(f"Prescreening to top {prescreen_k} by correlation...")
        predictors = correlation_prescreen(df, predictors, target, prescreen_k)

    print(f"Predictors after prescreen: {len(predictors)}")

    # ----------------------------
    # PREP DATA
    # ----------------------------
    cols = predictors + [target]
    df_sub = df[cols].copy()

    df_sub = df_sub.fillna(df_sub.mean())

    mean = df_sub.mean()
    std = df_sub.std().replace(0, 1)

    df_sub = (df_sub - mean) / std

    df_sub = df_sub.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"Matrix shape: {df_sub.shape}")

    # ----------------------------
    # METHOD LOGIC
    # ----------------------------
    use_millipede = (
        method == "millipede" or
        (method == "auto" and HAS_MILLIPEDE)
    )

    if use_millipede:
        try:
            print("Running millipede...")

            pip = run_millipede(df_sub, target)

            pip = pip.sort_values(ascending=False)
            selected = pip.index[:top_k].tolist()

            print("\nTop variables (millipede):")
            for v in selected:
                print(f"{v}: {pip[v]:.3f}")

            return {
                "selected": selected,
                "pip": pip[selected].to_dict(),
                "all_pip": pip,
                "method": "millipede"
            }

        except Exception as e:
            print(f"Millipede failed: {e}")
            print("Falling back to correlation...")

    # ----------------------------
    # FALLBACK
    # ----------------------------
    return correlation_selector(df_sub, target, predictors, top_k)