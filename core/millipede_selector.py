import numpy as np
import pandas as pd
import warnings

from millipede import NormalLikelihoodVariableSelector


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
# MAIN FUNCTION
# ----------------------------
def select_variables(
    df: pd.DataFrame,
    target: str,
    top_k: int = 5,
    prescreen_k: int = PRESCREEN_K
):
    """
    Pure variable selection via millipede.

    Parameters
    ----------
    df : DataFrame (T x N)
    target : str
    top_k : number of variables to return

    Returns
    -------
    dict with:
        - selected: list[str]
        - pip: dict[str, float]
        - all_pip: full sorted series (optional use)
    """

    assert target in df.columns, f"{target} not in dataframe"

    df = df.copy()

    predictors = [c for c in df.columns if c != target]

    print(f"\n[Millipede] Target: {target}")
    print(f"Initial predictors: {len(predictors)}")

    # ----------------------------
    # PRESCREEN (ALWAYS for large US case)
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

    # ----------------------------
    # 1. Impute
    # ----------------------------
    df_sub = df_sub.fillna(df_sub.mean())

    # ----------------------------
    # 2. STANDARDIZE (CRITICAL)
    # ----------------------------
    mean = df_sub.mean()
    std = df_sub.std().replace(0, 1)

    df_sub = (df_sub - mean) / std

    # ----------------------------
    # 3. DROP ANY REMAINING NaNs
    # ----------------------------
    df_sub = df_sub.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"Matrix shape passed to millipede: {df_sub.shape}")

    # ----------------------------
    # RUN MCMC
    # ----------------------------
    try:
        pip = run_millipede(df_sub, target)

    except MemoryError:
        warnings.warn("MemoryError — retrying with subset_size")
        pip = run_millipede(df_sub, target, subset_size=SUBSET_SIZE_FALLBACK)

    # ----------------------------
    # SORT + SELECT
    # ----------------------------
    pip = pip.sort_values(ascending=False)

    selected = pip.index[:top_k].tolist()

    print("\nTop variables:")
    for v in selected:
        print(f"{v}: {pip[v]:.3f}")

    return {
        "selected": selected,
        "pip": pip[selected].to_dict(),
        "all_pip": pip  # keep for later use if needed
    }