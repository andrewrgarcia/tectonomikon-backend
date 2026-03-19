import numpy as np
from sklearn.linear_model import LinearRegression

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