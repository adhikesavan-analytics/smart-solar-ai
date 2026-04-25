"""Generic linear-regression forecasting on any numeric column."""
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


def forecast(df: pd.DataFrame, x_col: str, y_col: str, steps: int = 6):
    df = df.copy()
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Selected columns not found")

    # If x is non-numeric (e.g. month label) treat it as an ordered index
    x_raw = df[x_col]
    if not pd.api.types.is_numeric_dtype(x_raw):
        ordered = sorted(x_raw.dropna().astype(str).unique())
        idx_map = {v: i for i, v in enumerate(ordered)}
        x_vals = x_raw.astype(str).map(idx_map).astype(float)
        x_labels = ordered
        future_labels = [f"+{i+1}" for i in range(steps)]
        is_label = True
    else:
        x_vals = pd.to_numeric(x_raw, errors="coerce")
        x_labels = None
        future_labels = None
        is_label = False

    y_vals = pd.to_numeric(df[y_col], errors="coerce")
    mask = (~x_vals.isna()) & (~y_vals.isna())
    x = x_vals[mask].values.reshape(-1, 1)
    y = y_vals[mask].values

    if len(x) < 2:
        raise ValueError("Not enough data to fit a model")

    if SKLEARN_OK:
        m = LinearRegression().fit(x, y)
        slope, intercept = float(m.coef_[0]), float(m.intercept_)
        y_pred = m.predict(x)
        r2 = float(r2_score(y, y_pred)) if len(y) > 1 else 0.0
    else:
        # numpy fallback
        coef = np.polyfit(x.flatten(), y, 1)
        slope, intercept = float(coef[0]), float(coef[1])
        y_pred = slope * x.flatten() + intercept
        ss_res = float(((y - y_pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
        r2 = 1 - ss_res / ss_tot

    history = [
        {"x": (x_labels[int(xi)] if is_label and 0 <= int(xi) < len(x_labels) else float(xi)),
         "y": float(yi)}
        for xi, yi in zip(x.flatten(), y)
    ]

    last_x = float(x.max())
    forecast_pts = []
    for i in range(1, steps + 1):
        nx = last_x + i
        ny = slope * nx + intercept
        label = future_labels[i - 1] if is_label else float(nx)
        forecast_pts.append({"x": label, "y": float(ny)})

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "history": history,
        "forecast": forecast_pts,
    }
