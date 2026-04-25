"""Statistical analysis: descriptive stats, correlations, plain-English insights."""
import pandas as pd
import numpy as np


def describe(df: pd.DataFrame) -> pd.DataFrame:
    nums = df.select_dtypes(include="number")
    if nums.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "count": nums.count(),
            "mean": nums.mean(),
            "median": nums.median(),
            "min": nums.min(),
            "max": nums.max(),
            "std": nums.std(),
        }
    )
    return out.round(3)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    nums = df.select_dtypes(include="number")
    if nums.shape[1] < 2:
        return pd.DataFrame()
    return nums.corr(numeric_only=True).round(3)


def correlation_insights(df: pd.DataFrame, top_n: int = 5):
    cm = correlation_matrix(df)
    if cm.empty:
        return []
    pairs = []
    cols = cm.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = cm.iloc[i, j]
            if pd.notna(r):
                pairs.append((cols[i], cols[j], float(r)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    out = []
    for a, b, r in pairs[:top_n]:
        if abs(r) >= 0.7:
            strength = "strongly"
        elif abs(r) >= 0.4:
            strength = "moderately"
        else:
            strength = "weakly"
        direction = "positively" if r >= 0 else "negatively"
        out.append(f"`{a}` is {strength} {direction} correlated with `{b}` (r = {r:.2f}).")
    return out
