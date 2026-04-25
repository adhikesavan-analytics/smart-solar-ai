"""Automatic data cleaning pipeline.

Steps:
 a) Standardize column names
 b) Type conversion (date + numeric)
 c) Drop fully-empty rows / drop rows missing critical columns
 d) Fill numeric NaNs with median, categorical NaNs with mode
 e) Remove duplicates
 f) Outlier handling (IQR cap or remove)
 g) Validation
 h) Logging — returns step-by-step log
"""
import re
import numpy as np
import pandas as pd


def _std_name(name: str) -> str:
    n = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip().lower())
    n = re.sub(r"_+", "_", n).strip("_")
    return n or "col"


def standardize_columns(df: pd.DataFrame, log: list) -> pd.DataFrame:
    new = [_std_name(c) for c in df.columns]
    if new != list(df.columns):
        log.append(f"Renamed columns to snake_case (e.g. `{df.columns[0]}` → `{new[0]}`).")
    df = df.copy()
    df.columns = new
    # ensure unique
    seen = {}
    final = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            final.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            final.append(c)
    df.columns = final
    return df


def convert_types(df: pd.DataFrame, log: list) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            stripped = df[col].astype(str).str.strip().replace(
                {"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "NULL": np.nan, "null": np.nan}
            )
            # try numeric
            num = pd.to_numeric(stripped, errors="coerce")
            if num.notna().sum() >= max(1, int(0.7 * stripped.notna().sum())):
                df[col] = num
                log.append(f"Converted `{col}` to numeric.")
                continue
            # try date if name suggests it
            if any(k in col for k in ("date", "time", "month", "year", "day")):
                d = pd.to_datetime(stripped, errors="coerce")
                if d.notna().sum() >= max(1, int(0.5 * stripped.notna().sum())):
                    df[col] = d
                    log.append(f"Converted `{col}` to datetime.")
                    continue
            df[col] = stripped
    return df


def drop_critical_missing(df: pd.DataFrame, required: list, log: list) -> pd.DataFrame:
    if not required:
        return df
    present = [c for c in required if c in df.columns]
    if not present:
        return df
    before = len(df)
    df = df.dropna(subset=present)
    removed = before - len(df)
    if removed:
        log.append(f"Dropped {removed} row(s) missing required columns: {', '.join(present)}.")
    return df


def fill_missing(df: pd.DataFrame, log: list) -> pd.DataFrame:
    df = df.copy()
    filled_num, filled_cat = [], []
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            if pd.notna(med):
                df[col] = df[col].fillna(med)
                filled_num.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        else:
            mode = df[col].mode(dropna=True)
            if len(mode) > 0:
                df[col] = df[col].fillna(mode.iloc[0])
                filled_cat.append(col)
    if filled_num:
        log.append(f"Filled missing numeric values with median in: {', '.join(filled_num)}.")
    if filled_cat:
        log.append(f"Filled missing categorical values with mode in: {', '.join(filled_cat)}.")
    return df


def remove_duplicates(df: pd.DataFrame, log: list) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        log.append(f"Removed {removed} duplicate row(s).")
    return df


def handle_outliers(df: pd.DataFrame, mode: str, log: list) -> pd.DataFrame:
    """mode: 'cap' | 'remove' | 'none'"""
    if mode == "none":
        return df
    df = df.copy()
    affected = []
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((df[col] < lo) | (df[col] > hi)).sum()
        if outliers == 0:
            continue
        affected.append(f"{col}({int(outliers)})")
        if mode == "cap":
            df[col] = df[col].clip(lower=lo, upper=hi)
        elif mode == "remove":
            df = df[(df[col] >= lo) & (df[col] <= hi)]
    if affected:
        verb = "Capped" if mode == "cap" else "Removed"
        log.append(f"{verb} outliers via IQR in: {', '.join(affected)}.")
    return df


def validate(df: pd.DataFrame, required: list) -> list:
    return [c for c in required if c not in df.columns] if required else []


def clean(
    df: pd.DataFrame,
    required: list = None,
    outlier_mode: str = "cap",
):
    """Run the full pipeline. Returns (cleaned_df, log_lines, missing_required)."""
    log = []
    required = [r.lower() for r in (required or [])]

    df = standardize_columns(df, log)

    # drop fully-empty rows early
    before = len(df)
    df = df.dropna(how="all")
    if before - len(df):
        log.append(f"Dropped {before - len(df)} fully-empty row(s).")

    df = convert_types(df, log)
    df = drop_critical_missing(df, required, log)
    df = fill_missing(df, log)
    df = remove_duplicates(df, log)
    df = handle_outliers(df, outlier_mode, log)

    missing = validate(df, required)
    if missing:
        log.append(f"⚠️ Missing required columns: {', '.join(missing)}.")
    if not log:
        log.append("Data was already clean — no changes applied.")
    return df.reset_index(drop=True), log, missing
