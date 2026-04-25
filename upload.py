"""Universal CSV ingestion: any schema, any industry."""
import io
import pandas as pd
import numpy as np


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, strip whitespace, attempt numeric conversion, handle nulls."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            stripped = df[col].astype(str).str.strip()
            stripped = stripped.replace(
                {"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan, "null": np.nan}
            )
            converted = pd.to_numeric(stripped, errors="ignore")
            df[col] = converted
    return df


def read_csv(file_or_bytes) -> pd.DataFrame:
    if isinstance(file_or_bytes, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(file_or_bytes))
    else:
        df = pd.read_csv(file_or_bytes)
    return normalize(df)


def numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
