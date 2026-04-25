"""Financial analysis: P&L, revenue vs cost, profit margin from any dataset."""
import pandas as pd
import numpy as np

REVENUE_HINTS = ["revenue", "sales", "income", "amount", "total"]
COST_HINTS = ["cost", "expense", "expenses", "cogs", "spend", "purchase", "purchases"]
DATE_HINTS = ["date", "month", "period", "timestamp"]


def _pick(df, hints):
    cols = [c.lower() for c in df.columns]
    for h in hints:
        for c in cols:
            if h == c:
                return c
    for h in hints:
        for c in cols:
            if h in c:
                return c
    return None


def financial_report(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    rev_col = _pick(df, REVENUE_HINTS)
    cost_col = _pick(df, COST_HINTS)
    date_col = _pick(df, DATE_HINTS)

    if rev_col is None and "unit_price" in df.columns and "quantity" in df.columns:
        df["__rev__"] = pd.to_numeric(df["unit_price"], errors="coerce") * pd.to_numeric(
            df["quantity"], errors="coerce"
        )
        rev_col = "__rev__"

    revenue = float(pd.to_numeric(df[rev_col], errors="coerce").sum()) if rev_col else 0.0
    cost = float(pd.to_numeric(df[cost_col], errors="coerce").sum()) if cost_col else revenue * 0.6
    profit = revenue - cost
    margin = (profit / revenue * 100) if revenue > 0 else 0.0

    timeline = []
    if date_col and rev_col:
        try:
            tdf = df.copy()
            if date_col != "month":
                tdf["__period__"] = pd.to_datetime(tdf[date_col], errors="coerce").dt.to_period("M").astype(str)
                key = "__period__"
            else:
                key = date_col
            tdf["__rev__"] = pd.to_numeric(tdf[rev_col], errors="coerce")
            if cost_col:
                tdf["__cost__"] = pd.to_numeric(tdf[cost_col], errors="coerce")
            else:
                tdf["__cost__"] = tdf["__rev__"] * 0.6
            grp = tdf.groupby(key).agg(revenue=("__rev__", "sum"), cost=("__cost__", "sum")).reset_index()
            grp["profit"] = grp["revenue"] - grp["cost"]
            for _, row in grp.iterrows():
                timeline.append(
                    {
                        "label": str(row[key]),
                        "revenue": float(row["revenue"] or 0),
                        "cost": float(row["cost"] or 0),
                        "profit": float(row["profit"] or 0),
                    }
                )
        except Exception:
            timeline = []

    return {
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "margin": margin,
        "revenue_column": rev_col if rev_col != "__rev__" else "unit_price × quantity",
        "cost_column": cost_col,
        "timeline": timeline,
    }
