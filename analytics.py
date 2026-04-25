import pandas as pd
import numpy as np


def compute_kpis(df: pd.DataFrame) -> dict:
    try:
        total_revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
        total_quantity = int(df["quantity"].sum()) if "quantity" in df.columns else 0
        total_customers = int(df["customer"].nunique()) if "customer" in df.columns else 0
        total_products = int(df["product"].nunique()) if "product" in df.columns else 0

        revenue_growth = 0.0
        if "month" in df.columns and "revenue" in df.columns:
            monthly = df.groupby("month")["revenue"].sum().reset_index()
            if len(monthly) >= 2:
                last = monthly["revenue"].iloc[-1]
                prev = monthly["revenue"].iloc[-2]
                if prev > 0:
                    revenue_growth = round(((last - prev) / prev) * 100, 2)

        return {
            "total_revenue": total_revenue,
            "total_quantity": total_quantity,
            "total_customers": total_customers,
            "total_products": total_products,
            "revenue_growth": revenue_growth,
        }
    except Exception:
        return {
            "total_revenue": 0, "total_quantity": 0,
            "total_customers": 0, "total_products": 0, "revenue_growth": 0,
        }


def revenue_by_product(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return (df.groupby("product")["revenue"].sum()
                .reset_index().sort_values("revenue", ascending=False))
    except Exception:
        return pd.DataFrame({"product": [], "revenue": []})


def revenue_by_district(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return (df.groupby("district")["revenue"].sum()
                .reset_index().sort_values("revenue", ascending=False))
    except Exception:
        return pd.DataFrame({"district": [], "revenue": []})


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if "month" not in df.columns:
            df = df.copy()
            df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
        return (df.groupby("month")["revenue"].sum()
                .reset_index().sort_values("month"))
    except Exception:
        return pd.DataFrame({"month": [], "revenue": []})


def pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    try:
        pivot = pd.pivot_table(
            df, values="revenue", index="product",
            columns="district", aggfunc="sum", fill_value=0,
        )
        pivot["Total"] = pivot.sum(axis=1)
        return pivot
    except Exception:
        return pd.DataFrame()


def growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        trend = monthly_trend(df)
        if len(trend) < 2:
            return pd.DataFrame()
        trend = trend.copy()
        trend["prev_revenue"] = trend["revenue"].shift(1)
        trend = trend.dropna()
        trend["growth_rate"] = ((trend["revenue"] - trend["prev_revenue"]) / trend["prev_revenue"] * 100).round(2)
        return trend[["month", "revenue", "growth_rate"]]
    except Exception:
        return pd.DataFrame()
