"""Universal KPI detection — works on any uploaded dataset, not just solar sales."""
import pandas as pd

REVENUE_HINTS = ["revenue", "sales", "income", "amount", "total"]
QTY_HINTS = ["quantity", "qty", "units", "volume"]
CUSTOMER_HINTS = ["customer", "client", "user", "account", "buyer"]
PRODUCT_HINTS = ["product", "sku", "item", "service"]
REGION_HINTS = ["district", "region", "area", "city", "country", "state", "location", "branch"]
DATE_HINTS = ["date", "month", "period", "timestamp"]


def _pick(df, hints, prefer_numeric=False):
    cols = {c.lower(): c for c in df.columns}
    for h in hints:
        if h in cols:
            c = cols[h]
            if not prefer_numeric or pd.api.types.is_numeric_dtype(df[c]):
                return c
    for h in hints:
        for k, c in cols.items():
            if h in k:
                if not prefer_numeric or pd.api.types.is_numeric_dtype(df[c]):
                    return c
    return None


def detect(df: pd.DataFrame) -> dict:
    return {
        "revenue": _pick(df, REVENUE_HINTS, prefer_numeric=True),
        "quantity": _pick(df, QTY_HINTS, prefer_numeric=True),
        "customer": _pick(df, CUSTOMER_HINTS),
        "product": _pick(df, PRODUCT_HINTS),
        "region": _pick(df, REGION_HINTS),
        "date": _pick(df, DATE_HINTS),
    }


def kpis(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"total_revenue": 0, "total_quantity": 0, "total_customers": 0,
                "total_products": 0, "revenue_growth": 0, "detected": {}}
    d = detect(df)
    rev = float(pd.to_numeric(df[d["revenue"]], errors="coerce").sum()) if d["revenue"] else 0.0
    qty = float(pd.to_numeric(df[d["quantity"]], errors="coerce").sum()) if d["quantity"] else 0.0
    cust = int(df[d["customer"]].nunique()) if d["customer"] else 0
    prod = int(df[d["product"]].nunique()) if d["product"] else 0
    growth = 0.0
    if d["revenue"] and d["date"]:
        try:
            tdf = df.copy()
            tdf["__rev__"] = pd.to_numeric(tdf[d["revenue"]], errors="coerce")
            if d["date"] != "month":
                tdf["__p__"] = pd.to_datetime(tdf[d["date"]], errors="coerce").dt.to_period("M").astype(str)
                key = "__p__"
            else:
                key = d["date"]
            grp = tdf.groupby(key)["__rev__"].sum().sort_index()
            grp = grp[grp.index.astype(str) != "NaT"]
            if len(grp) >= 2:
                last, prev = float(grp.iloc[-1]), float(grp.iloc[-2])
                if prev > 0:
                    growth = round((last - prev) / prev * 100, 2)
        except Exception:
            pass
    return {
        "total_revenue": rev,
        "total_quantity": int(qty),
        "total_customers": cust,
        "total_products": prod,
        "revenue_growth": growth,
        "detected": d,
    }
