import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


def demand_forecast(df: pd.DataFrame):
    try:
        if not SKLEARN_OK:
            return None, None

        df = df.copy()
        if "month" not in df.columns:
            df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

        monthly_prod = (
            df.groupby(["product", "month"])["revenue"]
            .sum().reset_index().sort_values("month")
        )

        if monthly_prod["month"].nunique() < 2:
            return None, None

        months_list = sorted(monthly_prod["month"].unique())
        month_map = {m: i for i, m in enumerate(months_list)}
        last_month_dt = datetime.strptime(months_list[-1], "%Y-%m")
        future_months = [(last_month_dt + timedelta(days=30 * i)).strftime("%Y-%m") for i in range(1, 4)]
        future_idxs = [len(months_list) + i for i in range(1, 4)]

        historical_list, forecast_list = [], []

        for product in monthly_prod["product"].unique():
            prod_data = monthly_prod[monthly_prod["product"] == product].copy()
            prod_data["month_idx"] = prod_data["month"].map(month_map)
            X = prod_data["month_idx"].values.reshape(-1, 1)
            y = prod_data["revenue"].values
            model = LinearRegression()
            model.fit(X, y)

            for _, row in prod_data.iterrows():
                historical_list.append({"product": product, "month": row["month"], "revenue": row["revenue"]})

            for fm, fi in zip(future_months, future_idxs):
                pred = max(0, float(model.predict([[fi]])[0]))
                forecast_list.append({"product": product, "month": fm, "predicted_revenue": round(pred, 2)})

        return pd.DataFrame(forecast_list), pd.DataFrame(historical_list)

    except Exception:
        return None, None


def inventory_optimization(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        if "month" not in df.columns:
            df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

        monthly_demand = df.groupby(["product", "month"])["quantity"].sum().reset_index()
        results = []

        for product in monthly_demand["product"].unique():
            vals = monthly_demand[monthly_demand["product"] == product]["quantity"]
            avg_demand = float(vals.mean())
            std_demand = float(vals.std()) if len(vals) > 1 else avg_demand * 0.2
            if np.isnan(std_demand):
                std_demand = avg_demand * 0.2
            safety_stock = 2 * std_demand
            reorder_point = avg_demand + safety_stock

            prod_rows = df[df["product"] == product]
            current_stock = float(prod_rows["stock_level"].iloc[-1]) if "stock_level" in df.columns and len(prod_rows) > 0 else avg_demand * 3

            results.append({
                "product": product,
                "avg_demand": round(avg_demand, 1),
                "std_demand": round(std_demand, 1),
                "safety_stock": round(safety_stock, 1),
                "reorder_point": round(reorder_point, 1),
                "current_stock": round(current_stock, 0),
                "status": "Low Stock" if current_stock < reorder_point else "OK",
            })

        return pd.DataFrame(results)

    except Exception:
        return pd.DataFrame()


def customer_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        reference_date = df["date"].max() + timedelta(days=1)

        rfm = df.groupby("customer").agg(
            recency=("date", lambda x: (reference_date - x.max()).days),
            frequency=("date", "count"),
            monetary=("revenue", "sum"),
        ).reset_index()

        for col in ["recency", "frequency", "monetary"]:
            if rfm[col].nunique() < 2:
                rfm[f"{col[0]}_score"] = 2
                continue
            if col == "recency":
                labels = [4, 3, 2, 1]
            else:
                labels = [1, 2, 3, 4]
            try:
                rfm[f"{col[0]}_score"] = pd.qcut(
                    rfm[col].rank(method="first"), q=4, labels=labels, duplicates="drop"
                ).astype(int)
            except Exception:
                rfm[f"{col[0]}_score"] = 2

        rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

        def segment(s):
            if s >= 10: return "VIP"
            elif s >= 7: return "Premium"
            elif s >= 5: return "Regular"
            else: return "New"

        rfm["segment"] = rfm["rfm_score"].apply(segment)
        rfm["recency"] = rfm["recency"].astype(int)
        rfm["frequency"] = rfm["frequency"].astype(int)
        rfm["monetary"] = rfm["monetary"].round(2)
        return rfm

    except Exception:
        return pd.DataFrame()


def business_health_score(df: pd.DataFrame) -> dict:
    try:
        df = df.copy()
        if "month" not in df.columns:
            df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

        monthly = df.groupby("month")["revenue"].sum()
        n_months = len(monthly)
        n_customers = int(df["customer"].nunique()) if "customer" in df.columns else 1
        n_products = int(df["product"].nunique()) if "product" in df.columns else 1

        score = 0
        recs = []
        growth_pct = 0.0

        if n_months >= 2:
            changes = monthly.pct_change().dropna()
            growth_pct = float(changes.mean() * 100)
            if growth_pct > 5:
                score += 30; recs.append("Strong revenue growth — maintain current strategy.")
            elif growth_pct > 0:
                score += 20; recs.append("Moderate growth — consider expanding top products.")
            else:
                score += 5; recs.append("Revenue declining — review pricing and marketing.")
        else:
            score += 15; recs.append("More data needed for trend analysis.")

        if n_months >= 3:
            cv = float(monthly.std() / monthly.mean()) if monthly.mean() > 0 else 1
            if cv < 0.2:
                score += 25; recs.append("Highly consistent revenue — excellent stability.")
            elif cv < 0.4:
                score += 15; recs.append("Some volatility — consider stabilization strategies.")
            else:
                score += 5; recs.append("High volatility — diversify customer base.")
        else:
            score += 15

        if n_customers >= 15:
            score += 25; recs.append("Good customer diversity.")
        elif n_customers >= 10:
            score += 15; recs.append("Moderate base — aim to acquire more customers.")
        else:
            score += 5; recs.append("Limited customer base — focus on acquisition.")

        if n_products >= 4:
            score += 20; recs.append("Strong product portfolio.")
        elif n_products >= 2:
            score += 12; recs.append("Consider expanding product range.")
        else:
            score += 5; recs.append("Single product — consider diversification.")

        score = min(100, score)
        if score >= 75: status = "Excellent"
        elif score >= 55: status = "Good"
        elif score >= 35: status = "Fair"
        else: status = "Critical"

        return {"score": score, "status": status, "recommendations": recs, "growth_pct": round(growth_pct, 2)}

    except Exception:
        return {"score": 50, "status": "Fair", "recommendations": ["Unable to compute health score."], "growth_pct": 0}
