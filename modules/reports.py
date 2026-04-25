import pandas as pd
import io

from modules.analytics import revenue_by_product, revenue_by_district, monthly_trend, compute_kpis
from modules.ai_modules import inventory_optimization, customer_segmentation


def export_excel(df: pd.DataFrame) -> bytes:
    try:
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sales Data", index=False)

            try:
                revenue_by_product(df).to_excel(writer, sheet_name="Products Summary", index=False)
            except Exception:
                pass

            try:
                revenue_by_district(df).to_excel(writer, sheet_name="District Summary", index=False)
            except Exception:
                pass

            try:
                monthly_trend(df).to_excel(writer, sheet_name="Monthly Trend", index=False)
            except Exception:
                pass

            try:
                inv_df = inventory_optimization(df)
                if inv_df is not None and not inv_df.empty:
                    inv_df.to_excel(writer, sheet_name="Inventory", index=False)
            except Exception:
                pass

            try:
                rfm_df = customer_segmentation(df)
                if rfm_df is not None and not rfm_df.empty:
                    cols = [c for c in ["customer", "recency", "frequency", "monetary", "segment"] if c in rfm_df.columns]
                    rfm_df[cols].to_excel(writer, sheet_name="Customer Intelligence", index=False)
            except Exception:
                pass

            try:
                kpis = compute_kpis(df)
                kpi_df = pd.DataFrame([
                    {"Metric": "Total Revenue ($)", "Value": round(kpis["total_revenue"], 2)},
                    {"Metric": "Total Sales Quantity", "Value": kpis["total_quantity"]},
                    {"Metric": "Total Customers", "Value": kpis["total_customers"]},
                    {"Metric": "Total Products", "Value": kpis["total_products"]},
                    {"Metric": "MoM Growth (%)", "Value": kpis["revenue_growth"]},
                ])
                kpi_df.to_excel(writer, sheet_name="KPI Summary", index=False)
            except Exception:
                pass

        output.seek(0)
        return output.read()

    except Exception as e:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sales Data", index=False)
        output.seek(0)
        return output.read()
