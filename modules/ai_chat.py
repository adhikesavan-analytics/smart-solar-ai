import pandas as pd
import plotly.express as px

from modules.analytics import revenue_by_product, revenue_by_district, monthly_trend, compute_kpis


def get_ai_response(question: str, df: pd.DataFrame):
    try:
        q = question.lower().strip()
        chart = None

        kpis = compute_kpis(df)
        prod_df = revenue_by_product(df)
        dist_df = revenue_by_district(df)
        trend_df = monthly_trend(df)

        top_product = prod_df.iloc[0]["product"] if not prod_df.empty else "N/A"
        top_product_rev = float(prod_df.iloc[0]["revenue"]) if not prod_df.empty else 0
        top_district = dist_df.iloc[0]["district"] if not dist_df.empty else "N/A"
        top_district_rev = float(dist_df.iloc[0]["revenue"]) if not dist_df.empty else 0
        total_rev = kpis["total_revenue"]

        if any(w in q for w in ["top product", "best product", "highest product", "top selling"]):
            pct = (top_product_rev / total_rev * 100) if total_rev > 0 else 0
            response = (f"The top-performing product is **{top_product}** with total revenue of "
                        f"**${top_product_rev:,.0f}** ({pct:.1f}% of total revenue).")
            if not prod_df.empty:
                chart = px.bar(prod_df, x="product", y="revenue", title="Revenue by Product",
                               color="revenue", color_continuous_scale="Blues")

        elif any(w in q for w in ["district", "region", "area", "location"]):
            pct = (top_district_rev / total_rev * 100) if total_rev > 0 else 0
            response = (f"The highest revenue district is **{top_district}** with "
                        f"**${top_district_rev:,.0f}** ({pct:.1f}% of total sales).")
            if not dist_df.empty:
                chart = px.pie(dist_df, names="district", values="revenue", title="Revenue by District")

        elif any(w in q for w in ["revenue", "total sales", "how much", "earnings"]):
            response = (f"Total revenue is **${total_rev:,.0f}** from "
                        f"**{kpis['total_quantity']:,}** units sold. "
                        f"MoM growth: **{kpis['revenue_growth']:+.1f}%**.")
            if not trend_df.empty:
                chart = px.area(trend_df, x="month", y="revenue", title="Monthly Revenue Trend")

        elif any(w in q for w in ["customer", "client", "buyer"]):
            avg = total_rev / kpis["total_customers"] if kpis["total_customers"] > 0 else 0
            response = (f"There are **{kpis['total_customers']}** unique customers. "
                        f"Average revenue per customer: **${avg:,.0f}**.")

        elif any(w in q for w in ["trend", "growth", "monthly", "month"]):
            if not trend_df.empty:
                last = trend_df.iloc[-1]
                response = (f"Latest month (**{last['month']}**): **${float(last['revenue']):,.0f}** in revenue. "
                             f"Overall MoM growth: **{kpis['revenue_growth']:+.1f}%**.")
                chart = px.line(trend_df, x="month", y="revenue", title="Monthly Revenue Trend", markers=True)
            else:
                response = "Not enough data to analyze trends."

        elif any(w in q for w in ["recommend", "suggest", "advice", "improve"]):
            response = (f"Based on your data:\n\n"
                        f"1. Focus on **{top_product}** — your highest revenue product.\n"
                        f"2. Expand in **{top_district}** — your top-performing district.\n"
                        f"3. Revenue growth: **{kpis['revenue_growth']:+.1f}%** MoM.")

        elif any(w in q for w in ["hello", "hi", "hey", "help"]):
            response = ("Hello! I'm your **Smart Solar AI** analyst. Ask me about:\n\n"
                        "- Revenue and sales\n- Top products and districts\n"
                        "- Monthly trends\n- Customer insights\n- Recommendations")

        else:
            response = (f"Quick summary:\n\n"
                        f"- **Revenue**: ${total_rev:,.0f}\n"
                        f"- **Top Product**: {top_product} (${top_product_rev:,.0f})\n"
                        f"- **Best District**: {top_district} (${top_district_rev:,.0f})\n"
                        f"- **Customers**: {kpis['total_customers']}\n"
                        f"- **MoM Growth**: {kpis['revenue_growth']:+.1f}%")

        return response, chart

    except Exception as e:
        return f"Error processing question: {e}", None
