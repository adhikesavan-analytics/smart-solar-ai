import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(
    page_title="Smart Solar AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# legacy modules
try:
    from modules.auth import authenticate
    from modules.data_generator import generate_sample_data
    from modules.analytics import (
        revenue_by_product, revenue_by_district, monthly_trend, pivot_table, growth_rates,
    )
    from modules.ai_modules import (
        demand_forecast, inventory_optimization,
        customer_segmentation, business_health_score,
    )
    from modules.ai_chat import get_ai_response
    from modules.reports import export_excel
    MODULES_OK = True
    MODULE_ERROR = ""
except Exception as e:
    MODULES_OK = False
    MODULE_ERROR = str(e)

# new modules
from modules import db
from modules.upload import read_csv, numeric_columns, categorical_columns
from modules.stats import describe, correlation_matrix, correlation_insights
from modules.financial import financial_report
from modules.predictions import forecast as ml_forecast
from modules.alerts import scan as scan_alerts, smtp_configured
from modules.dynamic import build as build_chart, CHART_TYPES
from modules.cleaning import clean as clean_dataset
from modules.kpi_universal import kpis as universal_kpis, detect as detect_cols

db.init_db()


# ============================================================
# Auth
# ============================================================
def login_page():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("## ☀️ Smart Solar AI")
        st.markdown("#### Business Intelligence Platform")
        st.markdown("---")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True, type="primary"):
            user = authenticate(username, password)
            if user:
                st.session_state["user"] = user
                st.session_state["logged_in"] = True
                st.session_state["company_id"] = user["company_id"]
                st.session_state["department_id"] = user["department_id"]
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.caption("There is no public sign-up. Contact your administrator for access.")


# ============================================================
# Sidebar
# ============================================================
def sidebar():
    user = st.session_state["user"]
    role = user["role"]

    with st.sidebar:
        st.markdown("## ☀️ Smart Solar AI")
        st.caption(
            f"**{user['username']}** · {role}  \n"
            f"{user.get('company_name') or '—'} / {user.get('department_name') or '—'}  \n"
            f"{user.get('email') or 'no email'}"
        )
        st.markdown("---")

        # Company / department picker (admin can switch; users locked to their own)
        companies = db.list_companies()
        if role == "admin":
            options = {c["id"]: c["name"] for c in companies}
        else:
            options = {c["id"]: c["name"] for c in companies if c["id"] == user["company_id"]}

        if not options:
            st.warning("No companies available. Ask an admin to create one.")
            st.stop()

        ids = list(options.keys())
        current = st.session_state.get("company_id") or ids[0]
        if current not in ids: current = ids[0]
        sel_co = st.selectbox("Company", ids, index=ids.index(current),
                              format_func=lambda i: options[i],
                              disabled=(role != "admin"))
        st.session_state["company_id"] = sel_co

        depts = db.list_departments(sel_co)
        if role == "admin":
            dept_opts = {None: "All departments"} | {d["id"]: d["name"] for d in depts}
            current_d = st.session_state.get("department_id")
            if current_d not in dept_opts: current_d = None
            sel_dept = st.selectbox("Department", list(dept_opts.keys()),
                                    index=list(dept_opts.keys()).index(current_d),
                                    format_func=lambda k: dept_opts[k])
        else:
            dept_opts = {user["department_id"]: user.get("department_name") or "(your dept)"}
            sel_dept = user["department_id"]
            st.selectbox("Department", list(dept_opts.keys()),
                         format_func=lambda k: dept_opts[k], disabled=True)
        st.session_state["department_id"] = sel_dept

        st.markdown("---")
        st.markdown("### 📂 Data source")
        st.session_state["auto_clean"] = st.toggle("Auto-clean on upload", value=True)

        saved = db.list_datasets(sel_co, sel_dept, role=role)
        ds_opts = {None: "— select —", "sample": "Sample data"} | {
            d["id"]: f"{d['name']} ({len(d['columns'])} cols)" for d in saved
        }
        chosen = st.selectbox("Saved datasets", list(ds_opts.keys()),
                              format_func=lambda k: ds_opts[k], key="ds_select")
        if chosen == "sample":
            st.session_state["df"] = generate_sample_data() if MODULES_OK else pd.DataFrame()
            st.session_state["df_name"] = "Sample data"
            st.session_state["dataset_id"] = None
        elif isinstance(chosen, int):
            payload = db.load_dataset_rows(chosen)
            if payload:
                st.session_state["df"] = pd.DataFrame(payload["rows"])
                st.session_state["df_name"] = payload["name"]
                st.session_state["dataset_id"] = chosen

        st.markdown("**Upload CSV(s)**")
        uploads = st.file_uploader(
            "Drop one or more CSV files", type=["csv"],
            accept_multiple_files=True, key="uploader",
        )
        if uploads:
            for up in uploads:
                try:
                    raw = read_csv(up)
                    if st.session_state.get("auto_clean", True):
                        cleaned, log, missing = clean_dataset(raw, required=[])
                        st.session_state["last_clean_log"] = log
                        st.session_state["last_clean_before"] = raw.head(20)
                        st.session_state["last_clean_after"] = cleaned.head(20)
                        df_to_save = cleaned
                    else:
                        df_to_save = raw
                        st.session_state["last_clean_log"] = ["Auto-clean was disabled — raw data saved as-is."]
                        st.session_state["last_clean_before"] = raw.head(20)
                        st.session_state["last_clean_after"] = raw.head(20)
                    new_id = db.save_dataset(
                        company_id=sel_co, department_id=sel_dept,
                        name=up.name, columns=list(df_to_save.columns),
                        rows=df_to_save.to_dict(orient="records"),
                    )
                    st.session_state["df"] = df_to_save
                    st.session_state["df_name"] = up.name
                    st.session_state["dataset_id"] = new_id
                    st.success(f"Saved {up.name} — {len(df_to_save)} rows × {len(df_to_save.columns)} cols")
                except Exception as e:
                    st.error(f"{up.name}: {e}")

        st.markdown("---")
        st.caption("📧 Email alerts: " + ("✅ configured" if smtp_configured() else "❌ not configured"))
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ============================================================
# Tabs
# ============================================================
def tab_dashboard(df):
    st.subheader("📊 Business Overview")
    k = universal_kpis(df)
    detected = k["detected"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Revenue", f"${k['total_revenue']:,.0f}")
    c2.metric("📦 Quantity", f"{k['total_quantity']:,}")
    c3.metric("👥 Customers", f"{k['total_customers']:,}")
    c4.metric("🛒 Products", f"{k['total_products']:,}")
    c5.metric("📈 MoM Growth", f"{k['revenue_growth']:+.1f}%")

    with st.expander("Detected columns"):
        st.json({kk: vv for kk, vv in detected.items() if vv})

    st.markdown("---")
    rev_col = detected.get("revenue")
    prod_col = detected.get("product")
    region_col = detected.get("region")
    date_col = detected.get("date")

    col1, col2 = st.columns(2)
    if rev_col and prod_col:
        with col1:
            grp = df.groupby(prod_col)[rev_col].sum().reset_index().sort_values(rev_col, ascending=False)
            fig = px.bar(grp.head(15), x=prod_col, y=rev_col, title=f"Revenue by {prod_col}",
                         color=rev_col, color_continuous_scale="Blues", text_auto=".2s")
            fig.update_layout(showlegend=False, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
    if rev_col and region_col:
        with col2:
            grp = df.groupby(region_col)[rev_col].sum().reset_index()
            fig = px.pie(grp, names=region_col, values=rev_col,
                         title=f"Revenue by {region_col}", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    if rev_col and date_col:
        try:
            tdf = df.copy()
            if date_col != "month":
                tdf["__period__"] = pd.to_datetime(tdf[date_col], errors="coerce").dt.to_period("M").astype(str)
                key = "__period__"
            else:
                key = date_col
            tr = tdf.groupby(key)[rev_col].sum().reset_index().sort_values(key)
            fig = px.line(tr, x=key, y=rev_col, title="Revenue Trend",
                          markers=True, line_shape="spline")
            fig.update_traces(line_color="#1f77b4", line_width=2)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    if not rev_col:
        st.info("No revenue-like column detected. Try the **Dynamic Charts** tab to build a custom view.")


def tab_clean(df):
    st.subheader("🧼 Data Cleaning")
    st.caption("Run the full cleaning pipeline on the active dataset.")

    c1, c2 = st.columns(2)
    with c1:
        outlier = st.selectbox("Outlier handling", ["cap", "remove", "none"], index=0)
    with c2:
        req_text = st.text_input("Required columns (comma separated, optional)", "")

    if st.button("🚿 Run cleaning pipeline", type="primary"):
        required = [r.strip() for r in req_text.split(",") if r.strip()]
        cleaned, log, missing = clean_dataset(df, required=required, outlier_mode=outlier)
        st.session_state["last_clean_log"] = log
        st.session_state["last_clean_before"] = df.head(50)
        st.session_state["last_clean_after"] = cleaned.head(50)

        if st.session_state.get("dataset_id"):
            try:
                db.save_dataset(
                    company_id=st.session_state["company_id"],
                    department_id=st.session_state["department_id"],
                    name=f"{st.session_state.get('df_name','data')} (cleaned)",
                    columns=list(cleaned.columns),
                    rows=cleaned.to_dict(orient="records"),
                )
            except Exception:
                pass
        st.session_state["df"] = cleaned
        st.success(f"Cleaning complete — {len(cleaned)} rows × {len(cleaned.columns)} cols.")

    log = st.session_state.get("last_clean_log")
    if log:
        st.markdown("##### Cleaning log")
        for line in log: st.markdown(f"- {line}")

    before = st.session_state.get("last_clean_before")
    after = st.session_state.get("last_clean_after")
    if before is not None and after is not None:
        cb, ca = st.columns(2)
        cb.markdown("**Before**"); cb.dataframe(before, use_container_width=True)
        ca.markdown("**After**"); ca.dataframe(after, use_container_width=True)

    st.markdown("---")
    st.download_button(
        "📥 Download cleaned CSV",
        data=df.to_csv(index=False),
        file_name=f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


def tab_dynamic(df):
    st.subheader("🧱 Dynamic Dashboard")
    st.caption("Pick any X / Y / color / chart type. Works with any dataset.")

    cols = list(df.columns)
    if not cols:
        st.info("No columns available."); return
    nums = numeric_columns(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1: chart_type = st.selectbox("Chart type", CHART_TYPES)
    with c2: x = st.selectbox("X axis", cols)
    with c3:
        y_options = ["(count)"] + cols
        y_sel = st.selectbox("Y axis", y_options)
        y = None if y_sel == "(count)" else y_sel
    with c4:
        color_options = ["(none)"] + cols
        color = st.selectbox("Color", color_options)
        color = None if color == "(none)" else color
    agg = st.selectbox("Aggregation", ["sum", "mean", "median", "min", "max", "count"], index=0)

    fig = build_chart(df, chart_type, x, y, color, agg)
    if fig is None: st.warning("Couldn't render that combination.")
    else: st.plotly_chart(fig, use_container_width=True)

    if nums:
        with st.expander("Auto-suggested charts"):
            cat = next((c for c in cols if c not in nums), None)
            if cat:
                for n in nums[:4]:
                    f = build_chart(df, "bar", cat, n, None, "sum")
                    if f: st.plotly_chart(f, use_container_width=True)


def tab_stats(df):
    st.subheader("📐 Statistical Analysis")
    desc = describe(df)
    if desc.empty: st.info("No numeric columns to analyze."); return
    st.markdown("##### Descriptive statistics")
    st.dataframe(desc, use_container_width=True)
    cm = correlation_matrix(df)
    if not cm.empty:
        st.markdown("##### Correlation matrix")
        st.plotly_chart(px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, aspect="auto"), use_container_width=True)
        st.markdown("##### Insights")
        for line in correlation_insights(df): st.markdown(f"- {line}")


def tab_financial(df):
    st.subheader("💵 Financial Analysis")
    rep = financial_report(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"${rep['revenue']:,.0f}")
    c2.metric("Cost", f"${rep['cost']:,.0f}")
    c3.metric("Profit", f"${rep['profit']:,.0f}")
    c4.metric("Margin", f"{rep['margin']:.1f}%")
    st.caption(f"Revenue: `{rep['revenue_column']}` · Cost: `{rep['cost_column'] or 'estimated 60% of revenue'}`")
    if rep["timeline"]:
        tl = pd.DataFrame(rep["timeline"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tl["label"], y=tl["revenue"], name="Revenue", marker_color="#1f77b4"))
        fig.add_trace(go.Bar(x=tl["label"], y=tl["cost"], name="Cost", marker_color="#d62728"))
        fig.add_trace(go.Scatter(x=tl["label"], y=tl["profit"], name="Profit",
                                 mode="lines+markers", line=dict(color="#2ca02c", width=3)))
        fig.update_layout(title="Profit & Loss Timeline", barmode="group")
        st.plotly_chart(fig, use_container_width=True)


def tab_predict(df):
    st.subheader("🔮 Predictive Analytics")
    st.caption("Linear-regression forecast on any numeric column.")
    cols = list(df.columns)
    nums = numeric_columns(df)
    if not nums: st.info("Need at least one numeric column."); return
    c1, c2, c3 = st.columns(3)
    with c1: x = st.selectbox("X axis (time / index)", cols, index=0)
    with c2: y = st.selectbox("Y axis (numeric)", nums, index=0)
    with c3: steps = st.slider("Forecast steps", 1, 24, 6)
    try:
        result = ml_forecast(df, x, y, steps)
    except Exception as e:
        st.error(f"Forecast error: {e}"); return
    a, b, c = st.columns(3)
    a.metric("Slope", f"{result['slope']:.3f}")
    b.metric("Intercept", f"{result['intercept']:.3f}")
    c.metric("R²", f"{result['r2']:.3f}")
    hist, fc = pd.DataFrame(result["history"]), pd.DataFrame(result["forecast"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["x"], y=hist["y"], name="Historical",
                             mode="lines+markers", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=fc["x"], y=fc["y"], name="Forecast",
                             mode="lines+markers", line=dict(color="#ff7f0e", dash="dash")))
    fig.update_layout(title=f"{y} forecast", xaxis_title=x, yaxis_title=y)
    st.plotly_chart(fig, use_container_width=True)


def tab_alerts(df):
    st.subheader("🚨 Alerts")
    company_id = st.session_state["company_id"]
    department_id = st.session_state["department_id"]
    user_email = st.session_state["user"].get("email")

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("🔍 Scan current data", type="primary"):
            alerts, sent_to = scan_alerts(
                df, company_id=company_id, department_id=department_id,
                dataset_id=st.session_state.get("dataset_id"), user_email=user_email,
            )
            if not alerts: st.info("No alerts triggered.")
            else:
                msg = f"Triggered {len(alerts)} alert(s)."
                if sent_to: msg += f" Email sent to: {', '.join(sent_to)}."
                elif not smtp_configured(): msg += " (Email not sent — SMTP not configured.)"
                st.success(msg)
    with c2:
        if st.button("Clear all"):
            db.clear_alerts(company_id, department_id if st.session_state["user"]["role"] != "admin" else None)
            st.rerun()
    with c3:
        st.caption(
            "Triggers: efficiency < 75% · revenue drop > 20% MoM · expense spike · 3σ anomaly · low stock · bad debt."
        )

    role = st.session_state["user"]["role"]
    rows = db.list_alerts(
        company_id=company_id,
        department_id=department_id if role != "admin" else None,
        limit=200,
    )
    if not rows: st.info("No alerts yet. Run a scan after uploading data."); return
    for a in rows:
        head = f"**{a['category']}** — {a['message']}"
        body = f"_{a['created_at']}_"
        if a.get("suggested_action"): body = f"💡 {a['suggested_action']}  \n" + body
        text = f"{head}  \n{body}"
        if a["severity"] == "critical": st.error(text)
        elif a["severity"] == "warning": st.warning(text)
        else: st.info(text)


def tab_ai_chat(df):
    st.subheader("🤖 AI Business Analyst")
    st.markdown("Ask questions about your data.")
    if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]): st.write(msg["content"])
    question = st.chat_input("Ask a business question...")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"): st.write(question)
        try:
            if MODULES_OK: response, chart = get_ai_response(question, df)
            else:
                k = universal_kpis(df)
                response = f"Total revenue: ${k['total_revenue']:,.0f}."; chart = None
        except Exception as e:
            response = f"Could not process question: {e}"; chart = None
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
            if chart is not None:
                try: st.plotly_chart(chart, use_container_width=True)
                except Exception: pass


def tab_download(df):
    st.subheader("📥 Download Reports")
    col1, col2 = st.columns(2)
    with col1:
        try:
            if MODULES_OK: excel_data = export_excel(df)
            else:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Data", index=False)
                buf.seek(0); excel_data = buf.read()
            st.download_button(
                "📊 Download Excel Report", data=excel_data,
                file_name=f"SmartBI_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, type="primary",
            )
        except Exception as e:
            st.error(f"Excel export error: {e}")
    with col2:
        st.download_button(
            "📄 Download CSV", data=df.to_csv(index=False),
            file_name=f"SmartBI_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True,
        )
    st.markdown("### Data Preview")
    st.dataframe(df.head(20), use_container_width=True)


def tab_forecast(df):
    try:
        st.subheader("📈 Sales Demand Forecast")
        st.markdown("AI-powered 3-month forecast (legacy product-level).")
        forecast_df, historical_df = (None, None)
        if MODULES_OK:
            try: forecast_df, historical_df = demand_forecast(df)
            except Exception: pass
        if forecast_df is not None and not forecast_df.empty:
            products = ["All Products"] + list(forecast_df["product"].unique())
            sel = st.selectbox("Select Product", products)
            if sel == "All Products":
                hist = historical_df.groupby("month")["revenue"].sum().reset_index()
                fore = forecast_df.groupby("month")["predicted_revenue"].sum().reset_index()
            else:
                hist = historical_df[historical_df["product"] == sel][["month", "revenue"]]
                fore = forecast_df[forecast_df["product"] == sel][["month", "predicted_revenue"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["month"], y=hist["revenue"], mode="lines+markers",
                                     name="Historical", line=dict(color="#1f77b4", width=2)))
            fig.add_trace(go.Scatter(x=fore["month"], y=fore["predicted_revenue"], mode="lines+markers",
                                     name="Forecast", line=dict(color="#ff7f0e", width=2, dash="dash")))
            fig.update_layout(title=f"Revenue Forecast — {sel}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Use the **Predict** tab to forecast any column on any dataset.")
    except Exception as e:
        st.error(f"Forecast error: {e}")


def tab_recommendations(df):
    try:
        st.subheader("💡 Business Recommendations")
        if MODULES_OK:
            try: health = business_health_score(df)
            except Exception:
                health = {"score": 60, "status": "Good", "recommendations": [], "growth_pct": 0}
        else:
            health = {"score": 60, "status": "Good", "recommendations": [], "growth_pct": 0}
        score, status = health["score"], health["status"]
        color = "green" if score >= 75 else "orange" if score >= 50 else "red"
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
                title={"text": "Health Score"},
            ))
            fig.update_layout(height=250); st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(f"#### Status: **{status}**")
            for rec in health.get("recommendations", []): st.markdown(f"- {rec}")
    except Exception as e:
        st.error(f"Recommendations error: {e}")


def tab_advanced(df):
    try:
        st.subheader("🔬 Advanced Analytics")
        if MODULES_OK:
            try:
                piv = pivot_table(df)
                if not piv.empty:
                    st.markdown("### Pivot: Revenue by Product × District")
                    st.dataframe(piv.style.format("${:,.0f}").background_gradient(cmap="Blues"),
                                 use_container_width=True)
            except Exception: pass
        d = detect_cols(df)
        if d.get("revenue") and d.get("date"):
            tdf = df.copy()
            tdf["__rev__"] = pd.to_numeric(tdf[d["revenue"]], errors="coerce")
            tdf["__p__"] = pd.to_datetime(tdf[d["date"]], errors="coerce").dt.to_period("M").astype(str)
            tr = tdf.groupby("__p__")["__rev__"].sum().reset_index()
            st.plotly_chart(px.area(tr, x="__p__", y="__rev__", title="Revenue (Area)"),
                            use_container_width=True)
    except Exception as e:
        st.error(f"Advanced error: {e}")


def tab_inventory(df):
    try:
        st.subheader("📦 Inventory Intelligence")
        inv_df = inventory_optimization(df) if MODULES_OK else pd.DataFrame()
        if inv_df is not None and not inv_df.empty:
            low = inv_df[inv_df["status"] == "Low Stock"] if "status" in inv_df.columns else pd.DataFrame()
            c1, c2 = st.columns(2)
            c1.metric("Total Products", len(inv_df))
            c2.metric("Low Stock Items", len(low))
            st.dataframe(inv_df, use_container_width=True)
        else:
            st.info("Upload data with `product`, `quantity`, `stock_level` columns.")
    except Exception as e:
        st.error(f"Inventory error: {e}")


def tab_customers(df):
    try:
        st.subheader("👥 Customer Intelligence")
        rfm_df = customer_segmentation(df) if MODULES_OK else pd.DataFrame()
        if rfm_df is not None and not rfm_df.empty:
            seg = rfm_df["segment"].value_counts().reset_index()
            seg.columns = ["segment", "count"]
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.pie(seg, names="segment", values="count", title="Customer Segments"),
                                use_container_width=True)
            with col2:
                st.markdown("#### Segment Breakdown")
                total = seg["count"].sum()
                for _, row in seg.iterrows():
                    pct = row["count"] / total * 100
                    st.markdown(f"**{row['segment']}**: {row['count']} ({pct:.1f}%)")
        else:
            st.info("Customer data unavailable.")
    except Exception as e:
        st.error(f"Customers error: {e}")


# ============================================================
# Admin
# ============================================================
def tab_admin():
    st.subheader("🛠️ Admin Console")
    tabs = st.tabs(["Users", "Companies", "Departments", "Compare Companies"])

    with tabs[0]:
        st.markdown("### Create user")
        companies = db.list_companies()
        c_opts = {None: "— none —"} | {c["id"]: c["name"] for c in companies}
        # Company + Department are OUTSIDE the form so the Department list
        # refreshes immediately when Company changes.
        col_a, col_b = st.columns(2)
        with col_a:
            cid = st.selectbox("Company", list(c_opts.keys()),
                               format_func=lambda k: c_opts[k], key="nu_cid")
        d_opts = {None: "— none —"}
        if cid:
            d_opts |= {d["id"]: d["name"] for d in db.list_departments(cid)}
        with col_b:
            did = st.selectbox("Department", list(d_opts.keys()),
                               format_func=lambda k: d_opts[k], key="nu_did")
        with st.form("new_user", clear_on_submit=True):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            email = st.text_input("Email")
            r = st.selectbox("Role", ["user", "admin"])
            if st.form_submit_button("Create user", type="primary"):
                if not u or not p:
                    st.error("Username and password are required.")
                else:
                    try:
                        db.create_user(u, p, r, cid, did, email or None)
                        st.success(f"Created user '{u}' in "
                                   f"{c_opts.get(cid, '—')} / {d_opts.get(did, '—')}")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.markdown("### Users")
        for user in db.list_users():
            cols = st.columns([3, 2, 3, 3, 3, 1])
            cols[0].write(f"**{user['username']}**")
            cols[1].write(user["role"])
            cols[2].write(user.get("company_name") or "—")
            cols[3].write(user.get("department_name") or "—")
            cols[4].write(user.get("email") or "—")
            if user["username"] != "admin":
                if cols[5].button("Delete", key=f"du_{user['id']}"):
                    db.delete_user(user["id"]); st.rerun()

    with tabs[1]:
        st.markdown("### Add company")
        with st.form("new_co"):
            name = st.text_input("Company name")
            ind = st.text_input("Industry")
            email = st.text_input("Contact / alert email")
            if st.form_submit_button("Add company", type="primary"):
                try:
                    db.create_company(name, ind or None, email or None)
                    st.success("Company added"); st.rerun()
                except Exception as e:
                    st.error(str(e))
        st.markdown("### Existing companies")
        for co in db.list_companies():
            st.write(f"**{co['name']}** — {co.get('industry') or '—'} — {co.get('email') or 'no email'}")

    with tabs[2]:
        st.markdown("### Add department")
        companies = db.list_companies()
        if not companies: st.info("Add a company first.")
        else:
            with st.form("new_dept"):
                cid = st.selectbox(
                    "Company", [c["id"] for c in companies],
                    format_func=lambda i: next(c["name"] for c in companies if c["id"] == i),
                )
                dname = st.text_input("Department name")
                if st.form_submit_button("Add department", type="primary"):
                    try:
                        db.create_department(cid, dname)
                        st.success("Department added"); st.rerun()
                    except Exception as e:
                        st.error(str(e))
            st.markdown("### Departments")
            for c in companies:
                ds = db.list_departments(c["id"])
                if ds: st.markdown(f"**{c['name']}**: " + ", ".join(d["name"] for d in ds))

    with tabs[3]:
        st.markdown("### Cross-company performance")
        rows = db.company_revenue_summary()
        if not rows: st.info("No data uploaded yet."); return
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        if df["revenue"].sum() > 0:
            fig = px.bar(df.sort_values("revenue", ascending=False),
                         x="company", y="revenue", color="industry",
                         title="Total Revenue by Company", text_auto=".2s")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Main
# ============================================================
def main():
    if not MODULES_OK:
        st.warning(f"Some legacy modules failed to load: `{MODULE_ERROR}`. Running in fallback mode.")

    if not st.session_state.get("logged_in"):
        login_page(); return

    sidebar()
    df = st.session_state.get("df")
    role = st.session_state["user"]["role"]

    no_data = df is None or (hasattr(df, "empty") and df.empty)
    if no_data:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.info("No dataset loaded yet. Upload a CSV from the sidebar, pick a saved dataset, or load sample data. The tabs below still work — Alerts and Admin are always available.")
        with c2:
            if st.button("📊 Load sample data", use_container_width=True, type="primary"):
                if MODULES_OK:
                    st.session_state["df"] = generate_sample_data()
                    st.session_state["df_name"] = "Sample data"
                    st.session_state["dataset_id"] = None
                    st.rerun()
        df = pd.DataFrame() if df is None else df
    st.caption(
        f"Active dataset: **{st.session_state.get('df_name', 'in-memory')}** · "
        f"{len(df):,} rows × {len(df.columns)} cols  ·  "
        f"Scope: **{st.session_state['user'].get('company_name') or '—'} / "
        f"{st.session_state['user'].get('department_name') or 'all depts' if role=='admin' else st.session_state['user'].get('department_name') or '—'}**"
    )

    base = ["📊 Dashboard", "🧱 Dynamic Charts", "🧼 Clean", "📐 Stats", "💵 Financial",
            "🔮 Predict", "🚨 Alerts", "🤖 AI Chat", "📥 Reports",
            "📈 Forecast", "💡 Recommendations", "🔬 Advanced", "📦 Inventory", "👥 Customers"]
    if role == "admin": base.append("🛠️ Admin")
    tabs = st.tabs(base)

    with tabs[0]: tab_dashboard(df)
    with tabs[1]: tab_dynamic(df)
    with tabs[2]: tab_clean(df)
    with tabs[3]: tab_stats(df)
    with tabs[4]: tab_financial(df)
    with tabs[5]: tab_predict(df)
    with tabs[6]: tab_alerts(df)
    with tabs[7]: tab_ai_chat(df)
    with tabs[8]: tab_download(df)
    with tabs[9]: tab_forecast(df)
    with tabs[10]: tab_recommendations(df)
    with tabs[11]: tab_advanced(df)
    with tabs[12]: tab_inventory(df)
    with tabs[13]: tab_customers(df)
    if role == "admin":
        with tabs[14]: tab_admin()


if __name__ == "__main__":
    main()
