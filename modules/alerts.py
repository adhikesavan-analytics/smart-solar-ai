"""Alert system. Smart triggers + email to user + admin (SMTP_USER)."""
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd

from modules import db


REVENUE_HINTS = ["revenue", "sales", "income"]
EFF_HINTS = ["efficiency", "yield", "performance_ratio", "uptime"]
EXP_HINTS = ["expense", "expenses", "cost", "spend"]
DEBT_HINTS = ["debt", "outstanding", "overdue", "receivable"]
STOCK_HINTS = ["stock_level", "stock", "inventory", "qty_on_hand"]
DATE_HINTS = ["date", "month", "period", "timestamp"]


def _col(df, hints):
    cols = {c.lower(): c for c in df.columns}
    for h in hints:
        if h in cols: return cols[h]
    for h in hints:
        for k in cols:
            if h in k: return cols[k]
    return None


def _build_alerts(df: pd.DataFrame):
    alerts = []
    eff = _col(df, EFF_HINTS)
    if eff:
        try:
            vals = pd.to_numeric(df[eff], errors="coerce").dropna()
            if len(vals):
                # treat 0-1 as fraction, scale to %
                if vals.max() <= 1.5: vals = vals * 100
                avg = float(vals.mean())
                if avg < 75:
                    alerts.append(dict(
                        severity="critical",
                        category="Efficiency",
                        message=f"Average {eff} is {avg:.1f}% — below the 75% threshold.",
                        action="Inspect under-performing assets, schedule preventive maintenance, and audit recent operational changes.",
                    ))
        except Exception:
            pass

    rev = _col(df, REVENUE_HINTS)
    date = _col(df, DATE_HINTS)
    if rev and date:
        try:
            tdf = df.copy()
            if date != "month":
                tdf["__period__"] = pd.to_datetime(tdf[date], errors="coerce").dt.to_period("M").astype(str)
                key = "__period__"
            else:
                key = date
            tdf["__rev__"] = pd.to_numeric(tdf[rev], errors="coerce")
            grp = tdf.groupby(key)["__rev__"].sum().sort_index()
            grp = grp[grp.index.astype(str) != "NaT"]
            if len(grp) >= 2:
                last, prev = float(grp.iloc[-1]), float(grp.iloc[-2])
                if prev > 0:
                    delta = (last - prev) / prev * 100
                    if delta < -20:
                        alerts.append(dict(
                            severity="critical",
                            category="Revenue Drop",
                            message=f"Revenue fell {delta:.1f}% vs the prior period (${prev:,.0f} → ${last:,.0f}).",
                            action="Review pricing, sales pipeline, and customer churn for the most recent period.",
                        ))
        except Exception:
            pass

    # Anomaly detection on numeric columns
    try:
        for col in df.select_dtypes(include="number").columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v) < 8: continue
            z = (v - v.mean()) / (v.std() or 1)
            spikes = int((z.abs() > 3).sum())
            if spikes:
                alerts.append(dict(
                    severity="warning",
                    category="Unusual Pattern",
                    message=f"{spikes} extreme value(s) detected in `{col}` (>3σ from mean).",
                    action=f"Inspect outliers in `{col}` — they may indicate data-entry errors or genuine anomalies.",
                ))
    except Exception:
        pass

    stock = _col(df, STOCK_HINTS)
    if stock:
        try:
            low = (pd.to_numeric(df[stock], errors="coerce") < 10).sum()
            if int(low):
                alerts.append(dict(
                    severity="warning",
                    category="Inventory",
                    message=f"{int(low)} record(s) show stock below 10 units.",
                    action="Reorder affected products and review lead times.",
                ))
        except Exception:
            pass

    exp = _col(df, EXP_HINTS)
    if exp and rev:
        try:
            e = pd.to_numeric(df[exp], errors="coerce").sum()
            r = pd.to_numeric(df[rev], errors="coerce").sum()
            if r > 0 and e / r > 0.85:
                alerts.append(dict(
                    severity="critical",
                    category="Finance",
                    message=f"Expenses are {e/r*100:.1f}% of revenue — margin at risk.",
                    action="Identify the top three expense categories and renegotiate or cut discretionary spend.",
                ))
        except Exception:
            pass

    debt = _col(df, DEBT_HINTS)
    if debt:
        try:
            bad = int((pd.to_numeric(df[debt], errors="coerce") > 0).sum())
            if bad:
                alerts.append(dict(
                    severity="critical" if bad > 10 else "warning",
                    category="Bad Debt",
                    message=f"{bad} record(s) carry outstanding debt.",
                    action="Trigger collections workflow on overdue accounts.",
                ))
        except Exception:
            pass

    return alerts


def scan(df: pd.DataFrame, company_id: int, department_id: int = None,
         dataset_id: int = None, user_email: str = None):
    alerts = _build_alerts(df)
    company = db.get_company(company_id) or {}
    dept = db.get_department(department_id) or {}
    company_name = company.get("name", "—")
    dept_name = dept.get("name", "All departments")

    for a in alerts:
        db.add_alert(
            company_id=company_id, department_id=department_id,
            dataset_id=dataset_id, severity=a["severity"],
            category=a["category"], message=a["message"],
            suggested_action=a["action"],
        )

    # Email dispatch
    if alerts:
        recipients = set()
        if user_email: recipients.add(user_email)
        for e in db.admin_emails(): recipients.add(e)
        admin_smtp = os.environ.get("SMTP_USER")
        if admin_smtp: recipients.add(admin_smtp)
        if company.get("email"): recipients.add(company["email"])
        recipients = [r for r in recipients if r]

        body_lines = [
            f"Company: {company_name}",
            f"Department: {dept_name}",
            f"Alerts triggered: {len(alerts)}",
            "",
        ]
        for a in alerts:
            body_lines += [
                f"[{a['severity'].upper()}] {a['category']}",
                f"  Issue:   {a['message']}",
                f"  Action:  {a['action']}",
                "",
            ]
        body = "\n".join(body_lines)
        subject = f"[Smart BI] {len(alerts)} alert(s) for {company_name} / {dept_name}"

        sent_to = []
        for r in recipients:
            try:
                if send_email(r, subject, body):
                    sent_to.append(r)
            except Exception:
                pass
        return alerts, sent_to

    return alerts, []


def send_email(to_addr: str, subject: str, body: str) -> bool:
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587") or "587")
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASSWORD")
    sender = os.environ.get("SMTP_FROM", user or "alerts@smartbi.local")
    if not host or not user or not pw:
        return False
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=20, context=ctx) as s:
            s.login(user, pw)
            s.sendmail(sender, [to_addr], msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls(context=ctx)
            s.login(user, pw)
            s.sendmail(sender, [to_addr], msg.as_string())
    return True


def smtp_configured() -> bool:
    return all([os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD")])
