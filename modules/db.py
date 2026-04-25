"""Persistence layer.

Backend is auto-selected:
 • If DATABASE_URL is set (e.g. on Streamlit Cloud + Neon/Supabase), use Postgres.
 • Otherwise fall back to local SQLite at app/smartbi.db.

All public function signatures match the previous SQLite-only version,
so the rest of the app needs no changes.
"""
import os
import json
import bcrypt
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ------------------------------------------------------------------
# Engine setup
# ------------------------------------------------------------------
_DB_URL = os.environ.get("DATABASE_URL")
if _DB_URL:
    # SQLAlchemy needs `postgresql://` (or `postgresql+psycopg2://`)
    if _DB_URL.startswith("postgres://"):
        _DB_URL = _DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)
    elif _DB_URL.startswith("postgresql://") and "+psycopg2" not in _DB_URL:
        _DB_URL = _DB_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
    IS_PG = True
    engine: Engine = create_engine(_DB_URL, pool_pre_ping=True, pool_recycle=300)
else:
    IS_PG = False
    sqlite_path = os.environ.get(
        "SMARTBI_DB",
        os.path.join(os.path.dirname(__file__), "..", "smartbi.db"),
    )
    engine = create_engine(f"sqlite:///{sqlite_path}", future=True)


# Helper: pick the right serial-PK and JSON storage column type per backend
PK = "SERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
JSONCOL = "JSONB" if IS_PG else "TEXT"


@contextmanager
def conn():
    with engine.begin() as c:
        yield c


def _exec(c, sql, params=None):
    return c.execute(text(sql), params or {})


def _ensure_column(c, table, col, ddl):
    if IS_PG:
        rows = _exec(c, """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = :t
        """, {"t": table}).fetchall()
        cols = [r[0] for r in rows]
    else:
        rows = _exec(c, f"PRAGMA table_info({table})").fetchall()
        cols = [r[1] for r in rows]
    if col not in cols:
        _exec(c, f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------
def init_db():
    with conn() as c:
        _exec(c, f"""
            CREATE TABLE IF NOT EXISTS companies (
                id {PK},
                name TEXT NOT NULL UNIQUE,
                industry TEXT,
                email TEXT,
                created_at TEXT NOT NULL
            )
        """)
        _exec(c, f"""
            CREATE TABLE IF NOT EXISTS departments (
                id {PK},
                company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                UNIQUE(company_id, name)
            )
        """)
        _exec(c, f"""
            CREATE TABLE IF NOT EXISTS users (
                id {PK},
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                company_id INTEGER REFERENCES companies(id) ON DELETE SET NULL,
                department_id INTEGER REFERENCES departments(id) ON DELETE SET NULL,
                email TEXT,
                created_at TEXT NOT NULL
            )
        """)
        _exec(c, f"""
            CREATE TABLE IF NOT EXISTS datasets (
                id {PK},
                company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
                department_id INTEGER REFERENCES departments(id) ON DELETE SET NULL,
                name TEXT NOT NULL,
                columns_json {JSONCOL} NOT NULL,
                rows_json {JSONCOL} NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        _exec(c, f"""
            CREATE TABLE IF NOT EXISTS alerts (
                id {PK},
                company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
                department_id INTEGER,
                dataset_id INTEGER,
                severity TEXT NOT NULL,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                suggested_action TEXT,
                created_at TEXT NOT NULL
            )
        """)
        # idempotent migrations for older databases
        _ensure_column(c, "users", "department_id", "INTEGER")
        _ensure_column(c, "datasets", "department_id", "INTEGER")
        _ensure_column(c, "alerts", "department_id", "INTEGER")
        _ensure_column(c, "alerts", "suggested_action", "TEXT")
    seed()


# ------------------------------------------------------------------
# JSON helpers (Postgres returns dict/list, SQLite returns str)
# ------------------------------------------------------------------
def _to_json(val):
    if IS_PG: return val  # JSONB column accepts dict/list directly via psycopg2 only with json.dumps. Safer to dumps:
    return json.dumps(val, default=str)


def _store_json(val):
    """Always serialize before insert; both backends accept TEXT/JSONB string."""
    return json.dumps(val, default=str)


def _load_json(val):
    if val is None: return None
    if isinstance(val, (dict, list)): return val
    return json.loads(val)


# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------
def hash_password(p): return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()


def verify_password(p, h):
    try: return bcrypt.checkpw(p.encode(), h.encode())
    except Exception: return False


# ------------------------------------------------------------------
# Seed default company / users
# ------------------------------------------------------------------
def seed():
    with conn() as c:
        n = _exec(c, "SELECT COUNT(*) FROM users").scalar()
        if n and int(n) > 0:
            return
        now = datetime.utcnow().isoformat()
        _exec(c,
            "INSERT INTO companies (name, industry, email, created_at) "
            "VALUES (:n, :i, :e, :t)",
            {"n": "Smart Solar AI", "i": "Renewable Energy",
             "e": os.environ.get("SMTP_FROM") or "ops@smartsolar.example", "t": now},
        )
        cid = _exec(c, "SELECT id FROM companies WHERE name = :n",
                    {"n": "Smart Solar AI"}).scalar()
        dept_ids = {}
        for d in ["Sales", "Finance", "Inventory", "Operations"]:
            _exec(c,
                "INSERT INTO departments (company_id, name) VALUES (:c, :n)",
                {"c": cid, "n": d},
            )
            dept_ids[d] = _exec(
                c, "SELECT id FROM departments WHERE company_id = :c AND name = :n",
                {"c": cid, "n": d},
            ).scalar()
        admin_email = os.environ.get("SMTP_USER") or "admin@example.com"
        _exec(c,
            "INSERT INTO users (username, password_hash, role, company_id, department_id, email, created_at) "
            "VALUES (:u, :p, :r, :c, :d, :e, :t)",
            {"u": "admin", "p": hash_password("admin123"), "r": "admin",
             "c": cid, "d": dept_ids["Operations"], "e": admin_email, "t": now},
        )
        _exec(c,
            "INSERT INTO users (username, password_hash, role, company_id, department_id, email, created_at) "
            "VALUES (:u, :p, :r, :c, :d, :e, :t)",
            {"u": "demo1", "p": hash_password("password123"), "r": "user",
             "c": cid, "d": dept_ids["Sales"], "e": "demo1@example.com", "t": now},
        )


# ------------------------------------------------------------------
# Users
# ------------------------------------------------------------------
def _row_to_dict(row):
    return dict(row._mapping) if row is not None else None


def get_user(username):
    with conn() as c:
        row = _exec(c,
            """SELECT u.*, co.name AS company_name, d.name AS department_name
               FROM users u
               LEFT JOIN companies co ON co.id = u.company_id
               LEFT JOIN departments d ON d.id = u.department_id
               WHERE LOWER(u.username) = LOWER(:u)""",
            {"u": username},
        ).fetchone()
    return _row_to_dict(row)


def list_users():
    with conn() as c:
        rows = _exec(c,
            """SELECT u.id, u.username, u.role, u.company_id, u.department_id,
                      u.email, u.created_at,
                      co.name AS company_name, d.name AS department_name
               FROM users u
               LEFT JOIN companies co ON co.id = u.company_id
               LEFT JOIN departments d ON d.id = u.department_id
               ORDER BY u.id"""
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def create_user(username, password, role, company_id=None, department_id=None, email=None):
    with conn() as c:
        _exec(c,
            """INSERT INTO users (username, password_hash, role, company_id, department_id, email, created_at)
               VALUES (:u, :p, :r, :c, :d, :e, :t)""",
            {"u": username.strip(), "p": hash_password(password), "r": role,
             "c": company_id, "d": department_id, "e": email,
             "t": datetime.utcnow().isoformat()},
        )


def delete_user(uid):
    with conn() as c:
        _exec(c, "DELETE FROM users WHERE id = :i", {"i": uid})


def admin_emails():
    with conn() as c:
        rows = _exec(c,
            "SELECT email FROM users WHERE role = 'admin' AND email IS NOT NULL AND email <> ''"
        ).fetchall()
    return [r[0] for r in rows]


# ------------------------------------------------------------------
# Companies
# ------------------------------------------------------------------
def list_companies():
    with conn() as c:
        rows = _exec(c, "SELECT * FROM companies ORDER BY name").fetchall()
    return [_row_to_dict(r) for r in rows]


def create_company(name, industry=None, email=None):
    with conn() as c:
        _exec(c,
            "INSERT INTO companies (name, industry, email, created_at) VALUES (:n, :i, :e, :t)",
            {"n": name.strip(), "i": industry, "e": email,
             "t": datetime.utcnow().isoformat()},
        )


def get_company(cid):
    with conn() as c:
        row = _exec(c, "SELECT * FROM companies WHERE id = :i", {"i": cid}).fetchone()
    return _row_to_dict(row)


# ------------------------------------------------------------------
# Departments
# ------------------------------------------------------------------
def list_departments(company_id):
    with conn() as c:
        rows = _exec(c,
            "SELECT * FROM departments WHERE company_id = :c ORDER BY name",
            {"c": company_id},
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_department(did):
    if did is None: return None
    with conn() as c:
        row = _exec(c, "SELECT * FROM departments WHERE id = :i", {"i": did}).fetchone()
    return _row_to_dict(row)


def create_department(company_id, name):
    sql = (
        "INSERT INTO departments (company_id, name) VALUES (:c, :n) "
        + ("ON CONFLICT DO NOTHING" if IS_PG else "")
    )
    if IS_PG:
        with conn() as c:
            _exec(c, sql, {"c": company_id, "n": name.strip()})
    else:
        with conn() as c:
            _exec(c,
                "INSERT OR IGNORE INTO departments (company_id, name) VALUES (:c, :n)",
                {"c": company_id, "n": name.strip()},
            )


# ------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------
def list_datasets(company_id, department_id=None, role="user"):
    with conn() as c:
        if department_id and role != "admin":
            rows = _exec(c,
                "SELECT id, company_id, department_id, name, columns_json, created_at "
                "FROM datasets WHERE company_id = :c "
                "AND (department_id = :d OR department_id IS NULL) "
                "ORDER BY id DESC",
                {"c": company_id, "d": department_id},
            ).fetchall()
        elif department_id and role == "admin":
            rows = _exec(c,
                "SELECT id, company_id, department_id, name, columns_json, created_at "
                "FROM datasets WHERE company_id = :c AND department_id = :d "
                "ORDER BY id DESC",
                {"c": company_id, "d": department_id},
            ).fetchall()
        else:
            rows = _exec(c,
                "SELECT id, company_id, department_id, name, columns_json, created_at "
                "FROM datasets WHERE company_id = :c ORDER BY id DESC",
                {"c": company_id},
            ).fetchall()
    out = []
    for r in rows:
        d = _row_to_dict(r)
        d["columns"] = _load_json(d.pop("columns_json"))
        out.append(d)
    return out


def save_dataset(company_id, department_id, name, columns, rows):
    sql = (
        "INSERT INTO datasets (company_id, department_id, name, columns_json, rows_json, created_at) "
        "VALUES (:c, :d, :n, :cj, :rj, :t)"
        + (" RETURNING id" if IS_PG else "")
    )
    with conn() as c:
        res = _exec(c, sql, {
            "c": company_id, "d": department_id, "n": name,
            "cj": _store_json(list(columns)),
            "rj": _store_json(rows),
            "t": datetime.utcnow().isoformat(),
        })
        if IS_PG:
            return res.scalar()
        return _exec(c, "SELECT last_insert_rowid()").scalar()


def load_dataset_rows(dataset_id):
    with conn() as c:
        row = _exec(c,
            "SELECT name, company_id, department_id, columns_json, rows_json "
            "FROM datasets WHERE id = :i", {"i": dataset_id},
        ).fetchone()
    if not row: return None
    d = _row_to_dict(row)
    return {
        "name": d["name"],
        "company_id": d["company_id"],
        "department_id": d["department_id"],
        "columns": _load_json(d["columns_json"]),
        "rows": _load_json(d["rows_json"]),
    }


def delete_dataset(did):
    with conn() as c:
        _exec(c, "DELETE FROM datasets WHERE id = :i", {"i": did})


# ------------------------------------------------------------------
# Alerts
# ------------------------------------------------------------------
def list_alerts(company_id=None, department_id=None, limit=200):
    with conn() as c:
        if company_id and department_id:
            rows = _exec(c,
                "SELECT * FROM alerts WHERE company_id = :c "
                "AND (department_id = :d OR department_id IS NULL) "
                "ORDER BY id DESC LIMIT :l",
                {"c": company_id, "d": department_id, "l": limit},
            ).fetchall()
        elif company_id:
            rows = _exec(c,
                "SELECT * FROM alerts WHERE company_id = :c ORDER BY id DESC LIMIT :l",
                {"c": company_id, "l": limit},
            ).fetchall()
        else:
            rows = _exec(c,
                "SELECT * FROM alerts ORDER BY id DESC LIMIT :l", {"l": limit},
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def add_alert(company_id, department_id, dataset_id, severity, category, message, suggested_action=None):
    with conn() as c:
        _exec(c,
            """INSERT INTO alerts (company_id, department_id, dataset_id, severity,
                                   category, message, suggested_action, created_at)
               VALUES (:c, :d, :ds, :s, :cat, :m, :a, :t)""",
            {"c": company_id, "d": department_id, "ds": dataset_id,
             "s": severity, "cat": category, "m": message,
             "a": suggested_action, "t": datetime.utcnow().isoformat()},
        )


def clear_alerts(company_id, department_id=None):
    with conn() as c:
        if department_id:
            _exec(c,
                "DELETE FROM alerts WHERE company_id = :c AND department_id = :d",
                {"c": company_id, "d": department_id},
            )
        else:
            _exec(c, "DELETE FROM alerts WHERE company_id = :c", {"c": company_id})


# ------------------------------------------------------------------
# Cross-company analytics (admin)
# ------------------------------------------------------------------
def company_revenue_summary():
    out = []
    with conn() as c:
        cos = _exec(c, "SELECT * FROM companies").fetchall()
        for co in cos:
            co_d = _row_to_dict(co)
            ds_rows = _exec(c,
                "SELECT rows_json FROM datasets WHERE company_id = :c", {"c": co_d["id"]}
            ).fetchall()
            total_rev, total_rows, ds_count = 0.0, 0, 0
            for ds in ds_rows:
                try:
                    rows = _load_json(ds[0])
                    ds_count += 1
                    total_rows += len(rows)
                    for row in rows:
                        for k, v in row.items():
                            kl = str(k).lower()
                            if any(h in kl for h in ("revenue", "sales", "income")):
                                try: total_rev += float(v)
                                except Exception: pass
                                break
                except Exception:
                    continue
            out.append({
                "company": co_d["name"],
                "industry": co_d.get("industry") or "—",
                "datasets": ds_count,
                "rows": total_rows,
                "revenue": round(total_rev, 2),
            })
    return out
