"""Authentication backed by the SQLite users table.

The original USERS dict is kept as a fallback so the legacy
check_login(username, password) -> role contract still works.
"""
from modules import db

USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "demo1": {"password": "password123", "role": "user"},
}


def check_login(username: str, password: str):
    """Returns the user's role string on success, else None."""
    try:
        u = str(username).strip()
        p = str(password).strip()
        try:
            db.init_db()
            user = db.get_user(u)
            if user and db.verify_password(p, user["password_hash"]):
                return user["role"]
        except Exception:
            pass
        # legacy fallback
        legacy = USERS.get(u.lower())
        if legacy and legacy["password"] == p:
            return legacy["role"]
        return None
    except Exception:
        return None


def authenticate(username: str, password: str):
    """Returns the full user dict on success, else None."""
    try:
        db.init_db()
        u = str(username).strip()
        p = str(password).strip()
        user = db.get_user(u)
        if user and db.verify_password(p, user["password_hash"]):
            return user
        return None
    except Exception:
        return None
