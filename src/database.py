"""SQLite store for uploaded retraining data + retrain history."""
import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "db" / "app.db"


def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH, timeout=30)


def init_db():
    with closing(_conn()) as c, c:
        c.execute("""CREATE TABLE IF NOT EXISTS uploads(
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, label TEXT, path TEXT,
            used_in_training INTEGER DEFAULT 0, uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS retrain_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            n_new_images INTEGER, accuracy REAL, f1 REAL, recall REAL, status TEXT)""")


def add_upload(filename, label, path):
    with closing(_conn()) as c, c:
        c.execute("INSERT INTO uploads(filename,label,path) VALUES(?,?,?)", (filename, label, str(path)))


def pending_count() -> int:
    with closing(_conn()) as c:
        return c.execute("SELECT COUNT(*) FROM uploads WHERE used_in_training=0").fetchone()[0]


def total_uploads() -> int:
    with closing(_conn()) as c:
        return c.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]


def mark_used():
    with closing(_conn()) as c, c:
        c.execute("UPDATE uploads SET used_in_training=1 WHERE used_in_training=0")


def log_run(n_new, acc, f1, recall, status):
    with closing(_conn()) as c, c:
        c.execute("INSERT INTO retrain_runs(n_new_images,accuracy,f1,recall,status) VALUES(?,?,?,?,?)",
                  (n_new, acc, f1, recall, status))


def run_history(limit=10):
    with closing(_conn()) as c:
        rows = c.execute("SELECT started_at,n_new_images,accuracy,f1,recall,status FROM retrain_runs "
                         "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(zip(["started_at", "n_new_images", "accuracy", "f1", "recall", "status"], r)) for r in rows]
