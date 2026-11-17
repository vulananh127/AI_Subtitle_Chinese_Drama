import sqlite3

from flask import g

PATH = None
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT,
  password_hash TEXT, google_sub TEXT UNIQUE, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS projects(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), title TEXT NOT NULL,
  src_lang TEXT, tgt_lang TEXT, storage TEXT, video_ref TEXT, video_url TEXT,
  rev INTEGER DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS cues(
  project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE, cue_id TEXT NOT NULL, pos INTEGER NOT NULL,
  start_s REAL, end_s REAL, src TEXT DEFAULT '', text TEXT DEFAULT '', mt TEXT DEFAULT '',
  ok INTEGER DEFAULT 0, origin TEXT DEFAULT 'manual', fmt TEXT DEFAULT '', PRIMARY KEY(project_id, cue_id));
CREATE TABLE IF NOT EXISTS revisions(
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE, user_id INTEGER,
  reason TEXT, n_cues INTEGER, n_ok INTEGER, snapshot TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status TEXT, stage TEXT, progress REAL DEFAULT 0, error TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_rev_project ON revisions(project_id, id);
"""


def connect():
    c = sqlite3.connect(PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init(path):
    global PATH
    PATH = str(path)
    c = connect()
    c.executescript(SCHEMA)
    if 'fmt' not in [r[1] for r in c.execute('PRAGMA table_info(cues)')]:   # nâng cấp DB cũ
        c.execute("ALTER TABLE cues ADD COLUMN fmt TEXT DEFAULT ''")
    c.execute("UPDATE jobs SET status='failed', error='Server đã khởi động lại' WHERE status IN ('queued','running')")
    c.commit()
    c.close()


def conn():
    if "db" not in g:
        g.db = connect()
    return g.db


def close(_=None):
    c = g.pop("db", None)
    if c:
        c.close()


def one(sql, *a):
    return conn().execute(sql, a).fetchone()


def all_(sql, *a):
    return conn().execute(sql, a).fetchall()


def run(sql, *a):
    c = conn()
    cur = c.execute(sql, a)
    c.commit()
    return cur.lastrowid
