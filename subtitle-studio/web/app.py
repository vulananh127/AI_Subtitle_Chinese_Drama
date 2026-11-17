r"""
B1: Tạo và chạy môi trường ảo:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1

Lệnh -Scope Process chỉ có hiệu lực trong cửa sổ này và mất khi đóng, nên không thay đổi cài đặt hệ thống. Khi kích hoạt thành công, đầu dòng lệnh sẽ có chữ (.venv).
Hoặc mở Command Prompt (cmd) thay vì PowerShell và chạy .venv\Scripts\activate.bat

B2: cài các thư viện
pip install -r requirements.txt

B3: chạy ctrinh
python app.py        # mở http://localhost:5000

"""
import json
import os
import re
import secrets
import threading
import uuid
from datetime import timedelta
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, abort, g, jsonify, request, send_file, send_from_directory, session  # noqa: E402
from werkzeug.security import check_password_hash, generate_password_hash  # noqa: E402

import db  # noqa: E402
import model_client  # noqa: E402
import storage  # noqa: E402

BASE = Path(__file__).resolve().parent
DATA = Path(os.getenv("DATA_DIR", BASE / "data"))
DATA.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")
_kf = DATA / "secret.key"
if not os.getenv("SECRET_KEY") and not _kf.exists():
    _kf.write_text(secrets.token_hex(32))
app.secret_key = os.getenv("SECRET_KEY") or _kf.read_text()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    MAX_CONTENT_LENGTH=int(os.getenv("MAX_UPLOAD_MB", "2048")) * 1024 * 1024,
)
db.init(DATA / "app.db")
app.teardown_appcontext(db.close)

REASONS = ("edit", "import", "model", "restore")


def err(msg, code=400):
    return jsonify(error=msg), code


@app.errorhandler(404)
def _404(_):
    return err("Không tìm thấy", 404)


@app.errorhandler(413)
def _413(_):
    return err("File quá lớn", 413)


def login_required(f):
    @wraps(f)
    def w(*a, **k):
        uid = session.get("uid")
        u = db.one("SELECT id,email,name FROM users WHERE id=?", uid) if uid else None
        if not u:
            return err("Chưa đăng nhập", 401)
        g.user = u
        return f(*a, **k)
    return w


def own_project(pid):
    p = db.one("SELECT * FROM projects WHERE id=? AND user_id=?", pid, g.user["id"])
    if not p:
        abort(404)
    return p


# ---------------- Đăng ký / đăng nhập ----------------
def start_session(uid):
    session.clear()
    session["uid"] = uid
    session.permanent = True
    u = db.one("SELECT id,email,name FROM users WHERE id=?", uid)
    return jsonify(id=u["id"], email=u["email"], name=u["name"])


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/config")
def config():
    return jsonify(google_client_id=os.getenv("GOOGLE_CLIENT_ID", ""))


@app.get("/api/me")
@login_required
def me():
    u = db.one("SELECT email,name,created_at,password_hash,google_sub FROM users WHERE id=?", g.user["id"])
    methods = [m for m, ok in (("Email và mật khẩu", u["password_hash"]), ("Google", u["google_sub"])) if ok]
    return jsonify(id=g.user["id"], email=u["email"], name=u["name"], created_at=u["created_at"], methods=methods)


@app.post("/api/auth/register")
def register():
    d = request.get_json(silent=True) or {}
    email = (d.get("email") or "").strip().lower()
    pw = d.get("password") or ""
    name = (d.get("name") or "").strip()[:80] or email.split("@")[0]
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return err("Email không hợp lệ")
    if len(pw) < 8:
        return err("Mật khẩu cần ít nhất 8 ký tự")
    if db.one("SELECT 1 FROM users WHERE email=?", email):
        return err("Email đã được đăng ký", 409)
    uid = db.run("INSERT INTO users(email,name,password_hash) VALUES(?,?,?)", email, name, generate_password_hash(pw))
    return start_session(uid)


@app.post("/api/auth/login")
def login():
    d = request.get_json(silent=True) or {}
    u = db.one("SELECT * FROM users WHERE email=?", (d.get("email") or "").strip().lower())
    if u and not u["password_hash"]:
        return err("Tài khoản này đăng nhập bằng Google", 401)
    if not u or not check_password_hash(u["password_hash"], d.get("password") or ""):
        return err("Email hoặc mật khẩu không đúng", 401)
    return start_session(u["id"])


@app.post("/api/auth/google")
def google_login():
    cid = os.getenv("GOOGLE_CLIENT_ID")
    if not cid:
        return err("Server chưa cấu hình đăng nhập Google", 501)
    from google.auth.transport import requests as greq
    from google.oauth2 import id_token
    try:
        info = id_token.verify_oauth2_token((request.get_json(silent=True) or {}).get("credential", ""), greq.Request(), cid)
    except ValueError:
        return err("Token Google không hợp lệ", 401)
    if not info.get("email_verified"):
        return err("Email Google chưa được xác minh", 401)
    sub, email = info["sub"], info["email"].lower()
    u = db.one("SELECT id FROM users WHERE google_sub=?", sub) or db.one("SELECT id FROM users WHERE email=?", email)
    if u:
        db.run("UPDATE users SET google_sub=COALESCE(google_sub,?) WHERE id=?", sub, u["id"])
        uid = u["id"]
    else:
        uid = db.run("INSERT INTO users(email,name,google_sub) VALUES(?,?,?)", email, info.get("name") or email.split("@")[0], sub)
    return start_session(uid)


@app.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


# ---------------- Cue + revision ----------------
def clean_fmt(f):
    """Định dạng riêng của cue; chỉ lưu các giá trị khác mặc định (dưới / căn giữa / không đậm, nghiêng / chữ trắng)."""
    if not isinstance(f, dict):
        return {}
    out = {}
    if f.get("pos") in ("top", "mid"):
        out["pos"] = f["pos"]
    if f.get("align") in ("left", "right"):
        out["align"] = f["align"]
    if f.get("b") is True:
        out["b"] = True
    if f.get("i") is True:
        out["i"] = True
    col = str(f.get("color") or "").lower()
    if re.fullmatch(r"#[0-9a-f]{6}", col) and col != "#ffffff":
        out["color"] = col
    return out


def clean_cues(raw):
    if not isinstance(raw, list) or len(raw) > 20000:
        return None, "Danh sách cue không hợp lệ"
    out, seen = [], set()
    for c in raw:
        try:
            cid, s, e = str(c["id"])[:40], float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            return None, "Cue thiếu id hoặc thời gian"
        if cid in seen:
            return None, "Trùng id cue: " + cid
        if not (0 <= s < e < 1e6):
            return None, "Thời gian cue không hợp lệ"
        seen.add(cid)
        out.append({"id": cid, "start": s, "end": e, "src": str(c.get("src", ""))[:5000],
                    "text": str(c.get("text", ""))[:5000], "mt": str(c.get("mt", ""))[:5000], "ok": bool(c.get("ok")),
                    "origin": c.get("origin") if c.get("origin") in ("model", "import", "manual") else "manual",
                    "fmt": clean_fmt(c.get("fmt"))})
    return out, None


def load_cues(pid):
    rows = db.all_("SELECT cue_id,start_s,end_s,src,text,mt,ok,origin,fmt FROM cues WHERE project_id=? ORDER BY pos", pid)
    return [{"id": r["cue_id"], "start": r["start_s"], "end": r["end_s"], "src": r["src"], "text": r["text"],
             "mt": r["mt"], "ok": bool(r["ok"]), "origin": r["origin"],
             "fmt": json.loads(r["fmt"]) if r["fmt"] else {}} for r in rows]


def store_cues(conn, pid, cues):
    conn.execute("DELETE FROM cues WHERE project_id=?", (pid,))
    conn.executemany("INSERT INTO cues(project_id,cue_id,pos,start_s,end_s,src,text,mt,ok,origin,fmt) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     [(pid, c["id"], i, c["start"], c["end"], c["src"], c["text"], c["mt"], int(c["ok"]), c["origin"],
                       json.dumps(c["fmt"]) if c.get("fmt") else "")
                      for i, c in enumerate(cues)])


def add_revision(conn, pid, uid, reason, cues):
    snap, n_ok = json.dumps(cues, ensure_ascii=False), sum(c["ok"] for c in cues)
    if reason == "edit":   # gộp các lần sửa liên tiếp trong 5 phút thành một revision
        r = conn.execute("SELECT id FROM revisions WHERE project_id=? AND user_id=? AND reason='edit' "
                         "AND created_at > datetime('now','-5 minutes') "
                         "AND id=(SELECT MAX(id) FROM revisions WHERE project_id=?)", (pid, uid, pid)).fetchone()
        if r:
            conn.execute("UPDATE revisions SET snapshot=?, n_cues=?, n_ok=? WHERE id=?", (snap, len(cues), n_ok, r["id"]))
            return
    conn.execute("INSERT INTO revisions(project_id,user_id,reason,n_cues,n_ok,snapshot) VALUES(?,?,?,?,?,?)",
                 (pid, uid, reason, len(cues), n_ok, snap))
    conn.execute("DELETE FROM revisions WHERE project_id=? AND id NOT IN "
                 "(SELECT id FROM revisions WHERE project_id=? ORDER BY id DESC LIMIT 200)", (pid, pid))


def project_json(p, with_cues=False):
    d = {k: p[k] for k in ("id", "title", "src_lang", "tgt_lang", "video_url", "rev", "created_at", "updated_at")}
    if with_cues:
        d["cues"] = load_cues(p["id"])
    return d


# ---------------- Dự án ----------------
@app.get("/api/projects")
@login_required
def list_projects():
    rows = db.all_("SELECT p.*, (SELECT COUNT(*) FROM cues c WHERE c.project_id=p.id) n, "
                  "(SELECT COUNT(*) FROM cues c WHERE c.project_id=p.id AND c.ok=1) n_ok "
                  "FROM projects p WHERE user_id=? ORDER BY updated_at DESC, id DESC", g.user["id"])
    return jsonify([{**project_json(r), "n": r["n"], "n_ok": r["n_ok"]} for r in rows])


@app.post("/api/projects")
@login_required
def create_project():
    f = request.files.get("video")
    if not f or not f.filename:
        return err("Thiếu file video")
    ext = Path(f.filename).suffix.lower()[:8]
    title = (request.form.get("title") or "").strip()[:120] or Path(f.filename).stem[:120]
    src, tgt = request.form.get("src_lang", "zh")[:8], request.form.get("tgt_lang", "vi")[:8]
    pid = db.run("INSERT INTO projects(user_id,title,src_lang,tgt_lang) VALUES(?,?,?,?)", g.user["id"], title, src, tgt)
    tmp_dir = DATA / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp = tmp_dir / f"{uuid.uuid4().hex}{ext}"
    try:
        f.save(tmp)
        storage.extract_audio(tmp, DATA / "audio" / f"{pid}.ogg")
        st, ref, url = storage.put_video(tmp, pid, ext, DATA)
        db.run("UPDATE projects SET storage=?, video_ref=?, video_url=? WHERE id=?", st, ref, url, pid)
    except Exception as e:  # noqa: BLE001
        db.run("DELETE FROM projects WHERE id=?", pid)
        (DATA / "audio" / f"{pid}.ogg").unlink(missing_ok=True)
        return err(f"Không xử lý được video: {e}", 500)
    finally:
        tmp.unlink(missing_ok=True)
    return jsonify(id=pid), 201


@app.get("/api/projects/<int:pid>")
@login_required
def get_project(pid):
    return jsonify(project_json(own_project(pid), with_cues=True))


@app.patch("/api/projects/<int:pid>")
@login_required
def rename_project(pid):
    own_project(pid)
    title = ((request.get_json(silent=True) or {}).get("title") or "").strip()[:120]
    if not title:
        return err("Tên dự án trống")
    db.run("UPDATE projects SET title=? WHERE id=?", title, pid)
    return jsonify(ok=True)


@app.delete("/api/projects/<int:pid>")
@login_required
def delete_project(pid):
    p = own_project(pid)
    storage.delete_video(p, DATA)
    (DATA / "audio" / f"{pid}.ogg").unlink(missing_ok=True)
    db.run("DELETE FROM projects WHERE id=?", pid)   # cues, revisions, jobs xóa theo (ON DELETE CASCADE)
    return jsonify(ok=True)


@app.get("/media/<int:pid>")
@login_required
def media(pid):
    p = own_project(pid)
    if p["storage"] != "local":
        abort(404)
    return send_file(DATA / "videos" / p["video_ref"], conditional=True)   # hỗ trợ Range để tua video


@app.put("/api/projects/<int:pid>/cues")
@login_required
def save_cues(pid):
    own_project(pid)
    d = request.get_json(silent=True) or {}
    cues, e = clean_cues(d.get("cues"))
    if e:
        return err(e)
    reason = d.get("reason") if d.get("reason") in ("edit", "import") else "edit"
    c = db.conn()
    with c:
        cur = c.execute("UPDATE projects SET rev=rev+1, updated_at=CURRENT_TIMESTAMP WHERE id=? AND rev=?", (pid, d.get("rev")))
        if cur.rowcount == 0:
            return err("Dự án đã được cập nhật ở nơi khác. Hãy tải lại trang.", 409)
        store_cues(c, pid, cues)
        add_revision(c, pid, g.user["id"], reason, cues)
    return jsonify(rev=int(d["rev"]) + 1)


@app.get("/api/projects/<int:pid>/revisions")
@login_required
def list_revisions(pid):
    own_project(pid)
    rows = db.all_("SELECT r.id, r.reason, r.n_cues, r.n_ok, r.created_at, u.name FROM revisions r "
                  "LEFT JOIN users u ON u.id=r.user_id WHERE r.project_id=? ORDER BY r.id DESC LIMIT 100", pid)
    return jsonify([dict(r) for r in rows])


@app.post("/api/projects/<int:pid>/revisions/<int:rid>/restore")
@login_required
def restore_revision(pid, rid):
    own_project(pid)
    r = db.one("SELECT snapshot FROM revisions WHERE id=? AND project_id=?", rid, pid)
    if not r:
        abort(404)
    cues = json.loads(r["snapshot"])
    c = db.conn()
    with c:
        c.execute("UPDATE projects SET rev=rev+1, updated_at=CURRENT_TIMESTAMP WHERE id=?", (pid,))
        store_cues(c, pid, cues)
        add_revision(c, pid, g.user["id"], "restore", cues)
    return jsonify(project_json(own_project(pid), with_cues=True))


# ---------------- Dịch sơ qua model server ----------------
def run_job(jid, pid, uid):
    conn = db.connect()

    def upd(**kw):
        conn.execute("UPDATE jobs SET " + ",".join(f"{k}=?" for k in kw) + " WHERE id=?", (*kw.values(), jid))
        conn.commit()
    try:
        p = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        upd(status="running", stage="Gửi tới model", progress=0.02)
        segs = model_client.run(DATA / "audio" / f"{pid}.ogg", p["src_lang"], p["tgt_lang"], lambda s, x: upd(stage=s, progress=x))
        cues = [{"id": f"c{i + 1}", "start": float(s["start"]), "end": float(s["end"]), "src": s.get("src", ""),
                 "text": s.get("text", ""), "mt": s.get("text", ""), "ok": False, "origin": "model"} for i, s in enumerate(segs)]
        with conn:
            conn.execute("UPDATE projects SET rev=rev+1, updated_at=CURRENT_TIMESTAMP WHERE id=?", (pid,))
            store_cues(conn, pid, cues)
            add_revision(conn, pid, uid, "model", cues)
        upd(status="done", stage="Xong", progress=1.0)
    except Exception as e:  # noqa: BLE001
        upd(status="failed", error=str(e)[:300])
    finally:
        conn.close()


@app.post("/api/projects/<int:pid>/translate")
@login_required
def translate(pid):
    own_project(pid)
    if db.one("SELECT 1 FROM jobs WHERE project_id=? AND status IN ('queued','running')", pid):
        return err("Dự án đang có job chạy", 409)
    jid = db.run("INSERT INTO jobs(project_id,status,stage) VALUES(?,?,?)", pid, "queued", "Đang chờ")
    threading.Thread(target=run_job, args=(jid, pid, g.user["id"]), daemon=True).start()
    return jsonify(job_id=jid), 202


@app.get("/api/projects/<int:pid>/job")
@login_required
def get_job(pid):
    own_project(pid)
    j = db.one("SELECT status,stage,progress,error FROM jobs WHERE project_id=? ORDER BY id DESC LIMIT 1", pid)
    return jsonify(dict(j) if j else {"status": "none"})


@app.post("/api/projects/<int:pid>/redo-asr")
@login_required
def redo_asr(pid):
    """Nghe lại một đoạn: chỉ chạy ASR cho khoảng thời gian của cue."""
    p = own_project(pid)
    d = request.get_json(silent=True) or {}
    try:
        s, e = float(d["start"]), float(d["end"])
    except (KeyError, TypeError, ValueError):
        return err("Thiếu thời gian cue")
    if not (0 <= s < e <= s + 120):
        return err("Đoạn cần nghe lại không hợp lệ (tối đa 120 giây)")
    try:
        text = model_client.asr_clip(DATA / "audio" / f"{pid}.ogg", s, e, p["src_lang"])
    except Exception as ex:  # noqa: BLE001
        return err(f"Model ASR lỗi: {ex}", 502)
    return jsonify(text=text)


@app.post("/api/projects/<int:pid>/redo-mt")
@login_required
def redo_mt(pid):
    """Dịch lại một câu: chỉ chạy MT cho văn bản gốc."""
    p = own_project(pid)
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()[:5000]
    if not text:
        return err("Văn bản gốc trống")
    try:
        out = model_client.mt_text(text, p["src_lang"], p["tgt_lang"])
    except Exception as ex:  # noqa: BLE001
        return err(f"Model MT lỗi: {ex}", 502)
    return jsonify(text=out)


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("DEBUG") == "1", threaded=True)
