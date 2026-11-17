import os
import time
from pathlib import Path

import requests

URL = os.getenv("MODEL_API_URL", "").rstrip("/")
KEY = os.getenv("MODEL_API_KEY", "")


def run(audio_path, src, tgt, progress):
    """Gọi model server, trả về [{'start','end','src','text'}].
    Chưa cấu hình MODEL_API_URL thì trả dữ liệu mẫu để thử luồng làm việc."""
    if not URL:
        for i, st in enumerate(["Tách audio", "Nhận dạng (ASR)", "Dịch (MT)"]):
            progress(st + " (mẫu)", 0.2 + i * 0.25)
            time.sleep(0.6)
        return [{"start": i * 4.0, "end": i * 4.0 + 3.6, "src": f"(ASR) câu nguồn số {i + 1}",
                 "text": f"(MT) bản dịch sơ số {i + 1}"} for i in range(5)]
    h = {"X-API-Key": KEY} if KEY else {}
    with open(audio_path, "rb") as f:
        r = requests.post(URL + "/api/jobs", files={"video": (Path(audio_path).name, f)},
                          data={"src_lang": src, "tgt_lang": tgt}, headers=h, timeout=600)
    r.raise_for_status()
    jid = r.json()["job_id"]
    deadline = time.time() + 7200
    while time.time() < deadline:
        time.sleep(2)
        s = requests.get(f"{URL}/api/jobs/{jid}", headers=h, timeout=30)
        s.raise_for_status()
        s = s.json()
        if s["status"] == "failed":
            raise RuntimeError(s.get("error") or "Model server báo lỗi")
        if s["status"] == "done":
            return s["cues"]
        progress(s.get("stage") or "Đang xử lý", 0.1 + 0.85 * (s.get("progress") or 0))
    raise RuntimeError("Model server xử lý quá lâu")


def _h():
    return {"X-API-Key": KEY} if KEY else {}


def asr_clip(audio_path, start, end, lang):
    """Chạy riêng ASR cho đoạn [start, end] của audio; trả về văn bản."""
    if not URL:
        return f"(ASR mẫu) nghe lại {start:.1f}s–{end:.1f}s"
    import tempfile

    import storage
    with tempfile.TemporaryDirectory() as d:
        clip = Path(d) / "clip.wav"
        storage.cut_clip(audio_path, clip, start, end)
        with open(clip, "rb") as f:
            r = requests.post(URL + "/api/asr", files={"audio": ("clip.wav", f)}, data={"lang": lang}, headers=_h(), timeout=120)
    r.raise_for_status()
    return r.json()["text"]


def mt_text(text, src, tgt):
    """Chạy riêng MT cho một câu; trả về bản dịch."""
    if not URL:
        return f"(MT mẫu) {text}"
    r = requests.post(URL + "/api/mt", json={"text": text, "src_lang": src, "tgt_lang": tgt}, headers=_h(), timeout=120)
    r.raise_for_status()
    return r.json()["text"]
