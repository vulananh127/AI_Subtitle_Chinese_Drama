"""Dữ liệu đánh giá end-to-end: mỗi mẫu = 1 video/audio + SRT tham chiếu phía nguồn (zh) + SRT tham chiếu tiếng Việt.

Hai cách khai báo (chọn một):
  1) Manifest CSV (mặc định metadata/{language}/metadata_e2e.csv hoặc --input), mỗi dòng một video, cột tự dò:
       id | media (video/audio) | ref_src_srt (SRT tiếng gốc) | ref_tgt_srt (SRT tiếng Việt) | [split] | [duration]
     đường dẫn tương đối được hiểu so với thư mục chứa CSV.
  2) Thư mục (--media-dir): mỗi video `X.mp4` đi kèm `X.{src}.srt` và `X.{tgt}.srt` (vd X.zh.srt, X.vi.srt) cùng thư mục.
Không có SRT tiếng gốc thì vẫn chạy được phần dịch/phụ đề (bỏ điểm ASR); không có SRT tiếng Việt thì bỏ điểm MT.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .data import ID_COLS, _check_unique, _fingerprint, _pick_col, _read
from .paths import drama_metadata_csv, putils_mod

log = logging.getLogger(__name__)

MEDIA_COLS = ("video_path", "video", "media_path", "media", "audio_path", "audio", "path", "file")
SRC_SRT_COLS = ("ref_src_srt", "src_srt", "source_srt", "origin_srt", "zh_srt", "srt_zh")
TGT_SRT_COLS = ("ref_tgt_srt", "tgt_srt", "target_srt", "vi_srt", "srt_vi")
MEDIA_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v", ".wmv", ".mp3", ".wav", ".m4a", ".aac",
             ".flac", ".ogg", ".opus"}


def read_srt(path: Path) -> list[dict]:
    """SRT -> [{start, end, text(một dòng)}] theo thời gian; bỏ cue rỗng."""
    u = putils_mod()
    cues = u.parse_srt_text(Path(path).read_text(encoding="utf-8-sig"), source=str(path), strict=False)
    out = [{"start": c["start"], "end": c["end"], "text": u.flatten_text(c["text"], strip_dash=True)} for c in cues]
    return sorted((c for c in out if c["text"].strip() and c["end"] > c["start"]), key=lambda c: (c["start"], c["end"]))


def _resolve(base: Path, v: str) -> Path | None:
    v = (v or "").strip()
    if not v:
        return None
    p = Path(v)
    return p if p.is_absolute() else base / p


def load_e2e_dataset(language, src_lang, tgt_lang, csv_path=None, media_dir=None, media_col=None, src_col=None,
                     tgt_col=None, id_col=None, split_col="split", split=None, limit=None):
    """-> (items, info). item: {id, media, ref_src: list|None, ref_tgt: list|None, duration}."""
    items = []
    if media_dir:
        base = Path(media_dir)
        media = sorted(p for p in base.rglob("*") if p.suffix.lower() in MEDIA_EXT)
        for m in media:
            sp, tp = m.with_suffix(f".{src_lang}.srt"), m.with_suffix(f".{tgt_lang}.srt")
            items.append({"id": m.relative_to(base).with_suffix("").as_posix().replace("/", "__"), "media": m,
                          "src_srt": sp if sp.exists() else None, "tgt_srt": tp if tp.exists() else None})
        stems = [it["id"] for it in items]
        for it, m in zip(items, media):      # ep1.mp4 + ep1.wav cùng thư mục -> id gồm cả đuôi file để không trùng
            if stems.count(it["id"]) > 1:
                it["id"] += m.suffix.lower().replace(".", "_")
        n_csv, source = len(items), str(base)
        items = items[:limit] if limit else items
    else:
        csv_path = Path(csv_path) if csv_path else drama_metadata_csv(language, "e2e")
        fields, rows, n_csv = _read(csv_path, split_col, split, limit)
        c_id = _pick_col(fields, id_col, ID_COLS, "id")
        c_media = _pick_col(fields, media_col, MEDIA_COLS, "video/audio")
        c_src = _pick_col(fields, src_col, SRC_SRT_COLS, "SRT tham chiếu tiếng gốc") if (
            src_col or any(c in fields for c in SRC_SRT_COLS)) else None
        c_tgt = _pick_col(fields, tgt_col, TGT_SRT_COLS, "SRT tham chiếu tiếng Việt") if (
            tgt_col or any(c in fields for c in TGT_SRT_COLS)) else None
        base = csv_path.parent
        for r in rows:
            items.append({"id": r[c_id].strip(), "media": _resolve(base, r[c_media]),
                          "src_srt": _resolve(base, r[c_src]) if c_src else None,
                          "tgt_srt": _resolve(base, r[c_tgt]) if c_tgt else None})
        source = str(csv_path)
    _check_unique([it["id"] for it in items])

    ok, missing = [], []
    for it in items:
        if not it["media"] or not Path(it["media"]).exists():
            missing.append(it["id"])
            continue
        it["ref_src"] = read_srt(it["src_srt"]) if it["src_srt"] and Path(it["src_srt"]).exists() else None
        it["ref_tgt"] = read_srt(it["tgt_srt"]) if it["tgt_srt"] and Path(it["tgt_srt"]).exists() else None
        ok.append(it)
    if missing:
        log.warning("Bỏ %d mẫu không thấy file media, vd %s", len(missing), missing[:5])
    if not ok:
        raise SystemExit("Không có mẫu e2e hợp lệ nào.")
    info = {"source": source, "n_in_source": n_csv, "n": len(ok), "n_missing_media": len(missing),
            "n_with_ref_src": sum(it["ref_src"] is not None for it in ok),
            "n_with_ref_tgt": sum(it["ref_tgt"] is not None for it in ok),
            "split": split, "limit": limit, "fingerprint": _fingerprint([it["id"] for it in ok])}
    return ok, info
