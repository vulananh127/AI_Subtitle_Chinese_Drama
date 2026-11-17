"""Đọc metadata_asr.csv / metadata_mt.csv thành danh sách mẫu để đánh giá.

Tên cột tự dò theo danh sách ứng viên bên dưới; nếu metadata của bạn đặt tên khác thì truyền
--audio-col / --text-col / --src-col / --ref-col / --id-col (hoặc sửa các tuple *_COLS).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from .common import MTSource
from .paths import (DATASET_DIR, DRAMA_AUDIO_CHUNKS_DIR, DRAMA_DIR, DRAMA_PROCESSED_DIR, PROJECT_ROOT,
                    drama_metadata_csv, read_csv_rows)

log = logging.getLogger(__name__)

ID_COLS = ("sample_id", "id", "utt_id", "uttid", "key")
AUDIO_COLS = ("audio_path", "audio", "audio_filepath", "wav_path", "path", "file")
ASR_TEXT_COLS = ("origin_text", "text", "transcript", "sentence")
MT_SRC_COLS = ("origin_text", "src_text", "source", "text")
MT_REF_COLS = ("vi_text", "vi", "tgt_text", "target", "translation")
DURATION_COLS = ("duration", "dur", "audio_duration")


def _find_col(fields, override, candidates):
    if override:
        if override not in fields:
            raise KeyError(f"Không có cột '{override}'. Các cột hiện có: {fields}")
        return override
    return next((c for c in candidates if c in fields), None)


def _pick_col(fields, override, candidates, what):
    col = _find_col(fields, override, candidates)
    if col is None:
        raise KeyError(f"Không tự dò được cột {what} (thử {candidates}). Các cột hiện có: {fields}. "
                       f"Hãy chỉ định bằng tham số dòng lệnh.")
    return col


def _read(csv_path: Path, split_col, split, limit):
    if not csv_path.exists():
        raise FileNotFoundError(f"Không thấy {csv_path}")
    fields, rows = read_csv_rows(csv_path)
    fields = [f.lstrip("﻿") for f in fields]
    rows = [{k.lstrip("﻿"): (v or "") for k, v in r.items() if k is not None} for r in rows]
    n_csv = len(rows)
    if split:
        if split_col not in fields:
            raise KeyError(f"--split '{split}' nhưng CSV không có cột '{split_col}'. Các cột: {fields}")
        rows = [r for r in rows if r[split_col].strip() == split]
        log.info("Lọc %s=%s: %d/%d dòng", split_col, split, len(rows), n_csv)
    else:
        log.warning("Không dùng --split: đánh giá trên TOÀN BỘ %d dòng của %s", n_csv, csv_path.name)
    if limit:
        rows = rows[:limit]
    return fields, rows, n_csv


def _fingerprint(ids) -> str:
    return hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()[:12]


def _check_unique(ids):
    seen, dup = set(), []
    for x in ids:
        if x in seen:
            dup.append(x)
        seen.add(x)
    if dup:
        raise ValueError(f"id bị trùng ({len(dup)}), vd {dup[:5]}. Chỉ định cột id khác bằng --id-col.")


class AudioResolver:
    """Đường dẫn audio trong CSV có thể tuyệt đối hoặc tương đối; thử lần lượt các thư mục gốc."""

    def __init__(self, language: str, csv_dir: Path):
        self.bases = [DRAMA_AUDIO_CHUNKS_DIR / language, DRAMA_PROCESSED_DIR, DRAMA_DIR, DATASET_DIR,
                      PROJECT_ROOT, csv_dir]
        self._hit = None

    def __call__(self, value: str):
        p = Path(value)
        if p.is_absolute():
            return str(p) if p.exists() else None
        for base in ([self._hit] if self._hit else []) + self.bases:
            c = base / p
            if c.exists():
                self._hit = base
                return str(c)
        return None


def _probe_duration(path: str):
    try:
        import soundfile as sf
        return float(sf.info(path).duration)
    except Exception:  # noqa: BLE001
        return None


def load_asr_dataset(language, csv_path=None, audio_col=None, text_col=None, id_col=None,
                     split_col="split", split=None, limit=None, allow_missing_audio=False):
    """-> (items, info). item = {id, audio_path, reference, duration}"""
    csv_path = Path(csv_path) if csv_path else drama_metadata_csv(language, "asr")
    fields, rows, n_csv = _read(csv_path, split_col, split, limit)
    a = _pick_col(fields, audio_col, AUDIO_COLS, "audio")
    t = _pick_col(fields, text_col, ASR_TEXT_COLS, "transcript tham chiếu")
    i = _find_col(fields, id_col, ID_COLS)
    d = _find_col(fields, None, DURATION_COLS)
    resolve = AudioResolver(language, csv_path.parent)

    items, missing, n_empty = [], [], 0
    for r in rows:
        ref = r[t].strip()
        if not ref:
            n_empty += 1
            continue
        p = resolve(r[a].strip())
        if p is None:
            missing.append(r[a])
            continue
        dur = None
        if d and r[d]:
            try:
                dur = float(r[d])
            except ValueError:
                pass
        items.append({"id": r[i].strip() if i else Path(p).stem, "audio_path": p, "reference": ref, "duration": dur})
    if missing:
        msg = f"{len(missing)} file audio không tồn tại, vd: {missing[:3]}. Đã thử các thư mục gốc: {[str(b) for b in resolve.bases]}"
        if not allow_missing_audio:
            raise FileNotFoundError(msg + " (dùng --allow-missing-audio để bỏ qua)")
        log.warning(msg)
    if items and any(it["duration"] is None for it in items):
        for it in items:
            if it["duration"] is None:
                it["duration"] = _probe_duration(it["audio_path"])
    _check_unique([it["id"] for it in items])
    info = {"csv": str(csv_path), "n_rows_csv": n_csv, "n_items": len(items), "n_empty_ref": n_empty,
            "n_missing_audio": len(missing), "split_col": split_col if split else None, "split": split,
            "columns": {"audio": a, "text": t, "id": i, "duration": d},
            "fingerprint": _fingerprint([it["id"] for it in items])}
    log.info("ASR: %d mẫu (bỏ %d ref rỗng, %d thiếu audio)", len(items), n_empty, len(missing))
    return items, info



def _attach_mt_context(items):
    """Resolve adjacent numbered lines by ID, never CSV position or batch order.

    Unknown IDs and missing sequence numbers receive empty context. Only items
    in the selected split are eligible; references are never used as context.
    """
    pattern = re.compile(r"^(.*_EP\d+)_(\d+)$")
    lookup = {}
    keys = []
    for it in items:
        match = pattern.fullmatch(it["id"])
        key = (match[1], int(match[2])) if match else None
        if key is not None and key in lookup:
            raise ValueError(f"Ambiguous episode/line ID: {it['id']}")
        keys.append(key)
        if key is not None:
            lookup[key] = str(it["src"])
    for it, key in zip(items, keys):
        prev_src = lookup.get((key[0], key[1] - 1), "") if key else ""
        next_src = lookup.get((key[0], key[1] + 1), "") if key else ""
        it["src"] = MTSource(it["src"], prev_src, next_src)


def load_mt_dataset(language, csv_path=None, src_col=None, ref_col=None, id_col=None,
                    split_col="split", split=None, limit=None):
    """-> (items, info). item = {id, src, reference}"""
    csv_path = Path(csv_path) if csv_path else drama_metadata_csv(language, "mt")
    fields, rows, n_csv = _read(csv_path, split_col, split, None)
    s = _pick_col(fields, src_col, MT_SRC_COLS, "câu nguồn (origin_text)")
    r_ = _pick_col(fields, ref_col, MT_REF_COLS, "câu đích (vi_text)")
    i = _find_col(fields, id_col, ID_COLS)
    items, n_empty = [], 0
    for idx, r in enumerate(rows):
        src, ref = r[s].strip(), r[r_].strip()
        if not src or not ref:
            n_empty += 1
            continue
        items.append({"id": r[i].strip() if i else f"{idx:07d}", "src": src, "reference": ref})
    _check_unique([it["id"] for it in items])
    _attach_mt_context(items)
    if limit:
        items = items[:limit]
    info = {"csv": str(csv_path), "n_rows_csv": n_csv, "n_items": len(items), "n_empty_pair": n_empty,
            "context": {"previous": 1, "next": 1, "scope": "same episode, consecutive ID, selected split"},
            "split_col": split_col if split else None, "split": split,
            "columns": {"src": s, "ref": r_, "id": i},
            "fingerprint": _fingerprint([it["id"] for it in items])}
    log.info("MT: %d cặp câu (bỏ %d cặp rỗng)", len(items), n_empty)
    return items, info
