"""
Lọc subtitle đặc biệt + alignment sơ bộ Trung/Hàn/Nhật - Việt (2 stage).

Chỉ ĐỌC raw SRT, không sửa/xóa dữ liệu gốc. Dùng config.py và utils.py hiện có.

STAGE 1 - music: python special_review.py --stage music
    - auto_excluded.csv: MỌI cue (VI và ZH) chỉ gồm nốt nhạc -> loại tự động (AUTO).
    - music_pairs_review.csv: cặp ZH-VI trùng thời gian, gồm đúng 2 loại:
        (1) một bên là nhạc thuần, bên kia KHÔNG có marker nào (ngoài html/ass);
        (2) có cue "có dấu nốt nhưng còn dòng khác" (NOTE_MIXED).
      Cue có marker khác (ngoặc, gạch đầu dòng, sound effect) không ghép cặp vì luôn keep.
      Cột decision điền sẵn theo auto_music_decision, review_status = AUTO, mã luật ở cột
      note (rule=...), ca kém chắc chắn có thêm check=low -> soát trước.
      Sai chỗ nào thì sửa decision và đổi review_status thành CONFIRMED.

STAGE 2 - candidates: python special_review.py --stage candidates
    - Bỏ các cue đã loại (AUTO/CONFIRMED, asr_action = mt_action = exclude).
    - Tính lại overlap cùng ngôn ngữ, prev/next trên dữ liệu đã lọc (cue_id giữ
      nguyên theo thứ tự raw).
    - Candidate khi: một trong hai bên có marker, hoặc cue gốc overlap cue gốc
      (chỉ tính khi cả hai cue gốc đều không có marker thuộc OVERLAP_IGNORE_MARKERS, mặc định [...]).
      Overlap CHỈ ở phía VI không tạo candidate, chỉ ghi vào vi_overlap_suspect.csv
      để lấy mẫu nghe audio.
    - Ghép cặp: giữ khi cả hai cue đều đặc biệt, hoặc IoU >= CANDIDATE_MIN_IOU, hoặc là cặp IoU
      cao nhất của một cue đặc biệt (cặp yếu như vậy ghi note = weak_match). Cue có marker thuộc
      CANDIDATE_IOU_EXEMPT_MARKERS (mặc định leading_dash) không bị lọc IoU: giữ mọi cặp đạt overlap.
    - Gom candidate thành nhóm (connected component) và gợi ý hành động.
    Output: special_alignment_candidates.csv, special_unmatched.csv,
            vi_overlap_suspect.csv

Tham số lọc: --language chinese --drama D0002 --episode 1 --reset --allow-todo
"""

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from config import (
    DRAMA_LANGUAGES,
    DRAMA_RAW_DIR,
    DRAMA_PROCESSED_DIR,
    MIN_OVERLAP_RATIO,
    SPEECH_SUPPORT_MIN_IOU,
    EXCLUDE_MIN_PLAIN_COVER,
    PAIR_FORMAT_ONLY_MARKERS,
    SAME_LANG_MIN_SEC,
    SAME_LANG_MIN_RATIO,
    AUTO_KEEP_MIN_IOU,
    CANDIDATE_MIN_IOU,
    CANDIDATE_IOU_EXEMPT_MARKERS,
    OVERLAP_IGNORE_MARKERS,
    MARKER_PATTERNS,
    MUSIC_CAPTION_RE,
    drama_raw_dir,
)
from utils import (
    parse_srt,
    seconds_to_srt_time,
    pad_drama_id,
    pad_episode,
    read_csv_rows,
    write_csv_rows,
)

# Các ngưỡng (SPEECH_SUPPORT_MIN_IOU, SAME_LANG_*, AUTO_KEEP_MIN_IOU, EXCLUDE_MIN_PLAIN_COVER,
# CANDIDATE_MIN_IOU, CANDIDATE_IOU_EXEMPT_MARKERS, OVERLAP_IGNORE_MARKERS, PAIR_FORMAT_ONLY_MARKERS) nằm trong config.py.

# Định nghĩa marker (MARKER_PATTERNS, MUSIC_CAPTION_RE) nằm trong config.py.


def normalize_for_detection(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def detect_markers(text: str) -> list[str]:
    text = normalize_for_detection(text)
    return [n for n, p in MARKER_PATTERNS.items() if p.search(text)]


def fmt_markers(o: dict, v: dict) -> str:
    """Gộp marker hai phía thành một chuỗi, ghi rõ bên nào: origin:xxx;vi:yyy"""
    return ";".join([f"origin:{m}" for m in o["markers"]] + [f"vi:{m}" for m in v["markers"]])


def count_dash_lines(text: str) -> int:
    lines = [l for l in (text or "").splitlines() if l.strip()]
    return sum(bool(MARKER_PATTERNS["leading_dash"].search(l)) for l in lines)


def is_music_only(text: str) -> bool:
    """Toàn bộ dòng có nội dung đều là dòng nhạc/caption nhạc."""
    lines = [l.strip() for l in normalize_for_detection(text).splitlines() if l.strip()]
    if not lines:
        return False
    return all(MARKER_PATTERNS["music_note"].search(l) or MUSIC_CAPTION_RE.fullmatch(l) for l in lines)


# ---------------------------------------------------------------------
# Hình học thời gian
# ---------------------------------------------------------------------
def overlap_sec(a: dict, b: dict) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def union_length(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def overlap_metrics(a: dict, b: dict) -> dict:
    ov = overlap_sec(a, b)
    da, db = a["duration"], b["duration"]
    shorter = min(da, db)
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return {
        "overlap_sec": ov,
        "overlap_ratio_min": ov / shorter if shorter > 0 else 0.0,
        "iou": ov / union if union > 0 else 0.0,
        # a (origin) nằm trong b (VI) bao nhiêu, và b được a phủ bao nhiêu
        "origin_cover": ov / da if da > 0 else 0.0,
        "vi_cover": ov / db if db > 0 else 0.0,
    }

"""
# ---------------------------------------------------------------------
# Đọc cue (ID theo thứ tự RAW, không bao giờ đánh số lại sau khi lọc)
# ---------------------------------------------------------------------
def load_cues(path: Path, lang_tag: str) -> list[dict]:
    cues = []
    for idx, cue in enumerate(parse_srt(path), start=1):
        start, end = float(cue["start"]), float(cue["end"])
        text = cue.get("text", "")
        cues.append({
            "cue_id": f"{lang_tag}_{idx:06d}",
            "cue_index": idx,
            "start": start,
            "end": end,
            "duration": max(0.0, end - start),
            "text_raw": text,
            "markers": detect_markers(text),
            "auto_music": is_music_only(text),
            "dash_line_count": count_dash_lines(text),
        })
    return cues
"""

def load_cues(path: Path, lang_tag: str) -> list[dict]:
    """cue_id = {lang_tag}_{id cue trong file SRT}, ví dụ zh_000546.
 
    id lấy từ parse_srt (cue["id"]) = số ghi trong file SRT đã đánh số bằng number_srt.py.
    Cue rỗng bị bỏ nhưng không làm lệch id của cue sau. parse_srt (strict) báo ValueError
    nếu file đánh số dở dang hoặc trùng số, nên cue_id luôn duy nhất.
    """
    cues = []
    for cue in parse_srt(path):
        idx = cue["id"]
        start, end = float(cue["start"]), float(cue["end"])
        text = cue.get("text", "")
        cues.append({
            "cue_id": f"{lang_tag}_{idx:06d}",
            "cue_index": idx,
            "start": start,
            "end": end,
            "duration": max(0.0, end - start),
            "text_raw": text,
            "markers": detect_markers(text),
            "auto_music": is_music_only(text),
            "dash_line_count": count_dash_lines(text),
        })
    return cues

# ---------------------------------------------------------------------
# Đọc cue (ID theo thứ tự RAW, không bao giờ đánh số lại sau khi lọc)
# ---------------------------------------------------------------------
def load_cuess(path: Path, lang_tag: str) -> list[dict]:
    """cue_id = {lang_tag}_{số cue GỐC trong file SRT}, ví dụ zh_000546.
 
    Số gốc lấy từ parse_srt (cue["index"]); cue rỗng bị bỏ không làm lệch số của cue sau.
    Nếu file SRT lỗi và lặp số cue thì các lần lặp được thêm hậu tố _dup1, _dup2... để
    cue_id vẫn duy nhất (và in cảnh báo).
    """
    cues, seen = [], {}
    for pos, cue in enumerate(parse_srt(path), start=1):
        idx = cue.get("index")
        if idx is None:                      # parse_srt cũ chưa trả index
            idx = pos
        n = seen.get(idx, 0)
        seen[idx] = n + 1
        cue_id = f"{lang_tag}_{idx:06d}" + (f"_dup{n}" if n else "")
        if n:
            print(f"[WARN] {Path(path).name}: số cue {idx} bị lặp, đặt id {cue_id}")
        start, end = float(cue["start"]), float(cue["end"])
        text = cue.get("text", "")
        cues.append({
            "cue_id": cue_id,
            "cue_index": idx,
            "start": start,
            "end": end,
            "duration": max(0.0, end - start),
            "text_raw": text,
            "markers": detect_markers(text),
            "auto_music": is_music_only(text),
            "dash_line_count": count_dash_lines(text),
        })
    return cues


def find_same_lang_overlaps(cues: list[dict]) -> list[tuple[dict, dict, float]]:
    """Các cặp cue cùng ngôn ngữ chồng nhau đủ ngưỡng (giây VÀ tỉ lệ)."""
    ordered = sorted(cues, key=lambda c: (c["start"], c["end"]))
    pairs = []
    for i, a in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            if b["start"] >= a["end"]:
                break
            ov = min(a["end"], b["end"]) - max(a["start"], b["start"])
            shorter = min(a["duration"], b["duration"])
            if ov >= SAME_LANG_MIN_SEC and shorter > 0 and ov / shorter >= SAME_LANG_MIN_RATIO:
                pairs.append((a, b, ov))
    return pairs


# ---------------------------------------------------------------------
# Đọc/ghi CSV review (dùng read_csv_rows / write_csv_rows của utils.py)
# ---------------------------------------------------------------------
def read_table(path: Path) -> list[dict]:
    """Đọc CSV review. Tự bỏ BOM ở tên cột đầu (Excel 'CSV UTF-8' thêm BOM khi lưu)."""
    path = Path(path).with_suffix(".csv")
    if not path.exists():
        return []
    fieldnames, rows = read_csv_rows(path)
    if fieldnames and fieldnames[0].startswith("\ufeff"):
        clean = fieldnames[0].lstrip("\ufeff")
        rows = [{(clean if k == fieldnames[0] else k): v for k, v in r.items()} for r in rows]
    return rows


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def excel_safe(value):
    """Excel coi ô bắt đầu bằng = + - @ là công thức (ra #NAME?).
    Thêm 1 dấu cách đầu ô để nó luôn là text. Idempotent: ô đã có dấu cách thì giữ nguyên.
    Chỉ ảnh hưởng cột chữ hiển thị để review; script không đọc lại các cột đó."""
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return " " + value
    return value


def append_table(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Thêm dòng vào CSV (đọc dòng cũ, nối dòng mới, ghi lại để giữ phần bạn đã review)."""
    path = Path(path).with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows = read_table(path)
    if old_rows and list(old_rows[0].keys()) != fieldnames:
        raise ValueError(
            f"Cột của {path.name} khác phiên bản script hiện tại; "
            f"dùng --reset hoặc đổi tên file cũ."
        )
    new_rows = [{k: r.get(k, "") for k in fieldnames} for r in rows]
    safe = [{k: excel_safe(v) for k, v in r.items()} for r in old_rows + new_rows]
    write_csv_rows(path, fieldnames, safe)


def load_done(path: Path) -> set[str]:
    return set(path.read_text(encoding="utf-8").split("\n")) - {""} if path.exists() else set()


def mark_done(path: Path, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(key + "\n")


# ---------------------------------------------------------------------
# Tìm tập
# ---------------------------------------------------------------------
def find_episodes(language_filter=None, drama_filter=None, episode_filter=None) -> list[tuple]:
    episodes = []
    for language, lang_tag in DRAMA_LANGUAGES.items():
        if language_filter and language != language_filter:
            continue
        lang_dir = drama_raw_dir(language)
        if not lang_dir.is_dir():
            continue
        pattern = re.compile(
            rf"^{re.escape(language)}_D(\d+)_EP(\d+)_{re.escape(lang_tag)}_manual_raw\.srt$"
        )
        for drama_dir in sorted(lang_dir.glob("D*")):
            if not drama_dir.is_dir():
                continue
            if drama_filter and drama_dir.name != drama_filter:
                continue
            for origin_path in sorted(drama_dir.glob(f"{language}_*_{lang_tag}_manual_raw.srt")):
                m = pattern.match(origin_path.name)
                if not m:
                    continue
                drama_num, ep_num = m.groups()
                episode_tag = pad_episode(ep_num)
                if episode_filter and episode_tag != episode_filter:
                    continue
                vi_path = drama_dir / f"{language}_{drama_dir.name}_{episode_tag}_vi_manual_raw.srt"
                if vi_path.exists():
                    episodes.append((language, lang_tag, drama_dir.name,
                                     pad_drama_id(drama_num), episode_tag,
                                     origin_path, vi_path))
    return episodes


# =====================================================================
# STAGE 1 - MUSIC
# =====================================================================
EXCLUDED_FIELDS = [
    "language", "drama_id", "episode_id", "subtitle_language", "cue_id",
    "start", "end", "text_raw", "reason",
    "asr_action", "mt_action", "review_status", "note",
]
PAIR_FIELDS = [
    "language", "drama_id", "episode_id", "origin_cue_id", "vi_cue_id",
    "origin_start", "origin_end", "vi_start", "vi_end",
    "origin_duration", "vi_duration",
    "overlap_sec", "overlap_ratio_min", "iou",
    "origin_text_raw", "vi_text_raw", "detected_markers",
    "other_speech_ids",
    "decision", "review_status", "note",
]


def has_note(c: dict) -> bool:
    return "music_note" in c["markers"]


def pair_state(c: dict) -> str:
    """MUSIC      = nhạc thuần (có dấu nốt, mọi dòng đều là nhạc)
    NOTE_MIXED = có dấu nốt nhưng còn dòng khác (thoại, gạch đầu dòng, [..])
    CLEAN      = không có marker nào (ngoài định dạng)
    OTHER      = có marker khác (ngoặc, gạch đầu dòng, sound effect...) mà không có dấu nốt
    """
    if has_note(c):
        return "MUSIC" if c["auto_music"] else "NOTE_MIXED"
    return "OTHER" if set(c["markers"]) - PAIR_FORMAT_ONLY_MARKERS else "CLEAN"


def auto_music_decision(so: str, sv: str, speech_ids: list[str],
                        plain_cover: float) -> tuple[str, str, str]:
    """Điền sẵn decision cho cặp nhạc. Trả về (decision, review_status, note).

    Chỉ nhận cặp thuộc 2 loại (cặp có cue OTHER không được tạo, vì luôn keep):
      - MUSIC x CLEAN      (nhạc thuần ghép cue không marker, một trong hai phía là ZH/VI)
      - NOTE_MIXED x (MUSIC / CLEAN / NOTE_MIXED)

    `exclude` loại cả hai cue khỏi ASR/MT nên chỉ chọn khi rõ là lời hát; còn lại keep.
      M_note_cue_has_other_lines : có cue NOTE_MIXED -> keep, check=low (loại sẽ mất phần thoại)
      S_speech_over_music_<vi|zh>: cue nhạc còn cue thoại cùng ngôn ngữ khác khớp cue kia -> keep
      W_weak_overlap             : cue nhạc phủ < EXCLUDE_MIN_PLAIN_COVER cue kia -> keep, check=low
      L_lyrics_<vi|zh>_music     : còn lại -> exclude (lời hát phía kia không đánh dấu).
                                   Nhánh zh_music chưa có dữ liệu kiểm chứng -> check=low.
    """
    if "NOTE_MIXED" in (so, sv):
        return "keep", "AUTO", "rule=M_note_cue_has_other_lines;check=low"
    side = "vi" if sv == "MUSIC" else "zh"
    if speech_ids:
        return "keep", "AUTO", f"rule=S_speech_over_music_{side}"
    if plain_cover < EXCLUDE_MIN_PLAIN_COVER:
        return "keep", "AUTO", "rule=W_weak_overlap;check=low"
    return "exclude", "AUTO", f"rule=L_lyrics_{side}_music" + (";check=low" if side == "zh" else "")


def music_stage(episode: tuple) -> tuple[list[dict], list[dict]]:
    """Trả về (cue_rows loại tự động, pair_rows cần review).

    Loại tự động mọi cue chỉ gồm nhạc, rồi tạo cặp để review phía còn lại.
    """
    language, lang_tag, _folder, drama_id, episode_id, origin_path, vi_path = episode
    origin = load_cues(origin_path, lang_tag)
    vi = load_cues(vi_path, "vi")

    # 1) Loại tự động mọi cue chỉ gồm nhạc (không cần cặp)
    cue_rows = []
    for tag, cues in ((lang_tag, origin), ("vi", vi)):
        for c in cues:
            if c["auto_music"]:
                cue_rows.append({
                    "language": language, "drama_id": drama_id, "episode_id": episode_id,
                    "subtitle_language": tag, "cue_id": c["cue_id"],
                    "start": seconds_to_srt_time(c["start"]),
                    "end": seconds_to_srt_time(c["end"]),
                    "text_raw": c["text_raw"], "reason": "music_marked",
                    "asr_action": "exclude", "mt_action": "exclude",
                    "review_status": "AUTO", "note": "",
                })

    # 2) Cặp ZH-VI trùng thời gian. Chỉ ghép khi mỗi cue là MUSIC / NOTE_MIXED / CLEAN
    #    (cue có marker khác -> luôn keep nên không cần cặp) và ít nhất một bên có dấu nốt.
    plain_vi = [v for v in vi if not has_note(v)]
    plain_origin = [o for o in origin if not has_note(o)]

    def support(pool, other_music_id, partner):
        out = []
        for pc in pool:
            if pc["cue_id"] == other_music_id:
                continue
            mm = overlap_metrics(pc, partner)
            if mm["overlap_ratio_min"] >= MIN_OVERLAP_RATIO and mm["iou"] >= SPEECH_SUPPORT_MIN_IOU:
                out.append(pc["cue_id"])
        return out

    pair_rows = []
    for o in origin:
        so = pair_state(o)
        if so == "OTHER":
            continue
        for v in vi:
            sv = pair_state(v)
            if sv == "OTHER" or (so == "CLEAN" and sv == "CLEAN") or (so == sv == "MUSIC"):
                continue
            m = overlap_metrics(o, v)
            if m["overlap_sec"] <= 0 or m["overlap_ratio_min"] < MIN_OVERLAP_RATIO:
                continue
            speech, plain_cover = [], 1.0
            if sv == "MUSIC" and so == "CLEAN":        # VI nhạc, ZH không marker
                speech = support(plain_vi, v["cue_id"], o)
                plain_cover = m["origin_cover"]
            elif so == "MUSIC" and sv == "CLEAN":      # ZH nhạc, VI không marker
                speech = support(plain_origin, o["cue_id"], v)
                plain_cover = m["vi_cover"]
            decision, status, rule_note = auto_music_decision(so, sv, speech, plain_cover)
            pair_rows.append({
                "language": language, "drama_id": drama_id, "episode_id": episode_id,
                "origin_cue_id": o["cue_id"], "vi_cue_id": v["cue_id"],
                "origin_start": seconds_to_srt_time(o["start"]),
                "origin_end": seconds_to_srt_time(o["end"]),
                "vi_start": seconds_to_srt_time(v["start"]),
                "vi_end": seconds_to_srt_time(v["end"]),
                "origin_duration": round(o["duration"], 3),
                "vi_duration": round(v["duration"], 3),
                "overlap_sec": round(m["overlap_sec"], 3),
                "overlap_ratio_min": round(m["overlap_ratio_min"], 4),
                "iou": round(m["iou"], 4),
                "origin_text_raw": o["text_raw"], "vi_text_raw": v["text_raw"],
                "detected_markers": fmt_markers(o, v),
                "other_speech_ids": ";".join(speech),
                "decision": decision,
                "review_status": status,
                "note": rule_note,
            })

    return cue_rows, pair_rows


# --- API cho stage 2 và data_process.py ---------------------------------
def exclusion_key(language, drama_id, episode_id, tag, cue_id) -> tuple:
    return (language, drama_id, episode_id, tag, cue_id)


def load_exclusions(excluded_path: Path, pairs_path: Path) -> tuple[set, int]:
    """(tập key cue bị loại, số cặp chưa có decision).

    Cue bị loại khi: (a) dòng AUTO/CONFIRMED trong auto_excluded.csv có
    asr_action = mt_action = exclude; hoặc (b) nó thuộc một cặp có decision = exclude
    (một cue nằm trong nhiều cặp: chỉ cần một cặp exclude là loại).
    """
    excluded, todo = set(), 0
    for r in read_table(excluded_path):
        status = (r.get("review_status") or "").strip().upper()
        if (status in {"AUTO", "CONFIRMED"}
                and (r.get("asr_action") or "").strip().lower() == "exclude"
                and (r.get("mt_action") or "").strip().lower() == "exclude"):
            excluded.add(exclusion_key(r["language"], r["drama_id"], r["episode_id"],
                                       r["subtitle_language"], r["cue_id"]))
    for r in read_table(pairs_path):
        decision = (r.get("decision") or "").strip().lower()
        if decision == "exclude":
            tag = DRAMA_LANGUAGES.get(r["language"], "")
            base = (r["language"], r["drama_id"], r["episode_id"])
            excluded.add(exclusion_key(*base, tag, r["origin_cue_id"]))
            excluded.add(exclusion_key(*base, "vi", r["vi_cue_id"]))
        elif decision != "keep":
            todo += 1
    return excluded, todo


# =====================================================================
# STAGE 2 - CANDIDATES
# =====================================================================
CANDIDATE_FIELDS = [
    "language", "drama_id", "episode_id", "special_side",
    "group_id", "group_type", "n_origin", "n_vi", "vi_turns",
    "origin_cue_id", "vi_cue_id",
    "origin_start", "origin_end", "vi_start", "vi_end",
    "origin_duration", "vi_duration", "overlap_sec",
    "overlap_ratio_min", "iou",
    "origin_text_raw", "vi_text_raw", "detected_markers",
    "origin_dash_line_count", "vi_dash_line_count",
    "origin_same_lang_overlap_ids", "vi_same_lang_overlap_ids",
    "origin_prev_text", "origin_next_text", "vi_prev_text", "vi_next_text",
    "case_type", "asr_action", "mt_action", "alignment_action", "suggested_action", "note",
]
UNMATCHED_FIELDS = [
    "language", "drama_id", "episode_id", "subtitle_language", "cue_id",
    "start", "end", "duration", "text_raw", "detected_markers", "dash_line_count",
    "same_lang_overlap_ids", "prev_text", "next_text",
    "case_type", "asr_action", "mt_action", "note",
]
SUSPECT_FIELDS = [
    "language", "drama_id", "episode_id",
    "vi_cue_a", "vi_cue_b", "a_start", "a_end", "b_start", "b_end", "overlap_sec",
    "a_text", "b_text", "linked_origin_a", "linked_origin_b", "review_status", "note",
]


def prepare_cues(cues: list[dict], excluded_ids: set[str], is_origin: bool) -> tuple[list[dict], list]:
    """Lọc cue đã loại, tính lại ngữ cảnh/overlap trên danh sách còn lại."""
    kept = [dict(c) for c in cues if c["cue_id"] not in excluded_ids]
    for i, c in enumerate(kept):
        c["prev_text"] = kept[i - 1]["text_raw"] if i > 0 else ""
        c["next_text"] = kept[i + 1]["text_raw"] if i + 1 < len(kept) else ""
        c["same_lang_overlap_ids"] = []      # mọi cue cùng ngôn ngữ chồng lên (để tham khảo)
        c["same_lang_overlap_sec"] = 0.0
        c["overlap_flag_ids"] = []           # chỉ các cue được tính cho luật origin_overlap
    pairs = find_same_lang_overlaps(kept)

    def counts_for_overlap(cue: dict) -> bool:
        # cue có marker thuộc OVERLAP_IGNORE_MARKERS (caption [...]) không tham gia luật overlap;
        # cue có marker khác (ngoặc tròn, gạch đầu dòng...) vẫn tính
        return not (set(cue["markers"]) & OVERLAP_IGNORE_MARKERS)

    for a, b, ov in pairs:
        both_count = counts_for_overlap(a) and counts_for_overlap(b)
        for cue, other in ((a, b), (b, a)):
            cue["same_lang_overlap_ids"].append(other["cue_id"])
            cue["same_lang_overlap_sec"] = max(cue["same_lang_overlap_sec"], ov)
            if both_count:
                cue["overlap_flag_ids"].append(other["cue_id"])
    for c in kept:
        reasons = []
        if c["markers"]:
            reasons.append("marker")
        # Overlap chỉ có giá trị làm cờ ở phía gốc; overlap VI một mình hay do timing.
        # Cả hai cue đều phải không có marker trong OVERLAP_IGNORE_MARKERS.
        if is_origin and c["overlap_flag_ids"]:
            reasons.append("origin_overlap")
        c["special_reasons"] = reasons
        c["is_special"] = bool(reasons)
    return kept, pairs


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def classify_group(n_o: int, n_v: int, vi_turns: int) -> str:
    if n_o == 1 and n_v == 1:
        return "one_to_one"
    if n_v == 1 and n_o >= 2:
        return "vi_merged_by_dash" if vi_turns == n_o else "vi_merged"
    if n_o == 1 and n_v >= 2:
        return "origin_merged"
    return "many_to_many"


def candidate_stage(episode: tuple, excluded: set) -> tuple[list, list, list]:
    language, lang_tag, _folder, drama_id, episode_id, origin_path, vi_path = episode

    def ex_ids(tag):
        return {k[4] for k in excluded
                if k[:3] == (language, drama_id, episode_id) and k[3] == tag}

    origin_all = load_cues(origin_path, lang_tag)
    vi_all = load_cues(vi_path, "vi")
    origin, _ = prepare_cues(origin_all, ex_ids(lang_tag), True)
    vi, vi_pairs = prepare_cues(vi_all, ex_ids("vi"), False)
    o_by_id = {c["cue_id"]: c for c in origin}
    v_by_id = {c["cue_id"]: c for c in vi}

    # --- cặp ứng viên theo thời gian ---
    all_pairs = []
    for o in origin:
        for v in vi:
            if not (o["is_special"] or v["is_special"]):
                continue
            m = overlap_metrics(o, v)
            if m["overlap_sec"] <= 0 or m["overlap_ratio_min"] < MIN_OVERLAP_RATIO:
                continue
            all_pairs.append((o, v, m))

    # IoU cao nhất của từng cue đặc biệt (tính trên mọi cặp đạt ngưỡng overlap)
    best_iou_o, best_iou_v = {}, {}
    for o, v, m in all_pairs:
        if o["is_special"]:
            best_iou_o[o["cue_id"]] = max(best_iou_o.get(o["cue_id"], 0.0), m["iou"])
        if v["is_special"]:
            best_iou_v[v["cue_id"]] = max(best_iou_v.get(v["cue_id"], 0.0), m["iou"])

    # Giữ cặp khi: cả hai cue đều đặc biệt; hoặc IoU >= CANDIDATE_MIN_IOU; hoặc là cặp tốt nhất
    # của một cue đặc biệt. Cặp chỉ được giữ nhờ luật "tốt nhất" (IoU thấp) -> note = weak_match.
    raw_rows, weak_pairs, matched_o, matched_v = [], set(), set(), set()
    for o, v, m in all_pairs:
        both_special = o["is_special"] and v["is_special"]
        # cue có marker thuộc CANDIDATE_IOU_EXEMPT_MARKERS (leading_dash) -> giữ mọi cặp đạt overlap
        exempt = bool((set(o["markers"]) | set(v["markers"])) & CANDIDATE_IOU_EXEMPT_MARKERS)
        strong = m["iou"] >= CANDIDATE_MIN_IOU
        best = ((o["is_special"] and m["iou"] >= best_iou_o[o["cue_id"]])
                or (v["is_special"] and m["iou"] >= best_iou_v[v["cue_id"]]))
        if not (both_special or strong or best or exempt):
            continue
        if not (both_special or strong or exempt):
            weak_pairs.add((o["cue_id"], v["cue_id"]))
        matched_o.add(o["cue_id"])
        matched_v.add(v["cue_id"])
        raw_rows.append((o, v, m))

    # --- gom nhóm ---
    uf = UnionFind()
    for o, v, _ in raw_rows:
        uf.union("o:" + o["cue_id"], "v:" + v["cue_id"])
    members = defaultdict(lambda: {"o": set(), "v": set()})
    for o, v, _ in raw_rows:
        root = uf.find("o:" + o["cue_id"])
        members[root]["o"].add(o["cue_id"])
        members[root]["v"].add(v["cue_id"])
    group_ids = {root: f"{language}_{drama_id}_{episode_id}_G{i:04d}"
                 for i, root in enumerate(sorted(members), start=1)}

    # --- dựng dòng CSV ---
    rows = []
    for o, v, m in raw_rows:
        root = uf.find("o:" + o["cue_id"])
        n_o, n_v = len(members[root]["o"]), len(members[root]["v"])
        vi_turns = (v_by_id[next(iter(members[root]["v"]))]["dash_line_count"]
                    if n_v == 1 else "")
        gtype = classify_group(n_o, n_v, vi_turns if n_v == 1 else 0)

        if gtype == "one_to_one" and m["iou"] >= AUTO_KEEP_MIN_IOU \
                and v["dash_line_count"] < 2 and o["dash_line_count"] < 2:
            suggestion = "keep_pair"
        elif gtype == "vi_merged_by_dash":
            suggestion = "split_by_dash"
        else:
            suggestion = "review"

        if o["is_special"] and v["is_special"]:
            side = "BOTH"
        elif o["is_special"]:
            side = "ORIGIN_ONLY"
        else:
            side = "VI_ONLY"

        rows.append({
            "language": language, "drama_id": drama_id, "episode_id": episode_id,
            "special_side": side,
            "group_id": group_ids[root], "group_type": gtype,
            "n_origin": n_o, "n_vi": n_v, "vi_turns": vi_turns,
            "suggested_action": suggestion,
            "origin_cue_id": o["cue_id"], "vi_cue_id": v["cue_id"],
            "origin_start": seconds_to_srt_time(o["start"]),
            "origin_end": seconds_to_srt_time(o["end"]),
            "vi_start": seconds_to_srt_time(v["start"]),
            "vi_end": seconds_to_srt_time(v["end"]),
            "origin_duration": round(o["duration"], 3),
            "vi_duration": round(v["duration"], 3),
            "overlap_sec": round(m["overlap_sec"], 3),
            "overlap_ratio_min": round(m["overlap_ratio_min"], 4),
            "iou": round(m["iou"], 4),
            "origin_text_raw": o["text_raw"], "vi_text_raw": v["text_raw"],
            "detected_markers": fmt_markers(o, v),
            "origin_dash_line_count": o["dash_line_count"],
            "vi_dash_line_count": v["dash_line_count"],
            "origin_same_lang_overlap_ids": ";".join(o["same_lang_overlap_ids"]),
            "vi_same_lang_overlap_ids": ";".join(v["same_lang_overlap_ids"]),
            "origin_prev_text": o["prev_text"], "origin_next_text": o["next_text"],
            "vi_prev_text": v["prev_text"], "vi_next_text": v["next_text"],
            "alignment_action": "review",
            "note": "weak_match" if (o["cue_id"], v["cue_id"]) in weak_pairs else "",
        })

    # --- unmatched ---
    unmatched = []
    for tag, cues, matched in ((lang_tag, origin, matched_o), ("vi", vi, matched_v)):
        for c in cues:
            if c["is_special"] and c["cue_id"] not in matched:
                unmatched.append({
                    "language": language, "drama_id": drama_id, "episode_id": episode_id,
                    "subtitle_language": tag, "cue_id": c["cue_id"],
                    "start": seconds_to_srt_time(c["start"]),
                    "end": seconds_to_srt_time(c["end"]),
                    "duration": round(c["duration"], 3), "text_raw": c["text_raw"],
                    "detected_markers": ";".join(c["markers"]),
                    "dash_line_count": c["dash_line_count"],
                    "same_lang_overlap_ids": ";".join(c["same_lang_overlap_ids"]),
                    "prev_text": c["prev_text"], "next_text": c["next_text"],
                    "note": "Không có cue phía kia đạt ngưỡng overlap/IoU; chưa kết luận là không có bản dịch/gốc.",
                })

    # --- overlap VI một mình (nghi timing), không vào candidate ---
    def linked_origin(v):
        return [o for o in origin
                if overlap_metrics(o, v)["overlap_ratio_min"] >= MIN_OVERLAP_RATIO]

    suspect = []
    for a, b, ov in vi_pairs:
        if a["is_special"] or b["is_special"]:   # đã có marker -> nằm trong candidate
            continue
        la, lb = linked_origin(a), linked_origin(b)
        if any(o["is_special"] for o in la + lb):  # phía gốc cũng overlap -> đã vào candidate
            continue
        suspect.append({
            "language": language, "drama_id": drama_id, "episode_id": episode_id,
            "vi_cue_a": a["cue_id"], "vi_cue_b": b["cue_id"],
            "a_start": seconds_to_srt_time(a["start"]), "a_end": seconds_to_srt_time(a["end"]),
            "b_start": seconds_to_srt_time(b["start"]), "b_end": seconds_to_srt_time(b["end"]),
            "overlap_sec": round(ov, 3), "a_text": a["text_raw"], "b_text": b["text_raw"],
            "linked_origin_a": ";".join(o["cue_id"] for o in la),
            "linked_origin_b": ";".join(o["cue_id"] for o in lb),
            "review_status": "TODO",
            "note": "Chỉ VI chồng nhau, phía gốc không chồng: nghi timing VI kéo dài. Nghe audio mẫu để xác nhận.",
        })
    return rows, unmatched, suspect


# =====================================================================
# main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Lọc subtitle đặc biệt origin-VI (2 stage).")
    ap.add_argument("--stage", choices=["music", "candidates"], required=True)
    ap.add_argument("--language", choices=list(DRAMA_LANGUAGES.keys()))
    ap.add_argument("--drama", help="Ví dụ D0002")
    ap.add_argument("--episode", help="Ví dụ 1 hoặc EP001")
    ap.add_argument("--reset", action="store_true",
                    help="Xóa output của stage này (kể cả kết quả review) trước khi chạy.")
    ap.add_argument("--allow-todo", action="store_true",
                    help="Stage 2: chạy dù auto_excluded.csv còn dòng TODO (TODO coi như chưa loại).")
    args = ap.parse_args()

    ep_filter = args.episode
    if ep_filter:
        ep_filter = ep_filter.upper()
        if not ep_filter.startswith("EP"):
            ep_filter = pad_episode(ep_filter)

    out_dir = DRAMA_PROCESSED_DIR / "review"
    excluded_path = out_dir / "auto_excluded"
    pairs_path = out_dir / "music_pairs_review"
    cand_path = out_dir / "special_alignment_candidates"
    unm_path = out_dir / "special_unmatched"
    sus_path = out_dir / "vi_overlap_suspect"
    done_path = out_dir / f".done_{args.stage}.txt"

    if args.reset:
        targets = [excluded_path, pairs_path] if args.stage == "music" else [cand_path, unm_path, sus_path]
        for p in targets:
            p.with_suffix(".csv").unlink(missing_ok=True)
        done_path.unlink(missing_ok=True)

    episodes = find_episodes(args.language, args.drama, ep_filter)
    if not episodes:
        print("Không tìm thấy cặp SRT manual_raw tương ứng.")
        print(f"Đã tìm trong: {DRAMA_RAW_DIR}")
        return

    excluded = set()
    if args.stage == "candidates":
        if not excluded_path.with_suffix(".csv").exists():
            print(f"Chưa có auto_excluded.csv. Hãy chạy --stage music và review trước.")
            return
        excluded, todo = load_exclusions(excluded_path, pairs_path)
        if todo and not args.allow_todo:
            print(f"music_pairs_review.csv còn {todo} cặp chưa có decision.")
            print("Điền decision = exclude/keep rồi chạy lại, hoặc dùng --allow-todo.")
            return
        print(f"Số cue bị loại (AUTO/CONFIRMED): {len(excluded)}")

    done = load_done(done_path)
    print(f"Tìm thấy {len(episodes)} tập có đủ SRT nguồn và VI.")
    totals = defaultdict(int)

    for ep in episodes:
        language, lang_tag, folder, drama_id, episode_id, _o, _v = ep
        key = f"{language}|{drama_id}|{episode_id}"
        if key in done:
            print(f"[skip] {folder} {episode_id}: đã xử lý (dùng --reset để chạy lại)")
            continue
        try:
            if args.stage == "music":
                cue_rows, pair_rows = music_stage(ep)
                append_table(excluded_path, EXCLUDED_FIELDS, cue_rows)
                append_table(pairs_path, PAIR_FIELDS, pair_rows)
                totals["auto"] += len(cue_rows)
                totals["pairs"] += len(pair_rows)
                totals["pairs_todo"] += sum(r["review_status"] == "TODO" for r in pair_rows)
                print(f"[{language} / {folder} / {episode_id}] auto_excluded={len(cue_rows)}, "
                      f"cặp nhạc={len(pair_rows)} (tự điền AUTO="
                      f"{sum(r['review_status'] == 'AUTO' for r in pair_rows)}, "
                      f"cần điền TODO={sum(r['review_status'] == 'TODO' for r in pair_rows)})")
            else:
                cands, unm, sus = candidate_stage(ep, excluded)
                append_table(cand_path, CANDIDATE_FIELDS, cands)
                append_table(unm_path, UNMATCHED_FIELDS, unm)
                append_table(sus_path, SUSPECT_FIELDS, sus)
                totals["candidates"] += len(cands)
                totals["unmatched"] += len(unm)
                totals["suspect"] += len(sus)
                totals["keep_pair"] += sum(r["suggested_action"] == "keep_pair" for r in cands)
                print(f"[{language} / {folder} / {episode_id}] candidate={len(cands)}, "
                      f"unmatched={len(unm)}, vi_overlap_suspect={len(sus)}")
            mark_done(done_path, key)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {folder} {episode_id}: {exc}")

    print("\nHoàn tất. Tổng:", dict(totals))
    if args.stage == "music":
        print(f"Tự động loại: {excluded_path}.csv\nReview cặp: {pairs_path}.csv  (kiểm tra decision; sửa thì đổi review_status thành CONFIRMED)")
    else:
        print(f"Candidate: {cand_path}.csv\nUnmatched: {unm_path}.csv\nVI overlap suspect: {sus_path}.csv")


if __name__ == "__main__":
    main()