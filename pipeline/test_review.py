"""
Lọc subtitle đặc biệt + alignment sơ bộ Trung/Hàn/Nhật - Việt (2 stage).

Chỉ ĐỌC raw SRT, không sửa/xóa dữ liệu gốc. Dùng config.py và utils.py hiện có.

STAGE 1 - music: python test_review.py --stage music
    - auto_excluded.csv: MỌI cue (VI và ZH) chỉ gồm nốt nhạc -> loại tự động (AUTO).
    - music_pairs_review.csv: cặp ZH-VI trùng thời gian, gồm đúng 2 loại:
        (1) một bên là nhạc thuần, bên kia KHÔNG có marker nào (ngoài html/ass);
        (2) có cue "có dấu nốt nhưng còn dòng khác" (NOTE_MIXED).
      Cue có marker khác (ngoặc, gạch đầu dòng, sound effect) không ghép cặp vì luôn keep.
      Cột alignment_action điền sẵn theo auto_music_alignment, review_status = AUTO, mã luật ở cột
      note (rule=...), ca kém chắc chắn có thêm check=low -> soát trước.
        keep_pair : cặp ZH-VI là cặp thật (lời bài hát ở cả hai phía) -> loại cả hai cue khỏi ASR/MT
        reject    : hai cue chỉ chồng thời gian, không phải một cặp (thoại đè nhạc...) -> không loại cue nào
                    ở bước này (cue nhạc thuần đã nằm trong auto_excluded.csv)
      Sai chỗ nào thì sửa alignment_action và đổi review_status thành CONFIRMED.
      (Dữ liệu cũ: decision exclude = keep_pair, decision keep = reject.)

STAGE 2 - candidates: python test_review.py --stage candidates
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
    - Cue có marker leading_dash KHÔNG vào special_alignment_candidates.csv mà được tách sang
      dash_pairs_review.csv: mỗi nhóm là MỘT dòng (xem dash_stage).
        dash_side = VI_ONLY / ORIGIN_ONLY / BOTH (phía nào có leading dash)
        suggested_action:
          merge_origin_and_vi : N cue gốc <-> 1 cue VI có N dash (hoặc ngược lại), đủ vế
                                -> gộp N cue thành 1, phía dash bỏ dash và nối liền câu
          merge_vi            : 1 cue gốc có N vế cách nhau bằng dấu cách <-> 1 cue VI có N dash
                                -> chỉ bỏ dash và nối liền câu VI
          review              : thiếu/thừa vế hoặc bất thường (lý do ở note)
      Mọi cue gốc - VI chồng thời gian >= DASH_MIN_OVERLAP_SEC (mặc định 0.1 giây) với cue leading_dash đều được gom vào nhóm.
      origin_lines / vi_lines đánh số các lượt thoại của cả zh và vi; origin_keep_lines / vi_keep_lines
      (bạn điền, tùy chọn) chọn lượt cần giữ và THỨ TỰ nối, để bỏ lượt thừa hoặc sắp lại zh/vi cho khớp.
      note để trống. Hàm gộp cue (thời gian + nối text) nằm trong utils.py: merge_cues, merge_cue_groups.
      case_type, asr_action, mt_action, alignment_action do bạn điền (có điền sẵn vài mặc định).
    Output: special_alignment_candidates.csv, special_unmatched.csv,
            vi_overlap_suspect.csv, dash_pairs_review.csv

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
    MAX_MERGED_DURATION,
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
    merge_cues,
    split_dash_turns,
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
    # Các giá trị tính cho 1 cặp (a = cue gốc, b = cue VI):
    #   ov (overlap_sec)    = số giây hai cue chồng nhau
    #   shorter             = thời lượng của cue NGẮN hơn
    #   union               = độ dài đoạn bao hai cue (từ start nhỏ nhất đến end lớn nhất)
    return {
        "overlap_sec": ov,                                              # giây chồng nhau
        "overlap_ratio_min": ov / shorter if shorter > 0 else 0.0,      # chồng / cue ngắn hơn (ngưỡng ghép cặp)
        "iou": ov / union if union > 0 else 0.0,                        # chồng / đoạn bao (độ khớp tổng thể)
        "origin_cover": ov / da if da > 0 else 0.0,                     # cue gốc bị cue VI phủ bao nhiêu
        "vi_cover": ov / db if db > 0 else 0.0,                         # cue VI bị cue gốc phủ bao nhiêu
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
# auto_excluded: mỗi dòng = 1 cue chỉ gồm nhạc.
#   detected_markers : marker của cue, dạng <ngôn ngữ>:<marker>, vd vi:music_note hoặc zh:music_note
#   special_side     : VI_ONLY nếu cue VI, ORIGIN_ONLY nếu cue gốc (zh/ko/ja)
#   case_type        : mặc định music;  asr_action / mt_action mặc định exclude;  note để trống
EXCLUDED_FIELDS = [
    "language", "drama_id", "episode_id", "subtitle_language", "cue_id",
    "start", "end", "text_raw", "detected_markers", "special_side",
    "case_type", "asr_action", "mt_action", "review_status", "note",
]
PAIR_FIELDS = [
    "language", "drama_id", "episode_id", "origin_cue_id", "vi_cue_id",
    "origin_start", "origin_end", "vi_start", "vi_end",
    "origin_duration", "vi_duration",
    "overlap_sec", "overlap_ratio_min", "iou",
    "origin_text_raw", "vi_text_raw", "detected_markers", "special_side",
    "other_speech_ids",
    "alignment_action", "review_status", "note",
]
# music_pairs_review: special_side = phía có dấu nốt ♪ (VI_ONLY / ORIGIN_ONLY / BOTH);
# other_speech_ids = cue thoại cùng ngôn ngữ khác khớp với cue còn lại (bằng chứng thoại đè nhạc).


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


def auto_music_alignment(so: str, sv: str, speech_ids: list[str],
                         plain_cover: float) -> tuple[str, str, str]:
    """Điền sẵn alignment_action cho cặp nhạc. Trả về (alignment_action, review_status, note).

    Chỉ nhận cặp thuộc 2 loại (cặp có cue OTHER không được tạo, vì luôn keep):
      - MUSIC x CLEAN      (nhạc thuần ghép cue không marker, một trong hai phía là ZH/VI)
      - NOTE_MIXED x (MUSIC / CLEAN / NOTE_MIXED)

    `keep_pair` (cặp lời hát thật) loại cả hai cue khỏi ASR/MT nên chỉ chọn khi rõ là lời hát;
    còn lại `reject` (không loại cue nào ở bước này).
      M_note_cue_has_other_lines : có cue NOTE_MIXED -> reject, check=low (loại sẽ mất phần thoại)
      S_speech_over_music_<vi|zh>: cue nhạc còn cue thoại cùng ngôn ngữ khác khớp cue kia -> reject
      W_weak_overlap             : cue nhạc phủ < EXCLUDE_MIN_PLAIN_COVER cue kia -> reject, check=low
      L_lyrics_<vi|zh>_music     : còn lại -> keep_pair (lời hát phía kia không đánh dấu).
                                   Nhánh zh_music chưa có dữ liệu kiểm chứng -> check=low.
    """
    if "NOTE_MIXED" in (so, sv):
        return "reject", "AUTO", "rule=M_note_cue_has_other_lines;check=low"
    side = "vi" if sv == "MUSIC" else "zh"
    if speech_ids:
        return "reject", "AUTO", f"rule=S_speech_over_music_{side}"
    if plain_cover < EXCLUDE_MIN_PLAIN_COVER:
        return "reject", "AUTO", "rule=W_weak_overlap;check=low"
    return "keep_pair", "AUTO", f"rule=L_lyrics_{side}_music" + (";check=low" if side == "zh" else "")


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
                    "text_raw": c["text_raw"],
                    # marker của cue, gắn tiền tố ngôn ngữ: vi:music_note / zh:music_note
                    "detected_markers": ";".join(f"{tag}:{mk}" for mk in c["markers"]),
                    "special_side": "VI_ONLY" if tag == "vi" else "ORIGIN_ONLY",
                    "case_type": "music",
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
            alignment, status, rule_note = auto_music_alignment(so, sv, speech, plain_cover)
            # special_side: phía có dấu nốt ♪ (cue "đặc biệt" của cặp)
            note_o, note_v = has_note(o), has_note(v)
            special_side = "BOTH" if (note_o and note_v) else ("ORIGIN_ONLY" if note_o else "VI_ONLY")
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
                "special_side": special_side,
                "other_speech_ids": ";".join(speech),
                "alignment_action": alignment,
                "review_status": status,
                "note": rule_note,
            })

    return cue_rows, pair_rows


# --- API cho stage 2 và data_process.py ---------------------------------
def exclusion_key(language, drama_id, episode_id, tag, cue_id) -> tuple:
    return (language, drama_id, episode_id, tag, cue_id)


def load_exclusions(excluded_path: Path, pairs_path: Path) -> tuple[set, int]:
    """(tập key cue bị loại, số cặp chưa có alignment_action).

    Cue bị loại khi: (a) dòng AUTO/CONFIRMED trong auto_excluded.csv có
    asr_action = mt_action = exclude; hoặc (b) nó thuộc một cặp có alignment_action = keep_pair
    (cặp lời hát thật; một cue nằm trong nhiều cặp: chỉ cần một cặp keep_pair là loại).
    alignment_action = reject không loại cue nào.
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
        alignment = (r.get("alignment_action") or "").strip().lower()
        if alignment == "keep_pair":
            tag = DRAMA_LANGUAGES.get(r["language"], "")
            base = (r["language"], r["drama_id"], r["episode_id"])
            excluded.add(exclusion_key(*base, tag, r["origin_cue_id"]))
            excluded.add(exclusion_key(*base, "vi", r["vi_cue_id"]))
        elif alignment != "reject":
            todo += 1
    return excluded, todo


# =====================================================================
# STAGE 2 - CANDIDATES
# =====================================================================
# Ý nghĩa các cột special_alignment_candidates (mỗi dòng = 1 cặp cue gốc - VI trùng thời gian):
#   special_side     : ORIGIN_ONLY / VI_ONLY / BOTH - phía nào là cue "đặc biệt" (có marker, hoặc origin_overlap)
#   group_id         : các dòng nối nhau qua cue chung (connected component) có cùng group_id
#   group_type       : one_to_one | vi_merged (N gốc - 1 VI) | origin_merged (1 gốc - N VI) | many_to_many
#   n_origin, n_vi   : số cue gốc / VI trong nhóm
#   *_start, *_end, *_duration : thời gian từng cue (SRT time; duration tính bằng giây)
#   overlap_sec      : số giây hai cue chồng nhau
#   overlap_ratio_min: overlap_sec / thời lượng cue NGẮN hơn   (>= MIN_OVERLAP_RATIO mới thành cặp)
#   iou              : overlap_sec / đoạn bao hai cue          (gần 1 = khớp cả đầu lẫn cuối)
#   detected_markers : marker từng phía, dạng origin:xxx;vi:yyy
#   *_same_lang_overlap_ids : cue cùng ngôn ngữ chồng lên cue này (tham khảo timing)
#   *_prev_text / *_next_text : text cue liền trước / sau (đã bỏ cue nhạc) để xem ngữ cảnh
#   suggested_action : keep_pair (1-1 và iou >= AUTO_KEEP_MIN_IOU) hoặc review
#   note             : weak_match = cặp chỉ được giữ vì là cặp IoU cao nhất của cue đặc biệt
#   case_type, asr_action, mt_action, alignment_action : bạn điền thủ công
# (Đã bỏ vi_turns và origin/vi_dash_line_count: cue leading_dash không còn vào bảng này nên các cột đó luôn 0/rỗng.)
CANDIDATE_FIELDS = [
    "language", "drama_id", "episode_id", "special_side",
    "group_id", "group_type", "n_origin", "n_vi",
    "origin_cue_id", "vi_cue_id",
    "origin_start", "origin_end", "vi_start", "vi_end",
    "origin_duration", "vi_duration", "overlap_sec",
    "overlap_ratio_min", "iou",
    "origin_text_raw", "vi_text_raw", "detected_markers",
    "origin_same_lang_overlap_ids", "vi_same_lang_overlap_ids",
    "origin_prev_text", "origin_next_text", "vi_prev_text", "vi_next_text",
    "case_type", "asr_action", "mt_action", "alignment_action", "suggested_action", "note",
]
# special_unmatched: cue đặc biệt không có cặp. special_side = VI_ONLY (cue VI) / ORIGIN_ONLY (cue gốc).
UNMATCHED_FIELDS = [
    "language", "drama_id", "episode_id", "cue_id",
    "start", "end", "duration", "text_raw", "detected_markers", "special_side",
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
        # prev_text / next_text: text cue liền trước / sau TRONG danh sách đã lọc (để xem ngữ cảnh)
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
    # is_special: cue có marker, hoặc (phía gốc) chồng lên cue gốc khác -> vào candidates/unmatched/dash
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


def classify_group(n_o: int, n_v: int) -> str:
    """Loại nhóm theo số cue mỗi phía: one_to_one / vi_merged / origin_merged / many_to_many."""
    if n_o == 1 and n_v == 1:
        return "one_to_one"
    if n_v == 1 and n_o >= 2:
        return "vi_merged"          # nhiều cue gốc ứng với 1 cue VI
    if n_o == 1 and n_v >= 2:
        return "origin_merged"      # 1 cue gốc ứng với nhiều cue VI
    return "many_to_many"


# ---------------------------------------------------------------------
# leading_dash -> dash_pairs_review.csv (mỗi NHÓM một dòng)
# ---------------------------------------------------------------------
# Mọi cue gốc - VI CHỒNG thời gian với một cue leading_dash tối thiểu DASH_MIN_OVERLAP_SEC giây
# đều được gom vào cùng nhóm, cộng thêm các cue thường chồng lên cue đã trong nhóm (mở rộng 1 bước); nhóm chỉ để bạn xem và quyết định, không tự loại cue nào.
# Ngưỡng chỉ còn dùng để gợi ý merge tự động (suggested_action), không ảnh hưởng việc gom nhóm.
# Ngưỡng riêng cho nhóm dash (có thể chuyển sang config.py nếu muốn gom chung).
DASH_MIN_OVERLAP_SEC = 0.1    # cue phía kia phải chồng cue dash >= số giây này ...
DASH_MIN_COVER = 0.2          # ... và được cue dash phủ >= 20% thời lượng của chính nó
"""
DASH_MIN_OVERLAP_SEC = 0.1    # NGƯỠNG GOM NHÓM: hai cue phải chồng nhau >= số giây này mới được gom vào nhóm
                              # (đặt 0 để gom mọi cặp chồng nhau; áp dụng cả bước gom đầu lẫn bước mở rộng 1 bước)
"""
DASH_MAX_GAP_SEC = 0.5        # khoảng hở lớn nhất giữa các cue sẽ gộp; vượt -> review
DASH_MAX_AUTO_LINES = 2       # cue có > 2 lượt thoại (dòng dash) -> luôn review
DASH_LIGHT_MIN_IOU = 0.8      # IoU sau khi gộp >= ngưỡng (và có gợi ý merge) -> light, ngược lại heavy
INLINE_SPLIT_RE = re.compile(r"[ \u3000]+")   # dấu cách thường / toàn giác ngăn cách các vế trong 1 cue gốc

# Ý nghĩa các cột dash_pairs_review (mỗi dòng = 1 NHÓM cue liên quan tới leading dash):
#   dash_side        : VI_ONLY / ORIGIN_ONLY / BOTH - phía nào có cue leading dash
#   dash_match       : full / inline / partial / mismatch / single_dash / both_dash (xem dash_stage)
#   overlap_severity : light = có gợi ý merge và iou sau gộp >= DASH_LIGHT_MIN_IOU; còn lại heavy
#   origin_cue_ids, vi_cue_ids : id các cue của nhóm, theo thứ tự thời gian (ngăn bằng ';')
#   n_origin, n_vi   : số cue mỗi phía;  n_dash_lines: số dòng dash của phía có dash
#   *_start, *_end, *_duration : ĐOẠN BAO (start nhỏ nhất, end lớn nhất) của các cue mỗi phía
#   max_gap_sec      : khoảng hở lớn nhất giữa hai cue liên tiếp trong cùng một phía (âm = chồng nhau)
#   overlap_sec, overlap_ratio_min, iou : tính trên hai ĐOẠN BAO, tức SAU khi gộp
#   *_text_raw       : text gốc, nhiều cue nối bằng " || "
#   origin_lines, vi_lines : MỌI lượt thoại của các cue mỗi phía, đánh số liên tục theo thứ tự thời gian,
#                      dạng "1: ... | 2: ...". Lượt = một dòng dash (dòng xuống hàng không dash nối vào
#                      lượt trước); cue không có dash là 1 lượt
#   merged_*_text    : bản gộp mặc định (mọi lượt, theo thứ tự thời gian, nối bằng 1 dấu cách) để xem trước
#   origin_keep_lines, vi_keep_lines : BẠN ĐIỀN (tùy chọn): số thứ tự các lượt cần giữ, theo THỨ TỰ MUỐN NỐI,
#                      ngăn bằng ';' (vd "2" hoặc "2;1"). Để trống = giữ mọi lượt theo thứ tự thời gian.
#                      Dùng để bỏ lượt thừa, hoặc đảo thứ tự zh/vi cho khớp nhau (chỉ ảnh hưởng text MT;
#                      text ASR luôn là mọi lượt zh theo thứ tự thời gian vì phải khớp audio)
#   suggested_action : merge_origin_and_vi / merge_vi / partial_match_by_dash / review
DASH_FIELDS = [
    "language", "drama_id", "episode_id",
    "group_id", "dash_side", "dash_match", "overlap_severity",
    "origin_cue_ids", "vi_cue_ids", "n_origin", "n_vi", "n_dash_lines",
    "origin_start", "origin_end", "vi_start", "vi_end",
    "origin_duration", "vi_duration", "max_gap_sec",
    "overlap_sec", "overlap_ratio_min", "iou",
    "origin_text_raw", "vi_text_raw",
    "origin_lines", "vi_lines",
    "merged_origin_text", "merged_vi_text",
    "detected_markers",
    "origin_same_lang_overlap_ids", "vi_same_lang_overlap_ids",
    "origin_prev_text", "origin_next_text", "vi_prev_text", "vi_next_text",
    "origin_keep_lines", "vi_keep_lines",
    "case_type", "asr_action", "mt_action", "alignment_action", "suggested_action", "note",
]


def has_dash(c: dict) -> bool:
    return "leading_dash" in c["markers"]


def inline_parts(text: str) -> int:
    """Số vế trong 1 cue gốc (1 dòng) tách nhau bằng dấu cách. Cue nhiều dòng -> 0 (không xét)."""
    text = normalize_for_detection(text).strip()
    if not text or len(text.splitlines()) > 1:
        return 0
    return len([p for p in INLINE_SPLIT_RE.split(text) if p])


def fmt_markers_multi(o_list: list[dict], v_list: list[dict]) -> str:
    def collect(lst):
        return [m for m in MARKER_PATTERNS if any(m in c["markers"] for c in lst)]
    return ";".join([f"origin:{m}" for m in collect(o_list)] + [f"vi:{m}" for m in collect(v_list)])

def dash_link(o: dict, v: dict, m: dict) -> bool:
    """Ngưỡng nới hơn candidate thường để không mất cue lân cận của một lượt thoại.
    Cue phía kia (không dash) phải được cue dash phủ đủ phần của chính nó."""
    if m["overlap_sec"] < DASH_MIN_OVERLAP_SEC:
        return False
    o_dash, v_dash = has_dash(o), has_dash(v)
    if o_dash and v_dash:
        return m["overlap_ratio_min"] >= MIN_OVERLAP_RATIO
    cover = m["origin_cover"] if v_dash else m["vi_cover"]
    return cover >= DASH_MIN_COVER

def numbered_turns(cues: list[dict]) -> str:
    """Đánh số liên tục các lượt thoại của các cue (đã sắp theo thời gian): '1: ... | 2: ...'.
    Cách chia lượt và thứ tự trùng với data_process.py, nên số bạn điền vào *_keep_lines luôn khớp."""
    turns = [t for c in cues for t in split_dash_turns(c["text_raw"])]
    return " | ".join(f"{k}: {t}" for k, t in enumerate(turns, start=1))


def dash_stage(language: str, drama_id: str, episode_id: str,
               origin: list[dict], vi: list[dict]) -> tuple[list[dict], set, set]:
    """Gom các cue liên quan tới leading_dash thành nhóm, mỗi nhóm một dòng.

    dash_side: VI_ONLY / ORIGIN_ONLY / BOTH (phía nào có cue leading_dash).
    dash_match:
      full        : phía dash đúng 1 cue, số cue phía kia == số dòng dash
      inline      : VI_ONLY, 1 cue gốc (1 dòng, không dash) có đúng n_dash vế cách nhau bằng dấu cách
      partial     : thiếu vế (số cue phía kia < số dòng dash)
      mismatch    : thừa vế, hoặc phía dash có > 1 cue trong nhóm (chuỗi cue nối nhau)
      single_dash : cue chỉ có 1 dòng dash
      both_dash   : cả hai phía đều có dash
    suggested_action (chỉ gợi ý; case_type / asr_action / mt_action / alignment_action và hai cột
    *_keep_lines do bạn điền):
      merge_origin_and_vi : match = full -> gộp các cue của phía nhiều cue thành 1 cue
                            (start = min, end = max, nối text), phía dash bỏ dash và nối liền câu
      merge_vi            : match = inline -> chỉ bỏ dash và nối liền câu VI, giữ nguyên cue gốc
      partial_match_by_dash : match = partial (một phía thiếu bớt vế, vd zh chỉ có vế 2 của
                            "- vế 1 / - vế 2"). Bạn điền *_keep_lines của phía thừa vế để chọn các lượt
                            ứng với phía kia; asr_action điền sẵn exclude khi dash ở VI (audio còn vế không
                            có text zh), mt_action keep
      review              : còn lại (thừa vế, gap lớn, > 2 lượt thoại, ...)
    Điều kiện gợi ý merge: <= DASH_MAX_AUTO_LINES lượt thoại; với full còn cần
    max_gap_sec <= DASH_MAX_GAP_SEC và đoạn gộp <= MAX_MERGED_DURATION.
    overlap_severity: light = có gợi ý merge và IoU sau gộp >= DASH_LIGHT_MIN_IOU; còn lại heavy.
    Trả về (rows, id cue gốc đã vào nhóm, id cue VI đã vào nhóm).
    """
    # --- 1) liên kết: MỌI cặp (cue gốc, cue VI) chồng thời gian >= DASH_MIN_OVERLAP_SEC có ít nhất một cue leading_dash ---
    links = {}
    for v in (c for c in vi if has_dash(c)):
        for o in origin:
            if dash_link(o, v, overlap_metrics(o, v)):
            # if overlap_metrics(o, v)["overlap_sec"] >= DASH_MIN_OVERLAP_SEC:
                links[(o["cue_id"], v["cue_id"])] = (o, v)
    for o in (c for c in origin if has_dash(c)):
        for v in vi:
            if (o["cue_id"], v["cue_id"]) in links:
                continue
            if dash_link(o, v, overlap_metrics(o, v)):
            # if overlap_metrics(o, v)["overlap_sec"] >= DASH_MIN_OVERLAP_SEC:
                links[(o["cue_id"], v["cue_id"])] = (o, v)
    if not links:
        return [], set(), set()
    
    """
    # --- 1b) mở rộng 1 bước: cue THƯỜNG (không marker) chồng (>= DASH_MIN_OVERLAP_SEC) lên BẤT KỲ cue nào đã trong nhóm ---
    # Vd cue VI "Không ác" kết thúc đúng lúc cue dash bắt đầu nên không chồng cue dash, nhưng chồng cue gốc
    # của nhóm -> cần vào nhóm, nếu không nó thành cue VI thừa. Chỉ mở rộng đúng 1 bước (tập thành viên tính
    # trước vòng lặp) để nhóm không lan dần cả tập; cue có marker khác đã có quyết định ở bảng candidates nên
    # không bị kéo vào đây.
    in_o = {oid for oid, _ in links}
    in_v = {vid for _, vid in links}
    for o in origin:
        for v in vi:
            key = (o["cue_id"], v["cue_id"])
            if key in links:
                continue
            o_in, v_in = o["cue_id"] in in_o, v["cue_id"] in in_v
            if not (o_in or v_in):
                continue
            if (not o_in and o["is_special"]) or (not v_in and v["is_special"]):
                continue
            if overlap_metrics(o, v)["overlap_sec"] >= DASH_MIN_OVERLAP_SEC:
                links[key] = (o, v)
    """

    # --- 2) gom nhóm (connected component): các cặp dùng chung một cue thì cùng nhóm ---
    uf = UnionFind()
    for oid, vid in links:
        uf.union("o:" + oid, "v:" + vid)
    groups = defaultdict(lambda: {"o": {}, "v": {}})
    for (oid, vid), (o, v) in links.items():
        root = uf.find("o:" + oid)
        groups[root]["o"][oid] = o
        groups[root]["v"][vid] = v
    ordered = sorted(groups.values(),
                     key=lambda g: min(c["start"] for c in [*g["o"].values(), *g["v"].values()]))

    rows, used_o, used_v = [], set(), set()
    for i, g in enumerate(ordered, start=1):
        o_list = sorted(g["o"].values(), key=lambda c: (c["start"], c["end"]))
        v_list = sorted(g["v"].values(), key=lambda c: (c["start"], c["end"]))
        used_o.update(c["cue_id"] for c in o_list)
        used_v.update(c["cue_id"] for c in v_list)
        o_lines = sum(c["dash_line_count"] for c in o_list)   # tổng số dòng dash phía gốc
        v_lines = sum(c["dash_line_count"] for c in v_list)   # tổng số dòng dash phía VI
        o_has = any(has_dash(c) for c in o_list)              # phía gốc có cue leading_dash không
        v_has = any(has_dash(c) for c in v_list)              # phía VI có cue leading_dash không

        # --- 3) gộp cue (tính thời gian + nối text) bằng hàm dùng chung trong utils.py ---
        # mo / mv: cue đã gộp mỗi phía -> start = min, end = max, text nối (bỏ dash nếu phía đó có dash)
        mo = merge_cues(o_list, "text_raw", strip_dash=o_has)
        mv = merge_cues(v_list, "text_raw", strip_dash=v_has)
        o_span = {"start": mo["start"], "end": mo["end"], "duration": mo["end"] - mo["start"]}
        v_span = {"start": mv["start"], "end": mv["end"], "duration": mv["end"] - mv["start"]}
        m = overlap_metrics(o_span, v_span)       # overlap_sec / overlap_ratio_min / iou giữa hai đoạn bao
        gap = max(mo["max_gap"], mv["max_gap"])   # khoảng hở lớn nhất giữa các cue liên tiếp cùng phía

        # --- 4) phân loại nhóm (dash_match) ---
        if o_has and v_has:
            side, match, n_dash = "BOTH", "both_dash", max(o_lines, v_lines)
        else:
            side = "VI_ONLY" if v_has else "ORIGIN_ONLY"
            anchor, multi = (v_list, o_list) if v_has else (o_list, v_list)   # anchor = phía có dash
            n_dash = v_lines if v_has else o_lines    # số lượt thoại của phía có dash
            # so số cue phía kia (len(multi)) với n_dash để biết đủ / thiếu / thừa vế
            if len(anchor) != 1:
                match = "mismatch"
            elif n_dash < 2:
                match = "single_dash"
            elif len(multi) == n_dash:
                match = "full"
            elif v_has and len(multi) == 1 and inline_parts(multi[0]["text_raw"]) == n_dash:
                match = "inline"
            elif len(multi) < n_dash:
                match = "partial"
            else:
                match = "mismatch"

        # --- 5) gợi ý hành động ---
        ok = match in ("full", "inline")          # đủ vế -> có thể merge tự động nếu qua các kiểm tra dưới
        if ok and n_dash > DASH_MAX_AUTO_LINES:
            ok = False
        if ok and match == "full":
            if gap > DASH_MAX_GAP_SEC:
                ok = False
            if (o_span if v_has else v_span)["duration"] > MAX_MERGED_DURATION:
                ok = False
        if ok:
            suggestion = "merge_origin_and_vi" if match == "full" else "merge_vi"
        elif match == "partial":
            suggestion = "partial_match_by_dash"
        else:
            suggestion = "review"
        # light chỉ khi có gợi ý merge TỰ ĐỘNG và iou sau gộp đủ cao
        severity = "light" if ok and m["iou"] >= DASH_LIGHT_MIN_IOU else "heavy"

        def same_ids(lst):
            ids = {x for c in lst for x in c["same_lang_overlap_ids"]}
            return ";".join(sorted(ids - {c["cue_id"] for c in lst}))

        rows.append({
            "language": language, "drama_id": drama_id, "episode_id": episode_id,
            "group_id": f"{language}_{drama_id}_{episode_id}_DG{i:04d}",
            "dash_side": side, "dash_match": match, "overlap_severity": severity,
            "origin_cue_ids": ";".join(c["cue_id"] for c in o_list),
            "vi_cue_ids": ";".join(c["cue_id"] for c in v_list),
            "n_origin": len(o_list), "n_vi": len(v_list), "n_dash_lines": n_dash,
            "origin_start": seconds_to_srt_time(o_span["start"]),
            "origin_end": seconds_to_srt_time(o_span["end"]),
            "vi_start": seconds_to_srt_time(v_span["start"]),
            "vi_end": seconds_to_srt_time(v_span["end"]),
            "origin_duration": round(o_span["duration"], 3),
            "vi_duration": round(v_span["duration"], 3),
            "max_gap_sec": round(gap, 3),
            "overlap_sec": round(m["overlap_sec"], 3),
            "overlap_ratio_min": round(m["overlap_ratio_min"], 4),
            "iou": round(m["iou"], 4),
            "origin_text_raw": " || ".join(c["text_raw"] for c in o_list),
            "vi_text_raw": " || ".join(c["text_raw"] for c in v_list),
            "origin_lines": numbered_turns(o_list),   # lượt thoại gốc đánh số, để điền origin_keep_lines
            "vi_lines": numbered_turns(v_list),       # lượt thoại VI đánh số, để điền vi_keep_lines
            "merged_origin_text": mo["text"],
            "merged_vi_text": mv["text"],
            "detected_markers": fmt_markers_multi(o_list, v_list),
            "origin_same_lang_overlap_ids": same_ids(o_list),
            "vi_same_lang_overlap_ids": same_ids(v_list),
            "origin_prev_text": o_list[0]["prev_text"], "origin_next_text": o_list[-1]["next_text"],
            "vi_prev_text": v_list[0]["prev_text"], "vi_next_text": v_list[-1]["next_text"],
            "origin_keep_lines": "", "vi_keep_lines": "",
            # partial với dash ở VI: audio còn vế không có text zh -> ASR không dùng được (điền sẵn exclude)
            "case_type": "dash_dialogue",
            "asr_action": "exclude" if (match == "partial" and v_has) else "",
            "mt_action": "keep", "alignment_action": "",
            "suggested_action": suggestion,
            "note": "",
        })
    return rows, used_o, used_v


def candidate_stage(episode: tuple, excluded: set) -> tuple[list, list, list, list]:
    language, lang_tag, _folder, drama_id, episode_id, origin_path, vi_path = episode

    def ex_ids(tag):
        return {k[4] for k in excluded
                if k[:3] == (language, drama_id, episode_id) and k[3] == tag}

    origin_all = load_cues(origin_path, lang_tag)
    vi_all = load_cues(vi_path, "vi")
    origin, _ = prepare_cues(origin_all, ex_ids(lang_tag), True)
    vi, vi_pairs = prepare_cues(vi_all, ex_ids("vi"), False)

    # --- cặp ứng viên theo thời gian ---
    all_pairs = []
    for o in origin:
        for v in vi:
            if not (o["is_special"] or v["is_special"]):
                continue
            if has_dash(o) or has_dash(v):   # cue leading_dash -> dash_pairs_review.csv (dash_stage)
                continue
            m = overlap_metrics(o, v)
            if m["overlap_sec"] <= 0 or m["overlap_ratio_min"] < MIN_OVERLAP_RATIO:
                continue
            all_pairs.append((o, v, m))

    # best_iou_o / best_iou_v: IoU cao nhất của từng cue đặc biệt trong mọi cặp đạt ngưỡng overlap
    # (dùng để giữ lại cặp tốt nhất của cue đó dù IoU thấp -> đánh dấu weak_match)
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

    # --- gom nhóm: các cặp dùng chung một cue thì cùng group_id (connected component) ---
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
        gtype = classify_group(n_o, n_v)            # n_o, n_v = số cue gốc / VI cùng nhóm

        # gợi ý hành động: cặp 1-1 khớp tốt (iou >= AUTO_KEEP_MIN_IOU) -> keep_pair, còn lại -> review
        suggestion = "keep_pair" if (gtype == "one_to_one" and m["iou"] >= AUTO_KEEP_MIN_IOU) else "review"

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
            "n_origin": n_o, "n_vi": n_v,
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
            "origin_same_lang_overlap_ids": ";".join(o["same_lang_overlap_ids"]),
            "vi_same_lang_overlap_ids": ";".join(v["same_lang_overlap_ids"]),
            "origin_prev_text": o["prev_text"], "origin_next_text": o["next_text"],
            "vi_prev_text": v["prev_text"], "vi_next_text": v["next_text"],
            "alignment_action": "review",
            "note": "weak_match" if (o["cue_id"], v["cue_id"]) in weak_pairs else "",
        })

    # --- leading_dash: tách file riêng, cue đã vào nhóm dash không tính là unmatched ---
    dash_rows, dash_o_ids, dash_v_ids = dash_stage(language, drama_id, episode_id, origin, vi)
    matched_o |= dash_o_ids
    matched_v |= dash_v_ids

    # --- unmatched ---
    unmatched = []
    for tag, cues, matched in ((lang_tag, origin, matched_o), ("vi", vi, matched_v)):
        for c in cues:
            if c["is_special"] and c["cue_id"] not in matched:
                unmatched.append({
                    "language": language, "drama_id": drama_id, "episode_id": episode_id,
                    "cue_id": c["cue_id"],
                    "start": seconds_to_srt_time(c["start"]),
                    "end": seconds_to_srt_time(c["end"]),
                    "duration": round(c["duration"], 3), "text_raw": c["text_raw"],
                    "detected_markers": ";".join(c["markers"]),
                    "special_side": "ORIGIN_ONLY" if tag == lang_tag else "VI_ONLY",
                    "same_lang_overlap_ids": ";".join(c["same_lang_overlap_ids"]),
                    "prev_text": c["prev_text"], "next_text": c["next_text"],
                    "case_type": "scene_text",
                    "asr_action": "exclude",
                    "mt_action": "exclude",
                    "note": "",
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
    return rows, unmatched, suspect, dash_rows


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
    dash_path = out_dir / "dash_pairs_review"
    done_path = out_dir / f".done_{args.stage}.txt"

    if args.reset:
        targets = ([excluded_path, pairs_path] if args.stage == "music"
                   else [cand_path, unm_path, sus_path, dash_path])
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
            print(f"music_pairs_review.csv còn {todo} cặp chưa có alignment_action.")
            print("Điền alignment_action = keep_pair/reject rồi chạy lại, hoặc dùng --allow-todo.")
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
                cands, unm, sus, dash = candidate_stage(ep, excluded)
                append_table(cand_path, CANDIDATE_FIELDS, cands)
                append_table(unm_path, UNMATCHED_FIELDS, unm)
                append_table(sus_path, SUSPECT_FIELDS, sus)
                append_table(dash_path, DASH_FIELDS, dash)
                totals["candidates"] += len(cands)
                totals["unmatched"] += len(unm)
                totals["suspect"] += len(sus)
                totals["dash_groups"] += len(dash)
                totals["dash_suggest_merge"] += sum(r["suggested_action"] != "review" for r in dash)
                totals["keep_pair"] += sum(r["suggested_action"] == "keep_pair" for r in cands)
                print(f"[{language} / {folder} / {episode_id}] candidate={len(cands)}, "
                      f"unmatched={len(unm)}, vi_overlap_suspect={len(sus)}, "
                      f"dash_groups={len(dash)} (gợi ý merge="
                      f"{sum(r['suggested_action'] != 'review' for r in dash)})")
            mark_done(done_path, key)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {folder} {episode_id}: {exc}")

    print("\nHoàn tất. Tổng:", dict(totals))
    if args.stage == "music":
        print(f"Tự động loại: {excluded_path}.csv\nReview cặp: {pairs_path}.csv  (kiểm tra alignment_action; sửa thì đổi review_status thành CONFIRMED)")
    else:
        print(f"Candidate: {cand_path}.csv\nUnmatched: {unm_path}.csv\nVI overlap suspect: {sus_path}.csv\n"
              f"Dash pairs (leading_dash): {dash_path}.csv")


if __name__ == "__main__":
    main()