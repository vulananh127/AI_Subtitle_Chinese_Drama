"""
Xử lý dữ liệu DRAMA: raw SRT + audio -> bilingual zh-vi (xem được cùng video) -> metadata ASR / MT.
KHÔNG ghi/sửa bất kỳ file nào trong raw/.

Luồng tổng thể (2 bước tách riêng)
----------------------------------
  test_review.py (review thủ công)       data_process.py
  ------------------------------------   ------------------------------------------------------
  stage 1: cue nhạc + cặp nhạc      -->  --stage bilingual  (chỉ lọc cue đặc biệt + clean + align)
  stage 2: cue có marker còn lại              1. Đọc SRT gốc, áp các quyết định trong processed/review/*.csv
           + nhóm leading dash                   (chỉ đọc id cue + case_type + asr/mt/alignment action,
                                                  TEXT luôn lấy từ SRT gốc)
                                              2. Cue thường: clean -> align zh-vi theo thời gian
                                              3. Xuất bilingual: SRT xem cùng video + bảng CSV
                                                 KHÔNG merge cue ở bước này
                                         --stage metadata  (sau khi bạn đã soát bilingual)
                                              4. ASR : merge theo GAP (dấu kết câu không chặn; đoạn quá dài được cắt ở chỗ
                                                       nghỉ tốt nhất; dash là điểm ngắt) -> cắt chunk -> metadata_asr.csv
                                              5. MT  : merge theo dấu câu (chưa gặp dấu kết câu thì còn nối tiếp,
                                                       giới hạn token nguồn/đích, KHÔNG chặn gap/thời lượng; BỎ điểm
                                                       ngắt dash) -> metadata_mt.csv
                                              Ngưỡng ASR và MT tách riêng (ASR_MERGE_GAP / MT_MERGE_GAP); dùng
                                              --debug-merge để xem vì sao từng chỗ nối được merge hay dừng

Bilingual = phụ đề thoại có tiếng nói, xem được cùng video. Chỉ loại cue scene_text (và cue cặp không hợp
lệ/vi_only/origin_only). Mỗi cue của SRT là 1 dòng bảng; cờ asr_keep / mt_keep quyết định dòng đó có vào
metadata ASR / MT hay không (dòng chỉ để xem thì cả hai cờ = 0).

Quy tắc theo case_type (cột case_type trong special_alignment_candidates / special_unmatched /
dash_pairs_review). Bilingual SRT hiển thị bản "MT" (đọc tự nhiên); ASR/MT được suy ra ở bảng CSV:
-------------------------------------------------------------------------------------------
  scene_text                  bị loại khỏi bilingual
  inner_monologue, phone_speech  bỏ ký tự ngoặc ở cả zh và vi, giữ nội dung (ASR và MT)
  inline_supplement           ASR: bỏ ngoặc VÀ nội dung trong ngoặc (zh);  MT: chỉ bỏ ký tự ngoặc (zh và vi)
  note (= pronunciation_note) ASR và MT: bỏ cả ngoặc lẫn nội dung trong ngoặc (zh và vi)
  dash_dialogue               SRT hiển thị GIỮ NGUYÊN dấu dash và xuống dòng như SRT gốc (mỗi cue phía kia
                              một dòng); text ASR/MT nối mọi lượt thành 1 dòng, bỏ dấu dash
  dash_partial                alignment_action = partial_match_by_dash (bạn review thủ công): origin_keep_lines /
                              vi_keep_lines (đánh số theo origin_lines / vi_lines) chọn lượt và thứ tự nối cho
                              MT. Text ASR luôn là mọi lượt zh theo thứ tự thời gian. Hai cột trống hoặc số
                              sai -> pending (cue phía dash bị loại). Hai cột keep_lines điền ở dòng khác
                              (đảo thứ tự, bỏ lượt thừa) vẫn có tác dụng, cặp vẫn là dash_dialogue
  music (nhạc/lời bài hát)    MUSIC_MODE = "label" (mặc định): lời bài hát KHÔNG vào bilingual, thay bằng cue
                              nhãn [nhạc] cho các khoảng nhạc thuần (đã trừ phần có thoại), dùng để thử cả hệ
                              thống trên video; "drop": bỏ hẳn. Cue nhãn luôn asr_keep = mt_keep = 0
asr_action / mt_action = keep -> dòng có trong tập ASR / MT; exclude -> chỉ để xem (không vào metadata).
alignment_action: keep_pair = cặp hợp lệ; reject = hai cue chỉ chồng thời gian, KHÔNG phải một cặp.
  - scene_text + keep_pair -> loại CẢ HAI cue (cue không ngoặc còn lại là bản dịch của caption)
  - reject                 -> chỉ loại cue đặc biệt (phía có marker), cue kia ở lại pool cue thường
  - chưa điền đủ action    -> "pending": loại cue đặc biệt, in cảnh báo (chạy lại sau khi review)
  - special_unmatched      -> luôn loại (cue không có cặp: origin_only / vi_only không vào bilingual)
Cue có marker mà không nằm trong bảng quyết định nào ("undecided") cũng bị loại và được đếm.
Nếu id cue không còn khớp text trong SRT (SRT bị đánh số lại/sửa), dòng quyết định đó bị bỏ qua và
cảnh báo; tắt kiểm tra bằng --skip-text-check.

Bảng bilingual (processed/bilingual/{language}/Dxxxx/{base}_bilingual.csv + .srt cùng tên)
  pair_idx, start, end, duration   : thời gian theo cue gốc (SRT cue thứ i = pair_idx i)
  origin_text, vi_text             : text HIỂN THỊ (đúng như trong SRT bilingual)
  origin_text_asr                  : text zh khớp audio, dùng cho ASR
  origin_text_mt, vi_text_mt       : text dùng cho MT
  asr_keep, mt_keep, source        : cờ 1/0 và loại cặp (normal / phone_speech / ... / music)
  overlap                          : none / music / speech - cue zh hoặc vi của cặp chồng (>= OVERLAP_MIN_SEC) lên cue
                                     nhạc (auto_excluded / music_pairs_review) -> music; cặp dash_dialogue /
                                     dash_partial -> speech; còn lại none (music được ưu tiên nếu có cả hai)
  origin_cue_ids, vi_cue_ids       : id cue SRT gốc đã tạo nên dòng này
  Bước metadata đọc BẢNG CSV (giữ được cờ và các bản text), nên sửa text/cờ trong CSV, không sửa trong SRT.

Metadata (processed/metadata/{language}/)   (không còn metadata_master.csv)
  metadata_asr.csv : id, language, drama_id, episode_id, audio_path, start, end, duration, origin_text, overlap
                     (overlap của đoạn merge = giá trị mạnh nhất trong các cặp thành phần: music > speech > none)
  metadata_mt.csv  : id, language, drama_id, episode_id, origin_text, vi_text
  Hai file merge độc lập nên cùng một "id" ở hai file là HAI đoạn khác nhau (id chỉ là số thứ tự trong file).

Resumable: --stage bilingual bỏ qua tập đã có bảng bilingual; --stage metadata bỏ qua tập đã có trong
metadata_asr.csv / metadata_mt.csv. Dùng --force để làm lại (cần khi review CSV thay đổi, hoặc khi chuyển
từ phiên bản data_process cũ sang bản này).

Yêu cầu: ffmpeg + ffprobe trong PATH (chỉ cho --stage metadata). Cần utils.py bản có merge_cues, split_dash_turns.

Cách dùng:
    python data_process.py                         # = --stage bilingual
    python data_process.py --stage metadata        # merge + cắt chunk + metadata từ bilingual đã soát
    python data_process.py --stage all             # làm cả hai
    python data_process.py --language chinese --drama D0002 --episode 1 --force --debug-srt
"""

import argparse
import re
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    DRAMA_RAW_DIR, DRAMA_LANGUAGES, DRAMA_PROCESSED_DIR, DATASET_DIR,
    AUDIO_FORMAT, CHUNK_SAMPLE_RATE, CHUNK_CHANNELS,
    SENTENCE_END_CHARS,
    MIN_OVERLAP_RATIO, MIN_CHUNK_DURATION, MAX_CHUNK_DURATION,
    DURATION_TOLERANCE, FFMPEG_BIN, FFPROBE_BIN, MARKER_PATTERNS,
    drama_raw_dir, drama_debug_dir, drama_audio_chunk_dir, drama_metadata_paths,
)
from utils import (
    parse_srt, write_srt, make_sample_id, read_csv_rows, write_csv_rows,
    join_text_parts, merge_cues, split_dash_turns,
)

REVIEW_DIR = DRAMA_PROCESSED_DIR / "review"          # cùng thư mục với test_review.py
BILINGUAL_DIR = DRAMA_PROCESSED_DIR / "bilingual"

ASR_FIELDNAMES = ["id", "language", "drama_id", "episode_id", "audio_path", "start", "end", "duration", "origin_text", "overlap"]
MT_FIELDNAMES = ["id", "language", "drama_id", "episode_id", "origin_text", "vi_text"]
BILINGUAL_FIELDS = ["pair_idx", "start", "end", "duration", "origin_text", "vi_text",
                    "origin_text_asr", "origin_text_mt", "vi_text_mt",
                    "asr_keep", "mt_keep", "source", "overlap", "origin_cue_ids", "vi_cue_ids"]

# ---- nhạc trong bilingual ----
MUSIC_MODE = "label"            # "label": cue nhãn [nhạc] cho khoảng nhạc thuần | "drop": không đưa nhạc vào bilingual
MUSIC_LABELS = {"zh": "[音乐]", "ko": "[음악]", "ja": "[音楽]"}    # nhãn phía gốc, theo lang_tag
MUSIC_LABEL_VI = "[nhạc]"
MUSIC_MERGE_GAP = 1.0           # giây: các cue nhạc cách nhau <= giá trị này gộp thành 1 khoảng nhạc
MUSIC_MIN_LABEL_SEC = 1.0       # khoảng nhạc (sau khi trừ phần có thoại) ngắn hơn -> không gắn nhãn
MUSIC_REASONS = {"music_auto", "music_lyrics"}   # lý do loại cue được coi là "nhạc thuần/lời hát" (xem plan_special)

# ---- cột overlap (none / music / speech): chất lượng audio của đoạn ----
OVERLAP_MIN_SEC = 0.1           # cue của cặp chồng cue nhạc >= số giây này mới tính là "music" (0 = mọi chồng nhau)
OVERLAP_RANK = {"none": 0, "speech": 1, "music": 2}   # khi gộp nhiều cặp: lấy giá trị mạnh nhất (music > speech > none)

# raw/{language}/Dxxxx/{language}_Dxxxx_EPyyy_{lang_tag}_manual_raw.srt -> bắt group drama_id, episode
def episode_file_re(language: str, lang_tag: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(language)}_D(\d+)_EP(\d+)_{lang_tag}_manual_raw\.srt$")


# =====================================================================
# 1. Làm sạch text
# =====================================================================
HTML_TAG_RE = re.compile(r"<[^>]+>")
ASS_TAG_RE = re.compile(r"\{[^}]*\}")
SPEAKER_LABEL_RE = re.compile(r"^(\s*-\s*|\s*\[[^\]]{1,20}\]\s*:?\s*|\s*[\w\u4e00-\u9fffÀ-ỹ]{1,20}\s*:\s*)")
ANNOTATION_RE = re.compile(r"[\[\(（【][^\]\)）】]{0,80}[\]\)）】]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{2,}")

# Ngoặc cho xử lý cue đặc biệt (chỉ dùng trên cue ĐÃ được review, không dùng cho cue thường)
BRACKET_PAIR_RE = re.compile(r"[(（\[【〔][^()（）\[\]【】〔〕]*[)）\]】〕]")
BRACKET_CHARS_RE = re.compile(r"[()（）\[\]【】〔〕]")


def strip_markup(text: str) -> str:
    text = HTML_TAG_RE.sub("", text)
    text = ASS_TAG_RE.sub("", text)
    return text


def clean_text(text: str) -> str:
    """Làm sạch cue THƯỜNG (không marker). Cue có marker đã được xử lý theo bảng review từ trước."""
    text = strip_markup(text)
    text = ANNOTATION_RE.sub("", text)
    lines = [SPEAKER_LABEL_RE.sub("", line) for line in text.splitlines()]
    text = "\n".join(lines)
    text = unicodedata.normalize("NFC", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def normalize_subtitle_text(text: str) -> str:
    """Xuống dòng và khoảng trắng liên tiếp -> 1 dấu cách."""
    return re.sub(r"\s+", " ", text).strip() if text else ""


def clean_cues(cues: list[dict]) -> list[dict]:
    """Clean cue thường; giữ cue_id để truy vết. Cue rỗng sau clean bị bỏ."""
    cleaned = []
    for c in cues:
        text = normalize_subtitle_text(clean_text(c["text"]))
        if text:
            cleaned.append({"cue_id": c["cue_id"], "start": c["start"], "end": c["end"], "text": text})
    return cleaned


def tidy_text(text: str) -> str:
    """Chuẩn hóa text đã xử lý ngoặc: bỏ markup, NFC, nối các dòng bằng đúng 1 dấu cách."""
    text = unicodedata.normalize("NFC", strip_markup(text or ""))
    return MULTI_SPACE_RE.sub(" ", join_text_parts([text])).strip()


def remove_bracket_chars(text: str) -> str:
    """Bỏ ký tự ngoặc, GIỮ nội dung bên trong."""
    return tidy_text(BRACKET_CHARS_RE.sub("", text))


def remove_bracket_content(text: str) -> str:
    """Bỏ ngoặc VÀ nội dung bên trong (lặp để xử lý ngoặc lồng đơn giản)."""
    prev = None
    while prev != text:
        prev, text = text, BRACKET_PAIR_RE.sub("", text)
    return tidy_text(BRACKET_CHARS_RE.sub("", text))


def special_markers(text: str) -> list[str]:
    """Marker làm cue "đặc biệt" cần có quyết định review. markup HTML/ASS đơn thuần không tính
    (đã bị strip_markup xử lý)."""
    text = unicodedata.normalize("NFC", text or "")
    return [n for n, p in MARKER_PATTERNS.items() if n != "html_or_ass_markup" and p.search(text)]


LEADING_DASH_RE = MARKER_PATTERNS["leading_dash"]


def display_lines(text: str) -> list[str]:
    """Các dòng của 1 cue đã clean để HIỂN THỊ: bỏ markup, NFC, gọn khoảng trắng, bỏ dòng rỗng."""
    text = unicodedata.normalize("NFC", strip_markup(text or ""))
    return [MULTI_SPACE_RE.sub(" ", ln).strip() for ln in text.splitlines() if ln.strip()]


def display_text(cues: list[dict]) -> str:
    """Text hiển thị của một nhóm cue (theo thứ tự thời gian): mỗi cue một dòng; cue có leading dash
    giữ nguyên các dòng (dấu dash + xuống dòng như SRT gốc), cue thường nối thành 1 dòng."""
    parts = []
    for c in sorted(cues, key=lambda c: (c["start"], c["end"])):
        lines = display_lines(c["text"])
        if lines:
            parts.append("\n".join(lines) if LEADING_DASH_RE.search(unicodedata.normalize("NFC", c["text"])) else join_text_parts(lines))
    return "\n".join(parts)


CLEAN_KINDS = {"inner_monologue", "phone_speech", "inline_supplement", "note"}
KIND_ALIASES = {"pronunciation_note": "note"}


def norm(s) -> str:
    return (s or "").strip().lower()


def norm_kind(s) -> str:
    k = norm(s)
    return KIND_ALIASES.get(k, k)


def split_ids(s) -> list[str]:
    return [x.strip() for x in (s or "").split(";") if x.strip()]


def squash(s: str) -> str:
    """Chuẩn hóa chỉ để SO SÁNH text SRT với text trong CSV (không dùng cho dữ liệu xuất ra):
    NFKC (gộp dạng toàn/bán giác), bỏ khoảng trắng mọi loại và ký tự định dạng vô hình
    (zero-width space, BOM, dấu chiều văn bản...) mà mắt không thấy nhưng làm hai chuỗi 'giống hệt' bị lệch."""
    s = unicodedata.normalize("NFKC", s or "")
    return "".join(ch for ch in s if not ch.isspace() and unicodedata.category(ch) != "Cf")


def read_review_csv(path: Path) -> list[dict]:
    fieldnames, rows = read_csv_rows(path)
    if fieldnames and fieldnames[0].startswith("\ufeff"):          # Excel 'CSV UTF-8' thêm BOM
        clean = fieldnames[0].lstrip("\ufeff")
        rows = [{(clean if k == fieldnames[0] else k): v for k, v in r.items()} for r in rows]
    return rows


class ReviewTables:
    """Đọc 1 lần các bảng review, tra theo (language, drama_id, episode_id)."""
    NAMES = ("auto_excluded", "music_pairs_review", "special_alignment_candidates",
             "special_unmatched", "dash_pairs_review")

    def __init__(self, review_dir: Path):
        self.dir = Path(review_dir)
        self.index = {}
        for name in self.NAMES:
            idx = defaultdict(list)
            path = self.dir / f"{name}.csv"
            if path.exists():
                for r in read_review_csv(path):
                    idx[(r.get("language", ""), r.get("drama_id", ""), r.get("episode_id", ""))].append(r)
            self.index[name] = idx

    def rows(self, name: str, language: str, drama: str, ep: str) -> list[dict]:
        return self.index[name].get((language, drama, ep), [])

    def has_episode(self, language: str, drama: str, ep: str) -> bool:
        return any((language, drama, ep) in idx for idx in self.index.values())


def load_cues(path: Path, tag: str) -> list[dict]:
    """cue_id = {tag}_{số cue trong file SRT, 6 chữ số}, ví dụ zh_000251 - trùng với test_special.py."""
    return [{"cue_id": f"{tag}_{c['id']:06d}", "start": c["start"], "end": c["end"], "text": c["text"]}
            for c in parse_srt(path)]


# Cách làm sạch text theo case_type: (zh cho ASR, zh cho MT, vi cho MT). Bản hiển thị dùng bản MT.
KIND_CLEANERS = {
    "inner_monologue": (remove_bracket_chars, remove_bracket_chars, remove_bracket_chars),
    "phone_speech": (remove_bracket_chars, remove_bracket_chars, remove_bracket_chars),
    "inline_supplement": (remove_bracket_content, remove_bracket_chars, remove_bracket_chars),
    "note": (remove_bracket_content, remove_bracket_content, remove_bracket_content),
}


def _ids(cues: list[dict]) -> list[str]:
    return [c["cue_id"] for c in sorted(cues, key=lambda c: (c["start"], c["end"]))]


def build_pair(kind: str, o_cues: list[dict], v_cues: list[dict], asr_keep: bool, mt_keep: bool) -> dict:
    """Một cặp đặc biệt (phone_speech / inner_monologue / inline_supplement / note).
    Thời gian = đoạn bao các cue gốc (audio cắt theo phía gốc). Text hiển thị = bản MT."""
    f_asr, f_mo, f_mv = KIND_CLEANERS[kind]
    o_cues = sorted(o_cues, key=lambda c: (c["start"], c["end"]))
    v_cues = sorted(v_cues, key=lambda c: (c["start"], c["end"]))
    span = merge_cues(o_cues)
    mt_o = [t for t in (f_mo(c["text"]) for c in o_cues) if t]
    mt_v = [t for t in (f_mv(c["text"]) for c in v_cues) if t]
    return {
        "start": span["start"], "end": span["end"], "kind": kind,
        "origin_text": "\n".join(mt_o), "vi_text": "\n".join(mt_v),                 # hiển thị
        "origin_text_asr": join_text_parts([f_asr(c["text"]) for c in o_cues]),     # ASR: khớp audio
        "origin_text_mt": join_text_parts(mt_o), "vi_text_mt": join_text_parts(mt_v),
        "asr_keep": asr_keep, "mt_keep": mt_keep,
        "origin_ids": _ids(o_cues), "vi_ids": _ids(v_cues),
    }


def build_dash_pair(kind: str, o_cues: list[dict], v_cues: list[dict], o_turns: list[str], o_pick: list[str],
                    v_pick: list[str], asr_keep: bool, mt_keep: bool) -> dict:
    """Cặp dash. Hiển thị: giữ dash + xuống dòng như SRT gốc. ASR: mọi lượt zh theo thời gian.
    MT: các lượt đã chọn (origin_keep_lines / vi_keep_lines) theo đúng thứ tự điền, nối thành 1 dòng."""
    span = merge_cues(o_cues)                                  # thời gian = đoạn bao các cue zh
    return {
        "start": span["start"], "end": span["end"], "kind": kind,
        "origin_text": display_text(o_cues), "vi_text": display_text(v_cues),
        "origin_text_asr": tidy_text(join_text_parts(o_turns)),
        "origin_text_mt": tidy_text(join_text_parts(o_pick)),
        "vi_text_mt": tidy_text(join_text_parts(v_pick)),
        "asr_keep": asr_keep, "mt_keep": mt_keep,
        "origin_ids": _ids(o_cues), "vi_ids": _ids(v_cues),
    }


def plan_special(tables: ReviewTables, language: str, drama: str, ep: str,
                 origin_by_id: dict, vi_by_id: dict, check_text: bool) -> dict:
    """Kế hoạch cho cue đặc biệt của 1 tập:
       drop_origin / drop_vi : {cue_id: lý do} cue phải loại khỏi pool cue thường
       pairs                 : cặp đặc biệt (vào bilingual, kèm cờ asr_keep / mt_keep)
       stats / warns         : đếm và cảnh báo"""
    plan = {"drop_origin": {}, "drop_vi": {}, "pairs": [], "stats": Counter(), "warns": []}
    stats, warns = plan["stats"], plan["warns"]

    def drop(side: str, cue_id: str, reason: str):
        plan["drop_origin" if side == "origin" else "drop_vi"].setdefault(cue_id, reason)

    mismatch_detail: list[str] = []          # chi tiết chỗ khác của lần so sánh thất bại gần nhất

    def match(cues: list[dict], csv_text: str) -> bool:
        if not check_text:
            return True
        a = squash("".join(c["text"] for c in cues))                 # text trong SRT gốc
        b = squash((csv_text or "").replace("||", ""))               # text trong CSV review
        if a == b:
            return True
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))

        def ch(t):
            return f"{t[i]!r} (U+{ord(t[i]):04X})" if i < len(t) else "(hết chuỗi)"
        ids = ";".join(c["cue_id"] for c in cues)
        mismatch_detail[:] = [f"{ids}: text khác từ vị trí {i}: SRT={ch(a)} / CSV={ch(b)} "
                              f"(độ dài SRT={len(a)}, CSV={len(b)})"]
        return False

    def bad(name: str, r: dict, why: str):
        stats["row_skipped_mismatch"] += 1
        detail = f" | {mismatch_detail[0]}" if mismatch_detail else ""
        mismatch_detail.clear()
        warns.append(f"[{name}] bỏ qua dòng ({r.get('origin_cue_id') or r.get('cue_id') or r.get('origin_cue_ids')}): {why}{detail}")

    # ---- music ----
    for r in tables.rows("auto_excluded", language, drama, ep):
        status = (r.get("review_status") or "").strip().upper()
        if status in {"AUTO", "CONFIRMED"} and norm(r.get("asr_action")) == "exclude" and norm(r.get("mt_action")) == "exclude":
            side = "vi" if r["subtitle_language"] == "vi" else "origin"
            cue = (vi_by_id if side == "vi" else origin_by_id).get(r["cue_id"])
            if cue is None or not match([cue], r.get("text_raw")):
                bad("auto_excluded", r, "id không khớp text SRT")
                continue
            drop(side, r["cue_id"], "music_auto")
            stats["music_auto"] += 1

    for r in tables.rows("music_pairs_review", language, drama, ep):
        o, v = origin_by_id.get(r["origin_cue_id"]), vi_by_id.get(r["vi_cue_id"])
        if o is None or v is None or not (match([o], r.get("origin_text_raw")) and match([v], r.get("vi_text_raw"))):
            bad("music_pairs_review", r, "id không khớp text SRT")
            continue
        align = norm(r.get("alignment_action"))
        side = (r.get("special_side") or "").upper()
        if align == "reject":
            # không phải một cặp (thoại đè nhạc): chỉ loại cue nhạc (phía có ♪), cue thoại ở lại pool
            if side in ("VI_ONLY", "BOTH"):
                drop("vi", r["vi_cue_id"], "music_vi_over_speech")
            if side in ("ORIGIN_ONLY", "BOTH"):
                drop("origin", r["origin_cue_id"], "music_origin_over_speech")
            stats["music_keep_origin"] += 1
        else:                                                      # keep_pair = lời hát thật; trống = chưa review
            reason = "music_lyrics" if align == "keep_pair" else "music_pending"
            drop("origin", r["origin_cue_id"], reason)             # cả hai cue loại (an toàn khi chưa review)
            drop("vi", r["vi_cue_id"], reason)
            stats["music_pair_exclude" if align == "keep_pair" else "pending_music"] += 1

    # ---- special_alignment_candidates ----
    links = []                                   # (oid, vid, kind, asr_keep, mt_keep)
    for r in tables.rows("special_alignment_candidates", language, drama, ep):
        oid, vid = r["origin_cue_id"], r["vi_cue_id"]
        o, v = origin_by_id.get(oid), vi_by_id.get(vid)
        if o is None or v is None or not (match([o], r.get("origin_text_raw")) and match([v], r.get("vi_text_raw"))):
            bad("candidates", r, "id không khớp text SRT")
            continue
        asr, mt, align, kind = norm(r.get("asr_action")), norm(r.get("mt_action")), norm(r.get("alignment_action")), norm_kind(r.get("case_type"))
        side = (r.get("special_side") or "").upper()
        sp_o = side in ("ORIGIN_ONLY", "BOTH") or not side
        sp_v = side in ("VI_ONLY", "BOTH") or not side

        def drop_special(reason):
            if sp_o:
                drop("origin", oid, reason)
            if sp_v:
                drop("vi", vid, reason)

        if not (asr and mt and align):
            drop_special("pending_review")
            stats["pending_candidates"] += 1
        elif kind == "scene_text":
            if align == "keep_pair":                # caption zh + bản dịch vi của chính nó -> bỏ cả hai cue
                drop("origin", oid, "scene_text")
                drop("vi", vid, "scene_text")
            else:                                   # caption chỉ chồng thời gian với thoại -> chỉ bỏ cue caption
                drop_special("scene_text")
            stats["dropped_scene_text"] += 1
        elif align != "keep_pair":                  # reject: cặp không hợp lệ, cue đặc biệt không được vào bilingual
            drop_special("reject")
            stats["dropped_reject"] += 1
        elif kind not in KIND_CLEANERS:
            drop_special("unknown_case_type")
            stats["pending_candidates"] += 1
            warns.append(f"[candidates] {oid}/{vid}: case_type '{r.get('case_type')}' chưa hỗ trợ -> loại cue đặc biệt")
        else:
            # cặp hợp lệ: vào bilingual; asr/mt = exclude chỉ có nghĩa là không vào metadata (cờ = 0)
            links.append((oid, vid, kind, asr == "keep", mt == "keep"))

    # gom các cặp keep_pair thành nhóm (thường 1-1; nhiều-nhiều thì gộp cue theo thời gian)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for oid, vid, *_ in links:
        parent[find("o:" + oid)] = find("v:" + vid)
    groups = defaultdict(list)
    for link in links:
        groups[find("o:" + link[0])].append(link)

    used_o, used_v = set(), set()

    def add_pair(kind, o_ids, v_ids, asr_keep, mt_keep, source_name, pair=None) -> bool:
        if set(o_ids) & used_o or set(v_ids) & used_v:
            stats["conflict_rows"] += 1
            warns.append(f"[{source_name}] cue {o_ids}/{v_ids} đã thuộc cặp khác -> bỏ qua cặp này")
            return False
        if pair is None:
            o_cues, v_cues = [origin_by_id[i] for i in o_ids], [vi_by_id[i] for i in v_ids]
            pair = build_pair(kind, o_cues, v_cues, asr_keep, mt_keep)
        plan["pairs"].append(pair)
        used_o.update(o_ids)
        used_v.update(v_ids)
        stats[f"pair_{kind}"] += 1
        return True

    for members in groups.values():
        o_ids = sorted({m[0] for m in members})
        v_ids = sorted({m[1] for m in members})
        kinds = {m[2] for m in members}
        if len(kinds) > 1:
            warns.append(f"[candidates] nhóm {o_ids}/{v_ids} có nhiều case_type {sorted(kinds)}, dùng '{sorted(kinds)[0]}'")
        asr_keep, mt_keep = all(m[3] for m in members), all(m[4] for m in members)
        add_pair(sorted(kinds)[0], o_ids, v_ids, asr_keep, mt_keep, "candidates")

    # ---- leading dash ----
    for r in tables.rows("dash_pairs_review", language, drama, ep):
        o_ids, v_ids = split_ids(r.get("origin_cue_ids")), split_ids(r.get("vi_cue_ids"))
        if (not o_ids or not v_ids or any(i not in origin_by_id for i in o_ids)
                or any(i not in vi_by_id for i in v_ids)
                or not (match([origin_by_id[i] for i in o_ids], r.get("origin_text_raw"))
                        and match([vi_by_id[i] for i in v_ids], r.get("vi_text_raw")))):
            bad("dash_pairs_review", r, "id không khớp text SRT")
            continue
        asr, mt, align = norm(r.get("asr_action")), norm(r.get("mt_action")), norm(r.get("alignment_action"))
        side = (r.get("dash_side") or "").upper()

        def drop_dash_side(reason):
            if side in ("ORIGIN_ONLY", "BOTH"):
                for i in o_ids:
                    drop("origin", i, reason)
            if side in ("VI_ONLY", "BOTH"):
                for i in v_ids:
                    drop("vi", i, reason)

        if not (asr and mt):
            drop_dash_side("pending_dash")
            stats["pending_dash"] += 1
        elif align == "reject":
            drop_dash_side("dash_reject")
            stats["dropped_dash"] += 1
        else:
            o_sel_raw, v_sel_raw = split_ids(r.get("origin_keep_lines")), split_ids(r.get("vi_keep_lines"))
            selecting = bool(o_sel_raw or v_sel_raw)
            # chỉ tin cột bạn review thủ công (alignment_action), không dùng cột tự động dash_match/suggested_action
            is_partial = align == "partial_match_by_dash"
            o_cues = [origin_by_id[i] for i in o_ids]                 # theo thứ tự thời gian như trong CSV
            v_cues = [vi_by_id[i] for i in v_ids]
            # lượt thoại mỗi phía, đánh số liên tục từ 1: trùng cách đánh số ở cột origin_lines / vi_lines
            o_turns = [t for c in o_cues for t in split_dash_turns(c["text"])]
            v_turns = [t for c in v_cues for t in split_dash_turns(c["text"])]

            def pick(sel_raw, turns):
                """Các lượt đã chọn theo ĐÚNG thứ tự điền; None nếu số không hợp lệ."""
                if not sel_raw:
                    return list(turns)                                # để trống = mọi lượt theo thời gian
                try:
                    nums = [int(x) for x in sel_raw]
                except ValueError:
                    return None
                return None if any(n < 1 or n > len(turns) for n in nums) else [turns[n - 1] for n in nums]

            o_pick, v_pick = pick(o_sel_raw, o_turns), pick(v_sel_raw, v_turns)
            if is_partial and not selecting:
                drop_dash_side("pending_dash")
                stats["pending_dash"] += 1
                warns.append(f"[dash] {o_ids}/{v_ids}: partial_match_by_dash cần origin_keep_lines/vi_keep_lines "
                             f"(zh có {len(o_turns)} lượt, vi có {len(v_turns)} lượt) -> chờ review")
            elif o_pick is None or v_pick is None or not o_pick or not v_pick:
                drop_dash_side("pending_dash")
                stats["pending_dash"] += 1
                warns.append(f"[dash] {o_ids}/{v_ids}: keep_lines không hợp lệ "
                             f"(origin={r.get('origin_keep_lines')!r}/{len(o_turns)} lượt, "
                             f"vi={r.get('vi_keep_lines')!r}/{len(v_turns)} lượt) -> chờ review")
            else:
                if is_partial and asr == "keep" and side == "VI_ONLY":
                    warns.append(f"[dash] {o_ids}/{v_ids}: zh thiếu vế mà asr_action=keep -> chunk có thể không khớp text")
                kind = "dash_partial" if is_partial else "dash_dialogue"      # nhãn ở cột source của bảng bilingual
                pair = build_dash_pair(kind, o_cues, v_cues, o_turns, o_pick, v_pick, asr == "keep", mt == "keep")
                add_pair(kind, o_ids, v_ids, asr == "keep", mt == "keep", "dash", pair=pair)

    # ---- special_unmatched: cue không có cặp -> không vào bilingual (origin_only / vi_only) ----
    for r in tables.rows("special_unmatched", language, drama, ep):
        side = "vi" if (r.get("special_side") or "").upper() == "VI_ONLY" else "origin"
        cue = (vi_by_id if side == "vi" else origin_by_id).get(r["cue_id"])
        if cue is None or not match([cue], r.get("text_raw")):
            bad("special_unmatched", r, "id không khớp text SRT")
            continue
        if not (norm(r.get("asr_action")) and norm(r.get("mt_action"))):
            stats["pending_unmatched"] += 1
        else:
            stats["dropped_unmatched"] += 1
        drop(side, r["cue_id"], norm_kind(r.get("case_type")) or "unmatched")

    # cue đã vào cặp đặc biệt thì không còn ở pool cue thường
    for i in used_o:
        plan["drop_origin"].setdefault(i, "in_special_pair")
    for i in used_v:
        plan["drop_vi"].setdefault(i, "in_special_pair")
    return plan


# =====================================================================
# 3. Align origin (zh/ko/ja) <-> VI trên cue THÔ (không merge)
# =====================================================================
# Giới hạn khi gom nhiều cue gốc cùng khớp một (hoặc vài) cue vi thành MỘT cặp; vượt giới hạn thì không gom
# mà gán mỗi cue vi cho đúng một cue gốc (xem align_origin_vi)
ALIGN_GROUP_MAX_CUES = 6               # tối đa số cue gốc trong một nhóm
ALIGN_GROUP_MAX_SEC = MAX_CHUNK_DURATION # MAX_MERGED_DURATION   # tối đa độ dài đoạn bao các cue gốc của nhóm
ALIGN_GROUP_MAX_GAP = 1.0              # khoảng hở lớn nhất giữa hai cue gốc liên tiếp trong nhóm (giây)


def align_origin_vi(origin_segs: list[dict], vi_segs: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Trả về (aligned, origin_only, vi_only, stats) - origin_only/vi_only là các
    segment KHÔNG tìm được cặp khớp bên kia; chúng không vào bilingual (xem debug 02_align).

    Cách align (không merge theo câu, chỉ gom theo quan hệ chồng thời gian):
      1. Cạnh: cue gốc - cue vi chồng nhau >= MIN_OVERLAP_RATIO của cue NGẮN hơn.
      2. Nhóm: các cue nối nhau qua cạnh (connected component) là một nhóm; mỗi nhóm là MỘT cặp
         (zh nối bằng dấu cách theo thời gian, vi nối bằng dấu cách) nên không cue vi nào bị lặp ở hai cặp.
         Vd 2 cue zh "先生" + "鉴定费..." cùng khớp một cue vi -> một cặp.
      3. Nhóm quá lớn (> ALIGN_GROUP_MAX_CUES cue gốc, đoạn bao > ALIGN_GROUP_MAX_SEC giây, hoặc hai cue
         gốc liên tiếp cách nhau > ALIGN_GROUP_MAX_GAP giây) thì KHÔNG gom: mỗi cue vi gán cho đúng một cue
         gốc có phần chồng lớn nhất; cue gốc không còn cue vi nào thành origin_only."""
    # 1) cạnh và nhóm (union-find)
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges: dict[tuple[int, int], float] = {}                 # (chỉ số cue gốc, chỉ số cue vi) -> số giây chồng nhau
    for i, o in enumerate(origin_segs):
        o_dur = o["end"] - o["start"]
        for j, vc in enumerate(vi_segs):
            v_dur = vc["end"] - vc["start"]
            overlap = min(o["end"], vc["end"]) - max(o["start"], vc["start"])
            if overlap <= 0 or o_dur <= 0 or v_dur <= 0:
                continue
            if overlap / min(o_dur, v_dur) >= MIN_OVERLAP_RATIO:
                edges[(i, j)] = overlap
                parent[find(("o", i))] = find(("v", j))
    comps: dict = defaultdict(lambda: {"o": set(), "v": set()})
    for i, j in edges:
        root = find(("o", i))
        comps[root]["o"].add(i)
        comps[root]["v"].add(j)

    def make_pair(o_idx, v_idx) -> dict:
        os_ = sorted((origin_segs[i] for i in o_idx), key=lambda c: (c["start"], c["end"]))
        vs_ = sorted((vi_segs[j] for j in v_idx), key=lambda c: (c["start"], c["end"]))
        o_text = join_text_parts([c["text"] for c in os_])
        v_text = join_text_parts([c["text"] for c in vs_])
        return {
            "start": min(c["start"] for c in os_), "end": max(c["end"] for c in os_), "kind": "normal",
            "origin_text": o_text, "vi_text": v_text,
            "origin_text_asr": o_text, "origin_text_mt": o_text, "vi_text_mt": v_text,
            "asr_keep": True, "mt_keep": True,
            "origin_ids": [c["cue_id"] for c in os_], "vi_ids": [c["cue_id"] for c in vs_],
        }

    def group_ok(o_idx) -> bool:
        os_ = sorted((origin_segs[i] for i in o_idx), key=lambda c: (c["start"], c["end"]))
        span = max(c["end"] for c in os_) - min(c["start"] for c in os_)
        gaps = [b["start"] - a["end"] for a, b in zip(os_, os_[1:])]
        return (len(os_) <= ALIGN_GROUP_MAX_CUES and span <= ALIGN_GROUP_MAX_SEC
                and (max(gaps) if gaps else 0.0) <= ALIGN_GROUP_MAX_GAP)

    aligned, matched_o, n_grouped, n_fallback = [], set(), 0, 0
    for comp in sorted(comps.values(), key=lambda c: min(origin_segs[i]["start"] for i in c["o"])):
        o_idx, v_idx = sorted(comp["o"]), sorted(comp["v"])
        if len(o_idx) == 1 or group_ok(o_idx):
            aligned.append(make_pair(o_idx, v_idx))
            matched_o.update(o_idx)
            n_grouped += len(o_idx) > 1
        else:                                        # nhóm quá lớn: mỗi cue vi gán cho đúng một cue gốc
            n_fallback += 1
            assigned = defaultdict(list)
            for j in v_idx:
                best = max(o_idx, key=lambda i: edges.get((i, j), 0.0))
                assigned[best].append(j)
            for i, js in assigned.items():
                aligned.append(make_pair([i], js))
                matched_o.add(i)

    origin_only = [o for i, o in enumerate(origin_segs) if i not in matched_o]
    matched_v = {j for _, j in edges}
    vi_only = [vc for j, vc in enumerate(vi_segs) if j not in matched_v]
    aligned.sort(key=lambda p: (p["start"], p["end"]))
    stats = {
        "n_origin": len(origin_segs), "n_vi": len(vi_segs), "n_aligned": len(aligned),
        "n_origin_only": len(origin_only), "n_vi_only": len(vi_only),
        "n_grouped": n_grouped, "n_group_fallback": n_fallback,
    }
    return aligned, origin_only, vi_only, stats


# =====================================================================
# 4. Nhãn nhạc cho bilingual (chỉ dùng để thử cả hệ thống trên video)
# =====================================================================
def subtract_busy(span: tuple[float, float], busy: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Phần của khoảng `span` KHÔNG bị các khoảng trong `busy` che."""
    segs = [span]
    for bs, be in busy:
        nxt = []
        for s, e in segs:
            if be <= s or bs >= e:
                nxt.append((s, e))
                continue
            if bs > s:
                nxt.append((s, bs))
            if be < e:
                nxt.append((be, e))
        segs = nxt
    return segs


def music_label_pairs(music_cues: list[dict], pairs: list[dict], lang_tag: str) -> list[dict]:
    """Cue nhãn [nhạc] cho các khoảng nhạc thuần: gộp các cue nhạc (zh và vi) cách nhau <= MUSIC_MERGE_GAP,
    trừ phần bị cặp thoại nào đó che, bỏ đoạn còn lại ngắn hơn MUSIC_MIN_LABEL_SEC."""
    spans = []
    for s, e in sorted((c["start"], c["end"]) for c in music_cues):
        if spans and s - spans[-1][1] <= MUSIC_MERGE_GAP:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    busy = sorted((p["start"], p["end"]) for p in pairs)
    label_o = MUSIC_LABELS.get(lang_tag, MUSIC_LABEL_VI)
    out = []
    for span in spans:
        for s, e in subtract_busy(span, busy):
            if e - s >= MUSIC_MIN_LABEL_SEC:
                out.append({
                    "start": s, "end": e, "kind": "music",
                    "origin_text": label_o, "vi_text": MUSIC_LABEL_VI,
                    "origin_text_asr": "", "origin_text_mt": "", "vi_text_mt": "",
                    "asr_keep": False, "mt_keep": False, "overlap": "none", "origin_ids": [], "vi_ids": [],
                })
    return out


def pair_overlap(pair: dict, origin_by_id: dict, vi_by_id: dict, music_cues: list[dict]) -> str:
    """Giá trị cột overlap của một cặp (tính trên từng cue, TRƯỚC khi merge):
      music  : một cue zh hoặc vi của cặp chồng (>= OVERLAP_MIN_SEC) lên một cue nhạc zh/vi
               (cue bị loại do auto_excluded / music_pairs_review, tức reason bắt đầu bằng "music_")
      speech : cặp thuộc dash_dialogue / dash_partial (nhiều người nói luân phiên)
      none   : còn lại.   Cả hai cùng có thì music được ưu tiên."""
    cues = [origin_by_id[i] for i in pair["origin_ids"]] + [vi_by_id[i] for i in pair["vi_ids"]]
    for c in cues:
        for m in music_cues:
            if min(c["end"], m["end"]) - max(c["start"], m["start"]) >= OVERLAP_MIN_SEC:
                return "music"
    return "speech" if pair["kind"] in ATOMIC_KINDS else "none"


# =====================================================================
# Stage bilingual: raw SRT + review -> bilingual (SRT xem cùng video + bảng CSV)
# =====================================================================
def episode_paths(language: str, lang_tag: str, drama_folder: str, episode: str) -> dict:
    ep_tag = f"EP{int(episode):03d}"
    base_name = f"{language}_{drama_folder}_{ep_tag}"
    drama_dir = drama_raw_dir(language) / drama_folder
    bil_dir = BILINGUAL_DIR / language / drama_folder
    return {
        "ep_tag": ep_tag, "base_name": base_name,
        "origin_srt": drama_dir / f"{base_name}_{lang_tag}_manual_raw.srt",
        "vi_srt": drama_dir / f"{base_name}_vi_manual_raw.srt",
        "audio": drama_dir / f"{base_name}.{AUDIO_FORMAT}",
        "bil_csv": bil_dir / f"{base_name}_bilingual.csv",
        "bil_srt": bil_dir / f"{base_name}_bilingual.srt",
    }


def write_bilingual(paths: dict, pairs: list[dict]) -> list[dict]:
    rows, srt_cues = [], []
    for idx, p in enumerate(pairs, 1):
        rows.append({
            "pair_idx": idx, "start": round(p["start"], 3), "end": round(p["end"], 3),
            "duration": round(p["end"] - p["start"], 3),
            "origin_text": p["origin_text"], "vi_text": p["vi_text"],
            "origin_text_asr": p["origin_text_asr"], "origin_text_mt": p["origin_text_mt"], "vi_text_mt": p["vi_text_mt"],
            "asr_keep": int(p["asr_keep"]), "mt_keep": int(p["mt_keep"]), "source": p["kind"],
            "overlap": p["overlap"],
            "origin_cue_ids": ";".join(p["origin_ids"]), "vi_cue_ids": ";".join(p["vi_ids"]),
        })
        text = p["origin_text"] + ("\n" + p["vi_text"] if p["vi_text"] else "")      # zh ở trên, vi ở dưới
        srt_cues.append({"start": p["start"], "end": p["end"], "text": text})
    paths["bil_csv"].parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(paths["bil_csv"], BILINGUAL_FIELDS, rows)
    write_srt(srt_cues, paths["bil_srt"])      # cue thứ i = pair_idx i
    return rows


def build_bilingual_episode(language: str, lang_tag: str, drama_folder: str, episode: str,
                            tables: ReviewTables, debug_srt: bool, check_text: bool) -> tuple[list[dict], dict]:
    paths = episode_paths(language, lang_tag, drama_folder, episode)
    ep_tag, base_name = paths["ep_tag"], paths["base_name"]
    if not (paths["origin_srt"].exists() and paths["vi_srt"].exists()):
        return [], {"skipped": f"thiếu file ({lang_tag}={paths['origin_srt'].exists()}, vi={paths['vi_srt'].exists()})"}

    origin_all = load_cues(paths["origin_srt"], lang_tag)
    vi_all = load_cues(paths["vi_srt"], "vi")
    origin_by_id = {c["cue_id"]: c for c in origin_all}
    vi_by_id = {c["cue_id"]: c for c in vi_all}

    # 1. quyết định review -> cue đặc biệt
    plan = plan_special(tables, language, drama_folder, ep_tag, origin_by_id, vi_by_id, check_text)
    pool_o = [c for c in origin_all if c["cue_id"] not in plan["drop_origin"]]
    pool_v = [c for c in vi_all if c["cue_id"] not in plan["drop_vi"]]

    # cue còn marker mà không có quyết định nào -> loại, đếm
    undecided = {"origin": [], "vi": []}
    keep_o, keep_v = [], []
    for side, pool, keep in (("origin", pool_o, keep_o), ("vi", pool_v, keep_v)):
        for c in pool:
            if special_markers(c["text"]):
                undecided[side].append(c)
            else:
                keep.append(c)
    n_undecided = len(undecided["origin"]) + len(undecided["vi"])

    # 2. cue thường: clean -> align (KHÔNG merge ở bước này); origin_only / vi_only không vào bilingual
    origin_cues, vi_cues = clean_cues(keep_o), clean_cues(keep_v)
    aligned, origin_only, vi_only, stats = align_origin_vi(origin_cues, vi_cues)
    pairs = aligned + plan["pairs"]

    # cột overlap cho từng cặp: có cue nhạc (theo auto_excluded / music_pairs_review) chồng lên không
    music_cue_list = ([origin_by_id[k] for k, v in plan["drop_origin"].items() if v.startswith("music_")]
                      + [vi_by_id[k] for k, v in plan["drop_vi"].items() if v.startswith("music_")])
    for p in pairs:
        p["overlap"] = pair_overlap(p, origin_by_id, vi_by_id, music_cue_list)

    # 3. nhãn nhạc cho các khoảng nhạc thuần (chỉ để thử cả hệ thống trên video)
    if MUSIC_MODE == "label":
        music_cues = ([origin_by_id[k] for k, v in plan["drop_origin"].items() if v in MUSIC_REASONS]
                      + [vi_by_id[k] for k, v in plan["drop_vi"].items() if v in MUSIC_REASONS])
        pairs = pairs + music_label_pairs(music_cues, pairs, lang_tag)
    pairs = sorted(pairs, key=lambda p: (p["start"], p["end"]))

    # 4. xuất bilingual
    rows = write_bilingual(paths, pairs)

    kinds = Counter(p["kind"] for p in pairs)
    stats.update({
        "n_pairs": len(pairs), "kinds": dict(kinds),
        "n_asr": sum(p["asr_keep"] for p in pairs), "n_mt": sum(p["mt_keep"] for p in pairs),
        "n_undecided": n_undecided, "special": dict(plan["stats"]),
        "n_removed": sum(r != "in_special_pair" for r in [*plan["drop_origin"].values(), *plan["drop_vi"].values()]),
        "no_review_rows": not tables.has_episode(language, drama_folder, ep_tag),
        "warns": plan["warns"],
    })

    if debug_srt:
        debug_dir = drama_debug_dir(language)
        for sub in ("00_special", "01_clean", "02_align"):
            (debug_dir / sub).mkdir(parents=True, exist_ok=True)
        removed = ([{"side": "origin", "cue_id": k, "reason": v, "text": origin_by_id[k]["text"]} for k, v in plan["drop_origin"].items()]
                   + [{"side": "vi", "cue_id": k, "reason": v, "text": vi_by_id[k]["text"]} for k, v in plan["drop_vi"].items()]
                   + [{"side": s, "cue_id": c["cue_id"], "reason": "undecided_marker", "text": c["text"]}
                      for s in ("origin", "vi") for c in undecided[s]])
        write_csv_rows(debug_dir / "00_special" / f"{base_name}_removed.csv", ["side", "cue_id", "reason", "text"], removed)
        special_cues = [{"start": p["start"], "end": p["end"],
                         "text": f"[{p['kind']}] {p['origin_text']}\n{p['vi_text']}"} for p in plan["pairs"]]
        write_srt(special_cues, debug_dir / "00_special" / f"{base_name}_special_pairs.srt") if special_cues else \
            (debug_dir / "00_special" / f"{base_name}_special_pairs.srt").unlink(missing_ok=True)
        write_srt(origin_cues, debug_dir / "01_clean" / f"{base_name}_{lang_tag}.srt")
        write_srt(vi_cues, debug_dir / "01_clean" / f"{base_name}_vi.srt")
        # ghi file khi có dữ liệu, ngược lại xóa file của lần chạy trước để debug không bị lỗi thời
        for name, data in ((f"{base_name}_bilingual.srt",
                            [{"start": a["start"], "end": a["end"], "text": a["origin_text"] + "\n" + a["vi_text"]}
                             for a in aligned]),
                           (f"{base_name}_{lang_tag}_only.srt", origin_only),
                           (f"{base_name}_vi_only.srt", vi_only)):
            target = debug_dir / "02_align" / name
            if data:
                write_srt(data, target)
            else:
                target.unlink(missing_ok=True)
    return rows, stats


# =====================================================================
# 5. Merge cho metadata (ASR và MT merge ĐỘC LẬP trên bảng bilingual)
# =====================================================================
# ASR: gap <= 0.15 giây; split_run giữ nguyên, chia run theo giới hạn 15 giây.
# Dấu câu không chặn merge ASR; chỉ còn là ưu tiên chọn điểm cắt trong split_run.
# MT: dấu câu + giới hạn token nguồn/đích; không dùng gap hay thời lượng.
ASR_MERGE_GAP = 0.15
MAX_MERGED_DURATION = 15.0     # chỉ áp dụng ASR
MT_TOKENIZER_PATH = "facebook/nllb-200-distilled-600M"
MT_MAX_SOURCE_TOKENS = 480
MT_MAX_TARGET_TOKENS = 512
_MT_TOKENIZER = None
_MT_TOKEN_COUNTS: dict[tuple[str, str], int] = {}

ASR_SHORT_PIECE_SEC = 3.0       # khi phải cắt một đoạn quá dài: tránh tạo mảnh ngắn hơn giá trị này nếu còn cách khác
ATOMIC_KINDS = {"dash_dialogue", "dash_partial"}      # ASR: đã là 1 lượt thoại hoàn chỉnh, không merge với ai
MT_FLEX_KINDS = {"normal"} | ATOMIC_KINDS             # MT: cặp dash gộp được với cặp thường và với nhau


def _ends_sentence(text: str) -> bool:
    return bool(text) and text[-1] in SENTENCE_END_CHARS


def _make_item(p: dict, okey: str) -> dict:
    return {"start": p["start"], "end": p["end"], "kind": p["kind"], "otext": p[okey], "vtext": p["vi_text_mt"],
            "overlap": p["overlap"], "origin_ids": list(p["origin_ids"]), "vi_ids": list(p["vi_ids"])}


def _absorb(last: dict, cur: dict) -> None:
    """Gộp cặp `cur` vào cuối `last` (thời gian, text, id, overlap lấy giá trị mạnh nhất)."""
    last["end"] = max(last["end"], cur["end"])
    if OVERLAP_RANK[cur["overlap"]] > OVERLAP_RANK[last["overlap"]]:
        last["overlap"] = cur["overlap"]
    last["otext"] = join_text_parts([last["otext"], cur["otext"]])
    # cue vi khớp với 2 cue gốc liên tiếp xuất hiện ở cả hai cặp: không nối text vi lần thứ hai
    if not (cur["kind"] == "normal" and cur["vi_ids"] and set(cur["vi_ids"]) <= set(last["vi_ids"])):
        last["vtext"] = join_text_parts([last["vtext"], cur["vtext"]])
    last["origin_ids"] += cur["origin_ids"]
    last["vi_ids"] += [i for i in cur["vi_ids"] if i not in last["vi_ids"]]


def _log_row(mode: str, prev: dict, nxt: dict, merged_duration: float, blocker: str) -> dict:
    return {
        "mode": mode, "prev_start": round(prev["start"], 3), "prev_end": round(prev["end"], 3),
        "next_start": round(nxt["start"], 3), "gap": round(nxt["start"] - prev["end"], 3),
        "merged_duration": round(merged_duration, 3),
        "decision": "merge" if not blocker else "break", "blocker": blocker,
        "prev_zh_tail": prev["otext"][-20:], "prev_vi_tail": prev["vtext"][-30:],
        "next_zh_head": nxt["otext"][:20], "next_vi_head": nxt["vtext"][:30],
    }


# ---------------------------------------------------------------------
# ASR: ưu tiên gap, dấu kết câu không chặn merge
# ---------------------------------------------------------------------
def split_run(items: list[dict]) -> list[tuple[int, int]]:
    """Chia một đoạn liền mạch (mọi gap <= ASR_MERGE_GAP) thành các mảnh có độ dài <= MAX_MERGED_DURATION,
    trả về danh sách (đầu, cuối_không_gồm) theo chỉ số phần tử.
    Đoạn không quá dài thì giữ nguyên một mảnh. Đoạn quá dài thì chọn các chỗ cắt bằng quy hoạch động
    sao cho tổng "chi phí cắt" nhỏ nhất, thay vì dừng tham lam tại giới hạn (dễ cắt giữa cụm từ, bỏ lại
    mảnh 1,5 câu):
      chi phí cắt sau phần tử k = 1 - 0.7 * min(gap / ASR_MERGE_GAP, 1) - (0.3 nếu k kết câu)
        -> chỗ nghỉ dài và có dấu kết câu rẻ nhất (0), chỗ liền mạch không dấu câu đắt nhất (1)
      + 0.3 mỗi mảnh (ưu tiên ít mảnh) + 1.5 nếu mảnh ngắn hơn ASR_SHORT_PIECE_SEC."""
    n = len(items)
    if n == 1 or items[-1]["end"] - items[0]["start"] <= MAX_MERGED_DURATION:
        return [(0, n)]

    def break_cost(k: int) -> float:
        gap = items[k + 1]["start"] - items[k]["end"]
        ends = _ends_sentence(items[k]["otext"]) or _ends_sentence(items[k]["vtext"])
        return 1.0 - 0.7 * min(max(gap, 0.0) / ASR_MERGE_GAP, 1.0) - (0.3 if ends else 0.0)

    inf = float("inf")
    dp, prev = [inf] * (n + 1), [0] * (n + 1)
    dp[0] = 0.0
    for i in range(1, n + 1):
        seg_end = items[i - 1]["end"]
        for j in range(i - 1, -1, -1):                       # mảnh gồm các phần tử j .. i-1
            seg_end = max(seg_end, items[j]["end"])
            dur = seg_end - items[j]["start"]
            if dur > MAX_MERGED_DURATION and j < i - 1:      # dài hơn nữa chỉ tệ thêm (phần tử đơn luôn được phép)
                break
            cost = (dp[j] + 0.3 + (1.5 if dur < ASR_SHORT_PIECE_SEC else 0.0)
                    + (break_cost(j - 1) if j > 0 else 0.0))
            if cost < dp[i]:
                dp[i], prev[i] = cost, j
    cuts, i = [], n
    while i > 0:
        cuts.append((prev[i], i))
        i = prev[i]
    return cuts[::-1]


def merge_asr(pairs: list[dict], log: list | None = None) -> list[dict]:
    """Merge cho ASR: ưu tiên gap. Hai cặp CÙNG loại liên tiếp (cặp dash là điểm ngắt) có gap <= ASR_MERGE_GAP
    thuộc cùng một đoạn liền mạch, KHÔNG xét dấu kết câu; đoạn liền mạch quá dài được cắt ở chỗ nghỉ tốt nhất
    (split_run). Lý do ngắt ghi ở log: kind | gap | duration."""
    items = [_make_item(p, "origin_text_asr") for p in sorted(pairs, key=lambda p: (p["start"], p["end"]))]
    if not items:
        return []
    runs, run_break = [[items[0]]], {}                       # run_break[k] = lý do ngắt trước phần tử k
    for k in range(1, len(items)):
        a, b = items[k - 1], items[k]
        kind_ok = a["kind"] == b["kind"] and a["kind"] not in ATOMIC_KINDS
        if kind_ok and b["start"] - a["end"] <= ASR_MERGE_GAP:
            runs[-1].append(b)
        else:
            run_break[k] = "kind" if not kind_ok else "gap"
            runs.append([b])
    out, seg_of, base = [], [], 0
    for run in runs:
        for s0, e0 in split_run(run):
            merged = {**run[s0], "origin_ids": list(run[s0]["origin_ids"]), "vi_ids": list(run[s0]["vi_ids"])}
            for it in run[s0 + 1:e0]:
                _absorb(merged, it)
            out.append(merged)
            seg_of += [len(out) - 1] * (e0 - s0)
            if s0 > 0:
                run_break[base + s0] = "duration"
        base += len(run)
    if log is not None:
        for k in range(1, len(items)):
            blocker = "" if seg_of[k] == seg_of[k - 1] else run_break.get(k, "gap")
            seg_start = out[seg_of[k - 1]]["start"]
            log.append(_log_row("asr", items[k - 1], items[k],
                                max(items[k - 1]["end"], items[k]["end"]) - seg_start, blocker))
    return out


# ---------------------------------------------------------------------
# MT: dấu câu quyết định điểm dừng
# ---------------------------------------------------------------------
def _mt_token_count(text: str, side: str) -> int:
    """Đếm bằng tokenizer thực, gồm special tokens; không truncate văn bản.

    Tokenizer NLLB mặc định là thước đo chung để tạo cùng mẫu cho benchmark.
    Ngân sách prompt/context/generation từng model vẫn kiểm tra ở bước inference.
    """
    global _MT_TOKENIZER
    key = (side, text)
    if key not in _MT_TOKEN_COUNTS:
        if _MT_TOKENIZER is None:
            try:
                from transformers import AutoTokenizer
                _MT_TOKENIZER = AutoTokenizer.from_pretrained(MT_TOKENIZER_PATH)
            except Exception as exc:
                raise RuntimeError(
                    f"Không tải được tokenizer MT {MT_TOKENIZER_PATH!r}. "
                    "Cài transformers/sentencepiece hoặc chỉ định --mt-tokenizer đường dẫn local."
                ) from exc
        kwargs = {"add_special_tokens": True, "truncation": False,
                  "return_attention_mask": False}
        encoded = (_MT_TOKENIZER(text_target=text, **kwargs) if side == "target"
                   else _MT_TOKENIZER(text, **kwargs))
        _MT_TOKEN_COUNTS[key] = len(encoded["input_ids"])
    return _MT_TOKEN_COUNTS[key]


def _mt_token_blocker(item: dict) -> str:
    if _mt_token_count(item["otext"], "source") > MT_MAX_SOURCE_TOKENS:
        return "source_tokens"
    if _mt_token_count(item["vtext"], "target") > MT_MAX_TARGET_TOKENS:
        return "target_tokens"
    return ""


def merge_blocker(last: dict, seg: dict) -> str:
    """MT dừng theo kind, dấu kết câu zh/vi hoặc token nguồn/đích.

    Dash có thể nối với normal/dash nếu chưa kết câu và còn ngân sách token.
    Không dùng gap hoặc MAX_MERGED_DURATION. Mô phỏng _absorb để đếm đúng
    cả trường hợp nhiều cue origin cùng dùng một cue vi (không lặp text vi).
    """
    ka, kb = last["kind"], seg["kind"]
    if ka != kb and not (ka in MT_FLEX_KINDS and kb in MT_FLEX_KINDS):
        return "kind"
    if _ends_sentence(last["otext"]) or _ends_sentence(last["vtext"]):
        return "sentence_end"
    candidate = {**last, "origin_ids": list(last["origin_ids"]), "vi_ids": list(last["vi_ids"])}
    _absorb(candidate, seg)
    return _mt_token_blocker(candidate)


def merge_pairs(pairs: list[dict], mode: str, log: list | None = None) -> list[dict]:
    """Merge các cặp liền mạch. mode = "asr" (text zh khớp audio; xem merge_asr) hoặc "mt" (text MT).
    Mỗi phần tử trả về: start, end, kind, otext (zh), vtext (vi), overlap, origin_ids, vi_ids.
    Nếu truyền `log` (list) thì ghi lại quyết định ở MỌI chỗ nối giữa hai cặp liên tiếp (dùng --debug-merge)."""
    if mode == "asr":
        return merge_asr(pairs, log)
    _MT_TOKEN_COUNTS.clear()  # giới hạn cache trong một lần merge (một tập)
    out: list[dict] = []
    prev_item = None
    for p in sorted(pairs, key=lambda p: (p["start"], p["end"])):
        cur = _make_item(p, "origin_text_mt")
        oversized = _mt_token_blocker(cur)
        if oversized:
            raise ValueError(
                f"MT unit vượt ngưỡng {oversized}: origin_ids={cur['origin_ids']}, "
                f"vi_ids={cur['vi_ids']}. Cần tách/align lại unit hoặc điều chỉnh ngưỡng; "
                "không tự truncate hay xuất mẫu vượt giới hạn."
            )
        blocker = merge_blocker(out[-1], cur) if out else "first"
        if log is not None and prev_item is not None:
            log.append(_log_row("mt", prev_item, cur, max(out[-1]["end"], cur["end"]) - out[-1]["start"], blocker))
        if out and not blocker:
            _absorb(out[-1], cur)
        else:
            out.append(cur)
        prev_item = cur
    return out


def cut_chunk(src_audio, start: float, end: float, out_path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",   # đặt TRƯỚC -i để seek nhanh (input seeking)
        "-i", str(src_audio),
        "-t", f"{duration:.3f}",
        "-ar", str(CHUNK_SAMPLE_RATE), "-ac", str(CHUNK_CHANNELS),
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and out_path.exists()


def probe_duration(path) -> float | None:
    cmd = [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def validate_chunk(path, expected_duration: float) -> tuple[bool, str]:
    if not path.exists():
        return False, "file không tồn tại"
    actual = probe_duration(path)
    if actual is None:
        return False, "không đọc được duration"
    if actual <= 0:
        return False, "duration <= 0"
    if actual < MIN_CHUNK_DURATION or actual > MAX_CHUNK_DURATION:
        return False, f"duration {actual:.2f}s ngoài khoảng cho phép"
    if abs(actual - expected_duration) > DURATION_TOLERANCE:
        return False, f"duration lệch quá nhiều (thực tế {actual:.2f}s, kỳ vọng {expected_duration:.2f}s)"
    return True, ""


def chunk_output_path(language: str, drama_folder: str, drama_id: str, episode: str, ep_tag: str, chunk_idx: int):
    sample_id = make_sample_id(language, drama_id, episode, chunk_idx)
    out_dir = drama_audio_chunk_dir(language, drama_folder, ep_tag)
    return out_dir / f"{sample_id}.wav", sample_id


def to_relative_path(path) -> str:
    """Lưu đường dẫn TƯƠNG ĐỐI so với dataset/ để metadata portable giữa các máy."""
    return path.resolve().relative_to(DATASET_DIR.resolve()).as_posix()


# =====================================================================
# Stage metadata: bảng bilingual -> merge ASR / MT -> chunk audio + metadata rows
# =====================================================================
def read_bilingual_pairs(csv_path: Path) -> list[dict]:
    """Đọc bảng bilingual (đã soát) thành list cặp để merge."""
    pairs = []
    for r in read_review_csv(csv_path):
        pairs.append({
            "start": float(r["start"]), "end": float(r["end"]), "kind": r["source"],
            "origin_text_asr": r["origin_text_asr"], "origin_text_mt": r["origin_text_mt"], "vi_text_mt": r["vi_text_mt"],
            "asr_keep": r["asr_keep"] == "1", "mt_keep": r["mt_keep"] == "1",
            "overlap": r.get("overlap") or "none",
            "origin_ids": split_ids(r["origin_cue_ids"]), "vi_ids": split_ids(r["vi_cue_ids"]),
        })
    return pairs


MERGE_LOG_FIELDS = ["mode", "prev_start", "prev_end", "next_start", "gap", "merged_duration", "decision", "blocker",
                    "prev_zh_tail", "prev_vi_tail", "next_zh_head", "next_vi_head"]


def metadata_episode(language: str, lang_tag: str, drama_folder: str, drama_id: str,
                     episode: str, debug_merge: bool = False) -> tuple[list[dict], list[dict], dict]:
    """Từ bảng bilingual của 1 tập: ASR = merge (asr_keep) + cắt chunk; MT = merge (mt_keep)."""
    paths = episode_paths(language, lang_tag, drama_folder, episode)
    ep_tag = paths["ep_tag"]
    if not paths["bil_csv"].exists():
        return [], [], {"skipped": f"chưa có bảng bilingual ({paths['bil_csv'].name}); chạy --stage bilingual trước"}
    pairs = read_bilingual_pairs(paths["bil_csv"])

    # ---- ASR: chỉ cặp asr_keep, merge kiểu ASR (dash là điểm ngắt), cắt chunk theo thời gian đã merge ----
    merge_log: list | None = [] if debug_merge else None
    asr_items = [it for it in merge_pairs([p for p in pairs if p["asr_keep"] and p["origin_text_asr"]], "asr", merge_log)
                 if it["otext"]]
    if asr_items and not paths["audio"].exists():
        return [], [], {"skipped": f"thiếu audio {paths['audio'].name}"}
    asr_rows, n_fail = [], 0
    for idx, it in enumerate(asr_items, 1):
        out_path, sample_id = chunk_output_path(language, drama_folder, drama_id, episode, ep_tag, idx)
        ok = cut_chunk(paths["audio"], it["start"], it["end"], out_path)
        if ok:
            ok, _ = validate_chunk(out_path, it["end"] - it["start"])
        if not ok:
            out_path.unlink(missing_ok=True)
            n_fail += 1
            continue
        asr_rows.append({
            "id": sample_id, "language": language, "drama_id": drama_id, "episode_id": episode,
            "audio_path": to_relative_path(out_path), "start": round(it["start"], 3), "end": round(it["end"], 3),
            "duration": round(it["end"] - it["start"], 3), "origin_text": it["otext"],
            "overlap": it["overlap"],
        })

    # ---- MT: chỉ cặp mt_keep, merge kiểu MT (bỏ điểm ngắt dash); không cần audio ----
    mt_items = merge_pairs([p for p in pairs if p["mt_keep"] and p["origin_text_mt"] and p["vi_text_mt"]], "mt", merge_log)
    if merge_log is not None:                      # mỗi dòng = một chỗ nối giữa hai cặp liên tiếp: vì sao merge / vì sao dừng
        log_dir = drama_debug_dir(language) / "03_merge"
        log_dir.mkdir(parents=True, exist_ok=True)
        write_csv_rows(log_dir / f"{paths['base_name']}_merge_decisions.csv", MERGE_LOG_FIELDS, merge_log)
    mt_rows = [{
        "id": make_sample_id(language, drama_id, episode, idx), "language": language,
        "drama_id": drama_id, "episode_id": episode, "origin_text": it["otext"], "vi_text": it["vtext"],
    } for idx, it in enumerate(mt_items, 1) if it["otext"] and it["vtext"]]

    return asr_rows, mt_rows, {"n_pairs": len(pairs), "asr_merged": len(asr_items), "asr_ok": len(asr_rows),
                               "asr_fail": n_fail, "mt_merged": len(mt_items), "mt_rows": len(mt_rows)}


# ==== Metadata (resumable) - theo từng ngôn ngữ ====
def load_existing_metadata(language: str) -> tuple[list[dict], list[dict], set]:
    paths = drama_metadata_paths(language)
    asr_rows = read_review_csv(paths["asr"]) if paths["asr"].exists() else []
    mt_rows = read_review_csv(paths["mt"]) if paths["mt"].exists() else []
    done = {(r["drama_id"], r["episode_id"]) for r in asr_rows + mt_rows}
    return asr_rows, mt_rows, done


def save_metadata(language: str, asr_rows: list[dict], mt_rows: list[dict]):
    paths = drama_metadata_paths(language)
    paths["asr"].parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(paths["asr"], ASR_FIELDNAMES, sorted(({k: r.get(k, "") for k in ASR_FIELDNAMES} for r in asr_rows), key=lambda r: r["id"]))
    write_csv_rows(paths["mt"], MT_FIELDNAMES, sorted(({k: r.get(k, "") for k in MT_FIELDNAMES} for r in mt_rows), key=lambda r: r["id"]))


def find_episodes():
    """Quét raw/{language}/Dxxxx/{language}_Dxxxx_EPyyy_{lang_tag}_manual_raw.srt ->
    list of (language, lang_tag, drama_folder, drama_id, episode)."""
    result = []
    for language, lang_tag in DRAMA_LANGUAGES.items():
        lang_raw_dir = drama_raw_dir(language)
        if not lang_raw_dir.is_dir():
            continue
        ep_re = episode_file_re(language, lang_tag)
        for drama_dir in sorted(lang_raw_dir.glob("D*")):
            if not drama_dir.is_dir():
                continue
            for f in sorted(drama_dir.glob(f"{language}_*_{lang_tag}_manual_raw.srt")):
                m = ep_re.match(f.name)
                if m:
                    result.append((language, lang_tag, drama_dir.name, m.group(1), m.group(2)))
    return result


def print_bilingual_stats(stats: dict, lang_tag: str):
    sp = stats["special"]
    kinds = ", ".join(f"{k}={v}" for k, v in sorted(stats["kinds"].items()))
    print(f"  {lang_tag}={stats['n_origin']} vi={stats['n_vi']} aligned={stats['n_aligned']} "
          f"({lang_tag}_only={stats['n_origin_only']}, vi_only={stats['n_vi_only']}) "
          f"-> cue bilingual={stats['n_pairs']} [{kinds}]")
    if stats["n_grouped"] or stats["n_group_fallback"]:
        print(f"  gom nhóm: {stats['n_grouped']} cặp gồm nhiều cue gốc cùng khớp một đoạn vi, "
              f"{stats['n_group_fallback']} nhóm quá lớn không gom (gán mỗi cue vi cho một cue gốc)")
    print(f"  vào ASR={stats['n_asr']}, vào MT={stats['n_mt']}, cue đặc biệt bị loại={stats['n_removed']}, "
          f"cue marker chưa có quyết định={stats['n_undecided']}")
    pending = {k: v for k, v in sp.items() if k.startswith("pending") or k in ("row_skipped_mismatch", "conflict_rows")}
    if pending:
        print(f"  [CẢNH BÁO] {pending}")
    if stats["no_review_rows"]:
        print("  [CẢNH BÁO] tập này chưa có dòng nào trong bảng review (test_review.py chưa chạy?)")
    for w in stats["warns"][:5]:
        print(f"    - {w}")
    if len(stats["warns"]) > 5:
        print(f"    ... còn {len(stats['warns']) - 5} cảnh báo khác")


def main():
    global MT_TOKENIZER_PATH, MT_MAX_SOURCE_TOKENS, MT_MAX_TARGET_TOKENS, _MT_TOKENIZER
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["bilingual", "metadata", "all"], default="bilingual")
    parser.add_argument("--debug-srt", action="store_true", help="Xuất thêm debug từng bước vào processed/debug/{language}/")
    parser.add_argument("--debug-merge", action="store_true",
                        help="Stage metadata: ghi processed/debug/{language}/03_merge/*_merge_decisions.csv "
                             "(mỗi chỗ nối giữa hai cặp: merge hay dừng, và lý do kind/sentence_end/gap/duration/source_tokens/target_tokens)")
    parser.add_argument("--mt-tokenizer", default=MT_TOKENIZER_PATH,
                        help="Tokenizer/path local dùng kiểm tra độ dài MT (mặc định NLLB 600M)")
    parser.add_argument("--mt-max-source-tokens", type=int, default=MT_MAX_SOURCE_TOKENS,
                        help="Giới hạn token nguồn MT gồm special tokens (mặc định 480)")
    parser.add_argument("--mt-max-target-tokens", type=int, default=MT_MAX_TARGET_TOKENS,
                        help="Giới hạn token đích MT gồm special tokens (mặc định 512)")
    parser.add_argument("--force", action="store_true", help="Làm lại cả tập đã xử lý")
    parser.add_argument("--skip-text-check", action="store_true",
                        help="Không so text SRT với text trong CSV review (mặc định có so để bắt id cue bị lệch)")
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--language", choices=list(DRAMA_LANGUAGES.keys()))
    parser.add_argument("--drama", help="VD D0002")
    parser.add_argument("--episode", type=int)
    args = parser.parse_args()
    if args.mt_max_source_tokens <= 0 or args.mt_max_target_tokens <= 0:
        parser.error("Ngưỡng token MT phải lớn hơn 0")
    MT_TOKENIZER_PATH = args.mt_tokenizer
    MT_MAX_SOURCE_TOKENS = args.mt_max_source_tokens
    MT_MAX_TARGET_TOKENS = args.mt_max_target_tokens
    _MT_TOKENIZER = None
    _MT_TOKEN_COUNTS.clear()

    episodes = [e for e in find_episodes()
                if (not args.language or e[0] == args.language)
                and (not args.drama or e[2] == args.drama)
                and (args.episode is None or int(e[4]) == args.episode)]
    if not episodes:
        print(f"Không tìm thấy file *_manual_raw.srt phù hợp trong {DRAMA_RAW_DIR}.")
        return

    do_bil = args.stage in ("bilingual", "all")
    do_meta = args.stage in ("metadata", "all")
    tables = ReviewTables(args.review_dir) if do_bil else None
    if do_bil and not args.review_dir.exists():
        print(f"[CẢNH BÁO] không thấy thư mục review {args.review_dir}: mọi cue có marker sẽ bị loại.")

    by_language: dict[str, list[tuple]] = {}
    for language, lang_tag, drama_folder, drama_id, episode in episodes:
        by_language.setdefault(language, []).append((lang_tag, drama_folder, drama_id, episode))
    print(f"Tìm thấy {len(episodes)} tập trên {len(by_language)} ngôn ngữ. Stage: {args.stage}\n")

    for language, lang_episodes in by_language.items():
        asr_all, mt_all, processed_eps = [], [], set()
        if do_meta:
            asr_all, mt_all, processed_eps = load_existing_metadata(language)
        print(f"=== {language} ===" + (f" (đã có metadata: {len(processed_eps)} tập)" if do_meta else ""))

        for lang_tag, drama_folder, drama_id, episode in lang_episodes:
            ep_label = f"{drama_folder} EP{episode}"
            if do_bil:
                bil_csv = episode_paths(language, lang_tag, drama_folder, episode)["bil_csv"]
                if bil_csv.exists() and not args.force:
                    print(f"--- {ep_label}: đã có bilingual, bỏ qua (dùng --force để làm lại) ---")
                else:
                    print(f"--- {ep_label}: bilingual ---")
                    try:
                        _, stats = build_bilingual_episode(language, lang_tag, drama_folder, episode, tables,
                                                           args.debug_srt, not args.skip_text_check)
                    except ValueError as e:                        # parse_srt strict: đánh số dở dang/trùng
                        print(f"  [LỖI] {e}")
                        continue
                    if stats.get("skipped"):
                        print(f"  [BỎ QUA] {stats['skipped']}")
                        continue
                    print_bilingual_stats(stats, lang_tag)

            if do_meta:
                if not args.force and (drama_id, episode) in processed_eps:
                    continue
                print(f"--- {ep_label}: metadata ---")
                time.sleep(5)
                asr_rows, mt_rows, stats = metadata_episode(language, lang_tag, drama_folder, drama_id, episode,
                                                            args.debug_merge)
                if stats.get("skipped"):
                    print(f"  [BỎ QUA] {stats['skipped']}")
                    continue
                print(f"  cue bilingual={stats['n_pairs']} -> ASR: {stats['asr_merged']} đoạn merge, chunk hợp lệ="
                      f"{stats['asr_ok']}, lỗi={stats['asr_fail']} | MT: {stats['mt_rows']} cặp")
                same = lambda r: r["drama_id"] == drama_id and r["episode_id"] == episode
                asr_all = [r for r in asr_all if not same(r)] + asr_rows
                mt_all = [r for r in mt_all if not same(r)] + mt_rows
                save_metadata(language, asr_all, mt_all)          # ghi ngay sau mỗi tập để resumable

        if do_meta:
            mp = drama_metadata_paths(language)
            print(f"  Xong {language}. ASR={len(asr_all)} dòng ({mp['asr']}), MT={len(mt_all)} dòng ({mp['mt']})\n")
        else:
            print()

    print("Hoàn tất.")


if __name__ == "__main__":
    main()
