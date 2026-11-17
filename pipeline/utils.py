"""Hàm tiện ích dùng chung cho cả pipeline."""

import re
from pathlib import Path

from config import DRAMA_ID_WIDTH, EPISODE_WIDTH, CHUNK_WIDTH, SPORTS_EVENT_WIDTH, SPORTS_CLIP_WIDTH

SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


# ==== Zero padding ====
# DRAMA
def pad_drama_id(drama_id) -> str:
    return f"D{int(drama_id):0{DRAMA_ID_WIDTH}d}"

def pad_episode(episode) -> str:
    return f"EP{int(episode):0{EPISODE_WIDTH}d}"

def pad_chunk(chunk_idx) -> str:
    return f"{int(chunk_idx):0{CHUNK_WIDTH}d}"

def make_sample_id(language, drama_id, episode, chunk_idx) -> str:
    return f"{language}_{pad_drama_id(drama_id)}_{pad_episode(episode)}_{pad_chunk(chunk_idx)}"

# SPORTS
def slugify(name: str) -> str:
    name = name.strip().lower()               # bỏ khoảng trắng thừa 2 đầu, chuyển thường
    name = re.sub(r"\s+", "_", name)           # thay khoảng trắng bằng "_"
    name = re.sub(r"[^a-z0-9_\-]", "", name)   # xóa mọi ký tự không phải chữ/số/_/-
    return name

def pad_event_id(event_id) -> str:
    return f"EV{int(event_id):0{SPORTS_EVENT_WIDTH}d}"

def pad_clip_id(clip_id) -> str:
    return f"C{int(clip_id):0{SPORTS_CLIP_WIDTH}d}"

# ==== Thời gian ====
def srt_time_to_seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def seconds_to_srt_time(t: float) -> str:
    ms_total = round(t * 1000)
    h, rem = divmod(ms_total, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ==== SRT ====
def parse_srt_text(text: str, source: str = "", strict: bool = True) -> list[dict]:
    """Parse NỘI DUNG SRT -> list cue, gồm CẢ cue rỗng, theo thứ tự trong file.
 
    Mỗi cue: id, id_in_file, id_line_no, time_line_no, start, end, text
      - id_in_file : số ghi ở dòng ngay trước dòng thời gian (None nếu không có)
      - id_line_no / time_line_no : chỉ số dòng (theo text.splitlines(keepends=True))
      - id         : ID dùng trong pipeline
    strict=True  (dùng để đọc):
        - mọi cue đều có số  -> id = số trong file (phải duy nhất, nếu trùng thì ValueError)
        - không cue nào có số -> id = vị trí 1..N (tính cả cue rỗng, đúng như number_srt.py sẽ đánh)
        - đánh số dở dang    -> ValueError (chạy number_srt.py)
    strict=False (dùng khi --renumber): id = vị trí 1..N, bỏ qua số cũ.
    """
    lines = text.splitlines(keepends=True)
    time_idx = [i for i, line in enumerate(lines) if SRT_TIME_RE.search(line)]
    cues = []
    for k, i in enumerate(time_idx):
        g = SRT_TIME_RE.search(lines[i]).groups()
 
        id_in_file, id_line_no = None, None
        if i > 0:
            token = lines[i - 1].strip().lstrip("\ufeff")
            if token.isdigit():
                id_in_file, id_line_no = int(token), i - 1
 
        stop = time_idx[k + 1] if k + 1 < len(time_idx) else len(lines)
        body, hit_blank = [], False
        for j in range(i + 1, stop):
            if not lines[j].strip():
                hit_blank = True
                break
            body.append(lines[j].rstrip("\r\n"))
        # thiếu dòng trống giữa 2 cue: dòng số cuối là ID của cue kế, không phải text
        if (not hit_blank and k + 1 < len(time_idx) and body
                and body[-1].strip().lstrip("\ufeff").isdigit()):
            body.pop()
 
        cues.append({
            "id": None,
            "id_in_file": id_in_file,
            "id_line_no": id_line_no,
            "time_line_no": i,
            "start": srt_time_to_seconds(*g[0:4]),
            "end": srt_time_to_seconds(*g[4:8]),
            "text": "\n".join(body).strip(),
        })
 
    n = len(cues)
    numbered = sum(c["id_in_file"] is not None for c in cues)
    if strict and n:
        if numbered == n:
            ids = [c["id_in_file"] for c in cues]
            seen, dup = set(), []
            for x in ids:
                if x in seen and x not in dup:
                    dup.append(x)
                seen.add(x)
            if dup:
                raise ValueError(f"{source}: trùng số cue {dup[:10]}{' ...' if len(dup) > 10 else ''}")
        elif numbered == 0:
            ids = list(range(1, n + 1))
        else:
            raise ValueError(
                f"{source}: đánh số dở dang ({numbered}/{n} cue có số); "
                f"chạy number_srt.py (thêm --renumber nếu cần)")
    else:
        ids = list(range(1, n + 1))
    for c, cid in zip(cues, ids):
        c["id"] = cid
    return cues

def parse_srt(path: Path, strict: bool = True) -> list[dict]:
    """Đọc 1 file .srt -> list of {id, start, end, text} (start/end tính bằng giây).
 
    id = số cue GỐC trong file (xem parse_srt_text). Cue rỗng bị lọc bỏ nhưng KHÔNG làm
    lệch id của cue sau. Muốn có cả cue rỗng hoặc thông tin dòng, dùng parse_srt_text.
    """
    content = Path(path).read_text(encoding="utf-8-sig", errors="ignore")   # utf-8-sig: bỏ BOM
    cues = parse_srt_text(content, source=str(path), strict=strict)
    return [{"id": c["id"], "start": c["start"], "end": c["end"], "text": c["text"]}
            for c in cues if c["text"]]


"""
def parse_srt(path: Path) -> list[dict]:
    # Đọc 1 file .srt -> list of {start, end, text} (start/end tính bằng giây).
    content = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", content.strip())
    cues = []
    for block in blocks:
        lines = block.strip().splitlines()
        time_match = None
        text_lines = []
        for line in lines:
            m = SRT_TIME_RE.search(line)
            if m:
                time_match = m
                continue
            if time_match is not None:
                text_lines.append(line)
        if time_match is None:
            continue
        g = time_match.groups()
        start = srt_time_to_seconds(*g[0:4])
        end = srt_time_to_seconds(*g[4:8])
        text = "\n".join(text_lines).strip()
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return cues
"""

def parse_srts(path: Path) -> list[dict]:
    """Đọc 1 file .srt -> list of {index, start, end, text} (start/end tính bằng giây).
 
    index: số thứ tự cue GỐC ghi trong file (dòng số đứng ngay trước dòng thời gian).
    Block không có số hợp lệ thì dùng vị trí block (1-based) trong file.
    Cue có text rỗng vẫn bị bỏ qua, nhưng KHÔNG làm lệch index của các cue sau.
    """
    content = Path(path).read_text(encoding="utf-8-sig", errors="ignore")  # utf-8-sig: bỏ BOM
    blocks = re.split(r"\n\s*\n", content.strip())
    cues = []
    for block_pos, block in enumerate(blocks, start=1):
        lines = block.strip().splitlines()
        time_match = None
        index = None
        text_lines = []
        for line in lines:
            m = SRT_TIME_RE.search(line)
            if m:
                time_match = m
                continue
            if time_match is None:
                token = line.strip().lstrip("\ufeff")
                if token.isdigit():          # dòng số đứng trước dòng thời gian
                    index = int(token)
                continue
            text_lines.append(line)
        if time_match is None:
            continue
        g = time_match.groups()
        start = srt_time_to_seconds(*g[0:4])
        end = srt_time_to_seconds(*g[4:8])
        text = "\n".join(text_lines).strip()
        if text:
            cues.append({
                "index": index if index is not None else block_pos,
                "start": start,
                "end": end,
                "text": text,
            })
    return cues

def write_srt(cues: list[dict], path: Path) -> None:
    lines = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(c['start'])} --> {seconds_to_srt_time(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ==== CSV log đơn giản (đọc/ghi list of dict, giữ nguyên thứ tự cột) ====

def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ==== Merge cue (dùng chung: hội thoại gạch đầu dòng, cue bị tách, ghép nhiều cue thành 1) ====
# Cùng bộ ký tự với MARKER_PATTERNS["leading_dash"] trong config.py
LEADING_DASH_STRIP_RE = re.compile(r"^\s*[-－–—]\s*")


def strip_leading_dash(text: str) -> str:
    """Bỏ dấu gạch đầu dòng (- － – —) ở đầu MỖI dòng, giữ nguyên số dòng.
    '- Có.\\n- Không.' -> 'Có.\\nKhông.'"""
    return "\n".join(LEADING_DASH_STRIP_RE.sub("", line) for line in (text or "").splitlines())


def join_text_parts(parts: list[str], sep: str = " ") -> str:
    """Nối nhiều đoạn text thành 1 dòng, giữa các đoạn luôn có đúng 1 dấu cách.

    Mỗi đoạn được tách theo dòng, bỏ khoảng trắng đầu/cuối từng dòng rồi nối bằng `sep`.
    Nhờ vậy cue thoại dài bị tách làm 2 dòng (hoặc 2 cue) khi nối sẽ:
      - chưa có dấu cách ở giữa -> được chèn 1 dấu cách;
      - đã có sẵn dấu cách (kể cả cuối dòng trước / đầu dòng sau) -> không bị đôi.
    Dòng rỗng bị bỏ qua.
    """
    pieces = []
    for part in parts:
        for line in (part or "").splitlines():
            line = line.strip()
            if line:
                pieces.append(line)
    return sep.join(pieces)


def flatten_text(text: str, strip_dash: bool = False) -> str:
    """Đưa text 1 cue (có thể nhiều dòng) về 1 dòng; strip_dash=True thì bỏ dash đầu dòng trước."""
    return join_text_parts([strip_leading_dash(text) if strip_dash else text])


def max_cue_gap(cues: list[dict]) -> float:
    """Khoảng hở lớn nhất giữa 2 cue liên tiếp (sau khi sắp theo thời gian).
    Âm nếu các cue chồng nhau; 0.0 nếu chỉ có 1 cue."""
    ordered = sorted(cues, key=lambda c: (c["start"], c["end"]))
    return max((b["start"] - a["end"] for a, b in zip(ordered, ordered[1:])), default=0.0)


def merge_cues(cues: list[dict], text_key: str = "text", strip_dash: bool = False) -> dict:
    """Gộp nhiều cue thành 1 cue.

    - Sắp theo (start, end); start = min start, end = max end.
    - text = các text nối bằng join_text_parts (đảm bảo có dấu cách giữa các đoạn);
      strip_dash=True thì bỏ dấu gạch đầu dòng trước khi nối.
    - cues: dict có "start", "end" (giây) và text ở khóa `text_key` ("text" cho parse_srt,
      "text_raw" cho cue của test_review.py).
    Trả về {"start", "end", "text", "n_cues", "max_gap"}; có thể đưa thẳng vào write_srt.
    """
    if not cues:
        raise ValueError("merge_cues: danh sách cue rỗng")
    ordered = sorted(cues, key=lambda c: (c["start"], c["end"]))
    texts = [strip_leading_dash(c.get(text_key, "")) if strip_dash else (c.get(text_key, "") or "")
             for c in ordered]
    return {
        "start": min(c["start"] for c in ordered),
        "end": max(c["end"] for c in ordered),
        "text": join_text_parts(texts),
        "n_cues": len(ordered),
        "max_gap": max_cue_gap(ordered),
    }


def merge_cue_groups(origin_cues: list[dict], vi_cues: list[dict], text_key: str = "text",
                     strip_dash_origin: bool = True, strip_dash_vi: bool = True) -> tuple[dict, dict]:
    """Gộp 1 nhóm cue gốc và 1 nhóm cue VI thành một cặp (origin_cue, vi_cue).
    Dùng cho cả hội thoại gạch đầu dòng (2 cue gốc <-> 1 cue VI 2 dash, hoặc ngược lại)
    lẫn các cặp N-M thông thường. Mỗi phía gộp độc lập bằng merge_cues."""
    return (merge_cues(origin_cues, text_key, strip_dash_origin),
            merge_cues(vi_cues, text_key, strip_dash_vi))

def split_dash_turns(text: str) -> list[str]:
    """Tách text 1 cue thành các LƯỢT THOẠI (mỗi lượt đã bỏ dấu gạch đầu dòng).
 
    - Dòng bắt đầu bằng dash (- － – —) mở một lượt mới.
    - Dòng không có dash nối vào lượt đang mở (cue dài bị tách 2 dòng) bằng join_text_parts,
      nên chắc chắn có đúng 1 dấu cách ở chỗ nối.
    - Dòng không dash nằm ĐẦU cue tự thành lượt 1.
    '- Được, đi nhé.\\n- Vốn dĩ chỉ là\\nmột vụ án.' -> ['Được, đi nhé.', 'Vốn dĩ chỉ là một vụ án.']
    """
    turns: list[list[str]] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        if LEADING_DASH_STRIP_RE.match(line):
            turns.append([LEADING_DASH_STRIP_RE.sub("", line)])
        elif turns:
            turns[-1].append(line)
        else:
            turns.append([line])
    return [t for t in (join_text_parts(parts) for parts in turns) if t]