"""Cấu hình chung cho pipeline thu thập & xử lý dữ liệu ASR/MT."""

import re
from pathlib import Path

# ==== Đường dẫn gốc ====
# pipeline/ và dataset/ nằm ngang hàng nhau, vd:
#   project/
#   ├── pipeline/   (chứa các file .py này)
#   └── dataset/    (source.csv, raw/, processed/...)
# Dùng __file__ để xác định đường dẫn tuyệt đối, không phụ thuộc vào
# thư mục đang đứng khi chạy lệnh (cwd).
PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

# ==== DRAMA ====
# Cấu trúc thư mục:
#   dataset/drama/
#   ├── raw/{language}/Dxxxx/Dxxxx_EPyyy.mp3
#   │                          Dxxxx_EPyyy_{lang_tag}_raw.srt
#   │                          Dxxxx_EPyyy_vi_raw.srt
#   ├── processed/
#   │   ├── audio/{language}/Dxxxx/EPyyy/Dxxxx_EPyyy_nnnnn.wav
#   │   ├── debug/{language}/01_clean|02_merge|03_align/...
#   │   └── metadata/{language}/metadata_master.csv
#   │                            metadata_asr.csv   (audio -> origin_text)
#   │                            metadata_mt.csv    (origin_text -> vi_text)
#   └── splits/{language}/train.csv, validation.csv, test.csv
DRAMA_DIR = DATASET_DIR / "drama"
DRAMA_SOURCE_CSV = DRAMA_DIR / "drama_source.csv"
DRAMA_RAW_DIR = DRAMA_DIR / "raw"
DRAMA_PROCESSED_DIR = DRAMA_DIR / "processed"
DRAMA_AUDIO_CHUNKS_DIR = DRAMA_PROCESSED_DIR / "audio"
DRAMA_DEBUG_DIR = DRAMA_PROCESSED_DIR / "debug"
DRAMA_METADATA_DIR = DRAMA_PROCESSED_DIR / "metadata"
DRAMA_SPLITS_DIR = DRAMA_DIR / "splits"

# Ngôn ngữ nguồn hỗ trợ cho DRAMA -> tag dùng trong tên file *_<{tag}>_raw.srt
DRAMA_LANGUAGES = {
    "chinese": "zh",
    "korean": "ko",
    "japanese": "ja",
}

def drama_raw_dir(language: str) -> Path:
    return DRAMA_RAW_DIR / language

def drama_debug_dir(language: str) -> Path:
    return DRAMA_DEBUG_DIR / language

def drama_audio_chunk_dir(language: str, drama_folder: str, ep_tag: str) -> Path:
    return DRAMA_AUDIO_CHUNKS_DIR / language / drama_folder / ep_tag

def drama_metadata_dir(language: str) -> Path:
    return DRAMA_METADATA_DIR / language

def drama_metadata_paths(language: str) -> dict:
    d = drama_metadata_dir(language)
    return {
        "master": d / "metadata_master.csv",
        "asr": d / "metadata_asr.csv",
        "mt": d / "metadata_mt.csv",
    }

def drama_splits_dir(language: str) -> Path:
    return DRAMA_SPLITS_DIR / language

# ==== SPORTS ====
SPORTS_DIR = DATASET_DIR / "sports"
SPORTS_SOURCE_CSV = SPORTS_DIR / "sports_source.csv"
SPORTS_RAW_DIR = SPORTS_DIR / "raw"         # raw/{category}/ (phẳng, không tách theo clip)
SPORTS_PROCESSED_DIR = SPORTS_DIR / "processed"
SPORTS_AUDIO_CHUNKS_DIR = SPORTS_PROCESSED_DIR / "audio"
SPORTS_METADATA_MASTER_CSV = SPORTS_PROCESSED_DIR / "sports_metadata_master.csv"
SPORTS_METADATA_SPLIT_CSV = SPORTS_PROCESSED_DIR / "sports_metadata_master_split.csv"

# ==== Zero padding ====
DRAMA_ID_WIDTH = 4
EPISODE_WIDTH = 3
CHUNK_WIDTH = 5

SPORTS_EVENT_WIDTH = 4
SPORTS_CLIP_WIDTH = 3

# ==== Ngôn ngữ (prefix nhận diện mã ngôn ngữ trên các nền tảng khác nhau) ====
ZH_PREFIXES = ("zh", "chi", "cmn")      # Tiếng Trung
KO_PREFIXES = ("ko", "kor")             # Tiếng Hàn
JA_PREFIXES = ("ja", "jpn")             # Tiếng Nhật
VI_PREFIXES = ("vi", "vie")
EN_PREFIXES = ("en",)


# ==== Audio ====
AUDIO_FORMAT = "mp3"       # định dạng audio gốc tải về (nguyên tập)
VIDEO_FORMAT = "mp4"
CHUNK_SAMPLE_RATE = 16000  # sample rate chuẩn cho ASR
CHUNK_CHANNELS = 1         # mono

# ==== Merge segment (cùng 1 ngôn ngữ) ====
MERGE_GAP_THRESHOLD = 0.3     # giây - coi là liên tục nếu gap <= giá trị này
MAX_MERGED_DURATION = 15.0    # giây - không merge vượt quá độ dài này
# Các dấu câu kết thúc câu - không merge tiếp nếu segment trước đã kết thúc bằng dấu này
SENTENCE_END_CHARS = set("。！？.!?…")

# ==== Align origin (zh/ko/ja) <-> VI ====
MIN_OVERLAP_RATIO = 0.3   # ngưỡng overlap_ratio tối thiểu để coi là 1 cặp hợp lệ

# ==== Lọc subtitle đặc biệt (test_review.py) ====
# Định nghĩa marker (chỉ để REVIEW, không phải kết luận). Tên khóa được dùng ở các tập bên dưới
# và ghi ra cột detected_markers, nên đổi tên khóa thì phải đổi cả những chỗ đó.
MARKER_PATTERNS = {
    # ngoặc vuông: [ ] ASCII + 【】〔〕 (ngoặc góc CJK, cũng dùng cho caption)
    "square_brackets": re.compile(r"[\[\]【】〔〕]"),
    # ngoặc tròn: ( ) ASCII + （） full-width (tiếng Trung)
    "round_brackets": re.compile(r"[()（）]"),
    "music_note": re.compile(r"[♪♫♬♩]"),   # dùng chung cho has_note và is_music_only
    "leading_dash": re.compile(r"(?m)^\s*[-－–—]\s*\S"),
    "html_or_ass_markup": re.compile(r"<[^>]+>|\{[^}]*\}"),
}

MUSIC_CAPTION_RE = re.compile(
    r"^\s*[\[（(【]\s*(?:nhạc(?: nền)?|tiếng nhạc|music|background music|"
    r"音樂|音乐|배경음악|音楽)\s*[\]）)】]\s*$", re.IGNORECASE
)

# Stage 1 - music
SPEECH_SUPPORT_MIN_IOU = 0.35   # cue thoại khác được tính là "hỗ trợ" khi IoU >= ngưỡng với cue kia
EXCLUDE_MIN_PLAIN_COVER = 0.7   # exclude chỉ khi cue nhạc phủ >= 70% thời lượng cue không phải nhạc
# Marker chỉ là định dạng/dấu nốt, KHÔNG làm cue thành "có marker khác" khi xét ghép cặp nhạc
PAIR_FORMAT_ONLY_MARKERS = {"music_note", "html_or_ass_markup"}
# Stage 2 - candidates
SAME_LANG_MIN_SEC = 0.3         # overlap cùng ngôn ngữ: tối thiểu (giây) ...
SAME_LANG_MIN_RATIO = 0.3       # ... và >= 30% thời lượng cue ngắn hơn
AUTO_KEEP_MIN_IOU = 0.7         # cặp 1-1 có IoU >= ngưỡng này -> gợi ý keep_pair
# Ghép cặp candidate: giữ cặp khi (a) cả hai cue đều có marker, hoặc (b) IoU >= ngưỡng này,
# hoặc (c) là cặp IoU cao nhất của một cue đặc biệt (để cue đó không bị mất hết cặp).
# Dòng chỉ được giữ nhờ (c) mà IoU < ngưỡng sẽ ghi note = "weak_match".
CANDIDATE_MIN_IOU = 0.3
# Cue có marker trong tập này KHÔNG bị lọc IoU / giữ best: mọi cặp đạt MIN_OVERLAP_RATIO đều được giữ.
# leading_dash: một cue VI gộp lời nhiều người (- A / - B) ghép với nhiều cue ZH ngắn, IoU từng cặp
# thấp nhưng đều là cặp đúng.
CANDIDATE_IOU_EXEMPT_MARKERS = {"leading_dash"}
# Cue có marker trong tập này bị bỏ qua khi xét luật origin_overlap (caption [...] hiện song song
# với thoại nên chồng nhau là bình thường). Marker khác (ngoặc tròn, gạch đầu dòng...) vẫn tính.
OVERLAP_IGNORE_MARKERS = {"square_brackets"}

# Kiểm tra tên marker trong các tập trên có tồn tại trong MARKER_PATTERNS (tránh gõ sai tên khóa)
assert PAIR_FORMAT_ONLY_MARKERS <= set(MARKER_PATTERNS), "PAIR_FORMAT_ONLY_MARKERS có tên marker lạ"
assert OVERLAP_IGNORE_MARKERS <= set(MARKER_PATTERNS), "OVERLAP_IGNORE_MARKERS có tên marker lạ"
assert CANDIDATE_IOU_EXEMPT_MARKERS <= set(MARKER_PATTERNS), "CANDIDATE_IOU_EXEMPT_MARKERS có tên marker lạ"

# ==== Validate audio chunk ====
MIN_CHUNK_DURATION = 0.3   # giây
MAX_CHUNK_DURATION = 15.0  # giây
DURATION_TOLERANCE = 0.3   # giây - sai lệch cho phép giữa audio thực tế và (end-start)

# ==== ffmpeg / ffprobe ====
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"