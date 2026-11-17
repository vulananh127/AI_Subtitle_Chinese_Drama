"""Đường dẫn dùng chung cho evaluation/ (nạp config.py + utils.py của pipeline/).

Cấu trúc project (evaluation/ nằm ngang hàng pipeline/ và dataset/):
    project/
    ├── pipeline/     (config.py, utils.py, ...)
    ├── evaluation/   (các file này)
    └── dataset/drama/processed/
        ├── metadata/{language}/metadata_asr.csv, metadata_mt.csv
        └── eval/{language}/{asr|mt}/{pretrained|fine-tuned}/{tên_model}/
                predictions.jsonl   (kết quả suy luận thô, dùng để --resume / --score-only)
                predictions.csv     (từng câu kèm điểm, để xem lỗi)
                metrics.json        (điểm tổng + cấu hình chạy)
"""

import re
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _CODE_DIR.parent / "pipeline"
if not (_PIPELINE_DIR / "config.py").exists():
    raise ImportError(
        f"Không thấy {_PIPELINE_DIR / 'config.py'}: thư mục evaluation/ phải nằm ngang hàng pipeline/"
    )
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from config import (  # noqa: E402
    DATASET_DIR,
    DRAMA_AUDIO_CHUNKS_DIR,
    DRAMA_DIR,
    DRAMA_LANGUAGES,
    DRAMA_PROCESSED_DIR,
    PROJECT_ROOT,
    drama_metadata_dir,
)
from utils import read_csv_rows, write_csv_rows  # noqa: E402,F401

TASKS = ("asr", "mt")
STAGES = ("pretrained", "fine-tuned")

DRAMA_EVAL_DIR = DRAMA_PROCESSED_DIR / "eval"


def safe_name(name: str) -> str:
    """Tên model -> tên thư mục an toàn ('Qwen/Qwen3-ASR-1.7B' -> 'Qwen__Qwen3-ASR-1.7B')."""
    name = re.sub(r"[^\w.+\-]+", "_", name.strip().replace("/", "__"))
    return name.lstrip(".") or "model"


def drama_metadata_csv(language: str, task: str) -> Path:
    """metadata_asr.csv / metadata_mt.csv trong drama_metadata_dir(language)."""
    return drama_metadata_dir(language) / f"metadata_{task}.csv"


def drama_eval_dir(language: str, task: str, stage: str = "pretrained", model_name: str | None = None) -> Path:
    """eval/{language}/{task}/{stage}[/{model_name}]"""
    if task not in TASKS:
        raise ValueError(f"task phải là một trong {TASKS}, nhận '{task}'")
    if stage not in STAGES:
        raise ValueError(f"stage phải là một trong {STAGES}, nhận '{stage}'")
    d = DRAMA_EVAL_DIR / language / task / stage
    return d / safe_name(model_name) if model_name else d
