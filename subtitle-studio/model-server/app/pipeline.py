"""Cầu nối giữa web và model của bạn. Chỉ cần sửa file này.

PIPELINE=mock (mặc định): trả dữ liệu mẫu để thử giao diện.
PIPELINE=real           : gọi ASR + MT thật (điền vào các TODO bên dưới).
"""
import os

MODE = os.getenv("PIPELINE", "mock")
_models = {}


def _load():
    """Nạp model đúng 1 lần (lần gọi đầu tiên), giữ trên GPU."""
    if _models:
        return _models
    # TODO: thay bằng code nạp model của bạn, ví dụ:
    # from my_project.asr import ASRModel
    # from my_project.mt import MTModel
    # _models["asr"] = ASRModel.from_pretrained(os.environ["ASR_CKPT"]).to("cuda")
    # _models["mt"] = MTModel.from_pretrained(os.environ["MT_CKPT"]).to("cuda")
    raise NotImplementedError("Chưa gắn model thật vào pipeline.py")


def transcribe(wav_path: str, lang: str) -> list[dict]:
    """Trả về [{'start': giây, 'end': giây, 'text': 'câu gốc'}, ...]"""
    if MODE == "mock":
        return [{"start": i * 4.0, "end": i * 4.0 + 3.6, "text": f"(ASR) câu nguồn số {i + 1}"} for i in range(5)]
    m = _load()
    # TODO: return m["asr"].transcribe(wav_path, language=lang)  # đảm bảo có start/end/text
    raise NotImplementedError


def translate(texts: list[str], src: str, tgt: str) -> list[str]:
    """Nhận danh sách câu nguồn, trả về danh sách bản dịch cùng độ dài."""
    if MODE == "mock":
        return [f"(MT) bản dịch sơ của: {t}" for t in texts]
    m = _load()
    # TODO: return m["mt"].translate(texts, src_lang=src, tgt_lang=tgt)
    raise NotImplementedError
