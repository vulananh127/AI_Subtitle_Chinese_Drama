"""Lớp nền + bảng đăng ký backend (ASR/MT). Thêm backend mới: kế thừa ASRBackend/MTBackend và gắn @register."""

from __future__ import annotations

import gc

ASR_BACKENDS: dict[str, type] = {}
MT_BACKENDS: dict[str, type] = {}


def register(task: str, name: str):
    table = {"asr": ASR_BACKENDS, "mt": MT_BACKENDS}[task]

    def deco(cls):
        table[name] = cls
        cls.backend_name = name
        return cls
    return deco


class Backend:
    """Mọi tham số của backend khai báo trong DEFAULTS; tham số lạ (gõ sai --set) bị từ chối thay vì lặng lẽ bỏ qua."""

    DEFAULTS: dict = {"batch_size": 16}

    def __init__(self, model_path: str, **cfg):
        unknown = set(cfg) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"{type(self).__name__}: tham số không hợp lệ {sorted(unknown)}. "
                             f"Hợp lệ: {sorted(self.DEFAULTS)}")
        self.model_path = model_path
        self.cfg = {**self.DEFAULTS, **cfg}

    def load(self):
        """Nạp model (gọi một lần trước khi suy luận)."""

    def close(self):
        """Giải phóng bộ nhớ GPU."""
        for attr in ("model", "pipe", "llm"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class ASRBackend(Backend):
    def transcribe(self, audio_paths: list[str], lang: str) -> list[str]:
        """audio_paths: danh sách file wav 16kHz mono; lang: mã ngôn ngữ nguồn ('zh'). Trả về transcript cùng thứ tự."""
        raise NotImplementedError


class MTSource(str):
    """Source text compatible with existing translate(list[str], ...) callers.

    Keep this object intact until PromptBuilder; str(src), strip(), JSON roundtrips
    remove context. Plain strings remain supported with empty context.
    """

    def __new__(cls, text, prev_src="", next_src=""):
        obj = super().__new__(cls, text)
        obj.prev_src = prev_src
        obj.next_src = next_src
        return obj


class MTBackend(Backend):
    def translate(self, sources: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Trả về bản dịch cùng thứ tự với sources."""
        raise NotImplementedError


def load_wav_16k(path: str):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{path}: sample rate {sr} != 16000 (pipeline chuẩn hóa chunk về 16kHz mono)")
    return wav


def with_dtype(fn, *args, dtype=None, **kwargs):
    """Gọi from_pretrained/pipeline với dtype, tương thích cả `torch_dtype` (cũ) lẫn `dtype` (mới) của transformers."""
    if dtype is None:
        return fn(*args, **kwargs)
    if dtype != "auto":
        import torch
        dtype = getattr(torch, dtype)
    try:
        return fn(*args, torch_dtype=dtype, **kwargs)
    except TypeError:
        return fn(*args, dtype=dtype, **kwargs)
