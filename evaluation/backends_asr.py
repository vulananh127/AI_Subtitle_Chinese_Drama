"""Backend ASR. Mỗi backend nhận danh sách đường dẫn wav -> danh sách transcript.

Muốn thêm họ model mới (vd FireRedASR2): viết một lớp ~20 dòng kế thừa ASRBackend, gắn @register("asr", "tên"),
rồi thêm một dòng vào registry.ASR_MODELS. Phần còn lại (dữ liệu, lưu kết quả, CER, so sánh) không đổi.
"""

from __future__ import annotations

import os
import re

from .common import ASRBackend, load_wav_16k, register, with_dtype
from .langs import lang_info

_TAG_RE = re.compile(r"<\|[^|>]*\|>")


@register("asr", "qwen3_asr")
class Qwen3ASR(ASRBackend):
    """Qwen3-ASR-0.6B/1.7B (pip install qwen-asr[vllm]). API theo README của Qwen3-ASR."""

    DEFAULTS = {"batch_size": 32, "device": "cuda:0", "dtype": "bfloat16", "max_new_tokens": 256,
                "force_language": True}

    def load(self):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_path, dtype=getattr(torch, self.cfg["dtype"]), device_map=self.cfg["device"],
            max_inference_batch_size=self.cfg["batch_size"], max_new_tokens=self.cfg["max_new_tokens"])

    def transcribe(self, audio_paths, lang):
        language = [lang_info(lang)["qwen"]] * len(audio_paths) if self.cfg["force_language"] else None
        res = self.model.transcribe(audio=audio_paths, language=language)
        return [r.text for r in res]


@register("asr", "funasr")
class FunASR(ASRBackend):
    """Paraformer / SenseVoice / các model trong FunASR (pip install funasr). Tag kiểu <|zh|><|NEUTRAL|> được xóa."""

    DEFAULTS = {"batch_size": 16, "device": "cuda:0", "batch_size_s": 300, "pass_language": False,
                "model_kwargs": {}, "generate_kwargs": {}}

    def load(self):
        from funasr import AutoModel
        self.model = AutoModel(model=self.model_path, device=self.cfg["device"], disable_update=True,
                               **self.cfg["model_kwargs"])

    def transcribe(self, audio_paths, lang):
        gk = dict(self.cfg["generate_kwargs"])
        if self.cfg["pass_language"]:
            gk["language"] = lang
        res = self.model.generate(input=audio_paths, batch_size_s=self.cfg["batch_size_s"], **gk)
        if len(res) != len(audio_paths):
            raise RuntimeError(f"FunASR trả {len(res)} kết quả cho {len(audio_paths)} file (có thể do VAD cắt đoạn); "
                               f"chunk đã được cắt sẵn nên đừng bật vad_model trong model_kwargs.")
        return [_TAG_RE.sub("", r.get("text", "")).strip() for r in res]


@register("asr", "hf_whisper")
class HFWhisper(ASRBackend):
    """Whisper và các bản fine-tune dạng Whisper trên HuggingFace (kể cả Belle-whisper...)."""

    DEFAULTS = {"batch_size": 16, "device": "cuda:0", "dtype": "float16", "generate_kwargs": {}}

    def load(self):
        from transformers import pipeline
        self.pipe = with_dtype(pipeline, "automatic-speech-recognition", model=self.model_path,
                               device=self.cfg["device"], dtype=self.cfg["dtype"])

    def transcribe(self, audio_paths, lang):
        audios = [{"raw": load_wav_16k(p), "sampling_rate": 16000} for p in audio_paths]
        gk = {"task": "transcribe", "language": lang_info(lang)["whisper"], **self.cfg["generate_kwargs"]}
        out = self.pipe(audios, batch_size=self.cfg["batch_size"], generate_kwargs=gk)
        return [o["text"].strip() for o in out]


@register("asr", "dummy")
class DummyASR(ASRBackend):
    """Chỉ để kiểm tra luồng chạy: trả về tên file. fail_after_calls=N: gây lỗi sau N lần gọi (thử --resume)."""

    DEFAULTS = {"batch_size": 16, "fail_after_calls": 0}

    def load(self):
        self._calls = 0

    def transcribe(self, audio_paths, lang):
        self._calls += 1
        if self.cfg["fail_after_calls"] and self._calls > self.cfg["fail_after_calls"]:
            raise RuntimeError("dummy: lỗi giả lập")
        return [os.path.splitext(os.path.basename(p))[0] for p in audio_paths]
