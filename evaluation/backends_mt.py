"""Backend MT. Khác biệt giữa các model dịch chủ yếu nằm ở PROMPT -> khai báo trong registry.MT_MODELS, không sửa code.

prompt_mode:
  chat          : bọc prompt vào chat template của tokenizer (Qwen, Hunyuan...)
  raw           : đưa nguyên prompt, không chat template (Seed-X, MiLMMT)
  translategemma: chat template riêng của TranslateGemma (content có source_lang_code/target_lang_code)
Placeholder trong prompt: {prev_src} {next_src} {src} {src_name} {tgt_name} {src_full} {tgt_full} {src_code} {tgt_code} {src_name_zh} {tgt_name_zh}
"""

from __future__ import annotations

import re

from .common import MTBackend, register, with_dtype
from .langs import lang_info, prompt_fields

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _postprocess(text: str, mode: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    if mode == "first_line":
        text = text.splitlines()[0].strip() if text else ""
    return text


class PromptBuilder:
    """Dựng token id của prompt, dùng chung cho HF và vLLM (đảm bảo hai engine nhận cùng một đầu vào)."""

    def __init__(self, tok, cfg: dict):
        if cfg["prompt_mode"] != "translategemma" and not cfg["prompt"]:
            raise ValueError("Thiếu 'prompt' trong cấu hình model (xem registry.MT_MODELS)")
        self.tok, self.cfg = tok, cfg

    def build_ids(self, src: str, src_lang: str, tgt_lang: str) -> list[int]:
        c, tok = self.cfg, self.tok
        mode = c["prompt_mode"]
        if mode == "translategemma":
            msgs = [{"role": "user", "content": [{"type": "text", "source_lang_code": src_lang,
                                                  "target_lang_code": tgt_lang, "text": src}]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return tok(text, add_special_tokens=False)["input_ids"]
        tmpl = c["prompt_xx"] if (c["prompt_xx"] and "zh" not in (src_lang, tgt_lang)) else c["prompt"]
        fields = prompt_fields(src, src_lang, tgt_lang)
        fields.update(prev_src=getattr(src, "prev_src", ""), next_src=getattr(src, "next_src", ""))
        prompt = tmpl.format(**fields)
        if mode == "chat":
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, **c["chat_template_kwargs"])
            return tok(text, add_special_tokens=False)["input_ids"]
        if mode == "raw":
            return tok(prompt, add_special_tokens=c["add_special_tokens"])["input_ids"]
        raise ValueError(f"prompt_mode không hợp lệ: {mode}")


_CAUSAL_DEFAULTS = {
    "batch_size": 16, "device": "auto", "dtype": "bfloat16", "max_new_tokens": 256,
    "prompt": None, "prompt_xx": None, "prompt_mode": "chat", "chat_template_kwargs": {},
    "add_special_tokens": True, "trust_remote_code": False, "postprocess": "strip",
}


@register("mt", "hf_causal")
class HFCausalMT(MTBackend):
    """LLM dịch / LLM tổng quát chạy bằng transformers (greedy). Hy-MT, Qwen3.x, TranslateGemma, Seed-X, MiLMMT..."""

    DEFAULTS = {**_CAUSAL_DEFAULTS, "auto_class": "AutoModelForCausalLM", "gen_kwargs": {}}

    def load(self):
        import transformers
        from transformers import AutoTokenizer
        c = self.cfg
        self.tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=c["trust_remote_code"])
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        cls = getattr(transformers, c["auto_class"])
        self.model = with_dtype(cls.from_pretrained, self.model_path, dtype=c["dtype"], device_map=c["device"],
                                trust_remote_code=c["trust_remote_code"])
        self.model.eval()
        self.builder = PromptBuilder(self.tok, c)

    def translate(self, sources, src_lang, tgt_lang):
        import torch
        c = self.cfg
        ids = [self.builder.build_ids(s, src_lang, tgt_lang) for s in sources]
        order = sorted(range(len(ids)), key=lambda k: len(ids[k]), reverse=True)  # dài trước: lỗi OOM lộ sớm
        out = [""] * len(ids)
        gen = {"do_sample": False, "max_new_tokens": c["max_new_tokens"], "pad_token_id": self.tok.pad_token_id,
               **c["gen_kwargs"]}
        for b in range(0, len(order), c["batch_size"]):
            idx = order[b:b + c["batch_size"]]
            batch = self.tok.pad({"input_ids": [ids[k] for k in idx]}, padding=True, return_tensors="pt")
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.inference_mode():
                res = self.model.generate(**batch, **gen)
            texts = self.tok.batch_decode(res[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            for k, t in zip(idx, texts):
                out[k] = _postprocess(t, c["postprocess"])
        return out


@register("mt", "vllm_causal")
class VLLMCausalMT(MTBackend):
    """Giống hf_causal nhưng chạy bằng vLLM (nhanh hơn nhiều với 16k câu). pip install vllm."""

    DEFAULTS = {**_CAUSAL_DEFAULTS, "max_model_len": 2048, "gpu_memory_utilization": 0.9,
                "engine_kwargs": {}, "sampling_kwargs": {}}

    def load(self):
        from transformers import AutoTokenizer
        from vllm import LLM
        c = self.cfg
        self.tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=c["trust_remote_code"])
        self.llm = LLM(model=self.model_path, dtype=c["dtype"], trust_remote_code=c["trust_remote_code"],
                       max_model_len=c["max_model_len"], gpu_memory_utilization=c["gpu_memory_utilization"],
                       **c["engine_kwargs"])
        self.builder = PromptBuilder(self.tok, c)

    def translate(self, sources, src_lang, tgt_lang):
        from vllm import SamplingParams
        c = self.cfg
        prompts = [{"prompt_token_ids": self.builder.build_ids(s, src_lang, tgt_lang)} for s in sources]
        sp = SamplingParams(temperature=0.0, max_tokens=c["max_new_tokens"], **c["sampling_kwargs"])
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [_postprocess(o.outputs[0].text, c["postprocess"]) for o in outs]


@register("mt", "hf_seq2seq")
class HFSeq2SeqMT(MTBackend):
    """NLLB-200 / M2M100... (encoder-decoder, dùng mã ngôn ngữ NLLB trong langs.LANGS)."""

    DEFAULTS = {"batch_size": 32, "device": "cuda:0", "dtype": "float16", "max_new_tokens": 256,
                "num_beams": 4, "max_source_len": 256, "gen_kwargs": {}}

    def load(self):
        from transformers import AutoModelForSeq2SeqLM
        c = self.cfg
        self.model = with_dtype(AutoModelForSeq2SeqLM.from_pretrained, self.model_path, dtype=c["dtype"])
        self.model.to(c["device"]).eval()

    def translate(self, sources, src_lang, tgt_lang):
        import torch
        from transformers import AutoTokenizer
        c = self.cfg
        tok = AutoTokenizer.from_pretrained(self.model_path, src_lang=lang_info(src_lang)["nllb"])
        bos = tok.convert_tokens_to_ids(lang_info(tgt_lang)["nllb"])
        order = sorted(range(len(sources)), key=lambda k: len(sources[k]), reverse=True)
        out = [""] * len(sources)
        for b in range(0, len(order), c["batch_size"]):
            idx = order[b:b + c["batch_size"]]
            enc = tok([sources[k] for k in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=c["max_source_len"]).to(self.model.device)
            with torch.inference_mode():
                res = self.model.generate(**enc, forced_bos_token_id=bos, num_beams=c["num_beams"],
                                          max_new_tokens=c["max_new_tokens"], **c["gen_kwargs"])
            for k, t in zip(idx, tok.batch_decode(res, skip_special_tokens=True)):
                out[k] = t.strip()
        return out


@register("mt", "dummy")
class DummyMT(MTBackend):
    """Chỉ để kiểm tra luồng chạy: trả lại nguyên câu nguồn. fail_after_calls=N để thử --resume."""

    DEFAULTS = {"batch_size": 16, "fail_after_calls": 0}

    def load(self):
        self._calls = 0

    def translate(self, sources, src_lang, tgt_lang):
        self._calls += 1
        if self.cfg["fail_after_calls"] and self._calls > self.cfg["fail_after_calls"]:
            raise RuntimeError("dummy: lỗi giả lập")
        return list(sources)
