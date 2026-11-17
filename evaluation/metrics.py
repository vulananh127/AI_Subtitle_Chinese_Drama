"""Công thức đánh giá DÙNG CHUNG cho mọi model (pretrained lẫn fine-tuned) -> điểm luôn so sánh được.

ASR : CER (ký tự) / WER (từ). Chuẩn hóa: NFKC, chữ thường, bỏ dấu câu + ký hiệu, bỏ khoảng trắng (CER).
      CER tổng = tổng lỗi / tổng số ký tự tham chiếu (micro); thêm cer_macro = trung bình CER từng câu.
MT  : BLEU (sacreBLEU, tokenizer theo ngôn ngữ đích), chrF, chrF++ (word_order=2), COMET (tham chiếu / QE).
      Văn bản MT được chuẩn hóa NFC + gộp khoảng trắng trước khi chấm (tiếng Việt dễ lệch dấu tổ hợp/dựng sẵn).

Đơn vị: CER/WER theo %, BLEU/chrF theo thang 0-100, COMET theo thang gốc của model (~0-1).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Sequence

from .langs import lang_info

log = logging.getLogger(__name__)


# ============================== ASR ==============================

def normalize_asr_text(text: str, *, lower: bool = True, strip_punct: bool = True, strip_space: bool = True) -> str:
    """Chuẩn hóa trước khi tính CER/WER. strip_punct bỏ ký tự thuộc nhóm Unicode P* (dấu câu) và S* (ký hiệu, vd ♪)."""
    t = unicodedata.normalize("NFKC", text or "")
    if lower:
        t = t.lower()
    if strip_punct:
        t = "".join(ch for ch in t if unicodedata.category(ch)[0] not in "PS")
    return "".join(t.split()) if strip_space else " ".join(t.split())


def edit_counts(ref: Sequence, hyp: Sequence) -> tuple[int, int, int]:
    """Khoảng cách Levenshtein giữa 2 chuỗi/danh sách token -> (số thay thế, số xóa, số chèn).
    Tổng S+D+I là duy nhất; cách chia S/D/I có thể khác nhau chút giữa các cách căn chỉnh hòa."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0
    if ref == hyp:
        return 0, 0, 0
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri, di, dp = ref[i - 1], d[i], d[i - 1]
        for j in range(1, m + 1):
            di[j] = min(dp[j - 1] + (ri != hyp[j - 1]), dp[j] + 1, di[j - 1] + 1)
    i, j, s, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele, i = dele + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return s, dele, ins


def compute_asr_metrics(refs: Sequence[str], hyps: Sequence[str], unit: str = "char"):
    """unit='char' -> CER, unit='word' -> WER. Trả về (summary, per_utt)."""
    if unit not in ("char", "word"):
        raise ValueError("unit phải là 'char' hoặc 'word'")
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) và hyps ({len(hyps)}) không cùng độ dài")
    key = "cer" if unit == "char" else "wer"
    per, rates = [], []
    tot = {"ref": 0, "s": 0, "d": 0, "i": 0}
    n_empty_hyp = n_over = n_skip = 0
    for r, h in zip(refs, hyps):
        rn = normalize_asr_text(r, strip_space=(unit == "char"))
        hn = normalize_asr_text(h, strip_space=(unit == "char"))
        rt = list(rn) if unit == "char" else rn.split()
        ht = list(hn) if unit == "char" else hn.split()
        s, d, i = edit_counts(rt, ht)
        e = s + d + i
        rate = None
        if rt:
            rate = e / len(rt)
            rates.append(rate)
            tot["ref"] += len(rt); tot["s"] += s; tot["d"] += d; tot["i"] += i
            n_over += rate > 1.0
        else:
            n_skip += 1  # tham chiếu rỗng sau chuẩn hóa -> không xác định được tỉ lệ
        n_empty_hyp += not ht
        per.append({"ref_norm": rn, "hyp_norm": hn, "ref_len": len(rt), "sub": s, "del": d, "ins": i,
                    "errors": e, key: None if rate is None else round(100 * rate, 4)})
    N = tot["ref"]
    pct = lambda x: round(100 * x / N, 4) if N else None  # noqa: E731
    summary = {
        key: pct(tot["s"] + tot["d"] + tot["i"]),
        f"{key}_macro": round(100 * sum(rates) / len(rates), 4) if rates else None,
        f"{key}_sub": pct(tot["s"]), f"{key}_del": pct(tot["d"]), f"{key}_ins": pct(tot["i"]),
        "n": len(rates), "ref_len_total": N,
        "n_empty_hyp": n_empty_hyp, "n_over_100pct": n_over, "n_skipped_empty_ref": n_skip,
    }
    return summary, per


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return compute_asr_metrics(refs, hyps, "char")[0]["cer"]


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return compute_asr_metrics(refs, hyps, "word")[0]["wer"]


# ============================== MT ==============================

def normalize_mt_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def _need(module: str, pip_name: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as e:
        raise ImportError(f"Thiếu thư viện (pip install {pip_name})") from e


def _sig(metric) -> str:
    try:
        return str(metric.get_signature())
    except Exception:  # noqa: BLE001
        return ""


def compute_bleu(hyps, refs, tgt_lang: str = "vi") -> dict:
    sb = _need("sacrebleu.metrics", "sacrebleu")
    m = sb.BLEU(tokenize=lang_info(tgt_lang)["bleu_tok"])
    return {"bleu": round(m.corpus_score(list(hyps), [list(refs)]).score, 4), "bleu_signature": _sig(m)}


def compute_chrf(hyps, refs, word_order: int = 0) -> dict:
    """word_order=0 -> chrF ; word_order=2 -> chrF++"""
    sb = _need("sacrebleu.metrics", "sacrebleu")
    m = sb.CHRF(word_order=word_order)
    key = "chrf" if word_order == 0 else "chrfpp"
    return {key: round(m.corpus_score(list(hyps), [list(refs)]).score, 4), f"{key}_signature": _sig(m)}


def sentence_chrf(hyps, refs, word_order: int = 0) -> list[float]:
    sb = _need("sacrebleu.metrics", "sacrebleu")
    m = sb.CHRF(word_order=word_order)
    return [round(m.sentence_score(h, [r]).score, 4) for h, r in zip(hyps, refs)]


def comet_key(model_name: str) -> str:
    return "comet_" + re.sub(r"[^\w.\-]+", "_", model_name.split("/")[-1])


def is_comet_qe(model_name: str) -> bool:
    n = model_name.lower()
    return "kiwi" in n or "qe" in n


def compute_comet(srcs, hyps, refs, model_name: str = "Unbabel/wmt22-comet-da", batch_size: int = 64,
                  gpus: int | None = None):
    """COMET có tham chiếu (wmt22-comet-da) hoặc QE không tham chiếu (tên chứa 'kiwi'/'qe'). -> (điểm hệ thống, điểm từng câu)"""
    comet = _need("comet", "unbabel-comet")
    if gpus is None:
        try:
            import torch
            gpus = 1 if torch.cuda.is_available() else 0
        except ImportError:
            gpus = 0
    model = comet.load_from_checkpoint(comet.download_model(model_name))
    qe = is_comet_qe(model_name)
    data = [{"src": s, "mt": h} if qe else {"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return float(out.system_score), [round(float(x), 5) for x in out.scores]


def compute_mt_metrics(srcs, hyps, refs, tgt_lang: str = "vi",
                       metrics: Sequence[str] = ("bleu", "chrf", "chrfpp", "comet"),
                       comet_models: Sequence[str] = ("Unbabel/wmt22-comet-da",), comet_batch_size: int = 64):
    """Trả về (summary, per_seg, errors). Lỗi của một metric (vd thiếu thư viện) được ghi vào `errors`,
    các metric còn lại vẫn chạy -> không mất kết quả suy luận đã tốn thời gian."""
    srcs = [normalize_mt_text(x) for x in srcs]
    hyps = [normalize_mt_text(x) for x in hyps]
    refs = [normalize_mt_text(x) for x in refs]
    summary: dict = {"n": len(hyps), "n_empty_hyp": sum(not h for h in hyps)}
    per_seg: dict[str, list] = {}
    errors: dict[str, str] = {}

    def attempt(name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"
            log.error("Metric '%s' lỗi: %s", name, errors[name])

    if "bleu" in metrics:
        attempt("bleu", lambda: summary.update(compute_bleu(hyps, refs, tgt_lang)))
    for name, wo in (("chrf", 0), ("chrfpp", 2)):
        if name in metrics:
            def _chrf(name=name, wo=wo):
                summary.update(compute_chrf(hyps, refs, wo))
                per_seg[f"sent_{name}"] = sentence_chrf(hyps, refs, wo)
            attempt(name, _chrf)
    if "comet" in metrics:
        for cm in comet_models:
            def _comet(cm=cm):
                sys_score, seg = compute_comet(srcs, hyps, refs, cm, comet_batch_size)
                summary[comet_key(cm)] = round(sys_score, 5)
                per_seg[comet_key(cm)] = seg
            attempt(comet_key(cm), _comet)
    return summary, per_seg, errors
