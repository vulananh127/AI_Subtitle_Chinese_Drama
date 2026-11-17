"""Luồng đánh giá chung cho ASR và MT: nạp dữ liệu -> suy luận (lưu dần, resume được) -> chấm điểm -> lưu kết quả.

Kết quả: eval/{language}/{task}/{stage}/{tên model}/{predictions.jsonl, predictions.csv, metrics.json}
"""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

from . import data, metrics
from .paths import DRAMA_LANGUAGES, drama_eval_dir, write_csv_rows
from .registry import create_backend, parse_overrides, resolve_model

log = logging.getLogger(__name__)

_PKGS = ("torch", "transformers", "sacrebleu", "unbabel-comet", "qwen-asr", "funasr", "vllm", "soundfile")


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for p in _PKGS:
        try:
            out[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def _read_jsonl(path: Path) -> dict:
    rows = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["id"]] = r
    return rows


def _append_jsonl(path: Path, rows: list[dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _progress(total: int):
    try:
        from tqdm import tqdm
        return tqdm(total=total, unit="mẫu")
    except ImportError:
        class _P:
            n = 0

            def update(self, k):
                self.n += k
                log.info("  %d/%d", self.n, total)

            def close(self):
                pass
        return _P()


def _infer(task, spec, todo, pred_path, src_lang, tgt_lang, chunk_size):
    log.info("Nạp model %s (%s) ...", spec["name"], spec["model_path"])
    backend = create_backend(task, spec["backend"], spec["model_path"], spec["cfg"])
    backend.load()
    bar = _progress(len(todo))
    try:
        for s in range(0, len(todo), chunk_size):
            batch = todo[s:s + chunk_size]
            t0 = time.perf_counter()
            if task == "asr":
                outs = backend.transcribe([b["audio_path"] for b in batch], src_lang)
            else:
                outs = backend.translate([b["src"] for b in batch], src_lang, tgt_lang)
            dt = time.perf_counter() - t0
            if len(outs) != len(batch):
                raise RuntimeError(f"Backend trả {len(outs)} kết quả cho {len(batch)} đầu vào")
            _append_jsonl(pred_path, [{"id": b["id"], "hyp": o, "sec": dt / len(batch)} for b, o in zip(batch, outs)])
            bar.update(len(batch))
    finally:
        bar.close()
        backend.close()


def _score_asr(items, preds, a):
    refs = [it["reference"] for it in items]
    hyps = [preds[it["id"]]["hyp"] for it in items]
    summary, per = metrics.compute_asr_metrics(refs, hyps, unit=a.unit)
    key = "cer" if a.unit == "char" else "wer"
    rows = []
    for it, p, pu in zip(items, hyps, per):
        rows.append({"id": it["id"], "audio_path": it["audio_path"], "duration": it["duration"],
                     "reference": it["reference"], "hypothesis": p, **pu})
    fields = ["id", "audio_path", "duration", "reference", "hypothesis", "ref_norm", "hyp_norm",
              "ref_len", "sub", "del", "ins", "errors", key]
    audio = [it["duration"] for it in items]
    audio_sec = sum(audio) if audio and all(x is not None for x in audio) else None
    return summary, {}, rows, fields, audio_sec


def _score_mt(items, preds, a, tgt_lang):
    srcs = [it["src"] for it in items]
    refs = [it["reference"] for it in items]
    hyps = [preds[it["id"]]["hyp"] for it in items]
    summary, per_seg, errors = metrics.compute_mt_metrics(
        srcs, hyps, refs, tgt_lang, metrics=[m.strip() for m in a.metrics.split(",") if m.strip()],
        comet_models=[m.strip() for m in a.comet_models.split(",") if m.strip()],
        comet_batch_size=a.comet_batch_size)
    rows = []
    for k, it in enumerate(items):
        r = {"id": it["id"], "source": it["src"], "reference": it["reference"], "hypothesis": hyps[k]}
        r.update({name: vals[k] for name, vals in per_seg.items()})
        rows.append(r)
    fields = ["id", "source", "reference", "hypothesis"] + list(per_seg)
    return summary, errors, rows, fields, None


def run_task(task: str, a) -> Path:
    """a: argparse.Namespace từ run_eval.py"""
    if a.language not in DRAMA_LANGUAGES:
        raise SystemExit(f"--language phải thuộc {sorted(DRAMA_LANGUAGES)}")
    src_lang, tgt_lang = DRAMA_LANGUAGES[a.language], getattr(a, "tgt_lang", "vi")
    spec = resolve_model(task, a.model, a.backend, a.model_path, parse_overrides(a.set, a.batch_size))
    name = a.name or spec["name"]
    out_dir = drama_eval_dir(a.language, task, a.stage, name)
    pred_path = out_dir / "predictions.jsonl"

    if (pred_path.exists() or (out_dir / "metrics.json").exists()) and not (a.resume or a.overwrite or a.score_only):
        raise SystemExit(f"Đã có kết quả ở {out_dir}. Dùng --resume (chạy tiếp), --score-only (chấm lại) "
                         f"hoặc --overwrite (chạy lại từ đầu), hoặc đặt --name khác.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.overwrite:
        for f in ("predictions.jsonl", "predictions.csv", "metrics.json"):
            (out_dir / f).unlink(missing_ok=True)

    if task == "asr":
        items, dinfo = data.load_asr_dataset(a.language, a.input, a.audio_col, a.text_col, a.id_col,
                                             a.split_col, a.split, a.limit, a.allow_missing_audio)
    else:
        items, dinfo = data.load_mt_dataset(a.language, a.input, a.src_col, a.ref_col, a.id_col,
                                            a.split_col, a.split, a.limit)
    if not items:
        raise SystemExit("Không có mẫu nào để đánh giá (kiểm tra --split / cột dữ liệu).")

    preds = _read_jsonl(pred_path)
    todo = [it for it in items if it["id"] not in preds]
    if a.score_only:
        if todo:
            raise SystemExit(f"--score-only nhưng còn {len(todo)}/{len(items)} mẫu chưa có kết quả suy luận")
    elif todo:
        log.info("Cần suy luận %d/%d mẫu (đã có %d)", len(todo), len(items), len(items) - len(todo))
        _infer(task, spec, todo, pred_path, src_lang, tgt_lang, a.chunk_size)
        preds = _read_jsonl(pred_path)
    else:
        log.info("Đã đủ kết quả suy luận, chỉ chấm điểm lại.")

    log.info("Chấm điểm ...")
    summary, errors, rows, fields, audio_sec = (_score_asr(items, preds, a) if task == "asr"
                                                 else _score_mt(items, preds, a, tgt_lang))
    write_csv_rows(out_dir / "predictions.csv", fields, rows)

    infer_sec = sum(preds[it["id"]].get("sec", 0.0) for it in items)
    timing = {"inference_sec": round(infer_sec, 2)}
    if task == "asr":
        timing.update(audio_sec=None if audio_sec is None else round(audio_sec, 2),
                      rtf=round(infer_sec / audio_sec, 4) if audio_sec else None)
    else:
        timing["sent_per_sec"] = round(len(items) / infer_sec, 2) if infer_sec else None

    result = {
        "task": task, "language": a.language, "stage": a.stage, "model_name": name,
        "backend": spec["backend"], "model_path": spec["model_path"], "backend_cfg": spec["cfg"],
        "src_lang": src_lang, "tgt_lang": tgt_lang if task == "mt" else None,
        "dataset": dinfo, "metrics": summary, "metric_errors": errors, "timing": timing,
        "created_at": datetime.now().isoformat(timespec="seconds"), "versions": _versions(),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log.info("Kết quả lưu ở %s", out_dir)
    for k, v in summary.items():
        if not k.endswith("_signature"):
            log.info("  %-22s %s", k, v)
    for k, v in errors.items():
        log.warning("  METRIC LỖI %s -> %s  (cài thư viện rồi chạy lại với --score-only)", k, v)
    return out_dir
