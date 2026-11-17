"""Đánh giá end-to-end: video/audio -> VAD -> ASR -> timestamp aligner -> MT -> phân đoạn phụ đề.

Chạy ĐÚNG pipeline của server (server.pipeline.SubtitlePipeline) nên điểm đo được là điểm của hệ thống thật.
Đổi model = đổi tham số dòng lệnh (--asr-model, --mt-model, --vad, --aligner ...) hoặc file --config của server.

Luồng chạy (để mỗi model chỉ nạp một lần dù có nhiều video):
  lượt A: mọi video chạy chế độ "asr" (audio -> VAD -> ASR -> aligner)
  lượt B: mọi video chạy chế độ "full" (stage ASR đã có artifact nên bỏ qua; chạy MT + dựng phụ đề)
  lượt C (--oracle-mt): dịch thẳng SRT tham chiếu tiếng gốc -> tách riêng chất lượng MT khỏi lỗi ASR
Kết quả: eval/{language}/e2e/{stage}/{run_name}/  (metrics.json, videos/*.json, srt/*.srt, groups_*.csv, work/)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from . import e2e_data, e2e_metrics
from .paths import DRAMA_LANGUAGES, SERVER_PARENT, drama_eval_dir, safe_name, write_csv_rows
from .registry import parse_overrides
from .runner import _versions

log = logging.getLogger(__name__)


def _server():
    """Nạp gói server/ (nằm trong subtitle-studio/ hoặc ngang hàng evaluation/)."""
    if SERVER_PARENT is None:
        raise SystemExit("Không thấy gói server/ (subtitle-studio/server hoặc ../server). Đặt biến môi trường "
                         "SUBTITLE_STUDIO_DIR trỏ tới thư mục chứa server/.")
    if str(SERVER_PARENT) not in sys.path:
        sys.path.insert(0, str(SERVER_PARENT))
    from server.models import ModelManager
    from server.pipeline import JobContext, SubtitlePipeline
    from server.settings import load_settings
    from server.stages import subtitle
    return ModelManager, JobContext, SubtitlePipeline, load_settings, subtitle


def build_overrides(a, out_dir: Path) -> dict:
    """Cờ dòng lệnh -> phần ghi đè settings của server (chỉ khóa nào được truyền)."""
    ov: dict = {"storage_dir": str(out_dir / "work"), "swap_models": bool(a.swap_models)}
    for kind, pre in (("asr", "asr"), ("mt", "mt")):
        d = {}
        for k, v in (("model", getattr(a, f"{pre}_model")), ("backend", getattr(a, f"{pre}_backend")),
                     ("model_path", getattr(a, f"{pre}_model_path"))):
            if v:
                d[k] = v
        sets = parse_overrides(getattr(a, f"{pre}_set"))
        if sets:
            d["set"] = sets
        if d:
            ov[kind] = d
    for kind, flag in (("vad", a.vad), ("aligner", a.aligner)):
        d = {}
        if flag:
            d["backend"] = flag
        params = parse_overrides(getattr(a, f"{kind}_set"))
        if params:
            d["params"] = params
        if d:
            ov[kind] = d
    for item in a.opt or []:                       # --opt subtitle.max_cps=15  /  chunking.max_duration=12
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise SystemExit(f"--opt cần dạng nhóm.khóa=giá_trị (vd subtitle.max_cps=15), nhận '{item}'")
        key, val = item.split("=", 1)
        sec, k = key.split(".", 1)
        if sec not in ("subtitle", "chunking"):
            raise SystemExit(f"--opt chỉ hỗ trợ nhóm subtitle/chunking, nhận '{sec}'")
        ov.setdefault(sec, {})[k] = parse_overrides([f"v={val}"])["v"]
    return ov


def default_run_name(s: dict) -> str:
    """vad+asr+aligner+mt; model có --*-model-path thì thêm @tên_thư_mục để run fine-tune không trùng tên."""
    def tag(c):
        return c["model"] + (f"@{Path(c['model_path']).name}" if c.get("model_path") else "")
    return safe_name("+".join([s["vad"]["backend"], tag(s["asr"]), s["aligner"]["backend"] or "none", tag(s["mt"])]))


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_timings(*ts) -> dict:
    out: dict = {}
    for t in ts:
        for k, v in (t or {}).items():
            if isinstance(v, (int, float)) and k != "total":
                out[k] = round(out.get(k, 0.0) + v, 2)
            else:
                out.setdefault(k, v)
    return out


# ---------------------------------------------------------------- suy luận
def _run_pipeline(pipe, JobContext, work: Path, mode: str, params: dict):
    ctx = JobContext(work, mode, params)
    result = pipe.run(ctx)
    return ctx, result


def _link_media(src: Path, work: Path) -> str:
    (work / "input").mkdir(parents=True, exist_ok=True)
    name = "media" + src.suffix.lower()
    dst = work / "input" / name
    if not dst.exists():
        try:
            os.symlink(Path(src).resolve(), dst)
        except OSError:
            shutil.copy(src, dst)
    return name


def infer(items, a, settings, out_dir: Path, src_lang: str, tgt_lang: str):
    ModelManager, JobContext, SubtitlePipeline, _, _ = _server()
    models = ModelManager(settings)
    pipe = SubtitlePipeline(settings, models)
    opts = {"keep_intermediate": True}
    params = {"language": a.language, "tgt_lang": tgt_lang, "options": opts}
    rec: dict[str, dict] = {}
    todo = []
    for it in items:
        f = out_dir / "videos" / f"{safe_name(it['id'])}.json"
        if f.exists() and not a.overwrite:
            rec[it["id"]] = _load(f)
        else:
            todo.append(it)
    log.info("Cần chạy %d/%d video (đã có %d)", len(todo), len(items), len(items) - len(todo))
    if not todo:
        models.release_all()
        return rec
    load_sec = {}

    def warm(*kinds):
        for k in kinds:
            t0 = time.perf_counter()
            models.get(k)
            load_sec[k] = round(load_sec.get(k, 0.0) + time.perf_counter() - t0, 2)

    state = {it["id"]: {"id": it["id"], "status": "pending", "error": None,
                        "work": out_dir / "work" / safe_name(it["id"])} for it in todo}
    try:
        # ---- lượt A: audio -> VAD -> ASR -> aligner
        warm("vad", "asr", "aligner")
        for n, it in enumerate(todo, 1):
            st = state[it["id"]]
            log.info("[A %d/%d] %s", n, len(todo), it["id"])
            try:
                p = {**params, "input_name": _link_media(Path(it["media"]), st["work"])}
                st["ctx_a"], _ = _run_pipeline(pipe, JobContext, st["work"], "asr", p)
                st["params"] = p
            except Exception as e:  # noqa: BLE001
                log.exception("Video %s lỗi ở lượt ASR", it["id"])
                st.update(status="failed", error=f"{type(e).__name__}: {e}")
        # ---- lượt B: dịch + dựng phụ đề
        warm("mt")
        for n, it in enumerate(todo, 1):
            st = state[it["id"]]
            if st["status"] == "failed":
                continue
            log.info("[B %d/%d] %s", n, len(todo), it["id"])
            try:
                ctx_b, result = _run_pipeline(pipe, JobContext, st["work"], "full", st["params"])
                st.update(status="done", result=result, ctx_b=ctx_b)
            except Exception as e:  # noqa: BLE001
                log.exception("Video %s lỗi ở lượt MT", it["id"])
                st.update(status="failed", error=f"{type(e).__name__}: {e}")
        # ---- lượt C: MT trên transcript tham chiếu
        if a.oracle_mt:
            for n, it in enumerate(todo, 1):
                st = state[it["id"]]
                if not it.get("ref_src") or st["status"] == "failed":
                    continue
                log.info("[C %d/%d] oracle MT %s", n, len(todo), it["id"])
                w = st["work"].parent / (st["work"].name + "__oracle")
                w.mkdir(parents=True, exist_ok=True)
                _dump(w / "mt_input.json", {"segments": it["ref_src"]})
                try:
                    _, res = _run_pipeline(pipe, JobContext, w, "mt", {k: v for k, v in params.items()})
                    st["oracle_cues"] = res["cues"]
                except Exception as e:  # noqa: BLE001
                    log.exception("Oracle MT lỗi ở %s", it["id"])
                    st["oracle_error"] = f"{type(e).__name__}: {e}"
    finally:
        models.release_all()

    for it in todo:
        st = state[it["id"]]
        r = {"id": it["id"], "status": st["status"], "error": st["error"], "load_sec": load_sec}
        if st["status"] == "done":
            res, ca, cb = st["result"], st["ctx_a"], st["ctx_b"]
            vad = _load(st["work"] / "vad.json") if (st["work"] / "vad.json").exists() else []
            r.update(duration=ca.info.get("duration"), timings=_merge_timings(ca.timings, cb.timings),
                     info={**ca.info, **cb.info}, segments=res["segments"], cues=res["cues"] or [], vad_chunks=vad,
                     oracle_cues=st.get("oracle_cues"), oracle_error=st.get("oracle_error"))
            srt = out_dir / "srt"
            srt.mkdir(parents=True, exist_ok=True)
            for kind, lang in (("source.srt", src_lang), ("target.srt", tgt_lang)):
                if (st["work"] / kind).exists():
                    shutil.copy(st["work"] / kind, srt / f"{safe_name(it['id'])}.{lang}.srt")
            if not a.keep_audio:
                (st["work"] / "audio.wav").unlink(missing_ok=True)
                for d in ("chunks", "chunks_align"):
                    shutil.rmtree(st["work"] / d, ignore_errors=True)
        _dump(out_dir / "videos" / f"{safe_name(it['id'])}.json", r)
        rec[it["id"]] = r
    return rec


# ---------------------------------------------------------------- chấm điểm
def _cues(xs):
    return [{"start": float(x["start"]), "end": float(x["end"]), "text": x.get("text", "")} for x in xs]


def score(items, rec, a, settings):
    subtitle = _server()[4]
    ok = [it for it in items if rec.get(it["id"], {}).get("status") == "done"]
    failed = [it["id"] for it in items if it not in ok]
    summary: dict = {"n_videos": len(items), "n_failed": len(failed), "failed_ids": failed[:20]}
    errors: dict = {}
    if not ok:
        return summary, errors, {}
    dur = sum(rec[it["id"]].get("duration") or 0 for it in ok)
    ov = a.min_overlap

    # nguồn: ASR + aligner
    src_v = [{"ref": it["ref_src"], "hyp": _cues(rec[it["id"]]["segments"])} for it in ok if it["ref_src"]]
    groups: dict[str, list] = {}
    if src_v:
        m, rows = e2e_metrics.asr_side_metrics(src_v, "char", ov)
        summary["asr"] = {**m, **{f"time_{k}": v for k, v in e2e_metrics.time_metrics(src_v, ov).items()}}
        groups["asr"] = rows
        vad_v = [{"ref": it["ref_src"], "hyp": _cues(rec[it["id"]]["vad_chunks"])} for it in ok if it["ref_src"]]
        vt = e2e_metrics.time_metrics(vad_v, ov)
        summary["vad"] = {"speech_precision": vt["speech_precision"], "speech_recall": vt["speech_recall"],
                          "speech_f1": vt["speech_f1"], "n_chunks": sum(len(rec[it["id"]]["vad_chunks"]) for it in ok),
                          "mean_chunk_sec": round(sum(c["end"] - c["start"] for it in ok for c in rec[it["id"]]["vad_chunks"])
                                                  / max(1, sum(len(rec[it["id"]]["vad_chunks"]) for it in ok)), 2)}
    # đích: dịch + phụ đề
    tgt_v = [{"ref": it["ref_tgt"], "hyp": _cues(rec[it["id"]]["cues"]), "src": it["ref_src"]}
             for it in ok if it["ref_tgt"]]
    if tgt_v:
        metrics = tuple(m for m in a.metrics.split(",") if m)
        comets = [c for c in a.comet_models.split(",") if c]
        m, rows, err = e2e_metrics.mt_side_metrics(tgt_v, a.tgt_lang, metrics, comets, a.comet_batch_size, ov)
        summary["mt"] = {**m, **{f"time_{k}": v for k, v in e2e_metrics.time_metrics(tgt_v, ov).items()}}
        errors.update(err)
        groups["mt"] = rows
        if a.oracle_mt:
            orc = [{"ref": it["ref_tgt"], "hyp": _cues(rec[it["id"]]["oracle_cues"]), "src": it["ref_src"]}
                   for it in ok if it["ref_tgt"] and rec[it["id"]].get("oracle_cues")]
            if orc:
                m, rows, err = e2e_metrics.mt_side_metrics(orc, a.tgt_lang, metrics, comets, a.comet_batch_size, ov)
                summary["mt_oracle"] = m
                errors.update({f"oracle_{k}": v for k, v in err.items()})
    # định dạng phụ đề + tốc độ
    cfg = settings["subtitle"]
    st = [subtitle.stats(_cues(rec[it["id"]]["cues"]), cfg) for it in ok if rec[it["id"]]["cues"]]
    n_cues = sum(s["n_cues"] for s in st)
    summary["format"] = {"n_cues": n_cues,
                         "cues_per_min": round(60 * n_cues / dur, 2) if dur else None,
                         "pct_over_cps": round(100 * sum(s["n_over_cps"] for s in st) / n_cues, 2) if n_cues else None,
                         "pct_too_short": round(100 * sum(s["n_short"] for s in st) / n_cues, 2) if n_cues else None,
                         "max_cps_seen": max((s["max_cps"] for s in st), default=None)}
    stage_sec: dict = {}
    for it in ok:
        for k, v in (rec[it["id"]].get("timings") or {}).items():
            if isinstance(v, (int, float)) and k != "total":
                stage_sec[k] = stage_sec.get(k, 0.0) + v
    total = sum(stage_sec.values())
    summary["efficiency"] = {"audio_sec": round(dur, 1), "pipeline_sec": round(total, 1),
                             "rtf": round(total / dur, 4) if dur else None,
                             "stage_sec": {k: round(v, 1) for k, v in stage_sec.items()},
                             "model_load_sec": next((r.get("load_sec") for r in rec.values() if r.get("load_sec")), {})}
    summary["n_align_failed"] = sum(rec[it["id"]]["info"].get("align_failed") or 0 for it in ok)
    summary["n_flagged_segments"] = sum(rec[it["id"]]["info"].get("n_flagged") or 0 for it in ok)
    return summary, errors, groups


def run_e2e(a) -> Path:
    if a.language not in DRAMA_LANGUAGES:
        raise SystemExit(f"--language phải thuộc {sorted(DRAMA_LANGUAGES)}")
    src_lang, tgt_lang = DRAMA_LANGUAGES[a.language], a.tgt_lang
    _, _, _, load_settings, _ = _server()
    # thư mục kết quả phụ thuộc tên run -> cần settings cuối cùng để đặt tên mặc định
    scratch = Path(tempfile.mkdtemp(prefix="e2e_name_"))
    try:
        tmp = load_settings(a.config, build_overrides(a, scratch))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    name = a.name or default_run_name(tmp)
    if a.stage == "fine-tuned" and not a.name:
        log.warning("Kết quả fine-tuned nên đặt --name rõ ràng (vd ..._ft_v1) để khỏi trùng tên run pretrained.")
    out_dir = drama_eval_dir(a.language, "e2e", a.stage, name)
    if (out_dir / "metrics.json").exists() and not (a.resume or a.overwrite or a.score_only):
        raise SystemExit(f"Đã có kết quả ở {out_dir}. Dùng --resume, --score-only, --overwrite hoặc --name khác.")
    if a.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings(a.config, build_overrides(a, out_dir))

    items, dinfo = e2e_data.load_e2e_dataset(a.language, src_lang, tgt_lang, a.input, a.media_dir, a.media_col,
                                             a.src_col, a.tgt_col, a.id_col, a.split_col, a.split, a.limit)
    if not any(it["ref_src"] or it["ref_tgt"] for it in items):
        raise SystemExit("Không mẫu nào có SRT tham chiếu -> không có gì để chấm điểm.")
    if a.score_only:
        rec = {}
        for it in items:
            f = out_dir / "videos" / f"{safe_name(it['id'])}.json"
            if not f.exists():
                raise SystemExit(f"--score-only nhưng thiếu kết quả của video '{it['id']}' ({f})")
            rec[it["id"]] = _load(f)
    else:
        rec = infer(items, a, settings, out_dir, src_lang, tgt_lang)

    log.info("Chấm điểm ...")
    summary, errors, groups = score(items, rec, a, settings)
    for k, rows in groups.items():
        if rows:
            write_csv_rows(out_dir / f"groups_{k}.csv", list(rows[0]), rows)
    result = {
        "task": "e2e", "language": a.language, "stage": a.stage, "model_name": name,
        "pipeline": {"vad": settings["vad"], "aligner": settings["aligner"],
                     "asr": {k: settings["asr"].get(k) for k in ("model", "backend", "model_path", "set")},
                     "mt": {k: settings["mt"].get(k) for k in ("model", "backend", "model_path", "set")},
                     "chunking": settings["chunking"], "subtitle": settings["subtitle"]},
        "scoring": {"min_overlap_ratio": a.min_overlap, "metrics": a.metrics, "comet_models": a.comet_models,
                   "oracle_mt": bool(a.oracle_mt)},
        "src_lang": src_lang, "tgt_lang": tgt_lang, "dataset": dinfo, "metrics": summary, "metric_errors": errors,
        "created_at": datetime.now().isoformat(timespec="seconds"), "versions": _versions(),
        "host": platform.node(),
    }
    _dump(out_dir / "metrics.json", result)
    log.info("Kết quả lưu ở %s", out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for k, v in errors.items():
        log.warning("METRIC LỖI %s -> %s  (cài thư viện rồi chạy lại với --score-only)", k, v)
    return out_dir
