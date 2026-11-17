"""Gom metrics.json của mọi model (pretrained + fine-tuned) thành một bảng để chọn model.

Cảnh báo '!' khi các dòng được chấm trên tập mẫu khác nhau (fingerprint khác) -> điểm không so sánh được.
"""

from __future__ import annotations

import json
from collections import Counter

from .paths import STAGES, drama_eval_dir, write_csv_rows

_ASR_COLS = ["cer", "cer_macro", "cer_sub", "cer_del", "cer_ins", "wer", "n_empty_hyp", "n_over_100pct"]
_MT_COLS = ["bleu", "chrf", "chrfpp"]


def collect(language: str, task: str) -> list[dict]:
    rows = []
    for stage in STAGES:
        base = drama_eval_dir(language, task, stage)
        for mj in sorted(base.glob("*/metrics.json")) if base.exists() else []:
            r = json.loads(mj.read_text(encoding="utf-8"))
            m, t = r["metrics"], r.get("timing", {})
            row = {"stage": stage, "model": r["model_name"], "n": m.get("n")}
            row.update({k: v for k, v in m.items() if k not in ("n",) and not k.endswith("_signature")
                        and isinstance(v, (int, float))})
            row.update({"rtf": t.get("rtf"), "sent_per_sec": t.get("sent_per_sec"),
                        "fingerprint": r["dataset"].get("fingerprint")})
            rows.append(row)
    return rows


def _sort_key(task, row):
    if task == "asr":
        return (row.get("cer") is None, row.get("cer") or row.get("wer") or 0)
    comet = [v for k, v in row.items() if k.startswith("comet_") and isinstance(v, (int, float))]
    score = comet[0] * 100 if comet else (row.get("chrfpp") or row.get("chrf") or 0)
    return (False, -score)


def show(language: str, task: str):
    rows = collect(language, task)
    if not rows:
        print(f"Chưa có kết quả ở {drama_eval_dir(language, task, 'pretrained').parent}")
        return
    rows.sort(key=lambda r: _sort_key(task, r))
    cols = _ASR_COLS if task == "asr" else _MT_COLS + sorted({k for r in rows for k in r if k.startswith("comet_")})
    cols = [c for c in cols if any(r.get(c) is not None for r in rows)]
    cols += ["rtf" if task == "asr" else "sent_per_sec"]
    header = ["stage", "model", "n"] + cols

    fp = Counter(r["fingerprint"] for r in rows)
    main_fp = fp.most_common(1)[0][0]
    table = [header] + [[("!" if r["fingerprint"] != main_fp else "") + str(r["model"]) if h == "model"
                         else ("-" if r.get(h) is None else (f"{r[h]:.2f}" if isinstance(r[h], float) else str(r[h])))
                         for h in header] for r in rows]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(header))]
    for k, row in enumerate(table):
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
        if k == 0:
            print("  ".join("-" * w for w in widths))
    if len(fp) > 1:
        print("\n! = chấm trên tập mẫu KHÁC với đa số các dòng (khác --split/--limit/dữ liệu) -> không so sánh trực tiếp được.")
    out = drama_eval_dir(language, task, "pretrained").parent / "summary.csv"
    allcols = list(dict.fromkeys(k for r in rows for k in r))
    write_csv_rows(out, allcols, [{k: r.get(k, "") for k in allcols} for r in rows])
    print(f"\nĐã lưu {out}")
