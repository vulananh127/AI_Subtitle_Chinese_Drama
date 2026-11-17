"""Metric end-to-end KHÔNG phụ thuộc cách phân đoạn: hệ thống và tham chiếu có thể chia cue khác nhau hoàn toàn.

Ghép cue giả thuyết (hyp) với cue tham chiếu (ref) theo thời gian: hai cue nối với nhau nếu chồng lấn đủ
(overlap >= min_overlap_ratio * cue ngắn hơn); mỗi thành phần liên thông (một nhóm ref + một nhóm hyp) là một đơn vị chấm:
văn bản của nhóm ref và nhóm hyp được nối theo thời gian rồi so sánh.
  - nhóm chỉ có ref  -> "bỏ sót" (hyp rỗng)          - nhóm chỉ có hyp -> "thừa" (chèn)
Dùng cho cả phía nguồn (CER, so với SRT zh) lẫn phía đích (BLEU/chrF/COMET, so với SRT vi).
Công thức CER/BLEU/chrF/COMET dùng lại metrics.py nên cùng chuẩn với đánh giá ASR/MT riêng lẻ.
"""

from __future__ import annotations

from statistics import median

from .metrics import compute_chrf, compute_mt_metrics, edit_counts, normalize_asr_text

MIN_OVERLAP_RATIO = 0.3


# ---------------------------------------------------------------- ghép cue theo thời gian
def pair_components(ref: list[dict], hyp: list[dict], min_overlap_ratio: float = MIN_OVERLAP_RATIO):
    """-> list[(ref_idx[], hyp_idx[])], mỗi cue thuộc đúng một thành phần; sắp theo thời gian bắt đầu."""
    parent = list(range(len(ref) + len(hyp)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order_h = sorted(range(len(hyp)), key=lambda j: hyp[j]["start"])
    pmax, m = [], float("-inf")          # pmax[k] = end lớn nhất trong các hyp order_h[0..k]
    for j in order_h:
        m = max(m, hyp[j]["end"])
        pmax.append(m)
    lo = 0
    for i, r in sorted(enumerate(ref), key=lambda t: t[1]["start"]):
        while lo < len(order_h) and pmax[lo] <= r["start"]:   # mọi hyp tới lo đã kết thúc trước ref này (và các ref sau)
            lo += 1
        for j in order_h[lo:]:
            h = hyp[j]
            if h["start"] >= r["end"]:
                break
            ov = min(r["end"], h["end"]) - max(r["start"], h["start"])
            if ov > 0 and ov >= min_overlap_ratio * min(r["end"] - r["start"], h["end"] - h["start"]):
                parent[find(i)] = find(len(ref) + j)
    groups: dict[int, tuple[list, list]] = {}
    for i in range(len(ref)):
        groups.setdefault(find(i), ([], []))[0].append(i)
    for j in range(len(hyp)):
        groups.setdefault(find(len(ref) + j), ([], []))[1].append(j)
    comps = []
    for ri, hj in groups.values():
        key = min([ref[i]["start"] for i in ri] + [hyp[j]["start"] for j in hj])
        comps.append((key, sorted(ri, key=lambda i: ref[i]["start"]), sorted(hj, key=lambda j: hyp[j]["start"])))
    return [(ri, hj) for _, ri, hj in sorted(comps, key=lambda c: c[0])]


def _join(cues, idx, sep):
    return sep.join(cues[i]["text"].strip() for i in idx)


# ---------------------------------------------------------------- phía nguồn: CER theo thời gian
def asr_side_metrics(videos: list[dict], unit: str = "char", min_overlap_ratio: float = MIN_OVERLAP_RATIO):
    """videos: [{"ref": cues, "hyp": cues}] -> CER/WER gộp trên mọi nhóm + tỉ lệ bỏ sót/thừa. Trả (summary, rows)."""
    sep = "" if unit == "char" else " "
    tot = {"ref": 0, "s": 0, "d": 0, "i": 0, "miss": 0, "spur": 0}
    rows = []
    for vi, v in enumerate(videos):
        for ri, hj in pair_components(v["ref"], v["hyp"], min_overlap_ratio):
            rt = _join(v["ref"], ri, sep)
            ht = _join(v["hyp"], hj, sep)
            rn, hn = (normalize_asr_text(x, strip_space=(unit == "char")) for x in (rt, ht))
            rtok, htok = (list(rn), list(hn)) if unit == "char" else (rn.split(), hn.split())
            s, d, i = edit_counts(rtok, htok)
            tot["ref"] += len(rtok); tot["s"] += s; tot["d"] += d; tot["i"] += i
            if not htok:
                tot["miss"] += len(rtok)
            if not rtok:
                tot["spur"] += len(htok)
            rows.append({"video": vi, "start": round(min([v["ref"][k]["start"] for k in ri] or
                                                         [v["hyp"][k]["start"] for k in hj]), 3),
                         "ref": rt, "hyp": ht, "errors": s + d + i, "ref_len": len(rtok)})
    n = tot["ref"]
    key = "cer" if unit == "char" else "wer"
    pct = lambda x: round(100 * x / n, 4) if n else None  # noqa: E731
    return {key: pct(tot["s"] + tot["d"] + tot["i"]), f"{key}_sub": pct(tot["s"]), f"{key}_del": pct(tot["d"]),
            f"{key}_ins": pct(tot["i"]), "miss_pct": pct(tot["miss"]), "spurious_pct": pct(tot["spur"]),
            "ref_len_total": n}, rows


# ---------------------------------------------------------------- thời gian
def _union(intervals):
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _inter_len(a, b):
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        tot += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def time_metrics(videos: list[dict], min_overlap_ratio: float = MIN_OVERLAP_RATIO):
    """Độ khớp thời gian: P/R/F1 theo thời lượng có thoại; sai lệch mốc đầu/cuối của các nhóm có cả ref và hyp."""
    inter = h_len = r_len = 0.0
    se, ee = [], []
    for v in videos:
        ru = _union([(c["start"], c["end"]) for c in v["ref"]])
        hu = _union([(c["start"], c["end"]) for c in v["hyp"]])
        inter += _inter_len(ru, hu)
        h_len += sum(e - s for s, e in hu)
        r_len += sum(e - s for s, e in ru)
        for ri, hj in pair_components(v["ref"], v["hyp"], min_overlap_ratio):
            if ri and hj:
                se.append(abs(min(v["hyp"][j]["start"] for j in hj) - min(v["ref"][i]["start"] for i in ri)))
                ee.append(abs(max(v["hyp"][j]["end"] for j in hj) - max(v["ref"][i]["end"] for i in ri)))
    p, r = (inter / h_len if h_len else None), (inter / r_len if r_len else None)
    f1 = 2 * p * r / (p + r) if p and r else None
    both = se + ee
    r4 = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {"speech_precision": r4(p and 100 * p), "speech_recall": r4(r and 100 * r), "speech_f1": r4(f1 and 100 * f1),
            "start_err_median": r4(median(se)) if se else None, "end_err_median": r4(median(ee)) if ee else None,
            "boundary_within_0.5s": r4(100 * sum(x <= 0.5 for x in both) / len(both)) if both else None,
            "n_matched_groups": len(se)}


# ---------------------------------------------------------------- phía đích: dịch
def mt_side_metrics(videos: list[dict], tgt_lang: str = "vi", metrics=("bleu", "chrf", "chrfpp"),
                    comet_models=("Unbabel/wmt22-comet-da",), comet_batch_size: int = 64,
                    min_overlap_ratio: float = MIN_OVERLAP_RATIO):
    """videos: [{"ref": cues tiếng Việt, "hyp": cues phụ đề, "src": cues tiếng gốc (tùy chọn, cho COMET)}].
    Điểm trên các nhóm có cả ref và hyp (nhóm bỏ sót tính hyp rỗng; nhóm thừa không có ref nên chỉ báo tỉ lệ).
    Thêm doc_chrf/doc_bleu: nối toàn bộ phụ đề mỗi video thành một văn bản."""
    srcs, hyps, refs, rows = [], [], [], []
    n_spur = n_miss = 0
    for vi, v in enumerate(videos):
        for ri, hj in pair_components(v["ref"], v["hyp"], min_overlap_ratio):
            if not ri:
                n_spur += 1
                continue
            rt, ht = _join(v["ref"], ri, " "), _join(v["hyp"], hj, " ")
            n_miss += not hj
            st = ""
            if v.get("src"):   # nguồn của nhóm: các cue nguồn chồng lấn với nhóm ref
                lo, hi = min(v["ref"][i]["start"] for i in ri), max(v["ref"][i]["end"] for i in ri)
                st = " ".join(c["text"] for c in v["src"] if c["end"] > lo and c["start"] < hi)
            srcs.append(st); hyps.append(ht); refs.append(rt)
            rows.append({"video": vi, "start": round(v["ref"][ri[0]]["start"], 3), "src": st, "ref": rt, "hyp": ht})
    summary, per_seg, errors = compute_mt_metrics(srcs, hyps, refs, tgt_lang, metrics, comet_models, comet_batch_size)
    summary.update(n_groups=len(refs), n_spurious_groups=n_spur, n_missed_groups=n_miss)
    for k, vals in per_seg.items():
        for r, x in zip(rows, vals):
            r[k] = x
    docs = [(_join(v["hyp"], range(len(v["hyp"])), " "), _join(v["ref"], range(len(v["ref"])), " ")) for v in videos]
    try:
        if "chrf" in metrics or "chrfpp" in metrics:
            summary["doc_chrf"] = compute_chrf([h for h, _ in docs], [r for _, r in docs], 0)["chrf"]
    except Exception as e:  # noqa: BLE001
        errors["doc_chrf"] = f"{type(e).__name__}: {e}"
    return summary, rows, errors
