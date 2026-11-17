"""Kiểm tra nhanh luồng đánh giá end-to-end bằng video tổng hợp + backend giả (không cần GPU/model, cần ffmpeg):
    python -m evaluation.e2e_smoke_test
Kết quả ghi vào thư mục tạm (không đụng dataset/ thật)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import paths, run_eval


def main():
    from .paths import SERVER_PARENT
    import sys
    sys.path.insert(0, str(SERVER_PARENT))
    from server import smoke_test as st   # đăng ký test_asr / test_mt / test_uniform

    tmp = Path(tempfile.mkdtemp(prefix="e2e_smoke_"))
    paths.DRAMA_EVAL_DIR = tmp / "eval"
    media = tmp / "media"
    media.mkdir()
    st.make_video(media / "ep1.mp4")
    (media / "ep1.wav").unlink()          # make_video để lại file wav trung gian
    (media / "ep1.zh.srt").write_text(
        f"1\n00:00:00,500 --> 00:00:05,000\n{st.ZH[0]}\n\n2\n00:00:06,000 --> 00:00:09,500\n{st.ZH[1]}\n\n"
        f"3\n00:00:10,000 --> 00:00:11,500\n{st.ZH[2]}\n", encoding="utf-8")
    (media / "ep1.vi.srt").write_text(
        "1\n00:00:00,500 --> 00:00:05,000\nXin chào, hôm nay thời tiết đẹp.\n\n"
        "2\n00:00:06,000 --> 00:00:09,500\nĐây là một câu rất dài, dùng để kiểm tra việc chia phụ đề.\n\n"
        "3\n00:00:10,000 --> 00:00:11,500\nĐược rồi.\n", encoding="utf-8")
    (media / "bad.mp4").write_text("không phải video")
    (media / "bad.zh.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")

    base = ["e2e", "--language", "chinese", "--media-dir", str(media), "--asr-model", "t", "--asr-backend", "test_asr",
            "--asr-model-path", "-", "--mt-model", "t", "--mt-backend", "test_mt", "--mt-model-path", "-",
            "--aligner", "test_uniform", "--oracle-mt", "--metrics", "bleu,chrf,chrfpp"]
    run_eval.main(base + ["--vad", "energy", "--name", "smoke_energy"])
    run_eval.main(base + ["--vad", "none", "--name", "smoke_novad"])
    run_eval.main(["compare", "--language", "chinese", "--task", "e2e"])

    d = paths.DRAMA_EVAL_DIR / "chinese" / "e2e" / "pretrained" / "smoke_energy"
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    ok = lambda c, msg: print(("  OK   " if c else "  FAIL ") + msg) or c or (_ for _ in ()).throw(SystemExit(1))  # noqa: E731
    ok(m["n_failed"] == 1 and m["failed_ids"] == ["bad"], f"video hỏng được ghi nhận, không làm hỏng cả lượt: {m['failed_ids']}")
    ok(m["asr"]["cer"] == 0.0 and m["asr"]["miss_pct"] == 0.0, f"CER 0 khi ASR giả trả đúng transcript ({m['asr']['cer']})")
    ok(m["vad"]["speech_recall"] > 95, f"VAD phủ lời thoại ({m['vad']['speech_recall']}%)")
    ok(m["mt"]["n_groups"] == 3 and m["mt"]["n_missed_groups"] == 0, f"ghép nhóm cue ref/hyp đúng ({m['mt']['n_groups']} nhóm)")
    ok(m["mt_oracle"]["n_groups"] == 3, "có nhánh MT oracle (dịch từ transcript tham chiếu)")
    if "chrfpp" not in m["mt"]:
        print("  (bỏ qua điểm BLEU/chrF: chưa cài sacrebleu)")
    ok((d / "srt" / "ep1.vi.srt").exists() and (d / "groups_mt.csv").exists(), "có SRT đầu ra và bảng từng nhóm")
    print(f"\nTất cả đạt. Thư mục tạm: {tmp}")


if __name__ == "__main__":
    main()
