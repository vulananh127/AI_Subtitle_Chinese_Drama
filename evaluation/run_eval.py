"""CLI đánh giá ASR / MT. Chạy từ thư mục gốc project (nơi có pipeline/, evaluation/, dataset/):

  # ASR: đổi model chỉ cần đổi --model
  python -m evaluation.run_eval asr --language chinese --model qwen3-asr-1.7b --split test
  python -m evaluation.run_eval asr --language chinese --model paraformer-zh  --split test

  # MT
  python -m evaluation.run_eval mt  --language chinese --model hy-mt2-7b --split test --metrics bleu,chrf,chrfpp,comet

  # Chạy tiếp khi bị ngắt / chấm điểm lại từ kết quả đã có (vd sau khi cài thêm COMET)
  python -m evaluation.run_eval mt --language chinese --model hy-mt2-7b --split test --resume
  python -m evaluation.run_eval mt --language chinese --model hy-mt2-7b --split test --score-only

  # Model fine-tune (cùng luồng, lưu vào .../fine-tuned/<name>/)
  python -m evaluation.run_eval asr --language chinese --model qwen3-asr-1.7b --stage fine-tuned \\
        --model-path /ckpt/qwen3asr_ft_v1 --name qwen3-asr-1.7b_ft_v1 --split test

  # Model chưa có trong registry
  python -m evaluation.run_eval mt --language chinese --model my-mt --backend hf_causal --model-path /path --set prompt_mode=raw ...

  python -m evaluation.run_eval compare --language chinese --task mt     # bảng so sánh các model
  python -m evaluation.run_eval list                                     # model/backend có sẵn
"""

from __future__ import annotations

import argparse
import logging

from . import common, compare, registry, runner
from .paths import DRAMA_LANGUAGES, STAGES


def _add_common(p):
    p.add_argument("--language", default="chinese", choices=sorted(DRAMA_LANGUAGES))
    p.add_argument("--stage", default="pretrained", choices=STAGES)
    p.add_argument("--model", help="tên trong registry.py (hoặc tên tự đặt nếu kèm --backend/--model-path)")
    p.add_argument("--backend", help="ghi đè backend")
    p.add_argument("--model-path", help="ghi đè HF id / thư mục checkpoint")
    p.add_argument("--name", help="tên thư mục kết quả (mặc định = --model)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="ghi đè tham số backend (lặp lại được)")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--chunk-size", type=int, default=128, help="số mẫu mỗi lần gọi backend; kết quả được lưu sau mỗi chunk")
    p.add_argument("--input", help="CSV khác thay cho metadata_{task}.csv")
    p.add_argument("--split-col", default="split")
    p.add_argument("--split", help="chỉ lấy dòng có split-col == giá trị này (vd test). Bỏ trống = dùng toàn bộ CSV")
    p.add_argument("--limit", type=int, help="chỉ lấy N dòng đầu (chạy thử)")
    p.add_argument("--id-col")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true", help="chạy tiếp, bỏ qua các mẫu đã có kết quả")
    g.add_argument("--overwrite", action="store_true", help="xóa kết quả cũ của model này rồi chạy lại")
    g.add_argument("--score-only", action="store_true", help="không suy luận, chỉ chấm lại từ predictions.jsonl")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("asr", help="đánh giá ASR (CER/WER)")
    _add_common(p)
    p.add_argument("--audio-col"); p.add_argument("--text-col")
    p.add_argument("--unit", choices=["char", "word"], default="char")
    p.add_argument("--allow-missing-audio", action="store_true")

    p = sub.add_parser("mt", help="đánh giá MT (BLEU/chrF/COMET)")
    _add_common(p)
    p.add_argument("--src-col"); p.add_argument("--ref-col")
    p.add_argument("--tgt-lang", default="vi")
    p.add_argument("--metrics", default="bleu,chrf,chrfpp,comet", help="danh sách con của bleu,chrf,chrfpp,comet")
    p.add_argument("--comet-models", default="Unbabel/wmt22-comet-da",
                   help="phân tách bằng dấu phẩy; tên chứa 'kiwi'/'qe' sẽ chạy kiểu không tham chiếu")
    p.add_argument("--comet-batch-size", type=int, default=64)

    p = sub.add_parser("compare", help="bảng so sánh các model đã chạy")
    p.add_argument("--language", default="chinese", choices=sorted(DRAMA_LANGUAGES))
    p.add_argument("--task", required=True, choices=["asr", "mt"])

    sub.add_parser("list", help="liệt kê model và backend có sẵn")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if a.cmd in ("asr", "mt"):
        runner.run_task(a.cmd, a)
    elif a.cmd == "compare":
        compare.show(a.language, a.task)
    else:
        from . import backends_asr, backends_mt  # noqa: F401
        for task, table, bes in (("asr", registry.ASR_MODELS, common.ASR_BACKENDS),
                                 ("mt", registry.MT_MODELS, common.MT_BACKENDS)):
            print(f"[{task}] backends: {sorted(bes)}")
            for n, s in table.items():
                print(f"   {n:34s} {s['backend']:12s} {s['model_path']}")


if __name__ == "__main__":
    main()
