"""
Thu thập dữ liệu DRAMA.

Đọc drama_source.csv (drama_id, drama_name, episode, url, status), với mỗi dòng pending/failed:
  - Chỉ chấp nhận phụ đề THỦ CÔNG (manual) - không bao giờ dùng auto-generated hay auto-translate.
  - Bắt buộc phải có ĐỦ CẢ 2: zh (thủ công) và vi (thủ công). Thiếu 1 trong 2 -> failed, bỏ qua, ghi rõ lý do.
  - Nếu đủ -> tải audio + 2 file srt, lưu vào raw/Dxxxx/ (phẳng, không tách thư mục theo từng tập):
        D{drama_id:04d}_EP{episode:03d}.mp3
        D{drama_id:04d}_EP{episode:03d}_zh_raw.srt
        D{drama_id:04d}_EP{episode:03d}_vi_raw.srt
  - Cập nhật status ngay sau mỗi dòng (ghi đè csv) để resumable nếu crash giữa chừng.

Yêu cầu: pip install yt-dlp   (và ffmpeg trong PATH)

Cách dùng: python data_collect_drama.py
"""

import sys
import time
import traceback

from config import DRAMA_RAW_DIR, DRAMA_SOURCE_CSV, AUDIO_FORMAT, VIDEO_FORMAT, ZH_PREFIXES, VI_PREFIXES, KO_PREFIXES, JA_PREFIXES
from utils import pad_drama_id, pad_episode, slugify, read_csv_rows, write_csv_rows
from yt_helpers import probe, match_langs, download_sub_track, download_audio, download_video

def choose_zh_track(lang_codes):
    preferred = (
        "zh-hans", "zh-cn", "zh-sg",  # Ưu tiên giản thể
        "zh", "chi", "cmn",          # Chưa xác định hệ chữ
        # "zh-hant", "zh-tw", "zh-hk", # Sau cùng mới phồn thể
    )

    normalized = {
        code.lower().replace("_", "-"): code
        for code in lang_codes
    }

    for code in preferred:
        if code in normalized:
            return normalized[code]

    # Không tự chọn một mã lạ mà chưa kiểm tra
    return None

def process_row(row: dict) -> tuple[str, str]:
    language = slugify(row["language"].strip())
    drama_id = row["drama_id"].strip()
    episode = row["episode"].strip()
    url = row["url"].strip().strip('"').strip("'").strip()

    drama_folder = pad_drama_id(drama_id)
    ep_tag = pad_episode(episode)
    outdir = DRAMA_RAW_DIR / language / drama_folder
    outdir.mkdir(parents=True, exist_ok=True)

    base_name = f"{language}_{drama_folder}_{ep_tag}"

    try:
        info = probe(url)
    except Exception as e:
        return " failed", f"probe lỗi: {e}"

    subs = info.get("subtitles", {}) or {}
    # print("Available subtitles:", list(subs))

    if language == "chinese":
        lang_tag = "zh"
        lang_codes = match_langs(subs, ZH_PREFIXES)
        lang_code = choose_zh_track(lang_codes)
        print("List các subs Trung Quốc:", lang_codes)
        print(f"Sub đã tải trước đó: {lang_codes[0]}")
        print(f"Chọn tải sub: {lang_code}")

    if language == "korean":
        lang_tag = "ko"
        lang_codes = match_langs(subs, KO_PREFIXES)

    if language == "japanese":
        lang_tag = "ja"
        lang_codes = match_langs(subs, JA_PREFIXES)

    vi_codes = match_langs(subs, VI_PREFIXES)
    # lang_code = choose_zh_track(lang_codes)

    if not lang_code or not vi_codes:
        return " failed", (
            f"thiếu sub thủ công ({lang_tag}={'có' if lang_code else 'không'}, "
            f"vi={'có' if vi_codes else 'không'}) - drama bắt buộc cả 2, không dùng auto"
        )

    # Kiểm tra file đã có trên máy: file rỗng/hỏng thì xóa để tải lại, file ổn thì không tải lại
    lang_path = outdir / f"{base_name}_{lang_tag}_manual_raw.srt"
    vi_path = outdir / f"{base_name}_vi_manual_raw.srt"
    audio_path = outdir / f"{base_name}.{AUDIO_FORMAT}"
    video_path = outdir / f"{base_name}.{VIDEO_FORMAT}"
    for p in (lang_path, vi_path, audio_path):
        if p.exists() and (
            p.stat().st_size == 0
            or (p.suffix == ".srt" and "-->" not in p.read_text(encoding="utf-8-sig", errors="ignore"))
        ):
            p.unlink()
    ok_lang = lang_path.exists()
    ok_vi = vi_path.exists()
    ok_audio = audio_path.exists()
    ok_video = video_path.exists()
    print(f"Đã có sẵn trên máy: vi={ok_vi}, {lang_tag}={ok_lang}, audio={ok_audio}, video={ok_video}")

    try:
        if not ok_lang:
            ok_lang = download_sub_track(url, lang_code, True, outdir, f"{base_name}_{lang_tag}_manual_raw")
        if not ok_vi:
            ok_vi = download_sub_track(url, vi_codes[0], True, outdir, f"{base_name}_vi_manual_raw")
        """
        if not ok_audio:
            ok_audio = download_audio(url, outdir, base_name, AUDIO_FORMAT)
        if not ok_video:
            ok_video = download_video(url, outdir, base_name, VIDEO_FORMAT)
        """
         # Tải video trước
        if not ok_video:
            ok_video = download_video(url, outdir, base_name, VIDEO_FORMAT)

        # Audio chưa có: ưu tiên trích từ video trên máy
        if not ok_audio:
            ok_audio = download_audio(url, outdir, base_name, AUDIO_FORMAT, VIDEO_FORMAT)
    except Exception as e:
        return " failed", f"tải lỗi: {e}\n{traceback.format_exc(limit=1)}"

    if not (ok_lang and ok_vi and ok_audio):
        return " failed", f"thiếu file sau khi tải ({lang_tag}={ok_lang}, vi={ok_vi}, audio={ok_audio}, video={ok_video})"

    note = f"vi=manual, {lang_tag}=manual, audio={ok_audio}, video={ok_video}"
    return " done", note


def main():
    if not DRAMA_SOURCE_CSV.exists():
        print(f"Không tìm thấy {DRAMA_SOURCE_CSV}", file=sys.stderr)
        sys.exit(1)

    fieldnames, rows = read_csv_rows(DRAMA_SOURCE_CSV)
    if "status" not in fieldnames:
        fieldnames.append("status")
    if "note" not in fieldnames:
        fieldnames.append("note")

    todo = [r for r in rows if r.get("status", "pending") in ("", " pending", " failed")]
    print(f"[DRAMA] Tổng {len(rows)} dòng, cần xử lý {len(todo)} dòng (pending/failed).\n")

    for i, row in enumerate(todo, 1):
        print(f"--- [{i}/{len(todo)}] {row['language']} {pad_drama_id(row['drama_id'])} {pad_episode(row['episode'])} ---")
        status, note = process_row(row)
        row["status"] = status
        row["note"] = note
        print(f"    -> {status}" + (f" ({note})" if note else ""))
        write_csv_rows(DRAMA_SOURCE_CSV, fieldnames, rows)

        if i < len(todo):
            time.sleep(2)  # nghỉ giữa các video, giảm khả năng bị 429

    n_done = sum(1 for r in rows if r.get("status") == " done")
    n_failed = sum(1 for r in rows if r.get("status") == " failed")
    print(f"\nXong. done={n_done} failed={n_failed} (xem cột note trong {DRAMA_SOURCE_CSV})")


if __name__ == "__main__":
    main()