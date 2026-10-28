"""Hàm yt-dlp dùng chung cho cả data_collect_drama.py và data_collect_sport.py."""

import time

import yt_dlp


def match_langs(sub_dict: dict, prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in sub_dict.keys() if c.lower().startswith(prefixes)]


def probe(url: str) -> dict:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def download_sub_track(url: str, lang: str, is_manual: bool, outdir, filename: str) -> bool:
    """Tải 1 track phụ đề cụ thể (thủ công hoặc tự động), lưu thành {filename}.srt."""
    opts = {
        "outtmpl": str(outdir / f"{filename}.%(ext)s"),
        "skip_download": True,
        "writesubtitles": is_manual,
        "writeautomaticsub": not is_manual,
        "subtitleslangs": [lang],
        "subtitlesformat": "srt/best",
        "convertsubtitles": "srt",
        "quiet": True,
        "no_warnings": True,
        "sleep_interval_requests": 1.0,
        "sleep_interval_subtitles": 10.0,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    candidates = sorted(outdir.glob(f"{filename}.*.srt"))
    if not candidates:
        return False
    candidates[0].rename(outdir / f"{filename}.srt")
    return True


def download_audio(url: str, outdir, filename: str, audio_format: str) -> bool:
    opts = {
        "outtmpl": str(outdir / f"{filename}.%(ext)s"),
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": audio_format}],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return (outdir / f"{filename}.{audio_format}").exists()

def download_video(url: str, outdir, filename: str, video_format: str = "mp4") -> bool:
    """Tải video đầy đủ (hình + tiếng) thay vì chỉ audio. Lấy luồng video +
    audio chất lượng cao nhất rồi ghép thành 1 file duy nhất (video_format)."""
    opts = {
        "outtmpl": str(outdir / f"{filename}.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": video_format,
        "quiet": True,
        "no_warnings": True,
        # **_common_opts(),
    }
 
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
 
    return (outdir / f"{filename}.{video_format}").exists()
