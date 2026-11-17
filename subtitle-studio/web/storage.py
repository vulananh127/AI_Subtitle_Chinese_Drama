import os
import shutil
import subprocess

MODE = os.getenv("STORAGE", "local")   # local | cloudinary


def extract_audio(src, dst):
    """Tách audio mono 16kHz (opus) để gửi cho model; nhỏ hơn nhiều so với video."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "libopus", "-b:a", "24k", str(dst)], check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("Server chưa cài ffmpeg")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg lỗi: " + e.stderr.decode(errors="ignore")[-200:])


def cut_clip(src, dst, start, end):
    """Cắt đoạn [start, end] giây thành wav mono 16kHz để gửi cho ASR."""
    try:
        subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
                        "-ac", "1", "-ar", "16000", str(dst)], check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("Server chưa cài ffmpeg")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg lỗi: " + e.stderr.decode(errors="ignore")[-200:])


def put_video(tmp, pid, ext, data_dir):
    """Trả về (storage, video_ref, video_url)."""
    if MODE == "cloudinary":
        import cloudinary.uploader
        r = cloudinary.uploader.upload_large(str(tmp), resource_type="video", folder="subtitle-studio", chunk_size=20_000_000)
        return "cloudinary", r["public_id"], r["secure_url"]
    d = data_dir / "videos"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{pid}{ext}"
    shutil.move(str(tmp), d / name)
    return "local", name, f"/media/{pid}"


def delete_video(p, data_dir):
    try:
        if p["storage"] == "cloudinary":
            import cloudinary.uploader
            cloudinary.uploader.destroy(p["video_ref"], resource_type="video")
        else:
            (data_dir / "videos" / p["video_ref"]).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - xóa file lỗi không nên chặn xóa dự án
        pass
