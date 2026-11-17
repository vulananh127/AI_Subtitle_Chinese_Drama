import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from . import pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("DATA_DIR", ROOT / "model-server" / "data"))
(DATA / "uploads").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Subtitle Studio - Model Server")
API_KEY = os.getenv("MODEL_API_KEY", "")   # nếu đặt, web backend phải gửi header X-API-Key


def check_key(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Sai API key")

jobs: dict[str, dict] = {}                       # trạng thái job (trong RAM)
pool = ThreadPoolExecutor(max_workers=int(os.getenv("GPU_WORKERS", "1")))  # 1 = xử lý lần lượt từng video


def run_job(job_id: str, video: Path, src: str, tgt: str):
    job = jobs[job_id]
    try:
        job.update(status="running", stage="Tách audio", progress=0.05)
        wav = video.with_suffix(".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
            check=True, capture_output=True,
        )
        job.update(stage="Nhận dạng giọng nói (ASR)", progress=0.2)
        segs = pipeline.transcribe(str(wav), src)
        job.update(stage="Dịch (MT)", progress=0.7)
        texts = pipeline.translate([s["text"] for s in segs], src, tgt)
        cues = [{"start": s["start"], "end": s["end"], "src": s["text"], "text": t} for s, t in zip(segs, texts)]
        job.update(status="done", progress=1.0, cues=cues)
    except subprocess.CalledProcessError as e:
        job.update(status="failed", error="ffmpeg lỗi: " + e.stderr.decode(errors="ignore")[-300:])
    except Exception as e:  # noqa: BLE001
        job.update(status="failed", error=f"{type(e).__name__}: {e}")


@app.post("/api/jobs", dependencies=[Depends(check_key)])
async def create_job(video: UploadFile = File(...), src_lang: str = Form("en"), tgt_lang: str = Form("vi")):
    job_id = uuid.uuid4().hex
    path = DATA / "uploads" / f"{job_id}{Path(video.filename or '').suffix}"
    with path.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    jobs[job_id] = {"status": "queued", "stage": "Đang chờ", "progress": 0}
    pool.submit(run_job, job_id, path, src_lang, tgt_lang)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(check_key)])
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Không có job này")
    return jobs[job_id]

