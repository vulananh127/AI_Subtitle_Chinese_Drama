# Subtitle Studio

```
Trình duyệt ──► web/ (Flask: đăng nhập, dự án, SQLite, video)  ──►  model-server/ (FastAPI, GPU: ASR + MT)
```

## 1. Chạy web (chưa cần model)
```powershell
cd web
python -m venv .venv ; .venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                                     # Linux/macOS: cp .env.example .env
python app.py                                              # mở http://localhost:5000
```
Cần cài `ffmpeg` (tách audio khi tạo dự án). Để trống `MODEL_API_URL` thì "Dịch sơ" trả 5 cue mẫu.
Chạy thật (Windows): `waitress-serve --port=5000 app:app` · (Linux): `gunicorn -w 1 --threads 8 -b 0.0.0.0:5000 app:app`.

## 2. Đăng nhập Google
Google Cloud Console → APIs & Services → Credentials → Create credentials → OAuth client ID → *Web application*.
Thêm **Authorized JavaScript origins**: `http://localhost:5000` (và domain thật). Dán Client ID vào `GOOGLE_CLIENT_ID` trong `.env`.

## 3. Lưu video
- `STORAGE=local` (mặc định): video nằm ở `web/data/videos/`.
- `STORAGE=cloudinary`: đặt `CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME`. Gói free giới hạn 100 MB mỗi video.
Audio đã tách (opus, nhỏ) luôn nằm ở `web/data/audio/` để gửi cho model.

## 4. Nối model server (GPU)
```bash
cd model-server && pip install -r requirements.txt
MODEL_API_KEY=chuoi-bi-mat PIPELINE=mock uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Điền ASR/MT thật trong `model-server/app/pipeline.py`. Trong `web/.env`: `MODEL_API_URL=http://<ip-gpu>:8000` và cùng `MODEL_API_KEY`.

## Dữ liệu (SQLite: web/data/app.db)
`users` · `projects` · `cues` (start, end, src, text, mt = bản gốc của model, ok = đã duyệt, origin) · `revisions` (ảnh chụp cue; các lần sửa liên tiếp trong 5 phút gộp làm một) · `jobs`.
