# Video Highlights Backend & Frontend

FastAPI + Celery + Redis (Memurai) application for resumable uploads, YouTube downloads, processing pipeline (extract → chunk → transcribe), and video highlights generation. Includes a Jinja2-based frontend for managing jobs and viewing clips.

## Features

- **Uploads**: Resumable, chunked uploads for large files.
- **YouTube**: Download and process videos directly from YouTube.
- **Pipeline**: Automated audio extraction, transcription (faster-whisper), rich signal analysis (energy, scene cuts, audio events), and highlight detection (LLM-guided with heuristic fallback).
- **Exports**: Generate clips in 1:1, 16:9, and 9:16 aspect ratios.
- **UI**:
    - **Dashboard**: View job status and progress.
    - **Clips List**: Browse generated highlights with low-res previews (speaker-centered, karaoke subtitles).
    - **Clip Detail**: Dedicated page for individual clips with full interactive transcript and download options.
    - **Social Sharing**: Direct links to share clips on YouTube, TikTok, Instagram, Twitter, and Facebook.

## Folder structure

- app/
  - main.py (app bootstrap, sessions, router includes)
  - config.py (centralized settings)
  - celery_app.py (Celery app; task_track_started enabled)
  - paths.py (BASE_DIR, storage paths)
  - routers/
    - uploads.py (init, chunk, complete)
    - youtube.py (/download/youtube)
    - pipeline.py (/process, /pipeline – extract → chunk → transcribe → highlights)
    - export.py (/export, /export/clip)
    - jobs.py (/jobs/{job_id})
    - highlights.py (/highlights/{job_id})
  - services/
    - youtube_downloader.py (yt_dlp logic)
    - highlight.py (highlight generation script)
    - speaker_centering.py (speaker auto-centering script)
  - templates/ (Jinja2 HTML templates)
    - dashboard.html
    - clips.html
    - clip_detail.html
    - job.html
    - login.html
  - utils/
    - audio.py (chunking + faster-whisper transcription)
    - export.py (FFmpeg scale+pad exports)
- storage/
  - videos/ (assembled uploads and downloads)
  - audio/ (audio outputs)
  - pipeline/ (<job_id>/ audio.wav, chunks, transcript.txt)
  - exports/ (<job_id>/ output clips)
  - data/ (.json/.csv inputs/outputs)
- downloads/ (yt-dlp target)
- scripts/ (one-off scripts kept runnable)
- docs/ (guides and notes)
- requirements.txt

## Highlight pipeline (simplified)

- **Audio extraction & chunking**  
  - Extract mono 16kHz audio from the source video using FFmpeg.  
  - Chunk audio into overlapping windows for transcription.
- **Transcription (faster-whisper)**  
  - Run faster-whisper on each chunk and merge to a structured transcript with per-line timestamps.
- **Signal analysis features**  
  - **Energy** per segment (loudness / intensity).  
  - **Scene / shot changes** from the source video (PySceneDetect).  
  - **Audio events** such as music / laughter via a YamNet-style classifier.  
  - Combine these into an **excitement score** per transcript line.
- **Highlight selection**  
  - Pass transcript text + features (energy, scene_id, near_cut, tags, excitement) into the LLM to propose highlight time ranges, then snap to transcript boundaries and enforce the UI duration preset.  
  - If LLM calls fail (e.g. API/network/region issues), fall back to a **pure heuristic** mode using the excitement score + scene cuts to select the top segments and still produce clips.

## Environment variables

- SESSION_SECRET: secret for session cookies
- REDIS_URL: redis://localhost:6379/0 (Memurai)
- REDIS_BACKEND: (default = REDIS_URL)
- GOOGLE_CLIENT_ID: (for /auth/login)
- GOOGLE_CLIENT_SECRET: (for /auth/login)
- GOOGLE_REDIRECT_URI: http://127.0.0.1:8000/auth/callback
- OPENAI_API_KEY: (used when LLM analysis is integrated)

## Install

```powershell
pip install -r requirements.txt
```

## Run

- Redis/Memurai: ensure 127.0.0.1:6379 is listening
- Celery worker (Windows):
```powershell
$env:REDIS_URL="redis://localhost:6379/0"
$env:REDIS_BACKEND=$env:REDIS_URL
celery -A app.celery_app.celery_app worker --loglevel=info --pool=solo
```
- Celery beat (scheduled tasks):
```powershell
celery -A app.celery_app.celery_app beat --loglevel=info
```
- API server:
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## API & Frontend Quickstart

### Frontend Routes
- **Login**: `/` (redirects to Google Auth)
- **Dashboard**: `/app` (Upload videos or submit YouTube links)
- **Clips List**: `/app/jobs/{job_id}/clips`
- **Clip Detail**: `/app/jobs/{job_id}/clips/{clip_id}`

### API Endpoints
- **Health**: `GET /health`
- **Resumable upload**:
  - `POST /uploads/init`
  - `POST /uploads/chunk`
  - `POST /uploads/complete`
- **YouTube download**: `POST /download/youtube`
- **Orchestrated pipeline**: `POST /pipeline`
- **Export**:
  - Full video: `POST /export`
  - Single clip: `POST /export/clip`
  - Preview: `POST /export/preview` (Cached)
- **Job Status**: `GET /jobs/{job_id}`

## Notes

- Celery is configured to show STARTED state during task execution and jobs can be resumed if they are in processing state.
- faster-whisper will download a model on first use; consider the "tiny" model for faster tests.
- FFmpeg must be on PATH.
- Preview clips and highlights are cached on disk so completed jobs load quickly and show as ready without reprocessing.

## Security

- Do not print or commit secrets. Rotate keys if exposed.
