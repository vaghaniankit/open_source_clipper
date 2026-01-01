import os
import shutil
import uuid
import logging
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .config import settings
from .paths import STORAGE_DIR, UPLOAD_TMP_DIR, VIDEOS_DIR
from .routers.uploads import router as uploads_router
from .routers.youtube import router as youtube_router
from .routers.pipeline import router as pipeline_router
from .routers.export import router as export_router
from .routers.jobs import router as jobs_router
from .routers.highlights import router as highlights_router

BASE_DIR = Path(__file__).resolve().parent.parent

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

app = FastAPI(title="Video Highlights Backend", version="0.1.0")
logger = logging.getLogger(__name__)
logger.info("Google OAuth redirect URI: %s", settings.GOOGLE_REDIRECT_URI)

# Sessions for OAuth login state
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SESSION_SECRET,
    same_site="lax",
    https_only=False,
)

try:
    from .auth import router as auth_router
    app.include_router(auth_router)
except Exception:
    # auth is optional if not configured
    pass

app.include_router(uploads_router)
app.include_router(youtube_router)
app.include_router(pipeline_router)
app.include_router(export_router)
app.include_router(jobs_router)
app.include_router(highlights_router)

# Serve generated media (previews, exports) and static assets
app.mount("/media", StaticFiles(directory=str(STORAGE_DIR)), name="media")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")

@app.get("/", response_class=RedirectResponse)
def index(request: Request):
    """Render a simple login page that starts Google OAuth."""
    # Temporary hack: if no user in session, create a demo user and jump into the app
    user = request.session.get("user")
    if not user:
        request.session["user"] = {"id": "demo", "email": "demo@example.com"}
    return RedirectResponse(url="/app/source", status_code=302)

@app.get("/app", response_class=RedirectResponse)
def app_dashboard(request: Request):
    """Redirect to the first step of the wizard."""
    user = request.session.get("user")
    if not user:
        user = {"id": "demo", "email": "demo@example.com"}
        request.session["user"] = user
    # Initialize or clear job draft
    request.session["job_draft"] = {}
    return RedirectResponse(url="/app/source", status_code=302)

@app.get("/app/projects", response_class=HTMLResponse)
def app_projects(request: Request):
    """Render the projects list."""
    user = request.session.get("user")
    if not user:
        user = {"id": "demo", "email": "demo@example.com"}
        request.session["user"] = user
    return templates.TemplateResponse("projects.html", {"request": request, "user": user})

@app.get("/app/source", response_class=HTMLResponse)
def step_source_get(request: Request):
    user = request.session.get("user")
    if not user:
        user = {"id": "demo", "email": "demo@example.com"}
        request.session["user"] = user
    return templates.TemplateResponse("step_source.html", {"request": request, "user": user})

@app.post("/app/source")
async def step_source_post(
    request: Request,
    youtube_url: Optional[str] = Form(None),
    video_file: Optional[UploadFile] = File(None)
):
    user = request.session.get("user")
    if not user: return RedirectResponse(url="/", status_code=302)

    draft = request.session.get("job_draft", {})
    
    if youtube_url and youtube_url.strip():
        draft["source_type"] = "youtube"
        draft["source_url"] = youtube_url.strip()

        # Try to probe YouTube metadata (duration) without downloading the video,
        # so we can show an accurate processing timeframe later.
        try:
            import yt_dlp

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(draft["source_url"], download=False)
            dur = info.get("duration")
            if isinstance(dur, (int, float)) and dur > 0:
                draft["video_duration_seconds"] = float(dur)
        except Exception:
            # If yt_dlp probing fails for any reason, just continue without
            # duration; the prompt step will fall back to a generic label.
            pass
    elif video_file and video_file.filename:
        # Save file temporarily
        file_id = str(uuid.uuid4())
        ext = Path(video_file.filename).suffix
        temp_name = f"{file_id}{ext}"
        temp_path = UPLOAD_TMP_DIR / temp_name
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
            
        draft["source_type"] = "upload"
        draft["local_path"] = str(temp_path)
        draft["original_filename"] = video_file.filename
    else:
        # No input provided
        return RedirectResponse(url="/app/source", status_code=302)

    request.session["job_draft"] = draft
    return RedirectResponse(url="/app/options", status_code=302)

@app.get("/app/options", response_class=HTMLResponse)
def step_options_get(request: Request):
    user = request.session.get("user")
    if not user:
        user = {"id": "demo", "email": "demo@example.com"}
        request.session["user"] = user
    return templates.TemplateResponse("step_options.html", {"request": request, "user": user})

@app.post("/app/options")
async def step_options_post(
    request: Request,
    duration: str = Form(...),
    category: str = Form(...)
):
    user = request.session.get("user")
    if not user: return RedirectResponse(url="/", status_code=302)
    
    draft = request.session.get("job_draft", {})
    draft["duration"] = duration
    draft["category"] = category
    request.session["job_draft"] = draft
    
    return RedirectResponse(url="/app/prompt", status_code=302)

@app.get("/app/prompt", response_class=HTMLResponse)
def step_prompt_get(request: Request):
    user = request.session.get("user")
    if not user:
        user = {"id": "demo", "email": "demo@example.com"}
        request.session["user"] = user
    draft = request.session.get("job_draft", {})
    video_duration_label = None

    # 1) If we already know the duration in seconds (e.g. from YouTube
    # metadata), prefer that.
    dur_seconds = draft.get("video_duration_seconds")
    if isinstance(dur_seconds, (int, float)) and dur_seconds > 0:
        try:
            total_dur = float(dur_seconds)
            hours = int(total_dur // 3600)
            minutes = int((total_dur % 3600) // 60)
            seconds = int(total_dur % 60)
            video_duration_label = f"{hours:01d}:{minutes:02d}:{seconds:02d}"
        except Exception:
            video_duration_label = None

    # 2) If we don't have duration yet and this is an uploaded file, probe it
    # with ffprobe using the temporary local_path saved in the draft.
    if video_duration_label is None:
        source_type = draft.get("source_type")
        if source_type == "upload":
            local_path = draft.get("local_path")
            if local_path and Path(local_path).exists():
                try:
                    probe = subprocess.run([
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        local_path,
                    ], capture_output=True, text=True, check=False)
                    total_dur = float(probe.stdout.strip() or 0) if probe.stdout else 0.0
                    if total_dur > 0:
                        hours = int(total_dur // 3600)
                        minutes = int((total_dur % 3600) // 60)
                        seconds = int(total_dur % 60)
                        video_duration_label = f"{hours:01d}:{minutes:02d}:{seconds:02d}"
                except Exception:
                    video_duration_label = None

    return templates.TemplateResponse("step_prompt.html", {
        "request": request,
        "user": user,
        "video_duration_label": video_duration_label,
    })

@app.post("/app/prompt")
async def step_prompt_post(
    request: Request,
    prompt: Optional[str] = Form(None)
):
    try:
        user = request.session.get("user")
        if not user:
            user = {"id": "demo", "email": "demo@example.com"}
            request.session["user"] = user
        
        draft = request.session.get("job_draft", {})
        draft["prompt"] = prompt or ""
        
        from .tasks import download_youtube_task, orchestrate_pipeline_task
        from celery import chain
        import logging
        logger = logging.getLogger(__name__)
        
        job_id = str(uuid.uuid4())
        
        # Log the draft for debugging
        logger.info(f"Processing job {job_id} with draft: {draft}")
        
        source_type = draft.get("source_type")
        
        if source_type == "youtube":
            # Chain: download YouTube video, then process it
            youtube_url = draft.get("source_url")
            if youtube_url:
                logger.info(f"Queueing YouTube download + processing for job {job_id} with URL: {youtube_url}")
                
                # Create a chain: download -> orchestrate_pipeline
                # The download task returns the path, which gets passed to orchestrate_pipeline
                task_chain = chain(
                    download_youtube_task.s(youtube_url),
                    orchestrate_pipeline_task.s(
                        prompt=draft.get("prompt"),
                        duration_preset=draft.get("duration")
                    )
                )
                
                # Apply the chain with the job_id
                task_chain.apply_async(task_id=job_id)
                logger.info(f"YouTube task chain queued successfully for job {job_id}")
            else:
                logger.error(f"No YouTube URL found in draft for job {job_id}")
                
        elif source_type == "upload":
            # Directly run orchestrate_pipeline on uploaded file
            local_path = draft.get("local_path")
            if local_path:
                logger.info(f"Queueing orchestrate_pipeline_task for job {job_id} with path: {local_path}")
                orchestrate_pipeline_task.apply_async(
                    kwargs={
                        "path": local_path,
                        "prompt": draft.get("prompt"),
                        "duration_preset": draft.get("duration")
                    },
                    task_id=job_id
                )
                logger.info(f"Task queued successfully for job {job_id}")
            else:
                logger.error(f"No local_path found in draft for upload job {job_id}")
        else:
            logger.error(f"Unknown source_type: {source_type} for job {job_id}")
            return JSONResponse({
                "success": False,
                "error": "Invalid source type. Please start over."
            }, status_code=400)
        
        if "job_draft" in request.session:
            del request.session["job_draft"]
        
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "redirect_url": f"/app/jobs/{job_id}"
        })
    except Exception as e:
        import traceback
        logger.error(f"Error in step_prompt_post: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)





@app.get("/app/jobs/{job_id}", response_class=HTMLResponse)
def app_job_status(job_id: str, request: Request):
    """Job status page that polls /jobs/{job_id}."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("job.html", {"request": request, "job_id": job_id})


@app.get("/app/jobs/{job_id}/clips", response_class=HTMLResponse)
def app_job_clips(job_id: str, request: Request):
    """Clips page for a completed job."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    # Create a simple job object with the job_id
    job = {"id": job_id, "source_name": job_id}
    
    return templates.TemplateResponse("clips.html", {"request": request, "job": job})

@app.get("/app/jobs/{job_id}/clips/{clip_id}", response_class=HTMLResponse)
def app_clip_detail(job_id: str, clip_id: str, request: Request):
    """Detail page for a specific clip."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    # Default fallback objects
    job = {"id": job_id, "source_name": job_id}
    clip = {"id": clip_id, "title": f"Clip {clip_id}", "start": 0, "end": 0, "index": 1}

    # Load real clip data
    import json
    highlights_path = STORAGE_DIR / "pipeline" / job_id / "highlights.json"
    print('\n\n XXXX ➡ app/main.py:355 highlights_path:', highlights_path)
    if highlights_path.exists():
        try:
            with open(highlights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                found_clip = next((c for c in data.get("clips", []) if c.get("id") == clip_id), None)
                if found_clip:
                    clip = found_clip
                    # Add index if missing (1-based)
                    if "index" not in clip:
                        clip["index"] = data.get("clips", []).index(found_clip) + 1
        except Exception as e:
            print(f"Error loading highlights for job {job_id}: {e}")

    # Load transcript segments for this clip
    transcript_path = STORAGE_DIR / "pipeline" / job_id / "transcript.json"
    print('\n\n XXXX ➡ app/main.py:371 transcript_path:', transcript_path)
    if transcript_path.exists() and clip.get("end") > 0:
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
                # Filter segments that overlap with the clip
                start = clip.get("start", 0)
                end = clip.get("end", 0)
                # Simple overlap check: segment end > clip start AND segment start < clip end
                segments = [t for t in transcript if t.get("end", 0) > start and t.get("start", 0) < end]
                clip["transcript_segments"] = segments
        except Exception as e:
            print(f"Error loading transcript for job {job_id}: {e}")
    
    return templates.TemplateResponse("clip_detail.html", {
        "request": request, 
        "job": job, 
        "clip": clip
    })

@app.get("/health")
def health():
    return {"status": "ok"}
