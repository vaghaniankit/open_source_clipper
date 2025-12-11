from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, HTTPException

from ..tasks import export_video_task, export_clip_task, speaker_center_video_task
from ..paths import STORAGE_DIR

router = APIRouter(prefix="/export", tags=["export"])  # endpoint: POST /


@router.post("")
def export_video(
    path: str = Form(...),
    aspect: Optional[str] = Form("9:16"),
):
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="file not found")
    if aspect not in ("1:1", "16:9", "9:16"):
        raise HTTPException(status_code=400, detail="invalid aspect")
    job = export_video_task.delay(path=path, aspect=aspect)
    return {"job_id": job.id}


@router.post("/clip")
def export_clip(
    job_id: str = Form(...),
    clip_id: str = Form(...),
    aspect: Optional[str] = Form("9:16"),
):
    if aspect not in ("1:1", "16:9", "9:16"):
        raise HTTPException(status_code=400, detail="invalid aspect")

    # Load highlights.json for the given job and find the requested clip
    job_dir = STORAGE_DIR / "pipeline" / job_id
    highlights_path = job_dir / "highlights.json"
    if not highlights_path.exists():
        raise HTTPException(status_code=404, detail="highlights not found for this job_id")

    import json

    try:
        data = json.loads(highlights_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read highlights.json")

    clips = data.get("clips") or []
    clip = next((c for c in clips if str(c.get("id")) == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="clip_id not found for this job_id")

    job = export_clip_task.delay(job_id=job_id, clip=clip, aspect=aspect)
    return {"job_id": job.id}


@router.post("/centered")
def export_centered_video(
    path: str = Form(...),
):
    """Enqueue speaker-centering for an existing video file.

    Typical usage: call this with the output_path from an earlier /export/clip
    job to get a speaker-centered version of a highlight clip.
    """
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="file not found")

    job = speaker_center_video_task.delay(path=path)
    return {"job_id": job.id}


@router.post("/preview")
def export_preview_clip(
    job_id: str = Form(...),
    clip_id: str = Form(...),
    aspect: Optional[str] = Form("9:16"),
):
    """Generate (if needed) a low-resolution preview clip with subtitles and speaker centering.

    The preview is stored under storage/exports/<job_id>/previews/ and returned as a
    URL that can be accessed via the /media mount.
    """
    job_dir = STORAGE_DIR / "pipeline" / job_id
    highlights_path = job_dir / "highlights.json"
    if not highlights_path.exists():
        raise HTTPException(status_code=404, detail="highlights not found for this job_id")

    import json

    try:
        data = json.loads(highlights_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read highlights.json")

    clips = data.get("clips") or []
    clip = next((c for c in clips if str(c.get("id")) == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="clip_id not found for this job_id")

    video_path = data.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="source video not found for this job_id")

    # Validate aspect
    if aspect not in ("1:1", "16:9", "9:16"):
        raise HTTPException(status_code=400, detail="invalid aspect")

    # Prepare directories
    exports_dir = STORAGE_DIR / "exports" / job_id / "previews"
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Determine final preview path. We key by aspect so each ratio has its own
    # cached preview file.
    base_name = clip.get("filename") or f"{clip_id}.mp4"
    aspect_suffix = aspect.replace(":", "x")
    preview_name = f"{Path(base_name).stem}_{aspect_suffix}_preview.mp4"
    preview_path = exports_dir / preview_name

    if preview_path.exists():
        rel = preview_path.relative_to(STORAGE_DIR)
        # from_cache=True means we reused an existing preview file
        return {"preview_url": f"/media/{rel.as_posix()}", "from_cache": True}

    # 1) Cut the clip from the original video
    start = float(clip.get("start", 0.0))
    end = float(clip.get("end", 0.0))
    if end <= start:
        raise HTTPException(status_code=400, detail="invalid clip times")

    raw_cut = exports_dir / f"{Path(base_name).stem}_cut_tmp.mp4"

    cut_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(raw_cut),
    ]

    import subprocess

    subprocess.run(cut_cmd, check=True)

    # 2) Build subtitles for this clip using transcript.json
    transcript_json = job_dir / "transcript.json"
    subtitle_path: Optional[str] = None
    if transcript_json.exists():
        try:
            from ..utils.subtitles import write_clip_srt

            srt_path = write_clip_srt(transcript_json, clip, exports_dir)
            subtitle_path = str(srt_path)
        except Exception:
            subtitle_path = None

    # 3) Apply speaker centering to the raw cut
    centered_path = exports_dir / f"{Path(base_name).stem}_centered_tmp.mp4"
    try:
        from ..services.speaker_centering import center_speaker

        ok = center_speaker(str(raw_cut), str(centered_path))
        src_for_export = centered_path if ok and centered_path.exists() else raw_cut
    except Exception:
        # If speaker centering fails, fall back to the raw cut
        src_for_export = raw_cut

    # 4) Export low-res preview with aspect and optional subtitles
    from ..utils.export import export_with_aspect

    export_with_aspect(str(src_for_export), str(preview_path), aspect=aspect, subtitle_path=subtitle_path)

    rel = preview_path.relative_to(STORAGE_DIR)
    # from_cache=False means this preview was just generated
    return {"preview_url": f"/media/{rel.as_posix()}", "from_cache": False}


@router.post("/clip/download")
def export_clip_download(
    job_id: str = Form(...),
    clip_id: str = Form(...),
    aspect: Optional[str] = Form("9:16"),
):
    """Generate a higher-quality downloadable clip with subtitles and speaker centering.

    This is synchronous and returns a direct download URL under /media.
    """
    if aspect not in ("1:1", "16:9", "9:16"):
        raise HTTPException(status_code=400, detail="invalid aspect")

    job_dir = STORAGE_DIR / "pipeline" / job_id
    highlights_path = job_dir / "highlights.json"
    if not highlights_path.exists():
        raise HTTPException(status_code=404, detail="highlights not found for this job_id")

    import json

    try:
        data = json.loads(highlights_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read highlights.json")

    clips = data.get("clips") or []
    clip = next((c for c in clips if str(c.get("id")) == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="clip_id not found for this job_id")

    video_path = data.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="source video not found for this job_id")

    exports_dir = STORAGE_DIR / "exports" / job_id / "hd"
    exports_dir.mkdir(parents=True, exist_ok=True)

    base_name = clip.get("filename") or f"{clip_id}.mp4"
    output_name = f"{Path(base_name).stem}_{aspect.replace(':', 'x')}_hd.mp4"
    output_path = exports_dir / output_name

    # 1) Cut the clip from the original video
    start = float(clip.get("start", 0.0))
    end = float(clip.get("end", 0.0))
    if end <= start:
        raise HTTPException(status_code=400, detail="invalid clip times")

    raw_cut = exports_dir / f"{Path(base_name).stem}_cut_tmp.mp4"

    cut_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(raw_cut),
    ]

    import subprocess

    subprocess.run(cut_cmd, check=True)

    # 2) Build subtitles for this clip using transcript.json
    transcript_json = job_dir / "transcript.json"
    subtitle_path: Optional[str] = None
    if transcript_json.exists():
        try:
            from ..utils.subtitles import write_clip_srt

            srt_path = write_clip_srt(transcript_json, clip, exports_dir)
            subtitle_path = str(srt_path)
        except Exception:
            subtitle_path = None

    # 3) Export HD clip with aspect and optional subtitles.
    #
    # NOTE: We intentionally do *not* run speaker centering here to keep this
    # endpoint fast and predictable for the user. Speaker centering can still
    # be triggered separately via the /export/centered endpoint using the
    # resulting HD file.
    src_for_export = raw_cut

    # 4) Export HD clip with aspect and optional subtitles
    from ..utils.export import export_with_aspect

    export_with_aspect(str(src_for_export), str(output_path), aspect=aspect, subtitle_path=subtitle_path)

    rel = output_path.relative_to(STORAGE_DIR)
    return {"download_url": f"/media/{rel.as_posix()}"}
