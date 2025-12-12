from pathlib import Path
import json
import shutil
import subprocess
from typing import List

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse

from ..celery_app import celery_app
from ..paths import STORAGE_DIR

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _status_for_job(job_id: str):
    async_result = celery_app.AsyncResult(job_id)
    info = async_result.info if isinstance(async_result.info, dict) else {"info": str(async_result.info)}

    raw_progress = info.get("progress") if isinstance(info, dict) else None
    if isinstance(raw_progress, (int, float)):
        progress = max(0, min(100, int(raw_progress)))
    else:
        if async_result.state == "PENDING":
            progress = 0
        elif async_result.state == "SUCCESS":
            progress = 100
        else:
            progress = 0

    return async_result, info, progress


def _ensure_clip_thumbnails(job_id: str, highlights: dict) -> dict:
    """Ensure each clip in highlights.json has a thumbnail_url.

    This is primarily used when re-opening jobs from the Projects page: older
    jobs may have been created before thumbnail generation was added, so we
    lazily generate thumbs here based on the stored video_path and clip start
    times.
    """
    clips = highlights.get("clips") or []
    if not clips:
        return highlights

    video_path = highlights.get("video_path")
    if not video_path or not Path(video_path).exists():
        return highlights

    job_exports_dir = STORAGE_DIR / "exports" / job_id / "thumbs"
    job_exports_dir.mkdir(parents=True, exist_ok=True)

    changed = False
    for clip in clips:
        if clip.get("thumbnail_url"):
            continue

        clip_id = clip.get("id") or "clip"
        try:
            start = float(clip.get("start", 0.0) or 0.0)
        except Exception:
            start = 0.0
        start = max(0.0, start)

        thumb_name = f"{clip_id}.jpg"
        thumb_path = job_exports_dir / thumb_name

        if not thumb_path.exists():
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-i",
                str(video_path),
                "-vframes",
                "1",
                "-q:v",
                "3",
                str(thumb_path),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                continue

        try:
            rel = thumb_path.relative_to(STORAGE_DIR)
            clip["thumbnail_url"] = f"/media/{rel.as_posix()}"
            changed = True
        except Exception:
            continue

    if changed:
        highlights["clips"] = clips
    return highlights


@router.get("/{job_id}")
def job_status(job_id: str):
    async_result, info, progress = _status_for_job(job_id)
    
    state = async_result.state
    result = async_result.result if async_result.successful() else None
    
    # Fallback: Check disk if Celery doesn't have the result or says PENDING but files exist
    job_dir = STORAGE_DIR / "pipeline" / job_id
    highlights_path = job_dir / "highlights.json"
    
    if highlights_path.exists():
        # Job is actually done
        state = "SUCCESS"
        progress = 100
        try:
            highlights = json.loads(highlights_path.read_text(encoding="utf-8"))
            highlights = _ensure_clip_thumbnails(job_id, highlights)

            # Construct result similar to what task returns
            result = {
                "job_id": job_id,
                "clips": highlights.get("clips", []),
                "highlights_path": str(highlights_path)
            }

            # Persist any new thumbnail_url fields back to disk so future
            # requests do not need to regenerate them.
            try:
                highlights_path.write_text(json.dumps(highlights, indent=2), encoding="utf-8")
            except Exception:
                pass
        except Exception:
            # If read fails, stick to Celery status
            pass

    return JSONResponse(
        {
            "id": job_id,
            "state": state,
            "ready": state in ["SUCCESS", "FAILURE"],
            "successful": state == "SUCCESS",
            "progress": progress,
            "result": result,
            "info": info,
        }
    )


@router.get("")
def list_jobs():
    """List all pipeline jobs with basic metadata and status.

    Jobs are discovered from storage/pipeline/<job_id>/ directories.
    """
    pipeline_dir = STORAGE_DIR / "pipeline"
    if not pipeline_dir.exists():
        return JSONResponse({"jobs": []})

    jobs = []
    for job_dir in sorted(pipeline_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        meta_path = job_dir / "job_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                import json

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        async_result, info, progress = _status_for_job(job_id)

        # If highlights.json exists, force status to SUCCESS just like job_status()
        highlights_path = job_dir / "highlights.json"
        state = async_result.state
        ready = async_result.ready()
        successful = async_result.successful() if async_result.ready() else False
        if highlights_path.exists():
            state = "SUCCESS"
            ready = True
            successful = True
            progress = 100

        jobs.append(
            {
                "id": job_id,
                "state": state,
                "progress": progress,
                "ready": ready,
                "successful": successful,
                "source_name": Path(meta.get("path" or "")).name if meta.get("path") else job_id,
                "prompt": meta.get("prompt"),
                "timeframe_percent": meta.get("timeframe_percent"),
            }
        )

    return JSONResponse({"jobs": jobs})


@router.delete("/{job_id}/clips/{clip_id}")
def delete_clip(job_id: str, clip_id: str):
    """Delete a single clip from a job.
    
    Removes the clip from highlights.json and deletes associated files.
    """
    job_dir = STORAGE_DIR / "pipeline" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    highlights_path = job_dir / "highlights.json"
    if not highlights_path.exists():
        raise HTTPException(status_code=404, detail="No highlights found")
    
    # Load highlights
    try:
        highlights = json.loads(highlights_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read highlights: {e}")
    
    clips = highlights.get("clips", [])
    
    # Find and remove the clip
    clip_to_delete = None
    updated_clips = []
    for clip in clips:
        if clip.get("id") == clip_id:
            clip_to_delete = clip
        else:
            updated_clips.append(clip)
    
    if clip_to_delete is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    # Delete associated files
    exports_dir = STORAGE_DIR / "exports" / job_id
    if exports_dir.exists():
        # Delete preview
        preview_dir = exports_dir / "previews"
        if preview_dir.exists():
            preview_file = preview_dir / f"{clip_id}.mp4"
            if preview_file.exists():
                preview_file.unlink()
        
        # Delete high-quality exports
        for aspect in ["1:1", "16:9", "9:16"]:
            export_file = exports_dir / f"{clip_id}_{aspect.replace(':', 'x')}.mp4"
            if export_file.exists():
                export_file.unlink()
    
    # Update highlights.json
    highlights["clips"] = updated_clips
    highlights_path.write_text(json.dumps(highlights, indent=2), encoding="utf-8")

    # If no clips remain, clean up the whole job so it no longer appears as a
    # project.
    if not updated_clips:
        # Remove exports for this job
        exports_dir = STORAGE_DIR / "exports" / job_id
        if exports_dir.exists():
            shutil.rmtree(exports_dir, ignore_errors=True)
        # Remove the pipeline job directory itself
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

    return JSONResponse({
        "success": True,
        "deleted_clip_id": clip_id,
        "remaining_clips": len(updated_clips)
    })


@router.post("/{job_id}/resume")
def resume_job(job_id: str):
    """Re-queue the main processing pipeline for a job using stored metadata.

    This is used when a job was interrupted or failed; it restarts the
    orchestrate_pipeline_task with the original arguments and the same job_id
    so the frontend can continue polling the same identifier.
    """
    job_dir = STORAGE_DIR / "pipeline" / job_id
    meta_path = job_dir / "job_meta.json"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="No metadata found for job")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read job metadata: {e}")

    video_path = meta.get("path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Source video for job not found")

    from ..tasks import orchestrate_pipeline_task

    # Re-run the pipeline with the same parameters, keeping the same job_id
    orchestrate_pipeline_task.apply_async(
        kwargs={
            "path": video_path,
            "chunk_seconds": meta.get("chunk_seconds", 30),
            "overlap_seconds": meta.get("overlap_seconds", 2),
            "model_size": meta.get("model_size", "base"),
            "prompt": meta.get("prompt"),
            "duration_preset": meta.get("duration_preset"),
            "timeframe_percent": meta.get("timeframe_percent"),
        },
        task_id=job_id,
    )

    return JSONResponse({"success": True, "job_id": job_id})


@router.post("/{job_id}/clips/delete-bulk")
def delete_clips_bulk(job_id: str, clip_ids: List[str] = Body(..., embed=True)):
    """Delete multiple clips from a job in one operation.
    
    Request body: {"clip_ids": ["clip_001", "clip_002", ...]}
    """
    job_dir = STORAGE_DIR / "pipeline" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    highlights_path = job_dir / "highlights.json"
    if not highlights_path.exists():
        raise HTTPException(status_code=404, detail="No highlights found")
    
    # Load highlights
    try:
        highlights = json.loads(highlights_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read highlights: {e}")
    
    clips = highlights.get("clips", [])
    clip_ids_set = set(clip_ids)
    
    # Filter out clips to delete
    deleted_count = 0
    updated_clips = []
    for clip in clips:
        if clip.get("id") in clip_ids_set:
            deleted_count += 1
            # Delete associated files
            clip_id = clip.get("id")
            exports_dir = STORAGE_DIR / "exports" / job_id
            if exports_dir.exists():
                # Delete preview
                preview_dir = exports_dir / "previews"
                if preview_dir.exists():
                    preview_file = preview_dir / f"{clip_id}.mp4"
                    if preview_file.exists():
                        preview_file.unlink()
                
                # Delete high-quality exports
                for aspect in ["1:1", "16:9", "9:16"]:
                    export_file = exports_dir / f"{clip_id}_{aspect.replace(':', 'x')}.mp4"
                    if export_file.exists():
                        export_file.unlink()
        else:
            updated_clips.append(clip)
    
    # Update highlights.json
    highlights["clips"] = updated_clips
    highlights_path.write_text(json.dumps(highlights, indent=2), encoding="utf-8")

    # If all clips were removed, clean up the job so it disappears from the
    # projects view.
    if not updated_clips:
        exports_dir = STORAGE_DIR / "exports" / job_id
        if exports_dir.exists():
            shutil.rmtree(exports_dir, ignore_errors=True)
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

    return JSONResponse({
        "success": True,
        "deleted_count": deleted_count,
        "remaining_clips": len(updated_clips)
    })
