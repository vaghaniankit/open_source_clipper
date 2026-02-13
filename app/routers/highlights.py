from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..paths import STORAGE_DIR

router = APIRouter(prefix="/highlights", tags=["highlights"])


@router.get("/{job_id}")
def get_highlights(job_id: str):
    job_dir = STORAGE_DIR / "pipeline" / job_id
    highlights_path = job_dir / "highlights.json"
    if not highlights_path.exists():
        raise HTTPException(status_code=404, detail="highlights not found for this job_id")

    try:
        import json

        data = json.loads(highlights_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read highlights.json")

    # Ensure we always return job_id along with the stored payload
    payload = {"job_id": job_id}
    if isinstance(data, dict):
        payload.update(data)
    else:
        payload["clips"] = data

    return JSONResponse(payload)


@router.get("/{job_id}/transcript")
def get_transcript(job_id: str):
    """Return transcript segments for a given job_id."""
    job_dir = STORAGE_DIR / "pipeline" / job_id
    transcript_path = job_dir / "transcript.json"
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="transcript not found for this job_id")

    try:
        import json

        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read transcript.json")

    return JSONResponse({"job_id": job_id, "segments": segments})


from ..utils.subtitles import write_clip_srt


@router.get("/{job_id}/clips/{clip_id}/subtitles")
def get_clip_subtitles(job_id: str, clip_id: str):
    """
    Look up a single clip by job_id and clip_id from its highlights.json,
    and generate subtitles for it in ASS format.

    ASS format is returned directly in the response payload.
    """
    from ..utils.subtitles import build_clip_ass
    import json

    job_dir = STORAGE_DIR / "pipeline" / job_id
    highlights_path = job_dir / "highlights.json"
    transcript_path = job_dir / "transcript.json"

    if not highlights_path.exists():
        raise HTTPException(status_code=404, detail="highlights.json not found")
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="transcript.json not found")

    # Load data
    try:
        h_data = json.loads(highlights_path.read_text(encoding="utf-8"))
        t_data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to parse json: {e}")

    # Find the specific clip
    clips = h_data.get("clips", [])
    clip = next((c for c in clips if str(c.get("id")) == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail=f"clip_id {clip_id} not found in highlights.json")

    # Generate ASS content for this clip
    clip_start = float(clip.get("start", 0.0))
    clip_end = float(clip.get("end", 0.0))
    segments = t_data if isinstance(t_data, list) else t_data.get("segments", [])
    
    ass_text = build_clip_ass(segments, clip_start, clip_end, clip=clip)

    return JSONResponse({
        "job_id": job_id,
        "clip_id": clip_id,
        "ass_subtitles": ass_text,
    })


@router.delete("/{job_id}/clips/{clip_id}")
def delete_clip(job_id: str, clip_id: str):
    """Delete a single clip from highlights.json for the given job_id.

    - Does NOT renumber remaining clip IDs.
    - Persists updated highlights.json back to disk.
    """
    job_dir = STORAGE_DIR / "pipeline" / job_id
    highlights_path = job_dir / "highlights.json"
    if not highlights_path.exists():
        raise HTTPException(status_code=404, detail="highlights not found for this job_id")

    try:
        import json

        raw = json.loads(highlights_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read highlights.json")

    # Normalize structure: expect a dict with a "clips" list.
    if isinstance(raw, dict):
        clips = raw.get("clips") or []
    else:
        clips = raw
        raw = {"clips": clips}

    # Filter out the target clip
    original_len = len(clips)
    remaining = [c for c in clips if str(c.get("id")) != clip_id]

    if len(remaining) == original_len:
        # Nothing removed -> clip_id not found
        raise HTTPException(status_code=404, detail="clip_id not found for this job_id")

    raw["clips"] = remaining

    try:
        highlights_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        raise HTTPException(status_code=500, detail="failed to write updated highlights.json")

    payload = {"job_id": job_id, "deleted_clip_id": clip_id}
    payload.update(raw)
    return JSONResponse(payload)
