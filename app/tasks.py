import os
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional

from .celery_app import celery_app

# Default transcription model can be overridden via TRANSCRIBE_MODEL env var
DEFAULT_TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "medium")

# Simple tasks. Replace with real implementations / orchestrator later.

@celery_app.task(bind=True)
def download_youtube_task(self, url: str, filename: Optional[str] = None):
    """Download YouTube video using yt_dlp into storage/videos.
    Returns local path.
    """
    base = Path(__file__).resolve().parent.parent
    videos_dir = base / "storage" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = base / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # Run download (into downloads/ via existing script), then move to videos/
    orig_cwd = os.getcwd()
    try:
        os.chdir(base)
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        from .services.youtube_downloader import download_youtube
        safe_name = filename or None
        download_youtube(url, choice="video", quality="best", filename=safe_name)
        # If filename provided, expect downloads/<filename>.mp4 else unknown title
        if safe_name:
            src = base / "downloads" / f"{safe_name}.mp4"
        else:
            # fallback: pick the newest mp4 from downloads
            cand = sorted((base / "downloads").glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cand:
                raise RuntimeError("download produced no mp4")
            src = cand[0]
        dst = videos_dir / src.name
        src.replace(dst)
        # Return plain path so this task can be chained directly into
        # orchestrate_pipeline_task, which expects a path string.
        return str(dst)
    finally:
        os.chdir(orig_cwd)

@celery_app.task(bind=True)
def process_video_task(self, path: str):
    """Stub processing task: extract audio then return paths. Replace with orchestrator."""
    base = Path(__file__).resolve().parent.parent
    orig_cwd = os.getcwd()
    try:
        os.chdir(base)
        out_path = base / "storage" / "videos" / (Path(path).stem + "_vocals.wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "32k",
            str(out_path),
        ]
        self.update_state(state="STARTED", meta={"stage": "extract_audio", "progress": 10})
        subprocess.run(cmd, check=True)
        return {"audio_path": str(out_path)}
    finally:
        os.chdir(orig_cwd)

@celery_app.task(bind=True)
def orchestrate_pipeline_task(
    self,
    path: str,
    chunk_seconds: int = 30,
    overlap_seconds: int = 2,
    model_size: str = DEFAULT_TRANSCRIBE_MODEL,
    prompt: Optional[str] = None,
    duration_preset: Optional[str] = None,
    timeframe_percent: Optional[int] = None,
):
    """End-to-end pipeline: extract audio -> chunk -> transcribe -> enrich -> highlights.

    Returns paths and an enriched transcript summary used for highlight generation.
    """
    base = Path(__file__).resolve().parent.parent
    storage = base / "storage"
    job_id = getattr(getattr(self, "request", None), "id", None) or Path(path).stem
    job_dir = storage / "pipeline" / str(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Persist simple job metadata (prompt, duration preset, original path)
    meta_path = job_dir / "job_meta.json"
    try:
        meta = {
            "path": str(path),
            "chunk_seconds": int(chunk_seconds),
            "overlap_seconds": int(overlap_seconds),
            "model_size": model_size,
            "prompt": prompt,
            "duration_preset": duration_preset,
            "timeframe_percent": int(timeframe_percent) if timeframe_percent is not None else None,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        # Non-critical; continue even if metadata write fails
        pass

    # 1) Extract audio via ffmpeg directly to avoid import issues
    audio_out = job_dir / "audio.wav"

    # If timeframe_percent is provided (0-100), limit the audio duration to that
    # percentage of the source video length.
    ffmpeg_cmd = ["ffmpeg", "-y"]
    try:
        if timeframe_percent is not None:
            try:
                tf = max(1, min(100, int(timeframe_percent)))
            except Exception:
                tf = 100
            # Probe video duration via ffprobe
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            total_dur = float(probe.stdout.strip() or 0) if probe.stdout else 0.0
            if total_dur > 0:
                limit = total_dur * tf / 100.0
                ffmpeg_cmd += ["-t", f"{limit:.3f}"]
    except Exception:
        # If ffprobe/limiting fails, fall back to full duration
        pass

    ffmpeg_cmd += [
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "32k",
        str(audio_out),
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    audio_path = str(audio_out)

    # 2) Chunk
    # Import local pipeline utilities
    from .utils.audio import chunk_audio, transcribe_chunks, compute_segment_energy
    from .utils.analysis import (
        detect_scene_cuts,
        attach_scene_metadata,
        analyze_audio_events,
        attach_audio_tags,
        compute_excitement_score,
    )

    chunks_dir = job_dir / "chunks"
    chunk_files = chunk_audio(Path(audio_path), chunks_dir, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds)
    self.update_state(state="STARTED", meta={"stage": "chunk_audio", "progress": 30})

    # 3) Transcribe
    tr = transcribe_chunks(chunk_files, model_size=model_size)
    self.update_state(state="STARTED", meta={"stage": "transcribe", "progress": 50})
    (job_dir / "transcript.txt").write_text(tr.get("transcript", ""), encoding="utf-8")

    # 4) Enrich structured transcript JSON for highlight generation
    segments = tr.get("segments", []) or []
    # basic audio energy per segment
    segments = compute_segment_energy(audio_path, segments)

    # scene / shot changes from the source video
    try:
        cut_times = detect_scene_cuts(str(path))
    except Exception:
        cut_times = []
    segments = attach_scene_metadata(segments, cut_times)

    # lightweight audio event tags (music / laughter) from the audio
    try:
        events = analyze_audio_events(audio_path)
    except Exception:
        events = []
    segments = attach_audio_tags(segments, events)

    # aggregate excitement score combining energy, cuts and tags
    segments = compute_excitement_score(segments)
    self.update_state(state="STARTED", meta={"stage": "enrich_transcript", "progress": 70})

    transcript_json_path = job_dir / "transcript.json"
    with open(transcript_json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # 5) Generate highlights in the same job
    from .services.highlight import generate_highlights, map_duration_preset

    # Resolve clip duration bounds from the UI preset (per job)
    min_sec, max_sec = map_duration_preset(duration_preset)

    highlights_result = generate_highlights(
        input_video=str(path),
        transcript_path=str(transcript_json_path),
        job_dir=str(job_dir),
        min_sec=min_sec,
        max_sec=max_sec,
        user_prompt=prompt,
    )

    self.update_state(state="STARTED", meta={"stage": "highlights", "progress": 90})

    return {
        "job_id": str(job_id),
        "audio_path": str(audio_path),
        "chunks": [str(p) for p in chunk_files],
        "transcript_path": str(job_dir / "transcript.txt"),
        "segments": segments,
        "highlights_path": highlights_result.get("highlights_path"),
        "clips": highlights_result.get("clips", []),
    }


@celery_app.task(bind=True)
def export_video_task(self, path: str, aspect: str = "9:16"):
    """Export the given video path to the requested aspect ratio, padding as needed.
    Returns the output path.
    """
    base = Path(__file__).resolve().parent.parent
    storage = base / "storage"
    job_id = getattr(getattr(self, "request", None), "id", None) or Path(path).stem
    print('\n\n➡ app/tasks.py:248 job_id:', job_id)
    out_dir = storage / "exports" / str(job_id)
    print('\n\n➡ app/tasks.py:249 out_dir:', out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename
    src = Path(path)
    suffix = src.suffix or ".mp4"
    name = f"{src.stem}_{aspect.replace(':', 'x')}" + suffix
    print('\n\nXXXX➡ app/tasks.py:255 name:', name)
    out_path = out_dir / name
    print('\n\nXXXX➡ app/tasks.py:256 out_path:', out_path)

    # Perform export
    from .utils.export import export_with_aspect
    self.update_state(state="STARTED", meta={"stage": "export_video", "progress": 50})
    result_path = export_with_aspect(str(src), str(out_path), aspect=aspect)
    return {"output_path": result_path}


@celery_app.task(bind=True)
def export_clip_task(self, job_id: str, clip: dict, aspect: str = "9:16"):
    """Export a single highlight clip to the requested aspect ratio.

    - job_id: pipeline job that produced highlights.json
    - clip: clip dict containing start, end, id, filename, etc.
    - aspect: "1:1", "16:9", or "9:16"
    """
    base = Path(__file__).resolve().parent.parent
    storage = base / "storage"
    job_dir = storage / "pipeline" / str(job_id)

    # Load highlights.json to get the source video path (authoritative)
    import json

    highlights_path = job_dir / "highlights.json"
    if not highlights_path.exists():
        raise RuntimeError("highlights.json not found for job_id")

    data = json.loads(highlights_path.read_text(encoding="utf-8"))
    video_path = data.get("video_path")
    if not video_path:
        raise RuntimeError("video_path missing in highlights.json")

    src = Path(video_path)
    if not src.exists():
        raise RuntimeError(f"source video not found: {src}")

    # Build output directory under storage/exports/<job_id>
    out_dir = storage / "exports" / str(job_id) / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use clip filename if present, otherwise derive one
    clip_id = clip.get("id") or "clip"
    base_name = clip.get("filename") or f"{clip_id}.mp4"
    name = f"{Path(base_name).stem}_{aspect.replace(':', 'x')}{src.suffix or '.mp4'}"
    raw_clip_path = out_dir / name

    start = float(clip.get("start", 0.0))
    end = float(clip.get("end", 0.0))
    if end <= start:
        raise RuntimeError("invalid clip times")

    # First cut the time range to a temporary file, then apply aspect export
    tmp_cut = out_dir / f"{Path(base_name).stem}_cut_tmp{src.suffix or '.mp4'}"

    cut_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(src),
        "-c:v", "libx264", "-c:a", "aac",
        str(tmp_cut),
    ]
    self.update_state(state="STARTED", meta={"stage": "cut_clip", "progress": 30})
    subprocess.run(cut_cmd, check=True)

    # Prepare subtitles for this clip using the pipeline transcript.json
    transcript_json = job_dir / "transcript.json"
    subtitle_path: Optional[str] = None
    ass_path: Optional[str] = None
    if transcript_json.exists():
        try:
            from .utils.subtitles import write_clip_subtitles

            srt_path, ass_path_obj = write_clip_subtitles(transcript_json, clip, out_dir)
            subtitle_path = str(srt_path)  # Keep using SRT for burnt-in subs
            ass_path = str(ass_path_obj)
        except Exception as e:
            print(f"⚠️ Subtitle generation failed: {e}")
            subtitle_path = None
            ass_path = None

    from .utils.export import export_with_aspect

    self.update_state(state="STARTED", meta={"stage": "export_clip", "progress": 80})
    result_path = export_with_aspect(
        str(tmp_cut), 
        str(raw_clip_path), 
        aspect=aspect, 
        subtitle_path=subtitle_path,
        ass_path=str(ass_path) if ass_path else None
    )

    # Optionally remove tmp_cut later; for now, keep for debugging
    return {
        "job_id": str(job_id),
        "clip_id": clip_id,
        "aspect": aspect,
        "output_path": result_path,
    }


@celery_app.task(bind=True)
def speaker_center_video_task(self, path: str):
    """Apply speaker centering to a video and save under storage/exports/<job_id>.

    This expects that the input path points to an existing video file (for example,
    an already-exported highlight clip). The output will keep the same filename
    with a `_centered` suffix.
    """
    base = Path(__file__).resolve().parent.parent
    storage = base / "storage"
    src = Path(path)
    if not src.exists():
        raise RuntimeError(f"source video not found: {src}")

    job_id = getattr(getattr(self, "request", None), "id", None) or src.stem
    out_dir = storage / "exports" / str(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build centered output filename
    suffix = src.suffix or ".mp4"
    name = f"{src.stem}_centered" + suffix
    out_path = out_dir / name

    from .services.speaker_centering import center_speaker

    ok = center_speaker(str(src), str(out_path))
    if not ok:
        raise RuntimeError("speaker centering failed")

    return {
        "job_id": str(job_id),
        "input_path": str(src),
        "output_path": str(out_path),
    }


@celery_app.task(bind=True)
def generate_highlights_task(self, video_path: str, transcript_path: str):
    """Generate highlight clips metadata using the LLM-based pipeline.

    This task expects that a transcript JSON already exists for the given video
    (for example produced by orchestrate_pipeline_task). It will write
    highlights.json into storage/pipeline/<job_id>/ and return the clips.
    """
    base = Path(__file__).resolve().parent.parent
    storage = base / "storage"
    job_id = getattr(getattr(self, "request", None), "id", None) or Path(video_path).stem
    job_dir = storage / "pipeline" / str(job_id)

    # Ensure job dir exists and resolve transcript path
    job_dir.mkdir(parents=True, exist_ok=True)
    transcript_abs = Path(transcript_path)
    if not transcript_abs.is_absolute():
        transcript_abs = job_dir / transcript_abs

    from .services.highlight import generate_highlights

    result = generate_highlights(
        input_video=str(video_path),
        transcript_path=str(transcript_abs),
        job_dir=str(job_dir),
    )
    return {
        "job_id": str(job_id),
        "highlights_path": result.get("highlights_path"),
        "clips": result.get("clips", []),
    }
