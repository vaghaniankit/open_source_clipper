from pathlib import Path
from typing import List, Dict


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _wrap_text(text: str, max_len: int = 42) -> str:
    """Very simple word-wrapping so subtitles stay to 1–2 short lines."""
    words = text.split()
    if not words:
        return ""
    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        # +1 for space when there is already content
        add_len = len(w) + (1 if cur else 0)
        if cur_len + add_len > max_len and cur:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add_len
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


MIN_SEG_ENERGY = 0.008  # slightly lower so normal speech is less likely to be treated as music
MIN_WORD_CONFIDENCE = 0.35  # suppress hallucinated words when Whisper is unsure
VERY_LOW_ENERGY = 0.004  # only this low is treated as near-silence for word-level suppression


def build_clip_srt(segments: List[Dict], clip_start: float, clip_end: float, clip: Dict | None = None) -> str:
    """Build an SRT subtitle track for a clip using **word-level** timings.

    - Prefer the exact transcript slice corresponding to the clip's
      start_idx/end_idx when available.
    - Flatten words from those segments and group them into short
      time windows so captions follow the speech closely.
    """
    # Decide which segments to consider: use clip's index range if present.
    chosen_segments: List[Dict]
    if clip is not None and "start_idx" in clip and "end_idx" in clip:
        try:
            si = int(clip.get("start_idx", 0))
            ei = int(clip.get("end_idx", si))
            si = max(0, si)
            ei = min(len(segments) - 1, ei)
            chosen_segments = segments[si : ei + 1]
        except Exception:
            chosen_segments = segments
    else:
        chosen_segments = segments

    # Ensure segments are processed in chronological order; some upstream
    # generators emit overlapping chunks out of time order.
    chosen_segments = sorted(chosen_segments, key=lambda s: float(s.get("start", 0.0)))

    # Collect words inside the clip window
    word_windows = []
    for seg in chosen_segments:
        try:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
        except Exception:
            continue
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        seg_energy = float(seg.get("energy", 0.0) or 0.0)
        words = seg.get("words") or []
        if not words:
            # fallback: use the whole segment text
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            # For segments without word timings, only drop if energy is extremely low
            if seg_energy < VERY_LOW_ENERGY:
                continue
            # Clamp segment to the clip window and convert to *relative* times so
            # subtitles line up with the cut clip rather than the full video.
            s_abs = max(seg_start, clip_start)
            e_abs = min(seg_end, clip_end)
            if e_abs <= s_abs:
                continue
            s_rel = s_abs - clip_start
            e_rel = e_abs - clip_start
            if e_rel <= s_rel:
                continue
            word_windows.append({"start": s_rel, "end": e_rel, "text": text})
            continue

        for w in words:
            try:
                ws = float(w.get("start", seg_start))
                we = float(w.get("end", seg_end))
            except Exception:
                continue
            if we <= clip_start or ws >= clip_end:
                continue
            ws_rel = max(ws, clip_start) - clip_start
            we_rel = min(we, clip_end) - clip_start
            if we_rel <= ws_rel:
                continue
            text = (w.get("word") or "").strip()
            if not text:
                continue
            confidence = w.get("confidence")
            # Only suppress when both energy is extremely low and confidence is low
            if confidence is not None and seg_energy < VERY_LOW_ENERGY and confidence < MIN_WORD_CONFIDENCE:
                continue
            word_windows.append({"start": ws_rel, "end": we_rel, "text": text})

    if not word_windows:
        return ""

    # Sort by word start time
    word_windows.sort(key=lambda w: w["start"])

    # Group consecutive words into very small subtitles so captions follow
    # the speech word-by-word.
    MIN_SUB_DURATION = 0.1
    MAX_SUB_DURATION = 1.0
    MAX_WORDS_PER_SUB = 1

    groups = []
    cur = None
    for w in word_windows:
        if cur is None:
            cur = {"start": w["start"], "end": w["end"], "words": [w["text"]]}
            continue

        proposed_start = cur["start"]
        proposed_end = max(cur["end"], w["end"])
        proposed_dur = proposed_end - proposed_start
        word_count = len(cur["words"]) + 1

        if proposed_dur <= MAX_SUB_DURATION and word_count <= MAX_WORDS_PER_SUB:
            cur["end"] = proposed_end
            cur["words"].append(w["text"])
        else:
            if cur["end"] - cur["start"] >= MIN_SUB_DURATION:
                groups.append(cur)
            cur = {"start": w["start"], "end": w["end"], "words": [w["text"]]}

    if cur is not None and cur["end"] - cur["start"] >= MIN_SUB_DURATION:
        groups.append(cur)

    if not groups:
        return ""

    # Build SRT entries from word-groups
    lines: List[str] = []
    idx = 1
    for g in groups:
        text = " ".join(g["words"]).strip()
        if not text:
            continue
        wrapped = _wrap_text(text)
        if not wrapped:
            continue
        start_ts = _format_srt_timestamp(g["start"])
        end_ts = _format_srt_timestamp(g["end"])
        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(wrapped)
        lines.append("")
        idx += 1

    return "\n".join(lines).strip() + ("\n" if lines else "")


def write_clip_srt(transcript_path: Path, clip: Dict, out_dir: Path) -> Path:
    """Load transcript.json and write an SRT file for the given clip.

    Returns the path to the written SRT file. If no usable segments, an empty
    SRT file is still created so ffmpeg has a valid input.
    """
    import json

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = data or []
    clip_start = float(clip.get("start", 0.0))
    clip_end = float(clip.get("end", 0.0))
    clip_id = clip.get("id") or "clip"
    srt_name = f"{clip_id}.srt"
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / srt_name

    srt_text = build_clip_srt(segments, clip_start, clip_end, clip=clip)
    srt_path.write_text(srt_text, encoding="utf-8")
    return srt_path


def escape_path_for_ffmpeg(path: str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter expression.

    - Use forward slashes
    - Escape drive colon (C: -> C\:)
    - Wrap in single quotes so spaces are handled.
    """
    p = path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + r"\:" + p[2:]
    return f"'{p}'"
