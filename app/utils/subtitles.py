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


def build_clip_srt_and_ass(
    segments: List[Dict], 
    clip_start: float, 
    clip_end: float, 
    clip: Dict | None = None,
    highlight_color: str = "&H0000FF&"     # Red
) -> (str, str):
    """Build SRT (for burnt-in display) and ASS (for karaoke effect) subtitle tracks.
    - SRT will contain full sentences.
    - ASS will contain word-by-word highlighting.
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

    # Ensure segments are processed in chronological order
    chosen_segments = sorted(chosen_segments, key=lambda s: float(s.get("start", 0.0)))

    # Collect all words and tags within the clip's time window
    all_items = []
    for seg in chosen_segments:
        try:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
        except Exception:
            continue
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        # Add words
        words = seg.get("words") or []
        for w in words:
            try:
                ws = float(w.get("start", seg_start))
                we = float(w.get("end", seg_end))
            except Exception:
                continue
            if we <= clip_start or ws >= clip_end:
                continue
            
            text = (w.get("word") or "").strip()
            if not text:
                continue
            
            all_items.append({
                "start": ws,
                "end": we,
                "text": text,
                "type": "word"
            })

        # Add tags as separate items
        tags = seg.get("tags") or []
        if tags:
            all_items.append({
                "start": seg_start,
                "end": seg_end,
                "text": f"[{', '.join(tags)}]",
                "type": "tag"
            })

    # Sort all items by start time
    all_items.sort(key=lambda x: x['start'])
    
    if not all_items:
        return "", ""

    # Assign all_items to all_words for the rest of the function
    all_words = all_items

    # Build SRT content (sentence-level)
    srt_lines = []
    srt_idx = 1
    
    # Simple sentence segmentation: split by periods, question marks, etc.
    sentence_ends = {'.', '?', '!'}
    
    current_sentence = []
    sentence_start_time = -1.0

    for i, item in enumerate(all_items):
        if item["type"] == "tag":
            if current_sentence:
                # End the current sentence before the tag
                full_sentence = " ".join(current_sentence)
                sentence_end_time = all_items[i-1]["end"]
                
                s_rel = max(sentence_start_time, clip_start) - clip_start
                e_rel = min(sentence_end_time, clip_end) - clip_start
                
                if e_rel > s_rel:
                    start_ts = _format_srt_timestamp(s_rel)
                    end_ts = _format_srt_timestamp(e_rel)
                    
                    srt_lines.append(str(srt_idx))
                    srt_lines.append(f"{start_ts} --> {end_ts}")
                    srt_lines.append(_wrap_text(full_sentence))
                    srt_lines.append("")
                    srt_idx += 1
                
                current_sentence = []

            # Add the tag as a standalone subtitle line
            s_rel = max(item["start"], clip_start) - clip_start
            e_rel = min(item["end"], clip_end) - clip_start
            if e_rel > s_rel:
                start_ts = _format_srt_timestamp(s_rel)
                end_ts = _format_srt_timestamp(e_rel)
                
                srt_lines.append(str(srt_idx))
                srt_lines.append(f"{start_ts} --> {end_ts}")
                srt_lines.append(item["text"])
                srt_lines.append("")
                srt_idx += 1
            continue

        if not current_sentence:
            sentence_start_time = item["start"]
        
        current_sentence.append(item["text"])
        
        is_last_item = (i == len(all_items) - 1)
        word_text = item["text"]
        
        # Check if this word ends a sentence
        if any(word_text.endswith(p) for p in sentence_ends) or is_last_item:
            full_sentence = " ".join(current_sentence)
            sentence_end_time = item["end"]
            
            # Align times to clip window
            s_rel = max(sentence_start_time, clip_start) - clip_start
            e_rel = min(sentence_end_time, clip_end) - clip_start
            
            if e_rel > s_rel:
                start_ts = _format_srt_timestamp(s_rel)
                end_ts = _format_srt_timestamp(e_rel)
                
                srt_lines.append(str(srt_idx))
                srt_lines.append(f"{start_ts} --> {end_ts}")
                srt_lines.append(_wrap_text(full_sentence))
                srt_lines.append("")
                srt_idx += 1
            
            # Reset for next sentence
            current_sentence = []
            sentence_start_time = -1.0

    srt_content = "\n".join(srt_lines)
    
    # --- Build ASS content (karaoke-style) ---
    ass_header = f"""[Script Info]
Title: Karaoke Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_lines = [ass_header]
    
    # Group words into lines for ASS subtitles
    ass_lines = [ass_header]
    line_words = []
    line_start_time = -1

    # Filter out tags for karaoke effect
    words_only = [item for item in all_items if item["type"] == "word"]

    for i, word in enumerate(words_only):
        if line_start_time < 0:
            line_start_time = word['start']

        line_words.append(word)

        # Check for pause after the current word or if the line is too long
        is_last_word = (i == len(words_only) - 1)
        force_break = False
        if not is_last_word:
            next_word = words_only[i+1]
            pause = next_word['start'] - word['end']
            if pause >= 0.5: # PAUSE_THRESHOLD
                force_break = True
        
        if is_last_word or force_break or len(line_words) >= 8: # MAX_LINE_WORDS
            # Create a dialogue line for the current group of words
            line_end_time = line_words[-1]['end']
            start_ts = _format_ass_timestamp(line_start_time - clip_start)
            end_ts = _format_ass_timestamp(line_end_time - clip_start)
            
            text_line = f"{{\\c{highlight_color}}}"
            for w in line_words:
                duration_centiseconds = max(1, int((w["end"] - w["start"]) * 100))
                text_line += f"{{\\k{duration_centiseconds}}}{w['text']} "

            dialogue = f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text_line.strip()}"
            ass_lines.append(dialogue)

            # Reset for the next line
            line_words = []
            line_start_time = -1

    # Add any remaining words as a final line
    if line_words:
        line_end_time = line_words[-1]['end']
        start_ts = _format_ass_timestamp(line_start_time - clip_start)
        end_ts = _format_ass_timestamp(line_end_time - clip_start)
        
        text_line = ""
        for w in line_words:
            duration_centiseconds = max(1, int((w["end"] - w["start"]) * 100))
            text_line += f"{{\\k{duration_centiseconds}}}{w['text']} "
            
        dialogue = f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text_line.strip()}"
        ass_lines.append(dialogue)

    ass_content = "\n".join(ass_lines)
    
    return srt_content, ass_content




def _format_ass_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    # ASS format is H:MM:SS.cc (centiseconds)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{ms // 10:02d}"


def write_clip_subtitles(transcript_path: Path, clip: Dict, out_dir: Path) -> (Path, Path):
    """Load transcript.json and write SRT and ASS files for the given clip.
    Returns the paths to the written SRT and ASS files.
    """
    import json

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = data or []
    clip_start = float(clip.get("start", 0.0))
    clip_end = float(clip.get("end", 0.0))
    clip_id = clip.get("id") or "clip"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_name = f"{clip_id}.srt"
    ass_name = f"{clip_id}.ass"
    srt_path = out_dir / srt_name
    ass_path = out_dir / ass_name

    srt_text, ass_text = build_clip_srt_and_ass(segments, clip_start, clip_end, clip=clip)
    
    srt_path.write_text(srt_text, encoding="utf-8")
    ass_path.write_text(ass_text, encoding="utf-8")
    
    return srt_path, ass_path



def escape_path_for_ffmpeg(path: str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter expression.

    - Use forward slashes
    - Escape drive colon (C: -> C\:)
    - Wrap in single quotes so spaces are handled.
    """
    p = str(path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + r"\:" + p[2:]
    return f"'{p}'"
