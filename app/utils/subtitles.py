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


def _wrap_text(text: str, max_len: int = 10) -> str:
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
    print('\n\n XXXX ➡ app/utils/subtitles.py:201 srt_content:', srt_content)
    
    # --- Build ASS content (karaoke-style) ---
    ass_header = """[Script Info]
Title: Karaoke Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,150,150,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_lines = [ass_header]
    
    # Filter out tags for karaoke effect
    words_only = [item for item in all_items if item["type"] == "word"]
    print('\n\n XXXX ➡ app/utils/subtitles.py:222 words_only:', words_only)

    # Adjust word timestamps to be relative to the clip start
    adjusted_words = []
    for w in words_only:
        adjusted_words.append({
            "word": w["text"],
            "start": w["start"] - clip_start,
            "end": w["end"] - clip_start
        })

    print('\n\n XXXX ➡ app/utils/subtitles.py:232 adjusted_words:', adjusted_words)
    
    # Group words into subtitle lines
    lines = []
    if adjusted_words:
        current_line = [adjusted_words[0]]
        for i in range(1, len(adjusted_words)):
            # new line on pause
            if adjusted_words[i]["start"] - adjusted_words[i-1]["end"] > 0.5:
                lines.append(current_line)
                current_line = []
            current_line.append(adjusted_words[i])
        lines.append(current_line)

    # Generate per-word karaoke events
    for line in lines:
        line_words_text = [w["word"] for w in line]

        for i, active_word in enumerate(line):
            start_time = _format_ass_timestamp(active_word["start"])
            end_time = _format_ass_timestamp(active_word["end"])

            parts = []
            for j, w_text in enumerate(line_words_text):
                if i == j:
                    # Active word: black text + yellow box + slight scale + soft edges
                    # parts.append(
                    #     "{"
                    #     "\\1c&H000000FF&"      # Black text
                    #     "\\4c&H0000FFFF&"      # Yellow background
                    #     "\\bord12"             # Box padding
                    #     "\\blur2"              # Soft edges
                    #     "\\fscx100\\fscy100"   # Base scale
                    #     "\\t(0,120,\\fscx110\\fscy110)"  # Slight scale animation
                    #     "}" + w_text + "{\\r}"
                    # )
                    
                    # Active word: smooth color + slight scale
                    parts.append(
                        "{"
                        "\\1c&H00FFFF00&"        # Yellow text
                        "\\fscx100\\fscy100"
                        # "\\t(0,120,\\fscx108\\fscy108)"  # Gentle pop
                        "}" + w_text + "{\\r}"
                    )
                    
                else:
                    # Inactive word: dim gray text, no background
                    parts.append("{\\1c&H888888FF&}" + w_text + "{\\r}")

            text = " ".join(parts)
            ass_lines.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}")

    ass_content = "\n".join(ass_lines)
    print('\n\n XXXX ➡ app/utils/subtitles.py:283 ass_content:', ass_content)
    print('\n\n XXXX ➡ app/utils/subtitles.py:284 srt_content:', srt_content)
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
    print('\n\n XXXX ➡ app/utils/subtitles.py:307 data:', data)
    segments = data or []
    print('\n\n XXXX ➡ app/utils/subtitles.py:309 segments:', segments)
    clip_start = float(clip.get("start", 0.0))
    clip_end = float(clip.get("end", 0.0))
    clip_id = clip.get("id") or "clip"
    print('\n\n XXXX ➡ app/utils/subtitles.py:313 clip_id:', clip_id)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_name = f"{clip_id}.srt"
    ass_name = f"{clip_id}.ass"
    srt_path = out_dir / srt_name
    ass_path = out_dir / ass_name

    srt_text, ass_text = build_clip_srt_and_ass(segments, clip_start, clip_end, clip=clip)
    print('\n\n XXXX ➡ app/utils/subtitles.py:323 srt_text:', srt_text)
    print('\n\n XXXX ➡ app/utils/subtitles.py:323 ass_text:', ass_text)
    srt_path.write_text(srt_text, encoding="utf-8")
    ass_path.write_text(ass_text, encoding="utf-8")
    
    print('\n\n  XXXX ➡ app/utils/subtitles.py:327 ass_path:', ass_path)
    print('\n\n  XXXX ➡ app/utils/subtitles.py:329 srt_path:', srt_path)
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


def make_karaoke_ass_from_words(words, ass_file, font="Arial", font_size=20):
    """
    Create ASS subtitles with per-word karaoke highlighting:
    - Active word has a yellow background + black text + slight scale animation
    - Inactive words are dimmed gray
    - Box padding and soft edges create a pill-style illusion
    """
    def ass_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:d}:{m:02d}:{s:05.2f}"

    with open(ass_file, "w", encoding="utf-8") as f:
        # --- Script Info ---
        f.write("""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
""")

        # --- Style ---
        f.write(
            f"Style: Default,{font},{font_size},"
            "&H00FFFFFF,"        # PrimaryColour → White (inactive default)
            "&H00FFFFFF,"        # SecondaryColour
            "&H00000000,"        # OutlineColour → Black
            "&H00000000,"        # BackColour → transparent by default
            "0,0,0,0,100,100,0,0,"
            "3,10,0,"            # BorderStyle=3, Outline=10 (padding)
            "2,150,150,150,1\n\n"    # Bottom-center, lifted up
        )

        # --- Events Header ---
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        # --- Group words into subtitle lines ---
        lines = []
        if words:
            current_line = [words[0]]
            for i in range(1, len(words)):
                if words[i]["start"] - words[i-1]["end"] > 0.5:  # new line on pause
                    lines.append(current_line)
                    current_line = []
                current_line.append(words[i])
            lines.append(current_line)

        # --- Generate per-word karaoke events ---
        for line in lines:
            line_words = [w["word"] for w in line]

            for i, active_word in enumerate(line):
                start = ass_time(active_word["start"])
                end = ass_time(active_word["end"])

                parts = []
                for j, w in enumerate(line_words):
                    if i == j:
                        # Active word: black text + yellow box + slight scale + soft edges
                        parts.append(
                            "{"
                            "\\1c&H000000FF&"      # Black text
                            "\\4c&H0000FFFF&"      # Yellow background
                            "\\bord12"             # Box padding
                            "\\blur2"              # Soft edges
                            "\\fscx100\\fscy100"   # Base scale
                            "\\t(0,120,\\fscx110\\fscy110)"  # Slight scale animation
                            "}" + w + "{\\r}"
                        )
                    else:
                        # Inactive word: dim gray text, no background
                        parts.append("{\\1c&H888888FF&}" + w + "{\\r}")

                text = " ".join(parts)
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
