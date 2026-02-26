#!/usr/bin/env python3
"""
Apply karaoke subtitles to highlight clips using aligned_transcript.json.
"""
import os
import json
import subprocess
import sys

def make_karaoke_ass_from_words(words, ass_file, font="Arial", font_size=20):
    """
    Create ASS subtitles with per-word karaoke highlighting:
    - Active word has a green background + black text + slight scale animation
    - Inactive words are clean white with thin black border
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
            "1,2,0,"             # BorderStyle=1, Outline=2, Shadow=0
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
                        # Active word: Green Box effect
                        # Black text (\1c&H000000&) + Thick Green Border (\3c&H0000FF00&)
                        parts.append(
                            "{"
                            "\\1c&H000000&"          # Black Text
                            "\\3c&H0000FF00&"        # Bright Green Border (BGR: 00 FF 00)
                            "\\bord10"               # Thick border to look like a box
                            "\\blur3"                # Slight blur for softer edges
                            "\\fscx105\\fscy105"     # Subtle scale up
                            "\\t(0,120,\\fscx110\\fscy110)"  # Slight scale animation
                            "}" + w + "{\\r}"
                        )
                    else:
                        # Inactive word: Clean White
                        # White text + Thin Black Border
                        parts.append(
                            "{"
                            "\\1c&HFFFFFF&"          # White Text
                            "\\3c&H00000000&"        # Black Border
                            "\\bord2"                # Thin border
                            "\\blur0"
                            "}" + w + "{\\r}"
                        )

                text = " ".join(parts)
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")


def format_ass_time(seconds):
    """Convert seconds to ASS time format (H:MM:SS.cc)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def words_for_segment(words, start, end):
    """Get words within a time segment."""
    return [w for w in words if start <= w.get("start", 0) <= end]

def apply_karaoke_to_clip(clip_path, highlights_json_path, aligned_transcript_path):
    """Apply karaoke subtitles to a single clip."""
    # Load highlights to get clip timing
    with open(highlights_json_path, "r", encoding="utf-8") as f:
        highlights = json.load(f)
    
    # Find matching clip
    clip_name = os.path.basename(clip_path)
    clip_info = None
    for clip in highlights.get("clips", []):
        if clip.get("filename") in clip_name or clip_name.endswith(clip.get("filename", "")):
            clip_info = clip
            break
    
    if not clip_info:
        print(f"⚠️  Could not find clip info for {clip_name}")
        return False
    
    clip_start = clip_info.get("start", 0)
    clip_end = clip_info.get("end", 0)
    
    # Load aligned transcript
    with open(aligned_transcript_path, "r", encoding="utf-8") as f:
        aligned = json.load(f)
    
    # Collect all words
    all_words = []
    for seg in aligned:
        for word in seg.get("words", []):
            all_words.append({
                "word": word.get("word", "").strip(),
                "start": word.get("start", 0),
                "end": word.get("end", 0)
            })
    
    # Get words for this clip
    clip_words = words_for_segment(all_words, clip_start, clip_end)
    
    if not clip_words:
        print(f"⚠️  No words found for clip {clip_name}")
        return False
    
    # Adjust word timestamps relative to clip start
    adjusted_words = []
    for w in clip_words:
        adjusted_words.append({
            "word": w["word"],
            "start": w["start"] - clip_start,
            "end": w["end"] - clip_start
        })
    
    # Create ASS file
    ass_file = clip_path.replace(".mp4", ".ass")
    make_karaoke_ass_from_words(adjusted_words, ass_file)
    
    # Burn subtitles into video
    output_path = clip_path.replace(".mp4", "_karaoke.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", clip_path,
        "-vf", f"subtitles={ass_file}",
        "-c:a", "copy",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  ✅ {os.path.basename(output_path)}")
        return True
    except subprocess.CalledProcessError:
        print(f"  ⚠️  Failed: {clip_path}")
        return False

if __name__ == "__main__":
    highlights_dir = "storage/highlights"
    highlights_json = "highlights.json"
    aligned_transcript = "storage/data/aligned_transcript.json"
    
    if not os.path.exists(highlights_dir):
        print(f"❌ {highlights_dir} directory not found")
        sys.exit(1)
    
    if not os.path.exists(highlights_json):
        print(f"❌ {highlights_json} not found. Run highlight generator first.")
        sys.exit(1)
    
    if not os.path.exists(aligned_transcript):
        print(f"❌ {aligned_transcript} not found. Run transcription first.")
        sys.exit(1)
    
    video_files = [f for f in os.listdir(highlights_dir) 
                  if f.endswith(".mp4") and not f.endswith("_karaoke.mp4")]
    
    if not video_files:
        print(f"❌ No video files found in {highlights_dir}")
        sys.exit(1)
    
    print(f"🎤 Applying karaoke subtitles to {len(video_files)} clips...")
    
    for video_file in video_files:
        clip_path = os.path.join(highlights_dir, video_file)
        print(f"  Processing: {video_file}...")
        apply_karaoke_to_clip(clip_path, highlights_json, aligned_transcript)
    
    print("\n✅ Karaoke subtitles applied!")

