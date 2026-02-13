import subprocess
from pathlib import Path
from typing import Literal, Optional
from app.utils.subtitles import escape_path_for_ffmpeg

Aspect = Literal["1:1", "16:9", "9:16"]


def export_with_aspect(input_path: str, output_path: str, aspect: Aspect = "9:16", subtitle_path: Optional[str] = None) -> str:
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Choose a standard canvas for each aspect (even dimensions)
    canvas = {
        "1:1": (1080, 1080),
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
    }
    if aspect not in canvas:
        raise ValueError("invalid aspect")
    W, H = canvas[aspect]

    # Scale to fit inside canvas, then pad to exact canvas
    base_vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"

    if subtitle_path:
        subs_escaped = escape_path_for_ffmpeg(subtitle_path)
        # Use the 'ass' filter for ASS subtitles
        vf = f"{base_vf},ass={subs_escaped}"
    else:
        vf = base_vf

    def _run_with(codec: str):
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-vf", vf,
            "-c:v", codec,
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            str(dst),
        ]
        subprocess.run(cmd, check=True)

    try:
        _run_with("libx264")
    except subprocess.CalledProcessError:
        try:
            _run_with("h264")
        except subprocess.CalledProcessError:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(src),
                "-vf", vf,
                "-c:a", "aac",
                str(dst),
            ]
            subprocess.run(cmd, check=True)

    return str(dst)
