import yt_dlp
import os


def download_youtube(url, choice="video", quality="best", filename=None):
    # Output format
    outtmpl = f"downloads/{filename}.%(ext)s" if filename else "downloads/%(title)s.%(ext)s"

    # Base yt-dlp options (we'll clone and tweak per-attempt)
    base_opts = {
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",  # always make proper MP4
        "http_headers": {
            # Reasonable desktop UA
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        },
        # Use Android client which is often more stable for 1080p/4k
        # "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }

    # If cookies.txt exists as a file, load it automatically
    # Check multiple locations: current dir, app root
    cookie_file = None
    if os.path.isfile("cookies.txt"):
        cookie_file = "cookies.txt"
    elif os.path.isfile("/app/cookies.txt"):
        cookie_file = "/app/cookies.txt"
    
    if cookie_file:
        base_opts["cookiefile"] = cookie_file
        print(f"🍪 Using cookies from {cookie_file} for authentication...")
    else:
        print("⚠️ No cookies.txt file found, or the path points to a directory. YouTube downloads may fail with 'Sign in' errors.")

    attempts = []

    if choice == "video":
        # Primary: generic best MP4 combo, letting yt-dlp choose
        fmt_primary = None
        if quality in ("best", "worst"):
            fmt_primary = quality
        else:
            fmt_primary = quality or "bv*+ba/best[ext=mp4]/best[ext=mp4]/best"

        attempts.append({"format": fmt_primary, "desc": "primary"})

        # Fallback 1: simple best MP4 if available
        attempts.append({"format": "best[ext=mp4]/best", "desc": "best-mp4"})

        # Fallback 2: well-known mp4 format code 18 or best
        attempts.append({"format": "18/best", "desc": "code-18-or-best"})

    elif choice == "audio":
        attempts.append({"format": "bestaudio/best", "desc": "audio"})

    # Configure audio post-processing if needed
    def _apply_audio_postproc(opts: dict):
        if choice == "audio":
            opts["postprocessors"] = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }]

    last_error = None
    for attempt in attempts:
        ydl_opts = dict(base_opts)
        ydl_opts["format"] = attempt["format"]
        _apply_audio_postproc(ydl_opts)
        try:
            print(f"▶️ Trying yt-dlp format '{attempt['format']}' ({attempt['desc']})...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return
        except Exception as e:
            last_error = e
            print(f"\n⚠️ yt-dlp attempt '{attempt['desc']}' failed: {e}")

    # If all attempts failed, re-raise the last error so the caller sees a clear failure
    if last_error is not None:
        raise last_error
