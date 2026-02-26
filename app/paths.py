from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_TMP_DIR = STORAGE_DIR / "tmp"
VIDEOS_DIR = STORAGE_DIR / "videos"
STATIC_DIR = BASE_DIR / "app" / "static"
FONTS_DIR = STATIC_DIR / "fonts"

for p in (STORAGE_DIR, UPLOAD_TMP_DIR, VIDEOS_DIR, FONTS_DIR):
    p.mkdir(parents=True, exist_ok=True)
