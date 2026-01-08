import os
from dataclasses import dataclass
from pathlib import Path

try:
    # Load .env from project root if available
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
except Exception:
    # dotenv is optional; ignore if not installed
    pass


@dataclass(frozen=True)
class Settings:
    # Redis / Memurai
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    REDIS_BACKEND: str = os.environ.get("REDIS_BACKEND") or os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # Sessions
    SESSION_SECRET: str = os.environ.get("SESSION_SECRET", "dev-secret")

    # Google OAuth
    GOOGLE_CLIENT_ID: str = os.environ.get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    GOOGLE_REDIRECT_URI: str = os.environ.get("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/callback")

    # YouTube Downloader Configuration
    YOUTUBE_COOKIE_FILE: str = os.environ.get("YOUTUBE_COOKIE_FILE", "")

settings = Settings()
