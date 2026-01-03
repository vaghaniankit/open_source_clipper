# Tech Stack and External APIs

This document outlines the tools, technologies, and external APIs used in this project.

## Backend

- **Programming Language:** Python 3.10
- **Framework:** FastAPI
- **Web Server:** Gunicorn with Uvicorn workers
- **Task Queue:** Celery
- **Message Broker:** Redis
- **Authentication:** Authlib

## Frontend

The frontend appears to be server-side rendered using a templating engine, as there is no `package.json` file to indicate the use of a JavaScript framework.

## Key Python Libraries

- **`openai`**: Interacts with the OpenAI API for video highlight generation.
- **`yt-dlp`**: Downloads videos from YouTube.
- **`ffmpeg-python`**: A Python wrapper for the FFmpeg multimedia framework.
- **`faster-whisper`**: A reimplementation of OpenAI's Whisper model for speech-to-text transcription.
- **`opencv-python`**: A computer vision library used for video processing.
- **`mediapipe`**: A cross-platform, customizable ML solutions for live and streaming media.
- **`tensorflow`**: An end-to-end open source platform for machine learning.
- **`torch`**: An open source machine learning framework that accelerates the path from research prototyping to production deployment.
- **`librosa`**: A python package for music and audio analysis.
- **`scenedetect`**: A library for detecting scene changes in videos.

## External APIs

- **OpenAI API**: Used for generating video highlights. The `gpt-4o-mini` model is used to analyze video transcripts and identify interesting segments.
- **YouTube**: Videos are downloaded from YouTube using `yt-dlp`. While this is not a direct API integration, it is an external service that the application relies on.
