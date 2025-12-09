# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# ffmpeg: for video processing
# libsndfile1: for audio processing (librosa)
# git: if needed for installing dependencies from git
# redis-server: Celery broker/backend
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=900
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir supervisor itsdangerous

# Copy project
COPY . .

# Create storage directories
RUN mkdir -p storage/videos storage/audio storage/pipeline storage/exports storage/data downloads

# Expose the port
EXPOSE 8000

# Copy supervisor config
COPY docker/supervisord.conf /etc/supervisor/supervisord.conf

ENV REDIS_URL=redis://localhost:6379/0 \
    REDIS_BACKEND=redis://localhost:6379/0

# Command to run the application stack (Redis + Celery worker + Celery beat + Uvicorn)
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
