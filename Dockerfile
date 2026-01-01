# # Use the official NVIDIA CUDA base image for the build stage
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS build

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV DEBIAN_FRONTEND=noninteractive

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3-pip \
#     python3-dev \
#     ffmpeg \
#     libsndfile1 \
#     libgl1 \
#     git \
#     redis-server \
#     && rm -rf /var/lib/apt/lists/*

# # Set work directory
# WORKDIR /app

# # Copy the requirements file
# COPY requirements.in .

# # Install Python dependencies
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir pip-tools
# RUN python3 -m piptools compile requirements.in
# RUN python3 -m piptools sync requirements.txt

# # Copy the application code
# COPY . .

# RUN mkdir -p storage/videos storage/audio storage/pipeline storage/exports storage/data downloads

# # Expose the port
# EXPOSE 8000

# # --- Final Stage ---
# FROM python:3.10-slim AS final

# WORKDIR /app

# # Install system dependencies needed by the application
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ffmpeg \
#     libsndfile1 \
#     libgl1 \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Copy the application code and requirements.in
# COPY --from=build /app .

# # Install pip-tools and compile requirements.txt for the final stage
# RUN pip install --no-cache-dir pip-tools
# RUN python3 -m piptools compile requirements.in
# RUN python3 -m piptools sync requirements.txt

# EXPOSE 8000

# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]



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
    libgl1 \
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
# CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]




