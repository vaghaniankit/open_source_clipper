# Use NVIDIA CUDA base image for GPU support
# We use 11.8 because it's widely supported by torch, faster-whisper, and whisperx
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# python3.10 is default in ubuntu 22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set work directory
WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# We install torch explicitly first to ensure we get the CUDA version
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install the rest of the requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Install specific production servers explicitly if not in requirements (gunicorn/uvicorn are there)
# RUN pip3 install --no-cache-dir gunicorn uvicorn

# Copy project
COPY . .

# Create storage directories
RUN mkdir -p storage/videos storage/audio storage/pipeline storage/exports storage/data downloads

# Expose the port
EXPOSE 8000

# Copy supervisor config (if you plan to use it, otherwise compose handles this)
COPY docker/supervisord.conf /etc/supervisor/supervisord.conf

# Default command
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]




