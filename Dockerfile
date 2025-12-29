# Use the official NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
ENV HOME=/home/user
WORKDIR /home/user/app

# Copy the requirements file
COPY --chown=user:user requirements.in .

# Install Python dependencies
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir pip-tools
RUN python3 -m piptools compile requirements.in
RUN python3 -m piptools sync requirements.txt

# Copy the application code
COPY --chown=user:user . .

# Expose the port
# EXPOSE 8000
EXPOSE 20673

# Set the entrypoint
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app"]
