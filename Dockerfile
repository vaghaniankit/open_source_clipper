# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables

# Prevent Python from writing .pyc files to disk (makes containers lighter and often unnecessary in production)
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure output is sent straight to terminal (useful for Docker logs, avoids buffering)
ENV PYTHONUNBUFFERED=1

# Avoids interactive prompts during OS-level package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR /home/user/app

# Copy the requirements file and install dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user:user . .

# Expose the port that gunicorn will run on
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]
