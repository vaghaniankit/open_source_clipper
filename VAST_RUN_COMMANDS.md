# Quick run commands for Vast.ai

Run these in order inside the Jupyter terminal on the Vast.ai instance.

1. **Go to workspace and clone the repo**
```bash
cd /workspace
rm -rf app
git clone https://github.com/tariqbaluch/open_source_clipper.git app
cd app
```

2. **Install system packages**
```bash
apt-get update && apt-get install -y \
  ffmpeg \
  libsndfile1 \
  redis-server \
  git \
  curl
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install itsdangerous
```

4. **Prepare folders, start Redis and Celery worker (background)**
```bash
mkdir -p storage/{videos,audio,pipeline,exports,data} downloads logs
redis-server --daemonize yes
nohup celery -A app.celery_app worker --loglevel=info > logs/celery.log 2>&1 &
```

5. **Start the FastAPI app (Uvicorn)**
```bash
cd /workspace/app
uvicorn app.main:app --host 0.0.0.0 --port 8384
```

Replace `8384` with the container port that Vast.ai mapped from the public port if it is different.
