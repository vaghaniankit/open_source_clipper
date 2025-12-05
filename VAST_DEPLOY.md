# Deploying Open Source Clipper on Vast.ai

This guide assumes you are renting a Vast.ai instance and want to run the app directly from GitHub (no custom Docker image).

---

## 1. Create the Vast.ai instance

1. In Vast.ai, create a new instance.
2. **Template / Image**: choose a PyTorch image, for example:
   - `pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime` (or the PyTorch template with CUDA 11.8).
3. **Disk space**: at least **100 GB**.
4. **Ports**: in the Ports panel, add one public port mapping, for example:
   - `PUBLIC_PORT  →  8384/tcp`  (container port 8384 is what Uvicorn will use).

Record your instance **IP** and **PUBLIC_PORT** from Vast.ai – you will need them later.

---

## 2. Set environment variables (in Vast.ai UI)

In the Vast.ai instance settings, add at least:

```text
OPENAI_API_KEY=<your-real-openai-api-key>
```

Optionally (defaults are fine if you skip these):

```text
REDIS_URL=redis://localhost:6379/0
REDIS_BACKEND=redis://localhost:6379/0
```

> Do **not** commit the real `OPENAI_API_KEY` to GitHub. Keep it only in Vast.ai env or a private `.env`.

---

## 3. Connect to the instance

Use **Open in Jupyter** or SSH from the Vast.ai UI, then open a terminal.

All commands below run **inside that terminal**.

---

## 4. Get the code from GitHub

```bash
cd /workspace
rm -rf app

git clone https://github.com/tariqbaluch/open_source_clipper.git app
cd app
```

---

## 5. Install system packages

```bash
apt-get update && apt-get install -y \
  ffmpeg \
  libsndfile1 \
  redis-server \
  git \
  curl
```

---

## 6. Install Python dependencies

```bash
# Inside /workspace/app
pip install -r requirements.txt

# Ensure compatible PyTorch (CUDA 11.8 wheel)
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

# Extra dependency used by FastAPI
pip install itsdangerous
```

---

## 7. Create storage folders

```bash
mkdir -p storage/{videos,audio,pipeline,exports,data} downloads logs
```

---

## 8. Start Redis

```bash
redis-server --daemonize yes
```

This will run Redis in the background.

---

## 9. Start the Celery worker

From `/workspace/app`:

```bash
nohup celery -A app.celery_app worker --loglevel=info \
  > logs/celery.log 2>&1 &
```

- This starts the Celery worker in the background.
- Logs will be written to `logs/celery.log`.

> On Vast.ai (Linux) we do **not** need `--pool=solo`; the default prefork pool is fine.

---

## 10. Start the FastAPI app (Uvicorn)

Still in `/workspace/app`, run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8384
```

- Use the **container port** you mapped in the Vast.ai ports panel. In this example it is `8384`.

Keep this process running (do **not** close the terminal tab).

---

## 11. Open the app in the browser

Use the IP and public port from Vast.ai. If Vast shows:

```text
IP:      79.xx.xx.xx
PORT:    29164
Mapping: 79.xx.xx.xx:29164  →  8384/tcp
```

Then open in your browser:

```text
http://79.xx.xx.xx:29164/
```

The app will open directly into the wizard (Google login is bypassed in this build).

---

## 12. Stopping / restarting

- To stop Uvicorn: press `Ctrl + C` in the Uvicorn terminal.
- To restart after pulling new code:
  ```bash
  cd /workspace/app
  git pull
  # (re-run pip install if requirements changed)
  pkill -f "celery"
  nohup celery -A app.celery_app worker --loglevel=info > logs/celery.log 2>&1 &
  uvicorn app.main:app --host 0.0.0.0 --port 8384
  ```

Share this file (`VAST_DEPLOY.md`) with anyone who needs to run the app on Vast.ai.
