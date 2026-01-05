# Deploying Open Source Clipper on Vast.ai

This guide assumes you are renting a Vast.ai instance and want to run the app using **Docker Compose** for a production-like environment (Redis, Celery, Web App all containerized).

---

## 1. Create the Vast.ai instance

1. In Vast.ai, create a new instance.
2. **Template / Image**: choose a plain **Ubuntu** or **NVIDIA CUDA** image. Since we are using Docker Compose to build our own image, the base OS image matters less, but `nvidia/cuda:11.8.0-base-ubuntu22.04` or similar is good.
   - *Note*: You can also just use the default `Ubuntu 22.04` template, as long as you install the NVIDIA Container Toolkit (usually pre-installed on Vast instances).
3. **Instance Type**: 1x RTX 3060 Ti (or better).
4. **Disk space**: at least **50 GB** (Docker images + video storage).
5. **Ports**:
   - The app listens on port `8000` inside the container.
   - In Vast.ai "Edit Image & Config" -> "Port Mapping", map a public port to `8000`.
   - Example: `8000/tcp` mapped to `(Random/Assigned Port)`.

Record your instance **IP** and **PUBLIC_PORT** from Vast.ai.

---

## 2. Connect to the instance

Use SSH from your terminal:

```bash
ssh -p <PORT> root@<IP>
```

---

## 3. Install Docker & Docker Compose (if not present)

Most Vast.ai images have Docker, but you might need the Compose plugin.

```bash
# Update apt
apt-get update

# Install Docker Compose plugin
apt-get install -y docker-compose-plugin

# Verify
docker compose version
```

If `docker compose` is not available, install it manually:

```bash
curl -SL https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
```

---

## 4. Get the code

```bash
cd /root
git clone https://github.com/tariqbaluch/open_source_clipper.git app
cd app
```

---

## 5. Configure Environment

Create a `.env` file from the example:

```bash
cp .env.example .env
nano .env
```

**Crucial Settings for Docker Compose:**
- `REDIS_URL=redis://redis:6379/0` (Use the service name `redis`)
- `REDIS_BACKEND=redis://redis:6379/0`
- `OPENAI_API_KEY=sk-...` (Your key)

---

## 6. Build and Run

Run the production compose file:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

- `--build`: Rebuilds the images.
- `-d`: Detached mode (runs in background).

---

## 7. Check Status

Check if containers are running:

```bash
docker compose -f docker-compose.prod.yml ps
```

View logs:

```bash
docker compose -f docker-compose.prod.yml logs -f
```

---

## 8. Access the App

Open your browser to:

`http://<VAST_IP>:<MAPPED_PORT_FOR_8000>`

---

## Troubleshooting

**YouTube "Sign in" or "Bot" Errors:**
If you see errors like `Sign in to confirm you’re not a bot`, you need to provide YouTube cookies.

1.  **Export Cookies:**
    - Use a browser extension like "Get cookies.txt LOCALLY" (Chrome/Firefox).
    - Go to YouTube, log in, and export cookies as `cookies.txt`.
2.  **Upload to Server:**
    - Upload `cookies.txt` to the root of your project folder (`/root/app/cookies.txt`) on Vast.ai (use SCP or drag-and-drop in Jupyter/VSCode if available).
3.  **Redeploy:**
    ```bash
    docker compose -f docker-compose.prod.yml up --build -d
    ```
    The `cookies.txt` file is mounted into the containers automatically.

**GPU Access Error (`could not select device driver "nvidia"`):**
If the worker fails to start with this error, it means Docker cannot find the NVIDIA driver.

1.  **Verify Host Driver:**
    ```bash
    nvidia-smi
    ```
    If this fails, the instance is broken. Destroy and create a new one.

2.  **Check NVIDIA Container Toolkit:**
    ```bash
    docker info | grep -i runtime
    ```
    Should list `nvidia`.

3.  **Fix in Docker Compose:**
    Sometimes explicitly requesting capabilities is enough (already configured in `docker-compose.prod.yml`):
    ```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ```
    If it still fails, try **reinstalling the NVIDIA Container Toolkit** on the host:
    ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    ```

**Re-deploying:**
```bash
git pull
docker compose -f docker-compose.prod.yml up --build -d
```
