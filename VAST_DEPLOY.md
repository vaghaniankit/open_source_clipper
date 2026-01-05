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

**GPU Access:**
If the worker container complains about GPU access, ensure the NVIDIA Container Toolkit is working on the host.
Verify inside the container:
```bash
docker exec -it open_source_clipper_worker nvidia-smi
```

**Re-deploying:**
```bash
git pull
docker compose -f docker-compose.prod.yml up --build -d
```
