# Quick run commands for Vast.ai (Docker Compose)

Run these in order inside the Jupyter terminal or SSH on the Vast.ai instance.

1. **Go to workspace and clone the repo**
```bash
cd /root
git clone https://github.com/tariqbaluch/open_source_clipper.git app
cd app
```

2. **Setup Environment**
```bash
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY, YOUTUBE_COOKIE_FILE and other settings
# nano .env
```

3. **Install Docker Compose (if missing)**
```bash
# Check if installed
docker compose version

# If not installed:
curl -SL https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
```

4. **Build and Start Production Stack**
```bash
docker compose -f docker-compose.prod.yml up --build -d
```

5. **Verify**
```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f
```

6. **Access App**
Open `http://<VAST_IP>:<MAPPED_PORT>` in your browser.
