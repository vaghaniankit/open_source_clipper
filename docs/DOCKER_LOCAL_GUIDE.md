# Running the Open Source Clipper Locally with Docker

This guide explains how to set up and run the Open Source Clipper project locally on your machine using Docker. Using Docker ensures a consistent environment, eliminating "it works on my machine" issues and simplifying dependency management.

## Prerequisites

Before getting started, please ensure your system meets the following requirements:

### System Requirements
- **Operating System**: Windows 10/11, macOS (Intel or Apple Silicon), or a modern Linux distribution.
- **System Resources**:
  - Minimum 8GB RAM (16GB recommended for faster video processing).
  - At least 10GB of free disk space for Docker images, video storage, and generated clips.
- **Available Ports**:
  - `8000`: Used by the FastAPI web interface.
  - `6379`: Used by the Redis message broker.
  - *Make sure no other applications are currently using these ports on your system.*

### Required Tools
1. **Git**: Required to download the project source code.
   - [Download Git](https://git-scm.com/downloads)
2. **Docker Desktop** (or Docker Engine + Docker Compose on Linux): This is the core engine required to run the application containers.
   - [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - *Note for Windows users: It is highly recommended to use the WSL 2 backend for Docker Desktop.*

---

## Step 1: Clone the Repository

First, you need to download the source code to your local machine.

Open your terminal (Command Prompt/PowerShell on Windows, Terminal on Mac/Linux) and run:

```bash
git clone <YOUR_REPOSITORY_URL_HERE>
cd open_source_clipper
```

*(Replace `<YOUR_REPOSITORY_URL_HERE>` with the actual Git URL of the project).*

---

## Step 2: Environment Configuration

The application requires specific environment variables (like API keys) to function correctly.

1. Locate the `.env.example` file in the root directory of the project.
2. Make a copy of this file and name it `.env`.
   - On Windows (Command Prompt): `copy .env.example .env`
   - On Mac/Linux: `cp .env.example .env`
3. Open the newly created `.env` file in a text editor (like Notepad or VS Code).
4. Fill in the required values:
   - `SESSION_SECRET`: Change this to any random, secure string.
   - `OPENAI_API_KEY`: Provide your OpenAI API key (required for LLM-based highlight generation).
   - `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`: (Optional) Required if you want to test Google OAuth login.
   - `TRANSCRIBE_MODEL`: You can leave this as the default (`small`) or change it to `tiny` for faster but less accurate transcription during local testing.

---

## Step 3: Running the Application (Local CPU Mode)

The project includes two Docker Compose configurations. For standard local development and testing, we use the CPU configuration (`docker-compose.local.yml`).

### Option A: Using the Helper Script (Mac/Linux)

If you are on macOS or Linux, a helper script is provided to automate the startup process.

```bash
# Make the script executable
chmod +x start_local.sh

# Run the script
./start_local.sh
```
This script will verify your Docker installation, check your `.env` file, clean up old containers, and start the application.

### Option B: Manual Startup (Windows / Mac / Linux)

If you are on Windows, or prefer to run the commands manually, use the following Docker Compose command:

```bash
docker compose -f docker-compose.local.yml up --build
```

**What this command does:**
- `--build`: Forces Docker to build the latest image using `Dockerfile.cpu`.
- `-f docker-compose.local.yml`: Tells Docker to use the local development configuration.
- It will start 4 containers: `web` (FastAPI), `worker` (Celery task worker), `beat` (Celery scheduler), and `redis` (Message queue).

*Note: The first time you run this command, it may take several minutes to download the base Docker images and install all Python dependencies.*

---

## Step 4: Accessing the Application

Once the terminal shows that the containers are running (you will see continuous log output), the application is ready.

1. Open your web browser.
2. Navigate to: [http://localhost:8000](http://localhost:8000)

You should now see the Open Source Clipper interface and can begin uploading videos or submitting YouTube links!

---

## Step 5: Stopping the Application

To gracefully stop the application and background workers:

1. Go to the terminal window where Docker is running.
2. Press `Ctrl + C`. Docker will send a stop signal to all containers.
3. Wait a few seconds for all containers to cleanly shut down.

Alternatively, you can open a new terminal window in the project directory and run:

```bash
docker compose -f docker-compose.local.yml down
```

---

## Advanced: Running with GPU Support

If your local machine has a compatible NVIDIA GPU and you want to leverage hardware acceleration (which significantly speeds up video processing and transcription), you can run the production configuration.

**Additional Prerequisites for GPU:**
- [NVIDIA Display Drivers](https://www.nvidia.com/download/index.aspx) installed.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed and configured for Docker.

**Command to run the GPU version:**

```bash
docker compose up --build
```
This will use the main `docker-compose.yml` file and `Dockerfile`, which are configured to reserve GPU resources for the Celery worker.
