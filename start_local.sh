#!/bin/bash
set -e

# --- Configuration ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Local Development Environment (CPU)...${NC}"

# --- Dependency Check ---
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: 'docker' is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker daemon is not running. Please start Docker.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed and running.${NC}"

# --- Configuration Check ---
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found.${NC}"
    if [ -f .env.example ]; then
        echo -e "${YELLOW}Copying .env.example to .env...${NC}"
        cp .env.example .env
        echo -e "${YELLOW}Please update .env with your specific configuration (API keys, etc.).${NC}"
    else
        echo -e "${RED}Error: .env.example not found. Cannot create configuration.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ .env configuration found.${NC}"
fi

# --- Cleanup ---
echo -e "${YELLOW}Cleaning up old containers...${NC}"
# Attempt to stop and remove containers, continuing if none exist or command fails slightly (though set -e is on, pipe fail isn't)
docker compose -f docker-compose.local.yml down --remove-orphans || true

# --- Start Application ---
echo -e "${GREEN}Building and starting containers...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the environment.${NC}"

docker compose -f docker-compose.local.yml up --build
