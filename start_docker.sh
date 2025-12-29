#!/bin/bash
# This script is to be run on the HOST machine

# List all Docker images
echo "Listing all Docker images before starting..."
docker images -a

# Stop all running containers
echo "Stopping all running containers..."
docker stop $(docker ps -aq)

# Prune the system
echo "Pruning Docker system..."
docker system prune -af

# Ensure nvidia-docker2 is installed

if ! dpkg -l | grep -q nvidia-docker2; then
  echo "Installing nvidia-docker2..."
  sudo apt-get install -y nvidia-docker2
else
  echo "nvidia-docker2 is already installed."
fi

# Restart docker to ensure NVIDIA toolkit is active
echo "Restarting Docker to activate NVIDIA Docker Toolkit..."
sudo systemctl restart docker

# Build the images
echo "Building Docker images with docker-compose (production config)..."
docker compose -f docker-compose.prod.yml build

# Fix storage permissions
echo "Fixing storage permissions..."
sudo chown -R 1000:1000 ./storage

# Start the services
echo "Starting services using docker-compose..."
docker compose -f docker-compose.prod.yml up -d

echo "All steps completed. Services should be running."

echo "Listing all Docker images..."
docker images -a
