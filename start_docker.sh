#!/bin/bash
# This script is to be run on the HOST machine
# Ensure nvidia-docker2 is installed

echo "Installing nvidia-docker2 (if not already installed)..."
sudo apt-get install -y nvidia-docker2

# Restart docker to ensure NVIDIA toolkit is active
echo "Restarting Docker to activate NVIDIA Docker Toolkit..."
sudo systemctl restart docker

# Build the images
echo "Building Docker images with docker-compose (production config)..."
docker compose -f docker-compose.prod.yml build

# Start the services
echo "Starting services using docker-compose..."
docker compose -f docker-compose.prod.yml up -d

echo "All steps completed. Services should be running."
