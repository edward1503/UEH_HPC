#!/bin/bash

# Initialize Docker Swarm if not already initialized
if ! docker info | grep -q "Swarm: active"; then
    echo "Initializing Docker Swarm..."
    docker swarm init
fi

# Build the ML worker image
echo "Building ML worker image..."
docker build -t credit-risk-ml-worker:latest -f Dockerfile.ml-worker ..

# Deploy the stack
echo "Deploying services..."
docker stack deploy -c docker-compose.swarm.yml credit-risk-ml

echo "Services deployed. Check status with: docker service ls" 