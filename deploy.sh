#!/bin/bash

# Exit on any error
set -e

# Set variables
REGISTRY=${REGISTRY:-localhost}
TAG=${TAG:-latest}
STACK_NAME=${STACK_NAME:-credit-risk}

# Display banner
echo "============================================================="
echo "Credit Default Risk Prediction System - Docker Swarm Deployment"
echo "============================================================="
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running or not accessible"
    exit 1
fi

# Check if we're in swarm mode
if [[ $(docker info | grep "Swarm: active" | wc -l) -eq 0 ]]; then
    echo "Docker is not in swarm mode. Initializing swarm..."
    docker swarm init --advertise-addr eth0 || true
    echo "Swarm initialized."
fi

# Build Docker images
echo "Building Docker images..."
docker build -t ${REGISTRY}/credit-risk-api:${TAG} -f docker/Dockerfile.api .
docker build -t ${REGISTRY}/credit-risk-app:${TAG} -f docker/Dockerfile.app .
docker build -t ${REGISTRY}/credit-risk-train:${TAG} -f docker/Dockerfile.train .
echo "Docker images built successfully."

# Push images if using a remote registry
if [[ "${REGISTRY}" != "localhost" ]]; then
    echo "Pushing images to registry ${REGISTRY}..."
    docker push ${REGISTRY}/credit-risk-api:${TAG}
    docker push ${REGISTRY}/credit-risk-app:${TAG}
    docker push ${REGISTRY}/credit-risk-train:${TAG}
    echo "Images pushed successfully."
fi

# Create and configure volumes if they don't exist
if [[ $(docker volume ls -q -f name=${STACK_NAME}_model_data | wc -l) -eq 0 ]]; then
    echo "Creating model data volume..."
    docker volume create ${STACK_NAME}_model_data
fi

# Check if models exist and train if necessary
echo "Checking if trained models exist..."
if [[ ! -d "models" || $(ls -1 models/*.pkl 2>/dev/null | wc -l) -eq 0 ]]; then
    echo "No trained models found. Running training container..."
    docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ${REGISTRY}/credit-risk-train:${TAG}
    echo "Training completed."
else
    echo "Trained models found."
fi

# Deploy the stack
echo "Deploying stack ${STACK_NAME}..."
export REGISTRY TAG
docker stack deploy -c docker-stack.yml ${STACK_NAME}

# Print information
echo
echo "============================================================="
echo "Deployment complete!"
echo "============================================================="
echo
echo "Services:"
echo "  - API: http://localhost:8000"
echo "  - UI: http://localhost:8501"
echo "  - Visualizer: http://localhost:8080"
echo
echo "Stack status:"
docker stack services ${STACK_NAME}
echo
echo "To remove the stack, run:"
echo "docker stack rm ${STACK_NAME}"
echo "=============================================================" 