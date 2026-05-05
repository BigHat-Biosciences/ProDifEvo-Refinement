#!/bin/bash
# Build and push the bh-rerd image to ECR.
# Usage:
#   bash scripts/bh-deploy.sh           # tag = latest
#   bash scripts/bh-deploy.sh v0.1      # tag = v0.1
set -euxo pipefail

ECR_URI="332120041740.dkr.ecr.us-east-1.amazonaws.com"
ECR_REPO="$ECR_URI/bh-rerd"
DOCKER_IMAGE_TAG="${1:-latest}"

aws ecr get-login-password | docker login --username AWS --password-stdin "$ECR_URI"

IMAGE_PATH="${ECR_REPO}:${DOCKER_IMAGE_TAG}"
docker build -f ./Dockerfile --platform=linux/amd64 . -t "$IMAGE_PATH"
docker push "$IMAGE_PATH"
