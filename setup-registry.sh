#!/bin/bash

# Quip App Registry Setup Script
# Usage: ./setup-registry.sh [registry-type] [registry-url]

set -e

REGISTRY_TYPE=${1:-"ghcr"}
REGISTRY_URL=${2:-"ghcr.io"}
APP_NAME="quip-app"
VERSION=${3:-"latest"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Quip App Registry Setup${NC}"
echo "Registry Type: $REGISTRY_TYPE"
echo "Registry URL: $REGISTRY_URL"
echo ""

# Function to setup GitHub Container Registry
setup_ghcr() {
    echo -e "${YELLOW}Setting up GitHub Container Registry...${NC}"

    # Check if GitHub CLI is installed
    if ! command -v gh &> /dev/null; then
        echo -e "${RED}❌ GitHub CLI (gh) is not installed. Please install it first.${NC}"
        echo "Visit: https://cli.github.com/"
        exit 1
    fi

    # Login to GitHub
    echo "Logging into GitHub..."
    gh auth login

    # Get repository info
    REPO=$(gh repo view --json owner,name -q '.owner.login + "/" + .name')
    FULL_IMAGE="ghcr.io/$REPO/$APP_NAME:$VERSION"

    echo -e "${GREEN}✅ GitHub setup complete${NC}"
    echo "Image will be: $FULL_IMAGE"

    return 0
}

# Function to setup Harbor registry
setup_harbor() {
    echo -e "${YELLOW}Setting up Harbor Registry...${NC}"

    if [ -z "$REGISTRY_URL" ] || [ "$REGISTRY_URL" = "ghcr.io" ]; then
        echo -e "${RED}❌ Please provide Harbor registry URL${NC}"
        echo "Usage: ./setup-registry.sh harbor registry.company.com"
        exit 1
    fi

    echo "Harbor Registry: $REGISTRY_URL"
    echo "Please ensure you have access to: https://$REGISTRY_URL"

    # Test connection
    echo "Testing registry connection..."
    if ! curl -s "https://$REGISTRY_URL" > /dev/null; then
        echo -e "${YELLOW}⚠️  Warning: Cannot reach registry at https://$REGISTRY_URL${NC}"
    fi

    FULL_IMAGE="$REGISTRY_URL/$APP_NAME:$VERSION"
    echo -e "${GREEN}✅ Harbor setup complete${NC}"
    echo "Image will be: $FULL_IMAGE"

    return 0
}

# Function to setup AWS ECR
setup_ecr() {
    echo -e "${YELLOW}Setting up AWS ECR...${NC}"

    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
        exit 1
    fi

    # Get AWS account ID and region
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGION=$(aws configure get region)

    if [ -z "$REGION" ]; then
        REGION="us-west-2"
        echo -e "${YELLOW}⚠️  No default region found, using us-west-2${NC}"
    fi

    # Create ECR repository if it doesn't exist
    echo "Creating ECR repository..."
    aws ecr create-repository \
        --repository-name $APP_NAME \
        --region $REGION \
        --image-scanning-configuration scanOnPush=true \
        || echo "Repository may already exist"

    FULL_IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$APP_NAME:$VERSION"

    echo -e "${GREEN}✅ ECR setup complete${NC}"
    echo "Image will be: $FULL_IMAGE"

    return 0
}

# Build and test the Docker image
build_and_test() {
    echo -e "${BLUE}🔨 Building Docker image...${NC}"

    # Build the image
    docker build -t $APP_NAME:local .

    echo -e "${GREEN}✅ Build complete${NC}"

    # Test the image
    echo -e "${BLUE}🧪 Testing image...${NC}"

    # Quick health check
    CONTAINER_ID=$(docker run -d -p 8501:8501 $APP_NAME:local)
    sleep 10

    if curl -s http://localhost:8501/_stcore/health > /dev/null; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Health check inconclusive${NC}"
    fi

    # Clean up test container
    docker stop $CONTAINER_ID > /dev/null
    docker rm $CONTAINER_ID > /dev/null

    return 0
}

# Push to registry
push_to_registry() {
    echo -e "${BLUE}📤 Pushing to registry...${NC}"

    # Tag for registry
    docker tag $APP_NAME:local $FULL_IMAGE

    case $REGISTRY_TYPE in
        "ghcr")
            echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_ACTOR --password-stdin
            ;;
        "harbor")
            echo "Please login to Harbor registry:"
            docker login $REGISTRY_URL
            ;;
        "ecr")
            aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
            ;;
    esac

    # Push the image
    docker push $FULL_IMAGE

    echo -e "${GREEN}✅ Push complete${NC}"
    echo -e "${GREEN}🎉 Your team can now pull the image with:${NC}"
    echo ""
    echo -e "${BLUE}docker pull $FULL_IMAGE${NC}"
    echo -e "${BLUE}docker run -p 8501:8501 -e OPENAI_API_KEY=your_key $FULL_IMAGE${NC}"

    return 0
}

# Main execution
case $REGISTRY_TYPE in
    "ghcr")
        setup_ghcr
        ;;
    "harbor")
        setup_harbor
        ;;
    "ecr")
        setup_ecr
        ;;
    *)
        echo -e "${RED}❌ Unknown registry type: $REGISTRY_TYPE${NC}"
        echo "Supported types: ghcr, harbor, ecr"
        exit 1
        ;;
esac

echo ""
read -p "Continue with build and push? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    build_and_test
    push_to_registry
else
    echo "Setup complete. Run this script again when ready to build and push."
fi