#!/bin/bash

# Secure Team Deployment Script for Quip App
# Usage: ./deploy-secure.sh [environment] [registry-type]

set -e

ENVIRONMENT=${1:-"development"}
REGISTRY_TYPE=${2:-"ghcr"}
APP_NAME="quip-app"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔐 Secure Quip App Deployment${NC}"
echo "Environment: $ENVIRONMENT"
echo "Registry: $REGISTRY_TYPE"
echo ""

# Security checks
check_security() {
    echo -e "${YELLOW}🛡️  Running security checks...${NC}"

    # Check for secrets in code
    if grep -r "sk-" . --exclude-dir=.git --exclude="*.md" 2>/dev/null; then
        echo -e "${RED}❌ Found potential API keys in code!${NC}"
        echo "Please remove all hardcoded secrets before deployment."
        exit 1
    fi

    # Check for .env files
    if [ -f ".env" ]; then
        echo -e "${YELLOW}⚠️  Found .env file. Ensure it's in .gitignore${NC}"
        if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
            echo ".env" >> .gitignore
            echo "Added .env to .gitignore"
        fi
    fi

    # Check Docker security
    if [ -f "Dockerfile" ]; then
        if grep -q "^USER root" Dockerfile; then
            echo -e "${YELLOW}⚠️  Dockerfile runs as root. Consider using non-root user.${NC}"
        fi
    fi

    echo -e "${GREEN}✅ Security checks passed${NC}"
}

# Setup environment-specific secrets
setup_secrets() {
    echo -e "${YELLOW}🔑 Setting up secrets for $ENVIRONMENT...${NC}"

    case $REGISTRY_TYPE in
        "ghcr")
            setup_github_secrets
            ;;
        "ecr")
            setup_aws_secrets
            ;;
        "harbor")
            setup_harbor_secrets
            ;;
        *)
            echo -e "${RED}❌ Unknown registry type: $REGISTRY_TYPE${NC}"
            exit 1
            ;;
    esac
}

setup_github_secrets() {
    if ! command -v gh &> /dev/null; then
        echo -e "${RED}❌ GitHub CLI not found. Install from: https://cli.github.com/${NC}"
        exit 1
    fi

    echo "Setting up GitHub secrets..."

    # Check if secrets exist
    if ! gh secret list | grep -q "OPENAI_API_KEY"; then
        echo -e "${YELLOW}⚠️  OPENAI_API_KEY secret not found${NC}"
        read -s -p "Enter your OpenAI API key: " OPENAI_KEY
        echo ""
        gh secret set OPENAI_API_KEY --body "$OPENAI_KEY"
        echo -e "${GREEN}✅ OPENAI_API_KEY secret added${NC}"
    fi

    if ! gh secret list | grep -q "QUIP_TOKEN"; then
        echo -e "${YELLOW}⚠️  QUIP_TOKEN secret not found${NC}"
        read -s -p "Enter your Quip API token: " QUIP_TOKEN
        echo ""
        gh secret set QUIP_TOKEN --body "$QUIP_TOKEN"
        echo -e "${GREEN}✅ QUIP_TOKEN secret added${NC}"
    fi

    # Set Quip base URL
    if ! gh secret list | grep -q "QUIP_BASE_URL"; then
        gh secret set QUIP_BASE_URL --body "https://platform.quip-apple.com"
        echo -e "${GREEN}✅ QUIP_BASE_URL secret added${NC}"
    fi
}

setup_aws_secrets() {
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}❌ AWS CLI not found${NC}"
        exit 1
    fi

    echo "Setting up AWS Secrets Manager..."

    # Create secrets if they don't exist
    create_aws_secret "quip-app/openai-key" "OpenAI API key"
    create_aws_secret "quip-app/quip-token" "Quip API token"
    create_aws_secret "quip-app/quip-base-url" "Quip base URL" "https://platform.quip-apple.com"
}

create_aws_secret() {
    local secret_name=$1
    local description=$2
    local default_value=$3

    if aws secretsmanager describe-secret --secret-id "$secret_name" &>/dev/null; then
        echo "Secret $secret_name already exists"
    else
        if [ -n "$default_value" ]; then
            aws secretsmanager create-secret \
                --name "$secret_name" \
                --description "$description" \
                --secret-string "$default_value"
        else
            read -s -p "Enter value for $secret_name: " secret_value
            echo ""
            aws secretsmanager create-secret \
                --name "$secret_name" \
                --description "$description" \
                --secret-string "$secret_value"
        fi
        echo -e "${GREEN}✅ Created secret: $secret_name${NC}"
    fi
}

setup_harbor_secrets() {
    echo "For Harbor registry, ensure your secrets are configured in your orchestration platform:"
    echo "- Kubernetes secrets"
    echo "- Docker Swarm secrets"
    echo "- Environment variables"
}

# Deploy the application
deploy_app() {
    echo -e "${BLUE}🚀 Deploying application...${NC}"

    case $REGISTRY_TYPE in
        "ghcr")
            deploy_github
            ;;
        "ecr")
            deploy_aws
            ;;
        "harbor")
            deploy_harbor
            ;;
    esac
}

deploy_github() {
    # Get repository info
    REPO=$(gh repo view --json owner,name -q '.owner.login + "/" + .name')
    IMAGE="ghcr.io/$REPO/$APP_NAME:latest"

    echo "Deploying from: $IMAGE"

    # Generate docker-compose for team
    cat > docker-compose.team.yml << EOF
version: '3.8'

services:
  quip-app:
    image: $IMAGE
    ports:
      - "8501:8501"
    environment:
      # Set these in your .env file or environment
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - QUIP_TOKEN=\${QUIP_TOKEN}
      - QUIP_BASE_URL=\${QUIP_BASE_URL:-https://platform.quip-apple.com}
    restart: unless-stopped

    # Optional: Add volume for persistence
    volumes:
      - ./data:/app/data

    # Optional: Add health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

    echo -e "${GREEN}✅ Created docker-compose.team.yml${NC}"
    echo ""
    echo "Team members can now run:"
    echo -e "${BLUE}docker-compose -f docker-compose.team.yml up -d${NC}"
}

deploy_aws() {
    # Create ECS task definition
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGION=$(aws configure get region || echo "us-west-2")
    IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$APP_NAME:latest"

    cat > ecs-task-definition.json << EOF
{
    "family": "quip-app-$ENVIRONMENT",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "256",
    "memory": "512",
    "executionRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/quip-app-task-role",
    "containerDefinitions": [
        {
            "name": "quip-app",
            "image": "$IMAGE",
            "portMappings": [
                {
                    "containerPort": 8501,
                    "protocol": "tcp"
                }
            ],
            "essential": true,
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/quip-app",
                    "awslogs-region": "$REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "secrets": [
                {
                    "name": "OPENAI_API_KEY",
                    "valueFrom": "arn:aws:secretsmanager:$REGION:$ACCOUNT_ID:secret:quip-app/openai-key"
                },
                {
                    "name": "QUIP_TOKEN",
                    "valueFrom": "arn:aws:secretsmanager:$REGION:$ACCOUNT_ID:secret:quip-app/quip-token"
                },
                {
                    "name": "QUIP_BASE_URL",
                    "valueFrom": "arn:aws:secretsmanager:$REGION:$ACCOUNT_ID:secret:quip-app/quip-base-url"
                }
            ]
        }
    ]
}
EOF

    echo -e "${GREEN}✅ Created ECS task definition${NC}"
    echo ""
    echo "Deploy to ECS with:"
    echo -e "${BLUE}aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json${NC}"
}

deploy_harbor() {
    echo "Harbor deployment requires your orchestration platform configuration."
    echo "Example Kubernetes deployment created in k8s-deployment.yml"

    cat > k8s-deployment.yml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quip-app
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: quip-app
  template:
    metadata:
      labels:
        app: quip-app
    spec:
      containers:
      - name: quip-app
        image: registry.company.com/quip-app:latest
        ports:
        - containerPort: 8501
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: quip-secrets
              key: openai-api-key
        - name: QUIP_TOKEN
          valueFrom:
            secretKeyRef:
              name: quip-secrets
              key: quip-token
        - name: QUIP_BASE_URL
          value: "https://platform.quip-apple.com"
---
apiVersion: v1
kind: Service
metadata:
  name: quip-app-service
spec:
  selector:
    app: quip-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8501
  type: LoadBalancer
EOF
}

# Generate team instructions
generate_instructions() {
    echo -e "${BLUE}📚 Generating team instructions...${NC}"

    cat > TEAM_DEPLOYMENT.md << EOF
# Team Deployment Instructions

## Quick Start

### For $REGISTRY_TYPE deployment:

1. **Login to registry:**
EOF

    case $REGISTRY_TYPE in
        "ghcr")
            cat >> TEAM_DEPLOYMENT.md << EOF
   \`\`\`bash
   echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
   \`\`\`

2. **Run the application:**
   \`\`\`bash
   docker-compose -f docker-compose.team.yml up -d
   \`\`\`

3. **Access the app:**
   Open http://localhost:8501 in your browser
EOF
            ;;
        "ecr")
            cat >> TEAM_DEPLOYMENT.md << EOF
   \`\`\`bash
   aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
   \`\`\`

2. **Deploy to ECS:**
   \`\`\`bash
   aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
   \`\`\`
EOF
            ;;
        "harbor")
            cat >> TEAM_DEPLOYMENT.md << EOF
   \`\`\`bash
   docker login registry.company.com
   \`\`\`

2. **Deploy to Kubernetes:**
   \`\`\`bash
   kubectl apply -f k8s-deployment.yml
   \`\`\`
EOF
            ;;
    esac

    cat >> TEAM_DEPLOYMENT.md << EOF

## Environment Variables

Make sure to set these environment variables:

- \`OPENAI_API_KEY\`: Your OpenAI API key
- \`QUIP_TOKEN\`: Your Quip API token
- \`QUIP_BASE_URL\`: Quip base URL (default: https://platform.quip-apple.com)

## Security Notes

- Never commit secrets to git
- Use environment-specific secret management
- Rotate API keys regularly
- Monitor access logs

## Support

For issues or questions, contact the development team.
EOF

    echo -e "${GREEN}✅ Created TEAM_DEPLOYMENT.md${NC}"
}

# Main execution
main() {
    check_security
    setup_secrets
    deploy_app
    generate_instructions

    echo ""
    echo -e "${GREEN}🎉 Deployment setup complete!${NC}"
    echo ""
    echo "Next steps for your team:"
    echo "1. Review TEAM_DEPLOYMENT.md"
    echo "2. Set up their environment variables"
    echo "3. Login to the registry"
    echo "4. Deploy the application"
    echo ""
    echo -e "${BLUE}Happy Quip crawling! 🕷️📚${NC}"
}

# Run main function
main