# 🔐 Secure Secret Management Guide

## 🚨 Immediate Security Actions

1. **Revoke the exposed API key**: https://platform.openai.com/api-keys
2. **Generate a new OpenAI API key**
3. **Never commit secrets to git**

## 🏢 Team Secret Management Options

### Option 1: GitHub Container Registry + Secrets (Recommended)

#### 1. Repository Setup
```bash
# Create a private repository for your team
gh repo create yourteam/quip-app --private
cd /path/to/your/quip-app
git init
git remote add origin https://github.com/yourteam/quip-app.git
```

#### 2. Add GitHub Secrets
```bash
# Add secrets via GitHub CLI
gh secret set OPENAI_API_KEY --body "your_new_openai_key"
gh secret set QUIP_TOKEN --body "your_quip_token"
gh secret set QUIP_BASE_URL --body "https://platform.quip-apple.com"
```

#### 3. GitHub Actions Workflow
Create `.github/workflows/deploy.yml`:
```yaml
name: Build and Deploy

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout
      uses: actions/checkout@v4

    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
```

### Option 2: AWS ECR + Secrets Manager

#### 1. Create Secrets in AWS
```bash
# Store OpenAI API key
aws secretsmanager create-secret \
    --name "quip-app/openai-key" \
    --description "OpenAI API key for Quip app" \
    --secret-string "your_new_openai_key"

# Store Quip token
aws secretsmanager create-secret \
    --name "quip-app/quip-token" \
    --description "Quip API token" \
    --secret-string "your_quip_token"
```

#### 2. IAM Role for ECS/Fargate
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue"
            ],
            "Resource": [
                "arn:aws:secretsmanager:region:account:secret:quip-app/*"
            ]
        }
    ]
}
```

#### 3. ECS Task Definition
```json
{
    "family": "quip-app",
    "taskRoleArn": "arn:aws:iam::account:role/quip-app-task-role",
    "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "quip-app",
            "image": "account.dkr.ecr.region.amazonaws.com/quip-app:latest",
            "portMappings": [{"containerPort": 8501}],
            "secrets": [
                {
                    "name": "OPENAI_API_KEY",
                    "valueFrom": "arn:aws:secretsmanager:region:account:secret:quip-app/openai-key"
                },
                {
                    "name": "QUIP_TOKEN",
                    "valueFrom": "arn:aws:secretsmanager:region:account:secret:quip-app/quip-token"
                }
            ]
        }
    ]
}
```

### Option 3: Harbor + HashiCorp Vault

#### 1. Vault Setup
```bash
# Start Vault dev server (for testing)
vault server -dev

# Set environment
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN="your_vault_token"

# Store secrets
vault kv put secret/quip-app \
    openai_key="your_new_openai_key" \
    quip_token="your_quip_token" \
    quip_base_url="https://platform.quip-apple.com"
```

#### 2. Docker Compose with Vault Agent
```yaml
version: '3.8'

services:
  vault-agent:
    image: vault:latest
    command: vault agent -config=/vault/config/agent.hcl
    volumes:
      - ./vault-config:/vault/config
      - vault-secrets:/vault/secrets
    environment:
      - VAULT_ADDR=https://vault.company.com

  quip-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - vault-secrets:/vault/secrets:ro
    environment:
      - VAULT_SECRETS_PATH=/vault/secrets
    depends_on:
      - vault-agent

volumes:
  vault-secrets:
```

## 🔒 Local Development Security

### 1. Use .env files (never commit!)
```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_new_openai_key
QUIP_TOKEN=your_quip_token
QUIP_BASE_URL=https://platform.quip-apple.com
EOF

# Add to .gitignore
echo ".env" >> .gitignore
```

### 2. Use direnv for automatic loading
```bash
# Install direnv
brew install direnv  # macOS
# or: apt install direnv  # Ubuntu

# Create .envrc
echo "dotenv" > .envrc
direnv allow
```

## 🚀 Team Deployment Commands

### GitHub Container Registry
```bash
# Team members login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull and run
docker pull ghcr.io/yourteam/quip-app:latest
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/yourteam/quip-app:latest
```

### AWS ECR with Secrets
```bash
# Login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-west-2.amazonaws.com

# Run with secrets from environment
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="$(aws secretsmanager get-secret-value --secret-id quip-app/openai-key --query SecretString --output text)" \
  123456789012.dkr.ecr.us-west-2.amazonaws.com/quip-app:latest
```

## 🛡️ Security Best Practices

1. **Rotate secrets regularly** (every 90 days)
2. **Use least-privilege access** (only required permissions)
3. **Monitor secret usage** (audit logs)
4. **Never log secrets** (sanitize logs)
5. **Use container scanning** (vulnerabilities)
6. **Network security** (VPC, firewall rules)

## 🚨 Emergency Response

If secrets are compromised:
1. **Immediately revoke** all exposed keys
2. **Generate new secrets**
3. **Update all deployments**
4. **Audit access logs**
5. **Notify team members**