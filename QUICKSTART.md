# 🚀 Quick Setup Guide

## 🔐 Step 1: Secure Your API Key (URGENT!)

1. **Revoke the exposed key**: https://platform.openai.com/api-keys
2. **Generate a new OpenAI API key**
3. **Keep it secret!**

## 🏗️ Step 2: Choose Your Registry Option

### Option A: GitHub Container Registry (Easiest)
```bash
# 1. Create private GitHub repo
gh repo create yourteam/quip-app --private

# 2. Push your code
git init
git add .
git commit -m "Initial commit with robust retry functionality"
git remote add origin https://github.com/yourteam/quip-app.git
git push -u origin main

# 3. Add secrets to GitHub
gh secret set OPENAI_API_KEY --body "your_new_openai_key"
gh secret set QUIP_TOKEN --body "your_quip_token"

# 4. GitHub Actions will auto-build and push to ghcr.io
```

### Option B: AWS ECR (Enterprise)
```bash
# 1. Run secure deployment script
./deploy-secure.sh production ecr

# 2. Script will setup AWS Secrets Manager and ECS
```

### Option C: Harbor/Nexus (Self-hosted)
```bash
# 1. Setup your Harbor instance
# 2. Run deployment script
./deploy-secure.sh production harbor
```

## 👥 Step 3: Team Access

Your team can now deploy with:

### GitHub Container Registry
```bash
# Login with GitHub token
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull and run
docker pull ghcr.io/yourteam/quip-app:latest
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="your_team_key" \
  -e QUIP_TOKEN="your_quip_token" \
  ghcr.io/yourteam/quip-app:latest
```

### Or use Docker Compose
```bash
# Create .env file with secrets
echo "OPENAI_API_KEY=your_key" > .env
echo "QUIP_TOKEN=your_token" >> .env

# Start the app
docker-compose -f docker-compose.team.yml up -d
```

## 🌐 Access Your App

- **Local**: http://localhost:8501
- **Team**: Share the URL after deployment

## 📁 What You Now Have

```
cap_quip_streamlit_app/
├── streamlit_app.py          # Your app with retry functionality
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Local development
├── .dockerignore            # Optimize builds
├── SECURITY.md              # Security guide
├── setup-registry.sh        # Registry setup script
├── deploy-secure.sh         # Secure deployment script
├── .github/workflows/       # Auto-deployment
│   └── deploy.yml
└── README.md                # Updated documentation
```

## 🔥 Key Features Added

✅ **Robust HTTP retry** with exponential backoff
✅ **Fail-soft processing** - partial success beats total failure
✅ **Clear error reporting** with retry buttons
✅ **Secure secret management** for teams
✅ **Auto-deployment** with GitHub Actions
✅ **Container security** scanning
✅ **Multi-registry support** (GitHub, AWS, Harbor)

## 🏃‍♂️ Ready to Go!

1. **Secure your API key** (most important!)
2. **Choose a registry option** above
3. **Run the setup script**: `./deploy-secure.sh`
4. **Share with your team**

Your Quip app is now ready for enterprise-grade deployment with bulletproof retry logic! 🎉