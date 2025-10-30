# Quip App - Team Setup Guide

Welcome to the CAP Country Folder Q&A app! This guide will help you get the containerized app running on your machine.

## 🛠️ Prerequisites

- Docker Desktop installed and running
- GitHub account with access to the repository
- OpenAI API key
- Quip API token and base URL

## 🔑 Step 1: Create GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name like "Quip App Container Access"
4. Select scopes: `read:packages`
5. Click "Generate token"
6. **Copy and save the token** (you won't see it again!)

## 🔐 Step 2: Login to Container Registry

Open your terminal and run:

```bash
docker login ghcr.io -u YOUR_GITHUB_USERNAME
```

When prompted for password, paste your Personal Access Token (not your GitHub password!).

## 📁 Step 3: Create Environment File

Create a new file called `.env` in any directory:

```bash
# On Windows (Command Prompt)
echo OPENAI_API_KEY=your_openai_key_here > .env
echo QUIP_TOKEN=your_quip_token_here >> .env
echo QUIP_BASE_URL=https://platform.quip-apple.com >> .env

# On Mac/Linux
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
QUIP_TOKEN=your_quip_token_here
QUIP_BASE_URL=https://platform.quip-apple.com
EOF
```

**Replace the placeholder values** with your actual API keys!

## 🚀 Step 4: Run the App

```bash
docker run -d -p 8501:8501 --env-file .env --name quip-app ghcr.io/disantoluca/quip-app:latest
```

## 🌐 Step 5: Access the App

Open your browser and go to: **http://localhost:8501**

## 🔄 Managing the Container

### Check if it's running:
```bash
docker ps
```

### View logs:
```bash
docker logs quip-app
```

### Stop the app:
```bash
docker stop quip-app
```

### Remove the container:
```bash
docker rm quip-app
```

### Update to latest version:
```bash
# Stop and remove old container
docker stop quip-app && docker rm quip-app

# Pull latest image
docker pull ghcr.io/disantoluca/quip-app:latest

# Run new container
docker run -d -p 8501:8501 --env-file .env --name quip-app ghcr.io/disantoluca/quip-app:latest
```

## ✨ Features Available

- ✅ Robust HTTP retry with exponential backoff
- ✅ Fail-soft batch processing
- ✅ Clear error reporting with retry buttons
- ✅ Quip folder crawling
- ✅ Local file upload (PDF, HTML, TXT)
- ✅ OpenAI-powered Q&A

## 🆘 Troubleshooting

### Container won't start?
- Check Docker Desktop is running
- Verify your API keys in the `.env` file
- Check logs: `docker logs quip-app`

### Can't pull the image?
- Verify you're logged in: `docker login ghcr.io`
- Check your GitHub token has `read:packages` permission
- Try logging out and back in

### Port already in use?
- Use a different port: `-p 8502:8501` (then access via http://localhost:8502)
- Or stop other containers using port 8501

### App shows errors?
- Verify your OpenAI API key is valid and has credits
- Check your Quip token and base URL are correct
- Some features require specific API permissions

## 🔒 Security Notes

- Keep your API keys secure and never share them
- The `.env` file contains sensitive information - don't commit it to version control
- Each team member needs their own API keys

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the container logs
3. Contact the development team
4. Check the GitHub repository for updates

---

**Repository**: https://github.com/disantoluca/quip-app
**Container Registry**: https://github.com/users/disantoluca/packages/container/package/quip-app