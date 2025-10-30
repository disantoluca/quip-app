# Quip Country Folder Q&A App

CAP Country Folder Q&A — Quip + Local Docs with robust retry functionality and fail-soft processing.

## 🚀 Quick Start with Docker

### Option 1: GitHub Container Registry (Recommended)

```bash
# Pull and run from GHCR
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key_here \
  ghcr.io/yourteam/quip-app:latest
```

### Option 2: Build locally

```bash
# Build the container
docker build -t quip-app .

# Run the container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key_here \
  quip-app
```

## 🏢 Team Deployment Options

### GitHub Container Registry (ghcr.io)
```bash
# Build and push
docker build -t ghcr.io/yourteam/quip-app:latest .
docker push ghcr.io/yourteam/quip-app:latest

# Team members pull and run
docker pull ghcr.io/yourteam/quip-app:latest
docker run -p 8501:8501 ghcr.io/yourteam/quip-app:latest
```

### Harbor/Nexus Private Registry
```bash
# Build and push to private registry
docker build -t registry.company.com/quip-app:latest .
docker push registry.company.com/quip-app:latest

# Team access
docker pull registry.company.com/quip-app:latest
docker run -p 8501:8501 registry.company.com/quip-app:latest
```

## 🔑 Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key for embeddings and chat
- `QUIP_TOKEN` - Default Quip API token (optional)
- `QUIP_BASE_URL` - Default Quip base URL (optional)

## 📊 Features

- ✅ Robust HTTP retry with exponential backoff
- ✅ Fail-soft batch processing
- ✅ Clear failure reporting with retry buttons
- ✅ Support for Quip folder crawling
- ✅ Local file upload (PDF, HTML, TXT)
- ✅ Vector search with FAISS
- ✅ OpenAI-powered Q&A

## 🛠️ Development

```bash
# Local development
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 🌐 Access

Once running, access the app at: http://localhost:8501

## Security
- Tokens entered in the UI are not stored to disk by the app.
- For production deployment, add authentication and environment secret management.
