# Conda Development Setup

## 🐍 Create Conda Environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate quip_app

# Run the app
streamlit run streamlit_app.py
```

## 🔄 Update Environment

```bash
# Update existing environment
conda env update -f environment.yml

# Or recreate from scratch
conda env remove -n quip_app
conda env create -f environment.yml
```

## 📦 Export Current Environment

```bash
# Export current environment
conda env export > environment.yml

# Export without build numbers (more portable)
conda env export --no-builds > environment.yml
```

## 🚀 Docker with Conda

The Dockerfile now uses conda instead of pip for better dependency management and reproducible builds.

```bash
# Build conda-based image
docker build -t quip-app-conda .

# Run with conda environment
docker run -p 8501:8501 --env-file .env quip-app-conda
```