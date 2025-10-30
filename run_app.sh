#!/usr/bin/env bash
# =======================================================
# 🚀 Quip App Launcher
# =======================================================

# Exit immediately on error
set -e

# Define environment name
ENV_NAME="quip_app"

echo "🔍 Checking Conda environment: $ENV_NAME"

# Activate conda if not already available
if ! command -v conda &> /dev/null; then
    echo "⚠️ Conda not found. Trying to load Miniforge..."
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
    else
        echo "❌ Conda not initialized. Please install Miniforge first."
        exit 1
    fi
fi

# Activate environment
conda activate $ENV_NAME || {
    echo "⚙️ Environment not found — creating it..."
    conda env create -f environment.yml
    conda activate $ENV_NAME
}

# Verify FAISS installation
python - <<'PY'
try:
    import faiss
    print(f"✅ FAISS loaded successfully — version {faiss.__version__}")
except Exception as e:
    print(f"❌ FAISS not found: {e}")
    exit(1)
PY

# Ensure data folders exist
mkdir -p data/persist data/index data/logs

# Start Streamlit
echo "🚀 Starting Streamlit app..."
streamlit run streamlit_app.py