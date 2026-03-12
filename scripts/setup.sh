#!/usr/bin/env bash
set -euo pipefail

echo "=== TuneForge Setup ==="

# Check Python version
python3 --version 2>/dev/null || { echo "ERROR: Python 3 is required"; exit 1; }

# ---------------------------------------------------------------------------
# System dependencies (ffmpeg libs required by PyAV / audiocraft)
# ---------------------------------------------------------------------------
echo "Checking system dependencies..."
if ! pkg-config --exists libavformat 2>/dev/null; then
    echo "Installing ffmpeg development libraries (requires sudo)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        ffmpeg \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
        python3-venv
else
    echo "ffmpeg libraries found"
fi

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
if [ ! -d "venv/bin" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv || {
        echo "ERROR: python3-venv package is required."
        echo "  Run: sudo apt install python3.10-venv"
        exit 1
    }
fi
source venv/bin/activate

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing TuneForge and dependencies (this may take several minutes)..."
pip install -e ".[all]"

# audiocraft hard-pins torch==2.1.0 which conflicts with numpy>=2.0.1 (required by bittensor).
# Install it with --no-deps to bypass the strict pin — it works fine with torch>=2.4.
echo "Installing audiocraft (with relaxed torch constraint)..."
pip install --no-deps audiocraft>=1.3.0

# ---------------------------------------------------------------------------
# Directories & config
# ---------------------------------------------------------------------------
echo "Creating directories..."
mkdir -p storage
mkdir -p /tmp/tuneforge

if [ ! -f ".env.miner" ]; then
    cp .env.miner.example .env.miner
    echo "Created .env.miner from example - please configure it"
fi
if [ ! -f ".env.validator" ]; then
    cp .env.validator.example .env.validator
    echo "Created .env.validator from example - please configure it"
fi

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
echo "Downloading models..."
bash scripts/download_models.sh

echo ""
echo "=== Setup Complete ==="
echo "1. Configure .env.miner or .env.validator"
echo "2. Run miner:     bash scripts/run_miner.sh"
echo "3. Run validator:  bash scripts/run_validator.sh"
