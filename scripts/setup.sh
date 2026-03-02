#!/usr/bin/env bash
set -euo pipefail

echo "=== TuneForge Setup ==="

# Check Python version
python3 --version 2>/dev/null || { echo "ERROR: Python 3 is required"; exit 1; }

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e ".[all]"

# Create directories
echo "Creating directories..."
mkdir -p storage
mkdir -p /tmp/tuneforge

# Copy example env files if not present
if [ ! -f ".env.miner" ]; then
    cp .env.miner.example .env.miner
    echo "Created .env.miner from example - please configure it"
fi
if [ ! -f ".env.validator" ]; then
    cp .env.validator.example .env.validator
    echo "Created .env.validator from example - please configure it"
fi

# Download models
echo "Downloading models..."
bash scripts/download_models.sh

echo ""
echo "=== Setup Complete ==="
echo "1. Configure .env.miner or .env.validator"
echo "2. Run miner:     bash scripts/run_miner.sh"
echo "3. Run validator:  bash scripts/run_validator.sh"
