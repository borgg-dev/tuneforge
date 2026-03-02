# TuneForge Miner Setup Guide

## Prerequisites

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3090, A100, etc.)
- **Python**: 3.10 or later
- **CUDA**: 11.8 or later
- **OS**: Ubuntu 20.04+ or similar Linux
- **RAM**: 16GB+ recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/tuneforge/tuneforge.git
cd tuneforge

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Download models (MusicGen + CLAP)
bash scripts/download_models.sh
```

## Configuration

Copy the example environment file:

```bash
cp .env.miner.example .env.miner
```

Edit `.env.miner` with your settings:

```bash
TF_WALLET_NAME=your_wallet_name
TF_WALLET_HOTKEY=your_hotkey
TF_SUBTENSOR_NETWORK=finney
TF_NETUID=<assigned_netuid>
TF_MODEL_NAME=facebook/musicgen-medium
TF_GPU_DEVICE=cuda:0
```

## Running

### Direct
```bash
python -m neurons.miner --env-file .env.miner
```

### PM2
```bash
pm2 start ecosystem.config.js --only tuneforge-miner
```

### Docker
```bash
docker compose up miner
```

## Model Options

| Model | VRAM | Quality | Speed |
|-------|------|---------|-------|
| musicgen-small | ~4GB | Good | Fast |
| musicgen-medium | ~8GB | Better | Medium |
| musicgen-large | ~16GB | Best | Slow |

## Monitoring

Check logs:
```bash
tail -f /tmp/tuneforge/miner.log
```

Check registration:
```bash
btcli wallet overview --wallet.name your_wallet --subtensor.network finney
```

## Troubleshooting

- **CUDA out of memory**: Use a smaller model (`TF_MODEL_NAME=facebook/musicgen-small`)
- **Registration failed**: Ensure wallet is registered on the subnet
- **Serving rate limit**: The miner retries with exponential backoff automatically
