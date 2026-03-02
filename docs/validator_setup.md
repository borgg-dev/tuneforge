# TuneForge Validator Setup Guide

## Prerequisites

- **CPU**: Modern multi-core processor (GPU optional but recommended for CLAP scoring)
- **Python**: 3.10 or later
- **RAM**: 8GB+ recommended
- **Stake**: Sufficient TAO staked for validator permit

## Installation

```bash
git clone https://github.com/tuneforge/tuneforge.git
cd tuneforge
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Configuration

```bash
cp .env.validator.example .env.validator
```

Edit `.env.validator`:

```bash
TF_MODE=validator
TF_WALLET_NAME=your_wallet_name
TF_WALLET_HOTKEY=your_hotkey
TF_SUBTENSOR_NETWORK=finney
TF_NETUID=<assigned_netuid>
TF_GENERATION_TIMEOUT=120
TF_VALIDATION_INTERVAL=300
TF_EMA_ALPHA=0.2
TF_STEEPEN_BASELINE=0.6
TF_STEEPEN_POWER=3.0
```

## Running

```bash
python -m neurons.validator --env-file .env.validator
```

### PM2
```bash
pm2 start ecosystem.config.js --only tuneforge-validator
```

### Docker
```bash
docker compose up validator
```

## Scoring Pipeline

The validator scores miners on five dimensions:

| Signal | Weight | Description |
|--------|--------|-------------|
| CLAP Adherence | 35% | Text-audio similarity via CLAP model |
| Audio Quality | 25% | Spectral analysis, clipping, dynamic range |
| Preference | 20% | Learned music preference model |
| Diversity | 10% | Inter-miner output uniqueness |
| Speed | 10% | Generation latency |

## Weight Setting

Weights are set every 175 blocks (~35 minutes). The validator uses an EMA leaderboard
with steepening to reward consistently high-performing miners.

## Monitoring

```bash
tail -f /tmp/tuneforge/validator.log
```
