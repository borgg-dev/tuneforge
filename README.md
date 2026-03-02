# TuneForge - Decentralized AI Music Generation on Bittensor

TuneForge is a Bittensor subnet for decentralized AI music generation. Miners run MusicGen models to generate high-quality music from text prompts. Validators evaluate the quality, prompt adherence, and musicality of generated audio to set weights on the network.

## Architecture

```
                    Bittensor Network
                         |
            +------------+------------+
            |                         |
        Validator                   Miner
    (Scoring Pipeline)        (Music Generation)
            |                         |
    +-------+-------+         +------+------+
    | CLAP Scoring  |         | MusicGen    |
    | Audio Quality |         | Audio Post  |
    | Musicality    |         | Processing  |
    | Novelty       |         | API Server  |
    +---------------+         +-------------+
```

**Miners** receive text prompts and generate music using MusicGen models. They serve an axon endpoint and optionally a REST API.

**Validators** send text prompts to miners, receive generated audio, score it across multiple dimensions (quality, prompt adherence, musicality, novelty), and set weights on the network.

## Quick Start

### Prerequisites

- Python 3.10-3.12
- NVIDIA GPU with 16GB+ VRAM (miners)
- Bittensor wallet

### Installation

```bash
git clone https://github.com/tuneforge-subnet/tuneforge.git
cd tuneforge
bash scripts/setup.sh
```

### Miner Setup

1. Configure your environment:
```bash
cp .env.miner.example .env.miner
# Edit .env.miner with your wallet details and GPU config
```

2. Run the miner:
```bash
bash scripts/run_miner.sh
# Or with PM2:
pm2 start ecosystem.config.js --only tuneforge-miner
```

### Validator Setup

1. Configure your environment:
```bash
cp .env.validator.example .env.validator
# Edit .env.validator with your wallet details
```

2. Run the validator:
```bash
bash scripts/run_validator.sh
# Or with PM2:
pm2 start ecosystem.config.js --only tuneforge-validator
```

## API Server

Miners can expose a REST API for direct music generation requests:

```bash
# Start API server (included with miner, or standalone):
pm2 start ecosystem.config.js --only tuneforge-api
```

### Endpoints

- `POST /generate` - Generate music from a text prompt
- `GET /health` - Health check
- `GET /status` - Model and queue status

## Docker

### Miner
```bash
docker compose up miner -d
```

### Validator
```bash
docker compose up validator -d
```

### Full Stack
```bash
docker compose up -d
```

## Configuration

All configuration uses `TF_` prefixed environment variables. See `.env.miner.example` and `.env.validator.example` for the full list.

| Variable | Description | Default |
|----------|-------------|---------|
| `TF_NETUID` | Subnet UID | `0` |
| `TF_WALLET_NAME` | Wallet name | `default` |
| `TF_WALLET_HOTKEY` | Hotkey name | `default` |
| `TF_SUBTENSOR_NETWORK` | Network | `finney` |
| `TF_MODEL_NAME` | MusicGen model | `facebook/musicgen-medium` |
| `TF_GPU_DEVICE` | GPU device | `cuda:0` |
| `TF_MODEL_PRECISION` | Inference precision | `float16` |
| `TF_GENERATION_MAX_DURATION` | Max audio duration (s) | `30` |
| `TF_VALIDATION_INTERVAL` | Seconds between rounds | `300` |
| `TF_AUDIO_QUALITY_WEIGHT` | Quality score weight | `0.3` |
| `TF_PROMPT_ADHERENCE_WEIGHT` | CLAP score weight | `0.4` |
| `TF_MUSICALITY_WEIGHT` | Musicality weight | `0.2` |
| `TF_NOVELTY_WEIGHT` | Novelty weight | `0.1` |

## Scoring

Validators score generated audio across four dimensions:

1. **Audio Quality (30%)** - Signal-to-noise ratio, spectral analysis, artifact detection
2. **Prompt Adherence (40%)** - CLAP text-audio similarity score
3. **Musicality (20%)** - Rhythm consistency, harmonic structure, tonal quality
4. **Novelty (10%)** - Diversity of outputs, penalizing memorization

## Project Structure

```
tuneforge/
  tuneforge/
    __init__.py          # Version, constants
    settings.py          # Pydantic settings (TF_ env vars)
    base/                # Base neuron classes
    core/                # Core miner/validator logic
    generation/          # MusicGen pipeline
    scoring/             # Audio scoring modules
    rewards/             # Reward computation
    validation/          # Validation challenge logic
    config/              # Additional config
    api/                 # FastAPI server
      routes/            # API route handlers
    utils/               # Logging, config helpers
  neurons/
    miner.py             # Miner entry point
    validator.py         # Validator entry point
  tests/                 # Test suite
  scripts/               # Setup and run scripts
  docs/                  # Documentation
```

## License

MIT License - see [LICENSE](LICENSE) for details.
