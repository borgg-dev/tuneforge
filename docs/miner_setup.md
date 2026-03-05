# TuneForge Miner Setup Guide

This guide covers everything you need to run a miner on the TuneForge subnet. A miner receives music generation challenges from validators, produces audio, and earns TAO rewards based on quality scores.

---

## What a Miner Does

A TuneForge miner performs the following:

- Receives `MusicGenerationSynapse` challenges from validators via the Bittensor axon.
- Generates audio using a configurable backend (MusicGen small/medium/large or Stable Audio).
- Returns base64-encoded WAV audio along with metadata (`sample_rate`, `generation_time_ms`, `model_id`).
- Earns TAO rewards proportional to quality scores assigned by validators.
- Handles organic queries from the SaaS API. These do not affect scoring. Organic queries are prioritized over validation challenges (`organic_boost = 1,000,000`).

---

## Hardware Requirements

Minimum hardware specifications (from `min_compute.yml`):

| Resource | Minimum |
|----------|---------|
| GPU | NVIDIA, 16 GB VRAM |
| Recommended GPU | RTX 4090 / A100 |
| CPU Cores | 8 |
| RAM | 32 GB |
| Disk | 100 GB SSD |
| Network | 100 Mbps up/down |

### Model-Specific VRAM Usage

| Model | HuggingFace ID / Key | VRAM |
|-------|----------------------|------|
| MusicGen Small | `facebook/musicgen-small` | ~4 GB |
| MusicGen Medium | `facebook/musicgen-medium` | ~8 GB |
| MusicGen Large | `facebook/musicgen-large` | ~16 GB |
| Stable Audio | `stable_audio` | ~6 GB |

MusicGen Medium is the default and recommended baseline.

---

## Prerequisites

- Python 3.10, 3.11, or 3.12
- CUDA 11.8 or later (the Docker image uses NVIDIA CUDA 12.1.1)
- A Bittensor wallet registered on the subnet
- System packages: `git`, `ffmpeg`, `libsndfile`

---

## Installation

```bash
git clone https://github.com/tuneforge-subnet/tuneforge.git
cd tuneforge
python3 -m venv venv
source venv/bin/activate
pip install -e .

# audiocraft is installed separately because it pins torch==2.1.0:
pip install audiocraft --no-deps
```

---

## Configuration

Copy the example environment file and edit it:

```bash
cp .env.miner.example .env.miner
```

### Full Configuration Reference

All variables use the `TF_` prefix and are loaded via pydantic-settings.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_NETUID` | int | `0` | Subnet UID (234 on testnet) |
| `TF_SUBTENSOR_NETWORK` | str | None | Network: `finney`, `test`, or `local` |
| `TF_SUBTENSOR_CHAIN_ENDPOINT` | str | None | Custom chain endpoint URL |
| `TF_WALLET_NAME` | str | `default` | Bittensor wallet name |
| `TF_WALLET_HOTKEY` | str | `default` | Hotkey name |
| `TF_WALLET_PATH` | str | `~/.bittensor/wallets` | Path to wallet directory |
| `TF_MODE` | str | `miner` | Runtime mode |
| `TF_NEURON_EPOCH_LENGTH` | int | `100` | Blocks between epochs |
| `TF_NEURON_TIMEOUT` | int | `120` | Forward timeout in seconds |
| `TF_AXON_PORT` | int | None | Axon serving port |
| `TF_MODEL_NAME` | str | `facebook/musicgen-medium` | Model to use for generation |
| `TF_GENERATION_MAX_DURATION` | int | `30` | Maximum audio duration in seconds |
| `TF_GENERATION_SAMPLE_RATE` | int | `32000` | Audio sample rate in Hz |
| `TF_GENERATION_TIMEOUT` | int | `120` | Generation timeout in seconds |
| `TF_GPU_DEVICE` | str | `cuda:0` | GPU device identifier |
| `TF_MODEL_PRECISION` | str | `float16` | Model precision: `float32`, `float16`, or `bfloat16` |
| `TF_GUIDANCE_SCALE` | float | `3.0` | Classifier-free guidance scale |
| `TF_TEMPERATURE` | float | `1.0` | Sampling temperature |
| `TF_TOP_K` | int | `250` | Top-K sampling parameter |
| `TF_TOP_P` | float | `0.0` | Nucleus sampling threshold (0 = disabled) |
| `TF_STORAGE_PATH` | str | `./storage` | Local storage directory |
| `TF_LOG_LEVEL` | str | `INFO` | Log level |
| `TF_LOG_DIR` | str | `/tmp/tuneforge` | Log output directory |

### Minimal Example

```bash
TF_NETUID=234
TF_SUBTENSOR_NETWORK=test
TF_WALLET_NAME=my_wallet
TF_WALLET_HOTKEY=my_hotkey
TF_MODEL_NAME=facebook/musicgen-medium
TF_GPU_DEVICE=cuda:0
TF_AXON_PORT=8091
```

---

## Running the Miner

### Direct Python

```bash
source venv/bin/activate
python -m neurons.miner --env-file .env.miner
```

The miner can also be invoked via the `tuneforge-miner` console entry point if the package is installed.

### PM2

```bash
pm2 start ecosystem.config.js --only tuneforge-miner-1
pm2 logs tuneforge-miner-1
pm2 save
```

### Docker

```bash
docker compose up miner -d
docker logs tuneforge-miner -f
```

The `Dockerfile.miner` uses an NVIDIA CUDA 12.1.1 base image with Python 3.11 and pre-downloads the `musicgen-medium` checkpoint.

---

## Model Selection Guide

| Model | VRAM | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| MusicGen Small | ~4 GB | Lower | Fastest | Good for testing or low-VRAM GPUs |
| MusicGen Medium | ~8 GB | Good | Moderate | Default; recommended baseline |
| MusicGen Large | ~16 GB | Highest | Slower | Best quality across all scoring signals |
| Stable Audio | ~6 GB | Good | Moderate | Different sonic aesthetic |

Speed accounts for only 5% of the total score. Quality signals collectively represent 60%. Use the largest model your GPU can handle.

Set the model via:

```bash
TF_MODEL_NAME=facebook/musicgen-large
```

---

## Multi-GPU Setup

To run multiple miners on a single machine with multiple GPUs:

1. Create separate environment files (e.g., `.env.miner` and `.env.miner2`).
2. Assign different GPU devices and axon ports in each file:

```bash
# .env.miner
TF_GPU_DEVICE=cuda:0
TF_AXON_PORT=8091

# .env.miner2
TF_GPU_DEVICE=cuda:1
TF_AXON_PORT=8092
```

3. The PM2 `ecosystem.config.js` already includes entries for `tuneforge-miner-1` and `tuneforge-miner-2`.

```bash
pm2 start ecosystem.config.js --only tuneforge-miner-1
pm2 start ecosystem.config.js --only tuneforge-miner-2
```

---

## How Scoring Works

Understanding the scoring system helps you maximize rewards.

### Scoring Signals

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| CLAP Adherence | 30% | Text-audio similarity -- does the audio match the prompt? |
| Musicality | 10% | Pitch, harmony, rhythm, arrangement quality |
| Neural Quality (MERT) | 10% | Learned music representations |
| Production Quality | 8% | Spectral balance, loudness (LUFS), dynamics |
| Melody Coherence | 7% | Melodic intervals, contour, structure |
| Structural Completeness | 7% | Section detection, song form |
| Audio Quality | 6% | Signal-level analysis (harmonic ratio, onsets, contrast) |
| Preference Model | 6% | Perceptual quality scoring |
| Vocal Quality | 6% | Vocal presence, clarity, pitch accuracy (genre-aware) |
| Diversity | 5% | CLAP embedding diversity across your recent outputs |
| Speed | 5% | Generation time (5s = 1.0, 30s = 0.3, 60s+ = 0.0) |

### Penalties

| Penalty | Trigger | Effect |
|---------|---------|--------|
| Silence | RMS below 0.01 | Score set to 0.0 |
| Timeout | Generation exceeds 120s | Score set to 0.0 |
| Plagiarism | Self-similarity above 0.80 | Score set to 0.0 |
| Duration | Off-target by more than 20% | Linear penalty, reaching 0.0 at 50% deviation |
| Artifacts | Clipping, loops, discontinuities | Multiplier applied to final score |

### Optimization Tips

- **Prompt adherence is the single largest factor at 30%.** Your model must faithfully follow the text prompt.
- **Quality signals total 60%.** Larger models consistently score higher across all quality dimensions.
- **Diversity (5%):** Do not recycle outputs. The system tracks your last 10 CLAP embeddings.
- **Speed (5%):** Under 5 seconds yields a perfect speed score. At 30 seconds the score drops to 0.3. Over 60 seconds yields 0.0.
- **Organic queries** (where `is_organic=True`) do not affect your validation score.
- **Weight perturbation:** Scoring weights shift by up to 20% each round to prevent gaming.
- **Genre-aware evaluation:** Quality targets vary by genre, so the system accounts for genre-specific characteristics.

---

## Monitoring

### Logs

Logs are written to the directory specified by `TF_LOG_DIR` (default: `/tmp/tuneforge`).

```bash
tail -f /tmp/tuneforge/miner.log
```

### PM2

```bash
pm2 logs tuneforge-miner-1
```

### Docker

```bash
docker logs tuneforge-miner -f
```

### Health Metrics

The miner exposes health information via `HealthReportSynapse`, which includes GPU utilization, memory usage, generation count, and error count.

### Weights and Biases (Optional)

Enable W&B logging for detailed tracking:

```bash
TF_WANDB_ENABLED=true
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce the model size or change precision:

```bash
TF_MODEL_NAME=facebook/musicgen-small
TF_MODEL_PRECISION=float16
```

### Subnet Registration

Register your wallet on the subnet before starting the miner:

```bash
btcli subnet register --netuid 234 --subtensor.network test
```

### Axon Port Conflict

If the default port is in use, set a different one:

```bash
TF_AXON_PORT=8092
```

### Model Download Failure

If automatic download fails, check your internet connection and try the manual download script:

```bash
bash scripts/download_models.sh
```

### Low Scores

Check logs for per-signal scoring breakdowns. The most common cause of low scores is poor prompt adherence. Ensure you are running the largest model your hardware supports.

### No Challenges Received

Verify the following:

1. Your wallet is registered on the correct subnet and network.
2. The axon port is reachable from the internet (check firewall rules).
3. The miner process is running without errors.

```bash
btcli wallet overview --wallet.name your_wallet --subtensor.network test
```

---

## Upgrading

Pull the latest code and restart:

```bash
cd tuneforge
git pull origin main
pip install -e .
```

Then restart via your preferred method:

```bash
# PM2
pm2 restart tuneforge-miner-1

# Docker
docker compose up miner -d --build
```
