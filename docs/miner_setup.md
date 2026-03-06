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
git clone https://github.com/tuneforge-ai/tuneforge.git
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

Note: Scoring weights are hardcoded on the validator side and are not configurable via environment variables. This ensures consensus across all validators.

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

Speed accounts for only 2% of the total score. Quality and adherence signals collectively dominate. Use the largest model your GPU can handle.

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

Understanding the scoring system helps you maximize rewards. The validator evaluates your audio across 18 scoring signals and applies 4 penalty multipliers.

### Scoring Signals (18 scorers)

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| CLAP Adherence | 15% | Text-audio similarity using `laion/larger_clap_music`. Raw cosine similarity mapped from [0.15, 0.75] to [0, 1]. |
| Attribute | 9% | Attribute-level quality assessment |
| Musicality | 9% | Pitch, harmony, rhythm, arrangement quality |
| Vocal Lyrics | 8% | Whisper-based lyrics intelligibility, vocal clarity, pitch accuracy, expressiveness, sibilance |
| Diversity | 8% | CLAP embedding diversity across your last 50 outputs (70% intra-miner, 30% population-level bonus) |
| Preference Model | 7% | Perceptual quality scoring (0% during bootstrap, auto-scales 2-20% once trained) |
| Melody Coherence | 6% | Melodic intervals, contour, structure |
| Structural Completeness | 6% | Section detection, song form, compositional arc |
| Production Quality | 5% | Spectral balance, loudness (LUFS), dynamics |
| Neural Quality (MERT) | 5% | Learned music representations via `m-a-p/MERT-v1-95M` |
| Timbral Naturalness | 5% | Spectral envelope, harmonic decay, transient quality |
| Vocal Quality | 4% | Vocal presence, clarity, pitch accuracy (genre-aware) |
| Mix Separation | 4% | Spectral clarity, frequency masking, spatial depth |
| Learned MOS | 3% | Multi-resolution perceptual quality estimation |
| Audio Quality | 2% | Signal-level analysis (harmonic ratio, onsets, contrast) |
| Speed | 2% | Duration-relative generation speed (see below) |
| Perceptual | 1% | Perceptual audio quality |
| Neural Codec | 1% | EnCodec reconstruction quality |

### Speed Scoring

Speed is evaluated relative to the requested duration, not as an absolute value:

- **ratio** = generation_time / requested_duration
- ratio <= 1.0: score 1.0 (generated faster than real-time)
- ratio = 3.0: score 0.3
- ratio >= 6.0: score 0.0

The generation time is measured by the validator via `dendrite.process_time`, not self-reported by the miner.

### Final Score Calculation

The composite score from all 18 scorers is multiplied by four penalty factors:

```
final_score = composite * duration_penalty * artifact_penalty * fad_penalty * soft_plagiarism_penalty
```

### Penalty Multipliers (4)

| Penalty | Trigger | Effect |
|---------|---------|--------|
| Duration | Off-target by more than 20% | Linear penalty reaching 0.0 at 50% deviation |
| Artifact | Clipping, loops, spectral discontinuities | Multiplier 0-1 applied to final score |
| FAD | Per-miner Frechet Audio Distance | Sigmoid curve, floor 0.5 |
| Soft Plagiarism | Cosine similarity in [0.65, 0.72] zone | Cosine-shaped penalty from 1.0 down to 0.05 |

### Hard Penalties (score = 0)

| Condition | Threshold |
|-----------|-----------|
| Silence | RMS below 0.01 |
| Timeout | Round-trip exceeds 300 seconds |
| Self-plagiarism | Self-similarity above 0.72 |
| Cross-miner plagiarism | Cross-miner similarity above 0.70 |

Plagiarism detection uses canonical-form comparison: audio is normalized to pitch C and tempo 120 BPM before computing CLAP embeddings.

### Multi-Scale Evaluation

The system adjusts scoring emphasis based on requested duration:

- **Short (<10s):** Higher weight on production, quality, and timbral naturalness.
- **Medium (10-30s):** Baseline weights.
- **Long (>=30s):** Higher weight on structure, melody, and composition. Bonuses for phrase coherence (+0.05) and compositional arc (+0.05).

### Genre-Aware Scoring

The system recognizes 9 genre families with per-genre quality targets:

- Vocal scorers return a neutral 0.5 for instrumental genres (ambient, electronic, classical-cinematic).
- When `vocals_requested=True`, the genre gate is overridden and vocal weights are boosted (2x vocal_lyrics, 1.5x vocal).

### Anti-Gaming Measures

- **Weight perturbation:** Scoring weights shift by up to 30% each round, seeded by `SHA256(challenge_id + validator_secret)`. The validator secret is never shared with miners.
- **Scorer dropout:** 10% of non-zero scorers are randomly dropped each round.
- **Hardcoded weights:** Scoring weights are not configurable via env vars, ensuring all validators score identically.

### Optimization Tips

- **Prompt adherence (CLAP, 15%) is the single largest signal.** Your model must faithfully follow the text prompt.
- **Quality signals dominate.** Larger models consistently score higher across all quality dimensions.
- **Diversity (8%):** Do not recycle outputs. The system tracks your last 50 CLAP embeddings and compares against the full population.
- **Speed (2%):** Generate faster than real-time if possible, but do not sacrifice quality. Speed is only 2% of the total score.
- **Duration accuracy matters.** Stay within 20% of the requested duration to avoid the duration penalty.
- **Organic queries** (where `is_organic=True`) do not affect your validation score.
- **New miners** start with an EMA score of 0.25 (not 0.0), so you will not be immediately penalized while ramping up.
- **MAX_DURATION is 180 seconds.** Validators may request audio up to 3 minutes long.

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

Check logs for per-signal scoring breakdowns. The most common causes of low scores are:

1. **Poor prompt adherence** -- the largest single signal at 15%.
2. **Running a small model** -- larger models score higher across nearly all quality dimensions.
3. **Duration mismatch** -- generating audio that is too short or too long relative to the request.
4. **Repeated outputs** -- diversity tracking covers 50 recent generations and includes cross-miner comparison.

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

For full scoring details, see the project README.
