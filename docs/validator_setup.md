# TuneForge Validator Setup Guide

This document covers everything needed to run a TuneForge validator: hardware requirements, installation, configuration, the scoring pipeline, weight mechanics, and troubleshooting.

---

## What a Validator Does

A TuneForge validator performs the following duties:

1. **Generates diverse music challenges** from a combinatorial space of genre, mood, tempo, key signature, instruments, and creative constraints (100,000+ unique combinations per challenge).
2. **Sends challenges to miners** in batches via Bittensor dendrite.
3. **Scores responses** across 11 independent dimensions plus a penalty system.
4. **Maintains an EMA leaderboard** per miner, smoothing raw scores over time.
5. **Applies a steepening function** to amplify high performers and zero out low performers.
6. **Submits normalized weights** to the Bittensor chain every 115 blocks.
7. **Handles organic SaaS queries** by fan-out to top miners, scoring responses, and updating EMA.
8. **Optionally pushes round data** to the platform API (`TF_VALIDATOR_API_URL`).
9. **Auto-updates the preference model** from the platform API (checks hourly, downloads if SHA256 changed).

Entry point:

```bash
python -m neurons.validator --env-file .env.validator
```

---

## Hardware Requirements

These values come from `min_compute.yml`.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU cores | 4 | 4+ |
| RAM | 16 GB | 16 GB |
| Disk | 50 GB SSD | 50 GB SSD |
| Network | 50 Mbps up/down | 50 Mbps up/down |
| GPU | Optional | 8 GB+ VRAM for fast CLAP/MERT scoring |

A GPU is not required but significantly reduces scoring latency for the CLAP and MERT models.

---

## Prerequisites

- Python 3.10, 3.11, or 3.12
- A Bittensor wallet with sufficient TAO stake for a validator permit
- Registration on the subnet (`btcli subnet register`)
- System packages: `git`, `ffmpeg`, `libsndfile`

---

## Installation

```bash
git clone https://github.com/tuneforge-subnet/tuneforge.git
cd tuneforge
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

---

## Configuration

Create a file named `.env.validator` in the project root. All variables use the `TF_` prefix and are loaded by pydantic-settings.

### Core Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_NETUID` | int | `0` | Subnet UID (234 on testnet) |
| `TF_SUBTENSOR_NETWORK` | str | None | Network name: `finney`, `test`, or `local` |
| `TF_SUBTENSOR_CHAIN_ENDPOINT` | str | None | Custom chain endpoint URL |
| `TF_WALLET_NAME` | str | `default` | Bittensor wallet name |
| `TF_WALLET_HOTKEY` | str | `default` | Hotkey name |
| `TF_WALLET_PATH` | str | `~/.bittensor/wallets` | Path to wallet directory |
| `TF_MODE` | str | `validator` | Runtime mode |
| `TF_AXON_PORT` | int | None | Axon port |

### Validation Timing and Batching

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_VALIDATION_INTERVAL` | int | `300` | Seconds between validation rounds |
| `TF_CHALLENGE_BATCH_SIZE` | int | `8` | Number of miners to challenge per round |
| `TF_MAX_CONCURRENT_VALIDATIONS` | int | `4` | Maximum concurrent validation tasks |
| `TF_GENERATION_TIMEOUT` | int | `120` | Timeout for miner responses in seconds |

### Weight and EMA Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_WEIGHT_UPDATE_INTERVAL` | int | `115` | Blocks between weight submissions |
| `TF_EMA_ALPHA` | float | `0.2` | EMA smoothing factor |
| `TF_STEEPEN_BASELINE` | float | `0.50` | Minimum EMA for nonzero weight |
| `TF_STEEPEN_POWER` | float | `2.0` | Steepening exponent |
| `TF_WEIGHT_PERTURBATION` | float | `0.20` | Per-round weight perturbation (plus or minus %) |

### Scoring Dimension Weights

These control the relative importance of each scoring dimension. They must sum to 1.0 (before perturbation).

| Variable | Default | Signal |
|----------|---------|--------|
| `TF_WEIGHT_CLAP` | `0.30` | CLAP adherence (prompt-audio similarity) |
| `TF_WEIGHT_MUSICALITY` | `0.10` | Musicality |
| `TF_WEIGHT_NEURAL_QUALITY` | `0.10` | Neural quality (MERT) |
| `TF_WEIGHT_PRODUCTION` | `0.08` | Production quality |
| `TF_WEIGHT_MELODY` | `0.07` | Melody coherence |
| `TF_WEIGHT_STRUCTURAL` | `0.07` | Structural completeness |
| `TF_WEIGHT_QUALITY` | `0.06` | Audio quality |
| `TF_WEIGHT_PREFERENCE` | `0.06` | Preference model |
| `TF_WEIGHT_VOCAL` | `0.06` | Vocal quality |
| `TF_WEIGHT_DIVERSITY` | `0.05` | Diversity |
| `TF_WEIGHT_SPEED` | `0.05` | Speed |
| `TF_WEIGHT_ATTRIBUTE` | `0.00` | Attribute verification (opt-in) |

### Scoring Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `TF_SELF_PLAGIARISM_THRESHOLD` | `0.80` | CLAP similarity above which audio is flagged as plagiarism |
| `TF_SILENCE_THRESHOLD` | `0.01` | RMS level below which audio is considered silent |
| `TF_DURATION_TOLERANCE` | `0.20` | Duration deviation within 20% incurs no penalty |
| `TF_DURATION_TOLERANCE_MAX` | `0.50` | Duration deviation at 50% or beyond results in score 0 |

### Models and Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `TF_CLAP_MODEL` | `laion/clap-htsat-unfused` | CLAP model identifier |
| `TF_MERT_MODEL` | `m-a-p/MERT-v1-95M` | MERT model identifier |
| `TF_PREFERENCE_MODEL_PATH` | None | Path to a trained preference model |
| `TF_STORAGE_PATH` | `./storage` | Storage directory for snapshots and audio |

### Platform API Integration

| Variable | Default | Description |
|----------|---------|-------------|
| `TF_VALIDATOR_API_URL` | `""` | Platform API URL for pushing round data |
| `TF_VALIDATOR_API_TOKEN` | `""` | Bearer token for platform API authentication |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `TF_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `TF_LOG_DIR` | `/tmp/tuneforge` | Directory for log files |

### Minimal Example

```env
TF_NETUID=234
TF_SUBTENSOR_NETWORK=test
TF_WALLET_NAME=default
TF_WALLET_HOTKEY=default
TF_MODE=validator
TF_VALIDATION_INTERVAL=300
TF_LOG_LEVEL=INFO
```

---

## Running the Validator

### Direct

```bash
source venv/bin/activate
python -m neurons.validator --env-file .env.validator
```

### PM2

```bash
pm2 start ecosystem.config.js --only tuneforge-validator
pm2 logs tuneforge-validator
```

Note: `ecosystem.config.js` sets `CUDA_VISIBLE_DEVICES=-1` (CPU-only) for the validator by default. Remove or change this if you want GPU-accelerated scoring.

### Docker

```bash
docker compose up validator -d
```

The `Dockerfile.validator` uses `python:3.11-slim` (no CUDA) and pre-downloads the CLAP model at build time.

---

## The Scoring Pipeline

Every validation round, the validator scores each miner response across 11 dimensions and applies penalties. The final score is a weighted sum of dimension scores, multiplied by any applicable penalty factors.

### 1. CLAP Adherence (30%)

Measures text-audio similarity using CLAP (`laion/clap-htsat-unfused`). The raw cosine similarity between the text prompt embedding and the audio embedding is remapped from the range [0.15, 0.60] to [0, 1]. This is the single most important signal: does the generated audio match the prompt?

### 2. Musicality (10%)

Evaluates pitch stability, harmonic structure, rhythm consistency, and arrangement quality. Scoring targets are genre-aware, applying different expectations per style.

### 3. Neural Quality / MERT (10%)

Uses MERT (`m-a-p/MERT-v1-95M`) learned music representations to assess temporal coherence, layer agreement, periodicity, and representation norm. Scoring uses a bell-curve distribution around calibrated centers.

### 4. Production Quality (8%)

Analyzes spectral balance, LUFS loudness (ITU-R BS.1770-4), and dynamics. Targets are genre-aware.

### 5. Melody Coherence (7%)

Evaluates melodic intervals, pitch contour, and melodic structure.

### 6. Structural Completeness (7%)

Performs section detection and song form analysis. Expectations vary by genre.

### 7. Audio Quality (6%)

Signal-level analysis with five sub-metrics: harmonic ratio (25%), onset quality (20%), spectral contrast (20%), dynamic range (15%), and temporal variation (20%). Sub-metric weights are configurable via `TF_QW_*` variables.

### 8. Preference Model (6%)

Perceptual quality scoring. Uses a bootstrap heuristic by default when no trained model is available. When `TF_PREFERENCE_MODEL_PATH` is set, loads a trained `PreferenceHead` that scores MERT embeddings.

### 9. Vocal Quality (6%)

Assesses vocal presence, clarity, and pitch accuracy. Genre-aware: vocals are expected in pop and rock but not in ambient or electronic genres.

### 10. Diversity (5%)

Tracks CLAP embeddings of each miner's outputs over time. Miners that recycle similar outputs receive low diversity scores. Computed per-batch across all responses.

### 11. Speed (5%)

Rewards fast generation. Scoring curve:

- 5 seconds or less: 1.0
- 30 seconds: 0.3
- 60 seconds or more: 0.0

Uses validator-measured round-trip time (`dendrite.process_time`), not miner-reported time. If no validator timing is available, defaults to 0.5.

### 12. Attribute Verification (0% by default)

Can verify specific musical attributes. Weight is 0 by default; opt in by raising `TF_WEIGHT_ATTRIBUTE`.

### Penalty System

Penalties are applied after dimension scoring. Hard-zero penalties override the final score entirely.

| Penalty | Trigger | Effect |
|---------|---------|--------|
| Silence | Audio RMS below 0.01 | Hard zero -- final score = 0.0 |
| Timeout | Generation exceeds 120s | Hard zero -- final score = 0.0 |
| Plagiarism | Self-similarity above 0.80 | Hard zero -- final score = 0.0 |
| Duration | Off-target by more than 20% | Linear penalty from 1.0 to 0.0 at 50% deviation |
| Artifacts | Spectral discontinuities, clipping, loops | Multiplier on final score |

### Weight Perturbation (Anti-Gaming)

Each round, every scoring dimension weight receives a deterministic perturbation of up to plus or minus 20%, seeded by the `challenge_id`. All weights are perturbed individually, then renormalized to sum to 1.0. This prevents miners from optimizing for a fixed weight distribution.

Set `TF_WEIGHT_PERTURBATION=0.0` to disable perturbation.

---

## EMA Leaderboard and Weight Setting

### EMA (Exponential Moving Average)

Each miner's score is smoothed over time using an EMA:

```
ema_new = alpha * raw_score + (1 - alpha) * ema_old
```

where `alpha = 0.2` by default (`TF_EMA_ALPHA`). The first score for a miner initializes the EMA directly. Higher alpha values make the EMA more responsive to recent rounds.

New miners start with EMA = 0.0 and ramp up gradually. A miner consistently scoring 0.7 takes ~8 rounds to cross the baseline.

### Steepening Function

The steepening function converts EMA scores into weights, amplifying top performers and zeroing out underperformers.

- Miners with EMA at or below the baseline (0.50) receive weight = 0.
- Above the baseline:

```
weight = ((ema - baseline) / (1 - baseline)) ^ power
```

With default settings (`baseline = 0.50`, `power = 2.0`):

| EMA Score | Calculation | Weight |
|-----------|-------------|--------|
| 0.50 | 0.0 | 0.000 |
| 0.60 | ((0.60 - 0.50) / 0.50)^2 = 0.20^2 | 0.040 |
| 0.70 | ((0.70 - 0.50) / 0.50)^2 = 0.40^2 | 0.160 |
| 0.90 | ((0.90 - 0.50) / 0.50)^2 = 0.80^2 | 0.640 |

Only consistently high-quality miners earn meaningful rewards.

### Weight Submission

- Weights are submitted to the chain every 115 blocks (`TF_WEIGHT_UPDATE_INTERVAL`).
- All weights are normalized to sum to 1.0.
- Uses `bt.utils.weight_utils.process_weights_for_netuid` for chain compatibility.
- The validator waits for finalization and inclusion before proceeding.

---

## Organic Generation

Organic requests from the SaaS backend are handled directly by the validator via its built-in HTTP API (port 8090 by default). The flow:

1. SaaS backend sends a POST to the validator's `/organic/generate` endpoint.
2. Validator selects the top 10 miners by EMA (best proven quality).
3. Fan-out: the prompt is sent to all 10 miners with a 30-second timeout.
4. All valid responses are scored with the same 11-signal pipeline used for challenges.
5. Scores update the EMA leaderboard (organic and challenge scores share the same EMA).
6. The top N results (by composite score) are returned to the SaaS backend.

Key design properties:
- **Low latency**: top-K selection (10 miners) + 30s timeout keeps response time ~20-25s.
- **Coexistence**: organic requests run concurrently with challenge rounds on the same async event loop. Scoring models are protected by an asyncio lock, and CPU-bound scoring runs in a thread pool executor.
- **Fair EMA**: organic scores feed the same EMA as challenges. Poor performance on organic requests lowers a miner's EMA just like a bad challenge round.

---

## Challenge Generation

Challenges are assembled from a large combinatorial space:

- 40 genres
- 35+ moods
- 24 key signatures
- 6 time signatures
- Genre-specific instrument pools and tempo ranges
- 8 duration options (5--30 seconds)
- 15 prompt templates with varied structural ordering
- 4 creative constraint pools (emotional arcs, structural requirements, sonic textures, dynamic instructions), each with 10 options

This yields over 100,000 unique combinations per challenge, making it infeasible for miners to memorize or cache responses.

---

## Preference Model

The preference model adds a human-aligned quality signal to the scoring pipeline.

- **Bootstrap mode** (default): a heuristic scorer runs when no trained model is available.
- **Trained mode**: when `TF_PREFERENCE_MODEL_PATH` is set, the validator loads a trained `PreferenceHead` that scores MERT embeddings.
- **Training pipeline**: `tools/export_and_train.py` exports human annotations, computes CLAP embeddings, and trains the preference head.
- **Auto-update**: the validator checks the platform API endpoint `/api/v1/annotations/model/latest` every hour. If the SHA256 of the latest model differs from the current one, it downloads and loads the new model automatically.

---

## Monitoring

- **Log files** are written to `TF_LOG_DIR` (default: `/tmp/tuneforge`).
- **Leaderboard summary** is logged each round: total miners scored, number above baseline, EMA mean and max.
- **Per-miner detail** is logged each round: raw reward, current EMA, and computed weight.
- **Leaderboard snapshot** is saved to `storage/leaderboard.json` after each round.

---

## Troubleshooting

### No miners responding

Check that the metagraph is syncing correctly, miners are registered on the subnet, and miners are actively serving on their advertised endpoints.

### All scores are zero

Verify that audio decoding is working (ffmpeg and libsndfile installed) and that the scoring models (CLAP, MERT) downloaded successfully. Check logs for model loading errors.

### Weight setting fails

Confirm that the validator hotkey has sufficient stake for a validator permit, the wallet is accessible, and the chain connection is stable. Check for errors in the subtensor communication logs.

### High scoring latency

If scoring is slow, use a GPU for CLAP and MERT inference. Ensure `CUDA_VISIBLE_DEVICES` is not set to `-1` in your environment. The `ecosystem.config.js` default disables GPU; remove that setting for GPU-accelerated scoring.

### Preference model download fails

Check that `TF_VALIDATOR_API_URL` is set correctly and the platform API is reachable. Verify that `TF_VALIDATOR_API_TOKEN` is valid. The validator will continue operating with the bootstrap heuristic if the download fails.
