# TuneForge Validator Setup Guide

This document covers everything needed to run a TuneForge validator: hardware requirements, installation, configuration, the scoring pipeline, weight mechanics, and troubleshooting.

---

## What a Validator Does

A TuneForge validator performs the following duties:

1. **Generates diverse music challenges** from a combinatorial space of genre, mood, tempo, key signature, instruments, vocals/lyrics requests, and creative constraints (100,000+ unique combinations per challenge). Duration range: 1--180 seconds.
2. **Sends challenges to miners** in batches via Bittensor dendrite.
3. **Scores responses** across 18 independent scoring dimensions, 3 penalty multipliers, and a set of hard-zero conditions.
4. **Applies multi-scale evaluation** that adjusts scorer weights based on the requested duration (short, medium, long).
5. **Applies genre-aware scoring** with per-genre targets across 9 genre families.
6. **Maintains an EMA leaderboard** per miner, smoothing raw scores over time. EMA state is persisted to disk.
7. **Applies a steepening function** with a soft sigmoid floor to amplify high performers and suppress low performers.
8. **Submits normalized weights** to the Bittensor chain every 115 blocks.
9. **Handles organic SaaS queries** by fan-out to top miners, scoring responses, and updating EMA.
10. **Optionally pushes round data** to the platform API (`TF_VALIDATOR_API_URL`).
11. **Auto-updates the preference model** from the platform API (checks hourly, downloads if SHA256 changed).

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

A GPU is not required but significantly reduces scoring latency for the CLAP, MERT, and Whisper models.

---

## Prerequisites

- Python 3.10, 3.11, or 3.12
- A Bittensor wallet with sufficient TAO stake for a validator permit
- Registration on the subnet (`btcli subnet register`)
- System packages: `git`, `ffmpeg`, `libsndfile`

---

## Installation

```bash
git clone https://github.com/tuneforge-ai/tuneforge.git
cd tuneforge
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

---

## Configuration

Create a file named `.env.validator` in the project root. All variables use the `TF_` prefix and are loaded by pydantic-settings.

**Important:** Scoring weights and thresholds are hardcoded in the source code for consensus. Validators cannot modify them via environment variables. Only operational parameters listed below are configurable.

### Core Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_NETUID` | int | `0` | Subnet UID |
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
| `TF_WEIGHT_UPDATE_INTERVAL` | int | `115` | Blocks between weight submissions |
| `TF_METAGRAPH_SYNC_INTERVAL` | int | `1200` | Seconds between metagraph syncs |

### EMA Persistence

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_EMA_STATE_PATH` | str | `./ema_state.json` | Path to persist EMA state |
| `TF_EMA_SAVE_INTERVAL` | int | `5` | Save EMA state every N blocks |

### Models and Storage

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_PREFERENCE_MODEL_PATH` | str | None | Path to a trained preference model |
| `TF_FAD_REFERENCE_STATS_PATH` | str | `./reference_fad_stats.npz` | Path to FAD reference statistics |
| `TF_STORAGE_PATH` | str | `./storage` | Storage directory for snapshots and audio |

### Security

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_VALIDATOR_PERTURBATION_SECRET` | str | auto-generated | Private nonce for weight perturbation seed. Never transmitted to miners. Auto-generated if not set, but set explicitly for reproducibility across restarts. |

### Platform API Integration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_VALIDATOR_API_URL` | str | `""` | Platform API URL for pushing round data and preference model updates |
| `TF_VALIDATOR_API_TOKEN` | str | `""` | Bearer token for platform API authentication |

### Organic API

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_ORGANIC_API_ENABLED` | bool | `true` | Enable the organic generation HTTP API |
| `TF_ORGANIC_API_PORT` | int | `8090` | Port for the organic generation API |

### Logging and Monitoring

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_LOG_LEVEL` | str | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `TF_LOG_DIR` | str | `/tmp/tuneforge` | Directory for log files |
| `TF_WANDB_ENABLED` | bool | `false` | Enable Weights & Biases logging |
| `TF_WANDB_ENTITY` | str | None | W&B entity (team or username) |
| `TF_WANDB_PROJECT` | str | None | W&B project name |

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

Every validation round, the validator scores each miner response across 18 dimensions, applies 3 penalty multipliers, checks hard-zero conditions, and applies anti-gaming perturbation. The final score formula is:

```
final = composite * duration_penalty * artifact_penalty * fad_penalty
```

All scoring weights and thresholds described below are **hardcoded for consensus**. Validators cannot modify them. This ensures all validators produce consistent scores.

### The 18 Scoring Dimensions

| # | Scorer | Weight | Description |
|---|--------|--------|-------------|
| 1 | **CLAP Adherence** | 0.15 | Text-audio similarity using `laion/larger_clap_music`. Raw cosine similarity between text prompt embedding and audio embedding is remapped from [0.15, 0.75] to [0, 1]. The primary signal: does the audio match the prompt? |
| 2 | **Attribute Verification** | 0.09 | Verifies specific musical attributes (tempo, key, instruments) via librosa analysis and CLAP zero-shot classification. |
| 3 | **Musicality** | 0.09 | Pitch stability, harmonic progression, chord coherence, rhythmic groove, and arrangement quality. Genre-aware targets. One-sided minimum floor (not bell curve). |
| 4 | **Vocal Lyrics** | 0.08 | Whisper-based lyrics intelligibility, vocal clarity, pitch accuracy, expressiveness, and sibilance control. Returns neutral 0.5 for instrumental genres unless `vocals_requested=True`. |
| 5 | **Diversity** | 0.08 | CLAP embedding diversity across a 50-entry history per miner. Computed as 70% intra-miner diversity + 30% population-level diversity bonus. Miners recycling similar outputs score low. |
| 6 | **Preference Model** | 0.07 base | Perceptual quality via trained preference head. Bootstrap mode returns neutral 0.5 (effective weight 0%). Auto-scales from 2% to 20% via PreferenceWeightScaler as the model trains and gains accuracy. |
| 7 | **Melody Coherence** | 0.06 | Interval quality, pitch contour analysis, repetition structure, and memorability. One-sided minimum floor. |
| 8 | **Structural Completeness** | 0.06 | Section count, section variety, intro/outro presence, and transition quality. One-sided minimum floor. No longer penalizes electronic music or strong openings. |
| 9 | **Production Quality** | 0.05 | Spectral balance, frequency fullness, loudness consistency (LUFS), dynamic expressiveness, and stereo width. Genre-aware targets. |
| 10 | **Neural Quality (MERT)** | 0.05 | Uses `m-a-p/MERT-v1-95M` learned representations to assess temporal coherence, activation strength, layer agreement, and periodicity. |
| 11 | **Timbral Naturalness** | 0.05 | Spectral envelope naturalness, harmonic decay characteristics, transient quality, temporal envelope, and spectral flux consistency. |
| 12 | **Vocal Quality** | 0.04 | Vocal presence, clarity, pitch consistency, and harmonic richness. Genre-aware: returns neutral 0.5 for instrumental genres. When `vocals_requested=True`, vocal scorer weight is boosted 1.5x and vocal_lyrics is boosted 2x. |
| 13 | **Mix Separation** | 0.04 | Spectral clarity, frequency masking analysis, spatial depth, low-end definition, and mid-range presence. |
| 14 | **Learned MOS** | 0.03 | Multi-resolution perceptual quality estimation: waveform quality, codec robustness, perceptual loudness, and harmonic richness. |
| 15 | **Audio Quality** | 0.02 | Signal-level analysis: harmonic ratio, onset quality, spectral contrast, dynamic range, and temporal variation. |
| 16 | **Speed** | 0.02 | Duration-relative speed scoring (see below). |
| 17 | **Perceptual Quality** | 0.01 | Bandwidth consistency, signal-to-noise ratio, harmonic noise ratio, and high-frequency presence. |
| 18 | **Neural Codec** | 0.01 | EnCodec reconstruction quality and naturalness assessment. |

### Speed Scoring

Speed is scored relative to the requested duration, not as an absolute time:

```
ratio = generation_time / requested_duration
```

| Ratio | Score |
|-------|-------|
| <= 1.0 | 1.0 |
| 3.0 | 0.3 |
| >= 6.0 | 0.0 |

Uses validator-measured `dendrite.process_time`, not miner-reported time. If no validator timing is available, defaults to 0.5.

### 3 Penalty Multipliers

These are applied multiplicatively to the composite score after dimension scoring.

| Penalty | Trigger | Effect |
|---------|---------|--------|
| **Duration** | Requested vs. actual duration mismatch | Within 20% tolerance: no penalty. Linear decay from 1.0 to 0.0 between 20% and 50% deviation. Beyond 50%: multiplier = 0.0. |
| **Artifact** | Clipping, loops, spectral discontinuities, spectral holes | Multiplier on final score based on artifact severity. |
| **FAD** | Per-miner Frechet Audio Distance | Sigmoid curve with midpoint=15, steepness=2, floor=0.5. Reference stats loaded from `TF_FAD_REFERENCE_STATS_PATH`. |

### Hard-Zero Conditions

These override the final score to 0.0 regardless of dimension scores:

| Condition | Trigger |
|-----------|---------|
| **Silence** | Audio RMS below 0.01 |
| **Timeout** | Round-trip time exceeds 300 seconds |

### Anti-Gaming Measures

**Weight Perturbation:** Each round, every non-zero scoring dimension weight receives a deterministic perturbation of up to +/-30%. The seed is computed as `SHA256(challenge_id + TF_VALIDATOR_PERTURBATION_SECRET)`. The secret is a private nonce that is never transmitted to miners. Without the secret, miners cannot reconstruct the perturbed weights from the open-source code. All weights are perturbed individually, then renormalized to sum to 1.0.

**Scorer Dropout:** Each round, 10% of non-zero scorers are randomly dropped (their weight redistributed). This adds further unpredictability to the scoring pipeline.

**Hardcoded Weights:** All scoring weights and thresholds are hardcoded in the source code, not configurable via environment variables. This ensures consensus across all validators.

### Multi-Scale Evaluation

Scorer weights are adjusted based on the requested audio duration:

**Short (<10s):** Production, quality, timbral, mix separation, and learned MOS weights are increased. Structural, musicality, and melody weights are decreased. Short clips are judged primarily on sonic quality.

**Medium (10--30s):** Baseline multipliers (all 1.0). No adjustment.

**Long (>=30s):** Structural weight is boosted to 1.8x, melody to 1.5x, musicality to 1.3x. Speed weight is reduced to 0.5x, production to 0.8x. Long-form bonuses are available:
- `phrase_coherence_bonus`: up to +0.05 added directly to the composite score
- `compositional_arc_bonus`: up to +0.05 added directly to the composite score

### Genre-Aware Scoring

The scoring system recognizes 9 genre families: electronic, rock, classical-cinematic, ambient, hip-hop, jazz-blues, folk-acoustic, groove-soul, and pop. Each family defines per-genre targets for:

- Dynamic range
- Onset density
- Rhythmic groove
- Spectral balance
- Loudness
- Other style-specific attributes

Vocal scorers return a neutral 0.5 for instrumental genres by default. When `vocals_requested=True` is set in the challenge synapse, vocal_lyrics weight is boosted 2x and vocal weight is boosted 1.5x.

### Conditional Targets and Progressive Difficulty

**ConditionalTargetDeriver:** Extracts quality targets from the prompt text via keyword matching, setting genre-specific expectations for each scorer.

**ProgressiveDifficultyManager:** Tracks the network's overall quality via an EMA and scales challenge difficulty accordingly. As the network improves, challenges become harder.

---

## Challenge Protocol

The challenge synapse includes the following fields:

- **prompt**: Text description of the music to generate
- **duration**: Requested duration in seconds (1--180)
- **vocals_requested**: Boolean indicating whether vocals are expected
- **lyrics**: Optional lyrics text when vocals are requested
- **genre**, **mood**, **tempo**, **key_signature**, **instruments**: Musical parameters
- **creative_constraints**: Additional structural or sonic requirements

Challenges are assembled from a large combinatorial space:

- 40 genres
- 35+ moods
- 24 key signatures
- 6 time signatures
- Genre-specific instrument pools and tempo ranges
- Duration range: 1--180 seconds
- 15 prompt templates with varied structural ordering
- 4 creative constraint pools (emotional arcs, structural requirements, sonic textures, dynamic instructions), each with 10 options

This yields over 100,000 unique combinations per challenge, making it infeasible for miners to memorize or cache responses.

---

## EMA Leaderboard and Weight Setting

### EMA (Exponential Moving Average)

Each miner's score is smoothed over time using an EMA:

```
ema_new = alpha * raw_score + (1 - alpha) * ema_old
```

Key parameters (hardcoded):

| Parameter | Value | Description |
|-----------|-------|-------------|
| EMA_ALPHA | 0.2 | Smoothing factor. Higher = more responsive to recent rounds. |
| EMA_NEW_MINER_SEED | 0.25 | New miners start at 0.25, not 0.0, giving them a fair ramp-up period. |

EMA state is persisted to disk at the path specified by `TF_EMA_STATE_PATH` (default: `./ema_state.json`), saved every 5 blocks (`TF_EMA_SAVE_INTERVAL`). This means validator restarts do not lose EMA history.

### Steepening Function

The steepening function converts EMA scores into weights, amplifying top performers and suppressing underperformers. It uses a soft sigmoid floor (not a hard cliff).

Key parameters (hardcoded):

| Parameter | Value |
|-----------|-------|
| STEEPEN_BASELINE | 0.45 |
| STEEPEN_POWER | 2.0 |

Miners with EMA at or below the baseline (0.45) receive near-zero weight. Above the baseline:

```
weight = ((ema - baseline) / (1 - baseline)) ^ power
```

Example calculations with baseline = 0.45, power = 2.0:

| EMA Score | Calculation | Weight |
|-----------|-------------|--------|
| 0.45 | 0.0 | 0.000 |
| 0.55 | ((0.55 - 0.45) / 0.55)^2 = 0.182^2 | 0.033 |
| 0.65 | ((0.65 - 0.45) / 0.55)^2 = 0.364^2 | 0.132 |
| 0.80 | ((0.80 - 0.45) / 0.55)^2 = 0.636^2 | 0.405 |
| 0.95 | ((0.95 - 0.45) / 0.55)^2 = 0.909^2 | 0.826 |

Only consistently high-quality miners earn meaningful rewards.

### Weight Submission

- Weights are submitted to the chain every 115 blocks (`TF_WEIGHT_UPDATE_INTERVAL`).
- All weights are normalized to sum to 1.0.
- Uses `bt.utils.weight_utils.process_weights_for_netuid` for chain compatibility.
- The validator waits for finalization and inclusion before proceeding.

---

## Organic Generation

Organic requests from the SaaS backend are handled directly by the validator via its built-in HTTP API (port 8090 by default, controlled by `TF_ORGANIC_API_PORT`). Disable with `TF_ORGANIC_API_ENABLED=false`.

The flow:

1. SaaS backend sends a POST to the validator's `/organic/generate` endpoint.
2. Validator selects the top miners by EMA using a composite miner selection strategy.
3. Fan-out: the prompt is sent to K=3 miners concurrently.
4. All valid responses pass through a quality gate using the same 18-scorer pipeline used for challenges.
5. Scores update the EMA leaderboard (organic and challenge scores share the same EMA).
6. The best result (by composite score) is returned to the SaaS backend.

Key design properties:

- **Low latency**: top-K fan-out with quality gating keeps response time manageable.
- **Coexistence**: organic requests run concurrently with challenge rounds on the same async event loop. Scoring models are protected by an asyncio lock, and CPU-bound scoring runs in a thread pool executor.
- **Fair EMA**: organic scores feed the same EMA as challenges. Poor performance on organic requests lowers a miner's EMA just like a bad challenge round.

---

## Preference Model

The preference model adds a human-aligned quality signal to the scoring pipeline.

- **Bootstrap mode** (default): returns a neutral 0.5 score. The preference weight is effectively 0% in bootstrap mode.
- **Trained mode**: when `TF_PREFERENCE_MODEL_PATH` is set, the validator loads a `PreferenceHead` (512-dim CLAP input) or `DualPreferenceHead` (1280-dim CLAP+MERT input). Training uses Bradley-Terry pairwise loss.
- **Auto-scaling**: `PreferenceWeightScaler` adjusts the preference scorer weight between 2% and 20% based on model accuracy (accuracy range 0.55--0.80). Low accuracy = low weight.
- **Training pipeline**: `tools/export_and_train.py` exports human annotations from the crowd annotation system, computes embeddings, and trains the preference head.
- **Auto-update**: the validator checks the platform API endpoint `/api/v1/annotations/model/latest` every hour. If the SHA256 of the latest model differs from the current one, it downloads and loads the new model automatically.

---

## Monitoring

- **Log files** are written to `TF_LOG_DIR` (default: `/tmp/tuneforge`).
- **Leaderboard summary** is logged each round: total miners scored, number above baseline, EMA mean and max.
- **Per-miner detail** is logged each round: raw reward, current EMA, and computed weight.
- **Leaderboard snapshot** is saved to `storage/leaderboard.json` after each round.
- **Weights & Biases**: enable with `TF_WANDB_ENABLED=true` for real-time dashboards of scoring metrics, EMA progression, and weight distributions.

---

## Troubleshooting

### No miners responding

Check that the metagraph is syncing correctly (`TF_METAGRAPH_SYNC_INTERVAL`), miners are registered on the subnet, and miners are actively serving on their advertised endpoints.

### All scores are zero

Verify that audio decoding is working (ffmpeg and libsndfile installed) and that the scoring models (CLAP, MERT, Whisper) downloaded successfully. Check logs for model loading errors. Also check that miners are not timing out (300s limit).

### Weight setting fails

Confirm that the validator hotkey has sufficient stake for a validator permit, the wallet is accessible, and the chain connection is stable. Check for errors in the subtensor communication logs.

### High scoring latency

If scoring is slow, use a GPU for CLAP, MERT, and Whisper inference. Ensure `CUDA_VISIBLE_DEVICES` is not set to `-1` in your environment. The `ecosystem.config.js` default disables GPU; remove that setting for GPU-accelerated scoring.

### EMA state lost after restart

Ensure `TF_EMA_STATE_PATH` points to a persistent location (not a tmpfs). The default `./ema_state.json` saves to the working directory. EMA is saved every 5 blocks by default.

### Preference model download fails

Check that `TF_VALIDATOR_API_URL` is set correctly and the platform API is reachable. Verify that `TF_VALIDATOR_API_TOKEN` is valid. The validator will continue operating in bootstrap mode (neutral 0.5, effective weight 0%) if the download fails.

### FAD penalty too aggressive

Ensure `TF_FAD_REFERENCE_STATS_PATH` points to a valid reference statistics file (`./reference_fad_stats.npz` by default). If the file is missing, the FAD penalty may not function correctly. The FAD penalty has a floor of 0.5, so it cannot reduce scores below half.
