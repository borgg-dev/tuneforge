# TuneForge Validator Setup Guide

This document covers everything needed to run a TuneForge validator: hardware requirements, installation, configuration, the scoring pipeline, weight mechanics, and troubleshooting.

---

## What a Validator Does

A TuneForge validator performs the following duties:

1. **Generates diverse music challenges** from a combinatorial space of genre, mood, tempo, key signature, instruments, vocals/lyrics requests, and creative constraints (100,000+ unique combinations per challenge). Duration range: 1--180 seconds.
2. **Sends challenges to miners** in batches via Bittensor dendrite.
3. **Scores responses** across 16 independent scoring dimensions, 4 penalty multipliers, and a set of hard-zero conditions.
4. **Applies multi-scale evaluation** that adjusts scorer weights based on the requested duration (short, medium, long).
5. **Applies genre-aware scoring** with per-genre targets across 9 genre families.
6. **Maintains an EMA leaderboard** per miner, smoothing raw scores over time. EMA state is persisted to disk.
7. **Applies tiered power-law weighting** where the top 10 miners share 80% of weight and the rest share 20%.
8. **Submits normalized weights** to the Bittensor chain every 115 blocks.
9. **Handles organic requests** by routing to a single top miner by EMA (no fan-out).
10. **Optionally pushes round data** to the platform API (`TF_VALIDATOR_API_URL`).
11. **Auto-updates the preference model** from the platform API (checks hourly, downloads if SHA256 changed).

Entry point:

```bash
python -m neurons.validator --env-file .env.validator
```

---

## Hardware Requirements

### Recommended Setups

| Setup | GPU | CPU | RAM | Disk | Scoring Time (10 miners) | Notes |
|-------|-----|-----|-----|------|--------------------------|-------|
| **GPU (recommended)** | RTX 3060+ (8 GB VRAM) | 4+ cores | 16 GB | 50 GB SSD | ~30-60s | CLAP, MERT, Whisper run on GPU — fits comfortably within the 240s round interval |
| **CPU-only** | None | 8+ cores | 32 GB | 50 GB SSD | ~120-180s | All scoring models on CPU — tight but feasible within the 240s round interval |

**Network:** 50 Mbps up/down minimum. The validator must be reachable on its axon port and needs stable connectivity to the Bittensor chain.

### GPU vs. CPU Scoring

The validator runs 16 scoring models per miner per round. Three of these models (CLAP, MERT, Whisper) are neural networks that benefit significantly from GPU acceleration:

| Scoring Model | GPU Time (per miner) | CPU Time (per miner) | VRAM |
|---------------|----------------------|----------------------|------|
| CLAP (text-audio similarity) | ~1-2s | ~5-10s | ~2.5 GB |
| MERT (neural quality) | ~1-2s | ~5-10s | ~1 GB |
| Whisper (vocal/lyrics) | ~2-3s | ~10-15s | ~1.5 GB |
| Other 13 scorers (librosa-based) | ~5-10s | ~5-10s | None (CPU-only) |
| **Total per miner** | **~10-15s** | **~25-45s** | **~5 GB** |

With 10 miners per round and a 240-second round interval, GPU scoring leaves plenty of headroom. CPU-only scoring is feasible but leaves little margin — if the subnet grows beyond ~15 active miners, CPU-only validators may not finish scoring in time.

**Recommendation:** Use a GPU if available. An 8 GB VRAM card (e.g., RTX 3060, T4) is sufficient — the validator does not run generation models, only scoring models. If you're already running a miner on the same machine, the scoring models share GPU memory alongside the generation model (total ~21 GB for MusicGen Large + scoring).

### Running Miner + Validator on One GPU

A single RTX 4090 (24 GB VRAM) can run both a MusicGen Large miner (~16 GB) and the validator scoring models (~5 GB) simultaneously. The miner generates audio during the round, then the validator scores it — the peak VRAM usage is ~21 GB since generation and scoring don't overlap.

For GPUs with less VRAM (16 GB), use a lighter miner model (Stable Audio ~6 GB or DiffRhythm ~6-8 GB) alongside the validator (~5 GB) for a total of ~11-13 GB.

### Disk Space Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| Python packages + PyTorch + CUDA | ~8 GB | Installed once |
| CLAP model | ~2.5 GB | Downloaded on first run |
| MERT model | ~0.5 GB | Downloaded on first run |
| Whisper model | ~1.5 GB | Downloaded on first run |
| OS + system packages | ~4-5 GB | Depends on base image |
| EMA state, logs | ~1 GB | Grows slowly over time |
| Audio storage (if no platform API) | ~5 GB/day | Cleaned automatically if using cron; not needed if `TF_VALIDATOR_API_URL` is set |
| Headroom | ~5 GB | Recommended buffer |

50 GB is sufficient. If also running a miner on the same machine, add the miner's model storage (see [Miner Setup Guide](miner_setup.md#disk-space-breakdown)).

### Deployment Options

**VPS or bare-metal (simplest):** Your machine has a public IP. Set `TF_AXON_PORT` and ensure the port is open in your firewall. No additional configuration needed.

**Docker / Vast.ai / NAT:** Your process runs behind a port mapping layer. Set these additional variables so the axon registers the correct public address on-chain:

```bash
TF_AXON_PORT=18384                  # Port the process listens on inside the container
TF_AXON_EXTERNAL_PORT=15941         # Public-facing port (e.g., Vast.ai mapped port)
TF_AXON_EXTERNAL_IP=203.0.113.10   # Your public IP
```

If running the organic API behind a reverse proxy (e.g., Caddy on Vast.ai), set `TF_ORGANIC_API_PORT` to the internal port that the proxy forwards to.

> **Vast.ai note:** Vast.ai runs Caddy, TensorBoard, and Jupyter on some mapped ports by default. You may need to stop these services and reclaim the ports. Disable them via `supervisorctl stop tensorboard jupyter` and set `autostart=false` in their supervisor config files under `/etc/supervisor/conf.d/`. Caddy can be reconfigured to proxy without authentication by editing `/etc/Caddyfile`.

---

## Prerequisites

- Python 3.10, 3.11, or 3.12
- A Bittensor wallet with sufficient α (alpha) stake for a validator permit
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
| `TF_AXON_EXTERNAL_PORT` | int | None | Public-facing port for Docker/NAT setups |
| `TF_AXON_EXTERNAL_IP` | str | None | Public IP for Docker/NAT setups |

### Epoch and Round Timing

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_ROUND_INTERVAL` | int | `240` | Seconds between rounds within an epoch |
| `TF_EPOCH_INTERVAL` | int | `1080` | Seconds per epoch (sync + rounds + cooldown) |
| `TF_WEIGHT_UPDATE_INTERVAL` | int | `115` | Blocks between weight submissions |
| `TF_METAGRAPH_SYNC_INTERVAL` | int | `1200` | Seconds between metagraph syncs |

Each epoch runs: 60s commit-reveal sync, then 4 rounds at 240s each, then 60s cooldown. Total epoch duration is 1080s (~18 minutes). These timing constants are defined in `tuneforge/__init__.py`:

| Constant | Value |
|----------|-------|
| `MAX_ROUNDS_PER_EPOCH` | 4 |
| `DEFAULT_ROUND_INTERVAL` | 240s |
| `EPOCH_SYNC` | 60s |
| `EPOCH_COOLDOWN` | 60s |
| `DEFAULT_EPOCH_INTERVAL` | 1080s |

### EMA Persistence

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_EMA_STATE_PATH` | str | `./ema_state.json` | Path to persist EMA state |
| `TF_EMA_SAVE_INTERVAL` | int | `5` | Save EMA state every N rounds |

### Models and Storage

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_PREFERENCE_MODEL_PATH` | str | None | Path to a trained preference model |
| `TF_FAD_REFERENCE_STATS_PATH` | str | `./reference_fad_stats.npz` | Path to FAD reference statistics |
| `TF_STORAGE_PATH` | str | `./storage` | Storage directory for snapshots and audio |
| `TF_ACOUSTID_API_KEY` | str | `""` | AcoustID API key for fingerprint penalty (optional) |

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

### Minimal Example

```env
TF_NETUID=234
TF_SUBTENSOR_NETWORK=test
TF_WALLET_NAME=default
TF_WALLET_HOTKEY=default
TF_MODE=validator
TF_ROUND_INTERVAL=240
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

Every round, the validator scores each miner response across 16 dimensions, applies 4 penalty multipliers, and checks hard-zero conditions. The final score formula is:

```
final = composite * duration_penalty * artifact_penalty * fad_penalty * fingerprint_penalty
```

All scoring weights and thresholds described below are **hardcoded for consensus**. Validators cannot modify them. This ensures all validators produce consistent scores.

### The 16 Scoring Dimensions

| # | Scorer | Weight | Description |
|---|--------|--------|-------------|
| 1 | **CLAP Adherence** | 0.19 | Text-audio similarity using `laion/clap-htsat-fused`. Raw cosine similarity between text prompt embedding and audio embedding is remapped from [0.05, 0.45] to [0, 1]. The primary signal: does the audio match the prompt? |
| 2 | **Attribute Verification** | 0.11 | Verifies specific musical attributes (tempo, key, instruments) via librosa analysis and CLAP zero-shot classification. |
| 3 | **Musicality** | 0.09 | Pitch stability, harmonic progression, chord coherence, rhythmic groove, and arrangement quality. Genre-aware targets. One-sided minimum floor (not bell curve). |
| 4 | **Vocal Lyrics** | 0.08 | Whisper-based lyrics intelligibility, vocal clarity, pitch accuracy, expressiveness, and sibilance control. Returns neutral 0.5 for instrumental genres unless `vocals_requested=True`. |
| 5 | **Preference Model** | 0.07 base | Perceptual quality via trained preference head. Bootstrap mode returns neutral 0.5 (effective weight 0%). Auto-scales from 2% to 20% via PreferenceWeightScaler as the model trains and gains accuracy. |
| 6 | **Melody Coherence** | 0.06 | Interval quality, pitch contour analysis, repetition structure, and memorability. One-sided minimum floor. |
| 7 | **Structural Completeness** | 0.06 | Section count, section variety, intro/outro presence, and transition quality. One-sided minimum floor. No longer penalizes electronic music or strong openings. |
| 8 | **Diversity** | 0.06 | CLAP embedding diversity across a 50-entry history per miner. Computed as 70% intra-miner diversity + 30% population-level diversity bonus. Miners recycling similar outputs score low. |
| 9 | **Production Quality** | 0.05 | Spectral balance, frequency fullness, loudness consistency (LUFS), dynamic expressiveness, and stereo width. Genre-aware targets. |
| 10 | **Neural Quality (MERT)** | 0.05 | Uses `m-a-p/MERT-v1-95M` learned representations to assess temporal coherence, activation strength, layer agreement, and periodicity. |
| 11 | **Vocal Quality** | 0.04 | Vocal presence, clarity, pitch consistency, and harmonic richness. Genre-aware: returns neutral 0.5 for instrumental genres. When `vocals_requested=True`, vocal scorer weight is boosted 1.5x and vocal_lyrics is boosted 2x. |
| 12 | **Mix Separation** | 0.04 | Spectral clarity, frequency masking analysis, spatial depth, low-end definition, and mid-range presence. |
| 13 | **Timbral Naturalness** | 0.03 | Spectral envelope naturalness, harmonic decay characteristics, transient quality, temporal envelope, and spectral flux consistency. |
| 14 | **Learned MOS** | 0.03 | Multi-resolution perceptual quality estimation: waveform quality, codec robustness, perceptual loudness, and harmonic richness. |
| 15 | **Audio Quality** | 0.02 | Signal-level analysis: harmonic ratio, onset quality, spectral contrast, dynamic range, and temporal variation. |
| 16 | **Speed** | 0.02 | Duration-relative speed scoring (see below). |

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

### 4 Penalty Multipliers

These are applied multiplicatively to the composite score after dimension scoring.

| Penalty | Trigger | Effect |
|---------|---------|--------|
| **Duration** | Requested vs. actual duration mismatch | Within 20% tolerance: no penalty. Linear decay from 1.0 to 0.0 between 20% and 50% deviation. Beyond 50%: multiplier = 0.0. |
| **Artifact** | Clipping, loops, spectral discontinuities, spectral holes | Multiplier on final score based on artifact severity. |
| **FAD** | Per-miner Frechet Audio Distance | Sigmoid curve with midpoint=15, steepness=2, floor=0.5. Reference stats loaded from `TF_FAD_REFERENCE_STATS_PATH`. |
| **Fingerprint** | AcoustID known-song match | Multiplier 0.0--1.0 based on match score (threshold 0.80). Requires `TF_ACOUSTID_API_KEY`. |

### Hard-Zero Conditions

These override the final score to 0.0 regardless of dimension scores:

| Condition | Trigger |
|-----------|---------|
| **Silence** | Audio RMS below 0.01 |
| **Timeout** | Round-trip time exceeds 300 seconds |

### Anti-Gaming Measures

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
| EMA_NEW_MINER_SEED | 0.0 | New miners start at 0.0 and build up from their first scored round. |

EMA state is persisted to disk at the path specified by `TF_EMA_STATE_PATH` (default: `./ema_state.json`), saved every 5 rounds (`TF_EMA_SAVE_INTERVAL`). This means validator restarts do not lose EMA history. The state file format is version 3 and includes per-UID hotkey tracking for UID recycling detection.

### UID Recycling Detection

Each time the metagraph syncs, the validator compares current hotkeys against previously stored values. When a UID's hotkey changes (indicating a new miner registered on that slot), its EMA and round count are reset to 0.0 so the new miner starts fresh. Stale UIDs no longer present in the metagraph are pruned from the leaderboard.

### Tiered Power-Law Weighting

Miners are ranked by EMA and split into two tiers for weight distribution:

Key parameters (hardcoded):

| Parameter | Value | Description |
|-----------|-------|-------------|
| ELITE_K | 10 | Number of miners in the elite tier |
| ELITE_POOL | 0.80 | Fraction of total weight reserved for elite tier |
| STEEPEN_POWER | 2.0 | Power-law exponent within each tier |

- **Elite tier** (top 10 miners by EMA): share **80%** of total weight
- **Remaining miners**: share **20%** of total weight

Within each tier, weight is distributed proportionally to `ema ^ power`:

| Tier | EMA Score | Weight Share (example with 20 miners) |
|------|-----------|---------------------------------------|
| Elite (#1) | 0.90 | ~14.5% |
| Elite (#5) | 0.80 | ~11.5% |
| Elite (#10) | 0.65 | ~7.6% |
| Rest (#11) | 0.60 | ~4.5% |
| Rest (#20) | 0.40 | ~2.0% |

The tier boundary creates a sharp incentive cliff: breaking into the top 10 provides roughly a 4x weight multiplier. When fewer than 10 miners are active, all share 100% of the weight pool.

### Weight Submission

- Weights are submitted to the chain every 115 blocks (`TF_WEIGHT_UPDATE_INTERVAL`).
- All weights are normalized to sum to 1.0.
- No coverage-gating is applied.
- Uses `bt.utils.weight_utils.process_weights_for_netuid` for chain compatibility.
- The validator waits for finalization and inclusion before proceeding.

---

## Organic Requests

Organic requests from the SaaS backend are handled directly by the validator via its built-in HTTP API (port 8090 by default, controlled by `TF_ORGANIC_API_PORT`). Disable with `TF_ORGANIC_API_ENABLED=false`.

The flow:

1. SaaS backend sends a POST to the validator's `/organic/generate` endpoint.
2. Validator selects a single miner by EMA using a weighted selection strategy. Only miners with EMA >= 0.45 (`MIN_EMA_THRESHOLD`) are eligible.
3. The prompt is sent to the selected miner (120s timeout).
4. On failure, the router falls back to the next-best available miner.
5. The result is returned to the SaaS backend.

Organic requests do **not** affect miner scores or weights. The challenge pipeline is the sole quality signal.

Key design properties:

- **No fan-out**: one miner per request to avoid wasting compute.
- **Load-balanced**: miners are selected weighted by EMA score, with active request tracking to avoid overloading individual miners.
- **Coexistence**: organic requests run concurrently with rounds on the same async event loop. Scoring models are protected by an asyncio lock, and CPU-bound scoring runs in a thread pool executor.

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
- **Leaderboard summary** is logged each round: total miners scored, number with weight, elite count, EMA mean and max.
- **Per-miner detail** is logged each round: raw reward, current EMA, and computed weight.
- **Leaderboard snapshot** is saved to `storage/leaderboard.json` after each round.
- **Score breakdowns** are stored in PostgreSQL and viewable in the mining dashboard.

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

Ensure `TF_EMA_STATE_PATH` points to a persistent location (not a tmpfs). The default `./ema_state.json` saves to the working directory. EMA is saved every 5 rounds by default. A `.bak` backup file is maintained alongside the primary state file.

### Preference model download fails

Check that `TF_VALIDATOR_API_URL` is set correctly and the platform API is reachable. Verify that `TF_VALIDATOR_API_TOKEN` is valid. The validator will continue operating in bootstrap mode (neutral 0.5, effective weight 0%) if the download fails.

### FAD penalty too aggressive

Ensure `TF_FAD_REFERENCE_STATS_PATH` points to a valid reference statistics file (`./reference_fad_stats.npz` by default). If the file is missing, the FAD penalty may not function correctly. The FAD penalty has a floor of 0.5, so it cannot reduce scores below half.

---

## License

TuneForge is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.
