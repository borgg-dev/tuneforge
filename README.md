<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/banner-dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="assets/brand/banner-light.png" />
    <img alt="TuneForge" src="assets/brand/banner-light.png" width="420" />
  </picture>

  <h2>Decentralized AI Music Generation on Bittensor</h2>

  <a href="#"><img src="https://img.shields.io/badge/python-3.10--3.12-blue" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC--BY--NC--4.0-green" /></a>
  <a href="#"><img src="https://img.shields.io/badge/bittensor-subnet-purple" /></a>
  <br/>

  <p>TuneForge is a Bittensor subnet where miners compete to generate music from text prompts.<br/>
  Validators issue challenges, score the returned audio across 16 quality signals with penalty<br/>
  multipliers, and set on-chain weights that determine α emissions. The scoring pipeline is<br/>
  model-agnostic — it evaluates the audio, not the architecture. Miners ship with MusicGen Large,<br/>
  Stable Audio Open, and ACE-Step 1.5 as baselines, but the ones earning real weight will be the ones who bring<br/>
  better models, fine-tune aggressively, or build something entirely new.</p>

  <p><strong>Testnet netuid: 234</strong> · <strong>Mainnet: TBD</strong></p>

  <p>
    <a href="docs/validator_setup.md"><strong>Run a Validator Node</strong></a> · <a href="docs/miner_setup.md"><strong>Run a Miner Node</strong></a>
  </p>
</div>

---

## Table of Contents

- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Scoring System](#scoring-system)
- [Anti-Gaming](#anti-gaming)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [License](#license)

## How It Works

The network runs in repeating validation rounds. Each round:

1. **Challenge** -- The validator generates a text-to-music prompt (tempo, key, genre, instruments, duration) from a combinatorial space of 100k+ possible challenges and sends it to a batch of miners via Bittensor dendrite.

2. **Generate** -- Each miner runs its generation backend (MusicGen, Stable Audio, ACE-Step, or any custom model), produces audio, and returns it via axon.

3. **Score** -- The validator runs the audio through 16 weighted scorers covering prompt adherence, composition, production quality, naturalness, and more. Penalty multipliers (duration, artifacts, FAD, fingerprint) are applied on top. Multi-scale evaluation adjusts weights based on audio duration, and genre-aware profiles set appropriate quality targets.

4. **Update** -- Round scores feed into a per-miner EMA (alpha=0.2). Miners are ranked by EMA and split into two tiers: the top 10 share 80% of total weight, everyone else shares 20%. Within each tier, weight follows a quadratic power law. Weights are submitted on-chain every 115 blocks.

5. **Organic** -- Real user requests from the SaaS platform flow through the validator, which routes them to top miners. Miners must handle organic requests the same way they handle challenges — there is no opt-out.

## Quick Start

### Prerequisites

- Python 3.10 - 3.12
- NVIDIA GPU with CUDA (miners)
- A registered Bittensor wallet with a hotkey on subnet 234 (testnet)

### Installation

```bash
git clone https://github.com/tuneforge-ai/tuneforge.git
cd tuneforge
pip install -e .
```

Or use the setup script:

```bash
bash scripts/setup.sh
```

### Run a Miner

```bash
cp .env.miner.example .env
# Edit .env with your wallet, netuid, and GPU settings
bash scripts/run_miner.sh
```

See [docs/miner_setup.md](docs/miner_setup.md) for the complete miner guide including GPU requirements, model selection, and tuning.

### Run a Validator

```bash
cp .env.validator.example .env
# Edit .env with your wallet and netuid
bash scripts/run_validator.sh
```

See [docs/validator_setup.md](docs/validator_setup.md) for the complete validator guide.

### Docker

```bash
# Miner (GPU)
docker build -f Dockerfile.miner -t tuneforge-miner .
docker run --gpus all --env-file .env tuneforge-miner

# Validator (CPU)
docker build -f Dockerfile.validator -t tuneforge-validator .
docker run --env-file .env tuneforge-validator
```

## Scoring System

Every round, miners are scored across 16 weighted signals grouped into five categories. Weights are consensus-critical constants hardcoded in `tuneforge/config/scoring_config.py` and sum to 1.0. Penalty multipliers are applied to the final composite, not as weighted components.

### Scoring Signals

| Signal | Weight | Category |
|--------|--------|----------|
| CLAP Adherence | 19% | Prompt Adherence |
| Attribute Verification | 11% | Prompt Adherence |
| Musicality | 9% | Composition |
| Melody Coherence | 6% | Composition |
| Structural Completeness | 6% | Composition |
| Vocal and Lyrics | 8% | Naturalness and Mix |
| Mix Separation | 4% | Naturalness and Mix |
| Timbral Naturalness | 3% | Naturalness and Mix |
| Learned MOS | 3% | Naturalness and Mix |
| Neural Quality (MERT) | 5% | Production and Fidelity |
| Production Quality | 5% | Production and Fidelity |
| Vocal Quality | 4% | Production and Fidelity |
| Audio Quality | 2% | Production and Fidelity |
| Preference Model | 7% base (0% bootstrap, 2-20% trained) | Learned Preference |
| Diversity | 6% | Other |
| Speed | 2% | Other |

**CLAP Adherence** (19%) is the biggest single signal. It measures text-audio cosine similarity using `laion/larger_clap_music`, mapped from a floor of 0.15 to a ceiling of 0.75. If your audio doesn't match the prompt, nothing else saves you.

**Attribute Verification** (11%) checks concrete prompt compliance -- tempo, key, instruments -- using librosa analysis and CLAP zero-shot classification.

**Composition** (21% total) covers musicality (pitch stability, harmonic progression, rhythmic groove), melody coherence (intervals, contour, memorability), and structural completeness (section detection, form, transitions).

**Naturalness and Mix** (18%) includes Whisper-based lyrics intelligibility, timbral envelope analysis, spectral clarity and frequency masking, and multi-resolution perceptual quality estimation. Vocal scorers are genre-aware: instrumental genres (ambient, electronic, classical-cinematic) receive neutral 0.5 so vocal absence doesn't penalize genuinely instrumental music. When the prompt requests vocals, genre gates are overridden and vocal weights are boosted (2x for vocal/lyrics, 1.5x for vocal quality) then renormalized.

**Neural Quality** (5%) uses MERT (`m-a-p/MERT-v1-95M`) to evaluate temporal coherence, activation strength, layer agreement, and structural periodicity.

**Preference Model** (7% base) starts in bootstrap mode returning neutral 0.5. As crowd annotations accumulate, the preference MLP trains via Bradley-Terry pairwise loss and its weight auto-scales from 2% (accuracy 0.55) to 20% (accuracy 0.80).

**Diversity** (6%) tracks CLAP embedding variety across a miner's recent 50 submissions with a population-level diversity bonus. Copy-paste strategies get punished.

**Speed** (2%) uses a duration-relative curve: real-time or faster scores 1.0, 3x realtime scores 0.3, 6x or slower scores 0.0. A 60-second track generated in 60 seconds scores the same as a 10-second track in 10 seconds.

### Penalties

```
final_score = composite * duration_penalty * artifact_penalty * fad_penalty * fingerprint_penalty
```

| Penalty | Trigger | Effect |
|---------|---------|--------|
| Silence | RMS below 0.01 | Hard zero |
| Timeout | Exceeds 300s | Hard zero |
| Duration | Off-target by >20% | Linear penalty (1.0 at 20% to 0.0 at 50%) |
| Artifacts | Clipping, loops, spectral discontinuities | Geometric mean of 4 checks (floor 0.1 each) |
| FAD | Frechet Audio Distance divergence | Sigmoid penalty (floor 0.5) |
| Fingerprint | Chromaprint dedup + AcoustID known-song match | Multiplier 0.0-1.0 |

### Multi-Scale Evaluation

Scoring weights shift based on audio duration:

- **Short** (< 10s) -- emphasizes production quality, audio fidelity, timbral quality
- **Medium** (10-30s) -- balanced baseline weights
- **Long** (>= 30s) -- emphasizes structure, melody, composition, vocals; up to +0.10 bonus for phrase coherence and compositional arc

## Anti-Gaming

**FAD Penalty.** Per-miner Frechet Audio Distance measures how far a miner's output distribution diverges from real music. Sigmoid penalty with a floor of 0.5.

**Fingerprint Detection.** Chromaprint deduplication catches identical submissions. AcoustID lookup catches audio ripped from commercial recordings. Both apply as multipliers on the final score.

**Diversity Tracking.** CLAP embeddings from each miner's last 50 outputs are tracked. Scoring combines intra-miner variety (70%) with population-level differentiation (30%).

**EMA Smoothing.** Alpha of 0.2 means one great round doesn't catapult you to the top. Consistent quality over time is required.

## Configuration Reference

All configuration uses environment variables with the `TF_` prefix. Set them in `.env` or pass directly.

### Network

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_NETUID` | int | 0 | Subnet network UID |
| `TF_SUBTENSOR_NETWORK` | str | None | Network (finney, test, local) |
| `TF_SUBTENSOR_CHAIN_ENDPOINT` | str | None | Custom chain endpoint URL |

### Wallet

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_WALLET_NAME` | str | default | Wallet name |
| `TF_WALLET_HOTKEY` | str | default | Hotkey name |
| `TF_WALLET_PATH` | str | ~/.bittensor/wallets | Wallet path |

### Neuron

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_MODE` | str | miner | Runtime mode (miner/validator) |
| `TF_NEURON_EPOCH_LENGTH` | int | 100 | Blocks between weight updates |
| `TF_NEURON_TIMEOUT` | int | 120 | Forward timeout (seconds) |
| `TF_NEURON_AXON_OFF` | bool | false | Disable axon serving |
| `TF_AXON_PORT` | int | None | Axon port |

### Generation (Miner)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HF_TOKEN` | str | None | HuggingFace token (required for gated models like Stable Audio) |
| `TF_MODEL_NAME` | str | facebook/musicgen-large | Generation model |
| `TF_GENERATION_MAX_DURATION` | int | 30 | Max duration (seconds) |
| `TF_GENERATION_SAMPLE_RATE` | int | 32000 | Sample rate (Hz) |
| `TF_GENERATION_TIMEOUT` | int | 120 | Generation timeout (seconds) |
| `TF_GPU_DEVICE` | str | cuda:0 | GPU device |
| `TF_MODEL_PRECISION` | str | float16 | Precision (float32/float16/bfloat16) |
| `TF_GUIDANCE_SCALE` | float | 3.0 | Classifier-free guidance scale |
| `TF_TEMPERATURE` | float | 1.0 | Sampling temperature |
| `TF_TOP_K` | int | 250 | Top-K sampling |
| `TF_TOP_P` | float | 0.0 | Nucleus sampling (0 = disabled) |

### Operational

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_ROUND_INTERVAL` | int | 240 | Seconds between validation rounds |
| `TF_EPOCH_INTERVAL` | int | 1140 | Seconds per epoch (sync + rounds + cooldown) |
| `TF_WEIGHT_SETTER_STEP` | int | 115 | Blocks between weight submissions |
| `TF_EMA_STATE_PATH` | str | ./ema_state.json | EMA persistence file |
| `TF_EMA_SAVE_INTERVAL` | int | 5 | Blocks between EMA saves |
| `TF_FAD_REFERENCE_STATS_PATH` | str | ./reference_fad_stats.npz | FAD reference statistics |
| `TF_PREFERENCE_MODEL_PATH` | str | None | Preference model checkpoint |


### Organic API (Validator)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_ORGANIC_API_ENABLED` | bool | true | Enable organic generation API |
| `TF_ORGANIC_API_PORT` | int | 8090 | Organic API port |

### API / Server

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_API_HOST` | str | 0.0.0.0 | API server host |
| `TF_API_PORT` | int | 8000 | API server port |
| `TF_STORAGE_PATH` | str | ./storage | Local storage path |
| `TF_FRONTEND_URL` | str | http://localhost:3000 | Frontend URL (CORS) |

### Logging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TF_LOG_LEVEL` | str | INFO | Log level |
| `TF_LOG_DIR` | str | /tmp/tuneforge | Log directory |
| `TF_WANDB_ENABLED` | bool | false | Enable W&B logging |
| `TF_WANDB_ENTITY` | str | None | W&B entity |
| `TF_WANDB_PROJECT` | str | tuneforge | W&B project |

### Consensus-Critical Constants

All scoring weights, thresholds, EMA parameters, and penalty curves are hardcoded in `tuneforge/config/scoring_config.py`. They are not configurable. All validators must use identical values to maintain consensus. Changing them will cause your validator to diverge from the network.

## Project Structure

```
tuneforge/
├── neurons/
│   ├── miner.py                    -- Miner entry point
│   └── validator.py                -- Validator entry point
├── tuneforge/
│   ├── __init__.py                 -- Version, constants
│   ├── settings.py                 -- Pydantic settings (TF_ env vars)
│   ├── config/
│   │   └── scoring_config.py       -- All scoring weights and thresholds
│   ├── base/
│   │   ├── neuron.py               -- Base neuron class
│   │   ├── miner.py                -- Base miner
│   │   ├── validator.py            -- Base validator
│   │   ├── protocol.py             -- Synapse definitions
│   │   └── dendrite.py             -- Dendrite response tracking
│   ├── core/
│   │   ├── miner.py                -- TuneForgeMiner implementation
│   │   └── validator.py            -- TuneForgeValidator implementation
│   ├── generation/
│   │   ├── model_manager.py        -- Backend manager (lazy load, GPU monitor)
│   │   ├── musicgen_backend.py     -- MusicGen backend (default)
│   │   ├── stable_audio_backend.py -- Stable Audio backend
│   │   ├── ace_step_backend.py     -- ACE-Step 1.5 backend
│   │   ├── audio_utils.py          -- Audio normalization, encoding
│   │   └── prompt_parser.py        -- Prompt builder
│   ├── rewards/
│   │   ├── reward.py               -- Composite scoring
│   │   ├── leaderboard.py          -- EMA leaderboard + tiered weighting
│   │   ├── weight_setter.py        -- On-chain weight submission
│   │   └── scoring.py              -- Task-level scorer
│   ├── scoring/                    -- 16 active scorers + penalties
│   │   ├── clap_scorer.py          -- CLAP text-audio similarity
│   │   ├── attribute_verifier.py   -- Prompt compliance
│   │   ├── musicality.py           -- Pitch, harmony, rhythm
│   │   ├── melody_coherence.py     -- Melodic intervals, contour
│   │   ├── structural_completeness.py -- Section detection, form
│   │   ├── neural_quality.py       -- MERT learned representations
│   │   ├── production_quality.py   -- Spectral balance, LUFS, dynamics
│   │   ├── vocal_quality.py        -- Vocal clarity, pitch
│   │   ├── vocal_lyrics.py         -- Whisper-based lyrics scoring
│   │   ├── audio_quality.py        -- Signal-level analysis
│   │   ├── timbral_naturalness.py  -- Spectral envelope, transients
│   │   ├── mix_separation.py       -- Frequency masking, spatial depth
│   │   ├── learned_mos.py          -- Perceptual quality estimation
│   │   ├── preference_model.py     -- Preference MLP + auto-scaler
│   │   ├── diversity.py            -- CLAP embedding diversity
│   │   ├── fad_scorer.py           -- Frechet Audio Distance
│   │   ├── artifact_detector.py    -- Clipping, loops, discontinuity
│   │   ├── fingerprint_scorer.py   -- Chromaprint + AcoustID
│   │   ├── genre_profiles.py       -- Genre-aware quality targets
│   │   └── multi_scale.py          -- Duration-based weight adjustment
│   ├── validation/
│   │   ├── prompt_generator.py     -- Challenge prompt generation
│   │   └── challenge_manager.py    -- Challenge tracking
│   ├── api/
│   │   ├── server.py               -- SaaS platform (FastAPI)
│   │   ├── validator_api.py        -- Organic API (validator-side)
│   │   ├── organic_router.py       -- Organic generation router
│   │   └── routes/                 -- API route handlers
│   └── utils/                      -- Logging, config, weight helpers
├── scripts/
│   ├── setup.sh                    -- Environment setup
│   ├── download_models.sh          -- Model download
│   ├── run_miner.sh                -- Miner launcher
│   └── run_validator.sh            -- Validator launcher
├── tools/
│   ├── train_preference.py         -- Preference model training
│   ├── build_embedding_cache.py    -- CLAP embedding cache
│   ├── build_reference_stats.py    -- FAD reference stats
│   └── calibrate_mert.py           -- MERT calibration
├── tests/                          -- Test suite (pytest)
├── docs/
│   ├── miner_setup.md              -- Miner guide
│   ├── validator_setup.md          -- Validator guide
│   └── setup.md                    -- Architecture reference
├── Dockerfile.miner                -- Miner container (NVIDIA CUDA)
├── Dockerfile.validator            -- Validator container
├── docker-compose.yml              -- Docker services
├── pyproject.toml                  -- Dependencies and metadata
├── .env.miner.example              -- Miner config template
└── .env.validator.example          -- Validator config template
```

## License

This project is licensed under [CC BY-NC 4.0](LICENSE) (Creative Commons Attribution-NonCommercial 4.0 International).
