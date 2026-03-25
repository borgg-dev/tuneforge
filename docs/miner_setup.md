# TuneForge Miner Setup Guide

This guide covers everything you need to run a miner on the TuneForge subnet. A miner receives music generation challenges from validators, produces audio, and earns α (alpha) rewards based on quality scores.

**The subnet's purpose is to incentivize competition and model improvement through Bittensor.** ACE-Step 1.5 is the recommended default backend -- it produces 48kHz stereo audio with vocals and lyrics support, requires under 4GB VRAM (turbo variant), and is MIT licensed. MusicGen Large, DiffRhythm, and HeartMuLa remain available as alternatives. Miners are strongly encouraged to bring their own models, fine-tune existing ones, or build entirely new generation pipelines. Any model that generates music from text prompts can be integrated. The scoring system rewards quality, not any specific model.

---

## What a Miner Does

A TuneForge miner performs the following:

- Receives `MusicGenerationSynapse` challenges from validators via the Bittensor axon.
- Generates audio using a configurable backend (ACE-Step 1.5, MusicGen Large, DiffRhythm, HeartMuLa, or your own custom model).
- Returns base64-encoded WAV audio along with metadata (`sample_rate`, `generation_time_ms`, `model_id`).
- Earns α (alpha) rewards proportional to quality scores assigned by validators.
- Handles organic requests from the SaaS API. Organic requests do not affect scoring.

---

## Hardware Requirements

### Recommended Setups by Model

| Setup | GPU | VRAM | CPU | RAM | Disk | Best For |
|-------|-----|------|-----|-----|------|----------|
| **Recommended (ACE-Step 1.5)** | RTX 3060 / T4 or better | 4 GB | 4 cores | 16 GB | 50 GB SSD | Default recommended backend, 48kHz stereo, vocals+lyrics, sub-10s generation |
| **Competitive (MusicGen Large)** | RTX 4090 / A100 | 24 GB | 8 cores | 32 GB | 50 GB SSD | MusicGen family, instrumental only, CC-BY-NC license |
| **Mid-range (MusicGen Medium)** | RTX 3090 / A10 | 16 GB | 4 cores | 16 GB | 50 GB SSD | Good balance of quality and cost |
| **Budget (DiffRhythm)** | RTX 3060 / T4 | 8 GB | 4 cores | 16 GB | 50 GB SSD | Lower VRAM models, still competitive on quality |
| **Entry (MusicGen Small)** | RTX 3060 | 6 GB | 4 cores | 16 GB | 30 GB SSD | Testing and development only |

**Network:** 50 Mbps up/down minimum. The miner must be reachable from the internet on its axon port.

### Baseline Model VRAM Usage

| Model | `TF_MODEL_NAME` | VRAM (fp16) | Speed (30s audio) | Sample Rate | Notes |
|-------|-----------------|-------------|--------------------| ------------|-------|
| **ACE-Step 1.5** (recommended) | `ace-step-1.5` | ~4 GB | sub-10s on 4090 | 48 kHz stereo | Recommended default. Vocals+lyrics, 50+ languages, up to 10 min, MIT license |
| MusicGen Large | `facebook/musicgen-large` | ~16 GB | ~20-40s on 4090 | 32 kHz mono | 3.3B params, instrumental only, no vocals. CC-BY-NC license (non-commercial) |
| DiffRhythm v1.2 (full) | `diffrhythm-full` | ~8-10 GB | ~5-10s on 4090 | 44.1 kHz stereo | Full-length songs up to 4m45s, vocal+lyrics support. Fast generation, decent quality |
| HeartMuLa 3B | `heartmula` | ~8-10 GB | ~10-30s on 4090 | 48 kHz | Vocals+lyrics. Open-source 3B has weak prompt adherence, mainly useful for vocal generation |
| HeartMuLa 7B | `heartmula-7b` | ~16-20 GB | ~20-60s on 4090 | 48 kHz | Higher quality vocals+lyrics, better prompt adherence than 3B |
| MusicGen Medium | `facebook/musicgen-medium` | ~8 GB | ~10-20s on 4090 | 32 kHz mono | Reduced quality vs. Large. CC-BY-NC license |
| MusicGen Small | `facebook/musicgen-small` | ~4 GB | ~5-10s on 4090 | 32 kHz mono | Lowest quality, for testing only. CC-BY-NC license |

These are baseline models to get you started. The scoring system is model-agnostic -- it evaluates audio quality, prompt adherence, musicality, and many other signals. Miners who develop or integrate superior models will earn higher scores and more α (alpha).

### Disk Space Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| Python packages + PyTorch + CUDA | ~8 GB | Installed once |
| ACE-Step 1.5 repo + checkpoints | ~5 GB | Repo clone + auto-downloaded from HuggingFace on first run |
| MusicGen Large model weights | ~7 GB | Only if using MusicGen |
| DiffRhythm model weights | ~4 GB | Only if using DiffRhythm |
| Lyrics generator (GPT-2) | ~0.5 GB | Loaded automatically for vocal support |
| OS + system packages | ~4-5 GB | Depends on base image |
| Logs, temp files, headroom | ~5 GB | Recommended buffer |

50 GB is sufficient for any single model. If running multiple model backends, increase to 80 GB.

### Deployment Options

**VPS or bare-metal (simplest):** Your machine has a public IP. Set `TF_AXON_PORT` and ensure the port is open in your firewall. No additional configuration needed.

**Docker / Vast.ai / NAT:** Your process runs behind a port mapping layer. The internal port differs from the external port. Set these additional variables so the axon registers the correct public address on-chain:

```bash
TF_AXON_PORT=8080                   # Port the process listens on inside the container
TF_AXON_EXTERNAL_PORT=15822         # Public-facing port (e.g., Vast.ai mapped port)
TF_AXON_EXTERNAL_IP=203.0.113.10   # Your public IP
```

Without `TF_AXON_EXTERNAL_PORT` and `TF_AXON_EXTERNAL_IP`, other nodes will try to connect to the internal container port and fail.

> **Vast.ai note:** Vast.ai runs Caddy, TensorBoard, and Jupyter on some mapped ports by default. You may need to stop these services and reclaim the ports for your miner. Disable them via `supervisorctl stop tensorboard jupyter` and set `autostart=false` in their supervisor config files under `/etc/supervisor/conf.d/`.

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
```

### ACE-Step 1.5 Setup (Recommended Default)

ACE-Step 1.5 is the recommended default backend. It produces 48kHz stereo audio with vocals and lyrics support in 50+ languages, generates up to 10 minutes of audio, requires under 4GB VRAM (turbo variant), and runs with sub-10s generation times. Quality is between Suno v4.5 and v5. It is MIT licensed (fully commercial).

```bash
# Clone the ACE-Step repo
git clone https://github.com/ace-step/ACE-Step-1.5.git ~/ace-step-repo
```

Model checkpoints are auto-downloaded from HuggingFace (`ACE-Step/Ace-Step1.5`) on the first run. No manual weight download is required.

Set `TF_MODEL_NAME=ace-step-1.5`, `TF_GENERATION_SAMPLE_RATE=48000`, and `TF_GENERATION_MAX_DURATION=180`.

### DiffRhythm v1.2 Setup (Alternative)

DiffRhythm is a latent diffusion model that generates full-length songs at 44.1kHz stereo. It uses ~8-10GB VRAM and supports vocals with lyrics. The full variant generates songs up to 4m45s.

```bash
# Clone the DiffRhythm repo
git clone https://github.com/ASLP-lab/DiffRhythm ~/DiffRhythm

# Install system dependency (required for lyrics phonemizer)
apt install espeak-ng

# Install DiffRhythm dependencies into the tuneforge venv
pip install -r ~/DiffRhythm/requirements.txt
```

The DiffRhythm repo must be located at `~/DiffRhythm` (the default path). To use a custom location, set the `DIFFRHYTHM_PATH` environment variable:

```bash
export DIFFRHYTHM_PATH=/path/to/DiffRhythm
```

Pre-download model weights before starting the miner to avoid timeout on first startup:

```bash
cd ~/DiffRhythm
source /path/to/tuneforge/venv/bin/activate

python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ASLP-lab/DiffRhythm-1_2-full', filename='cfm_model.pt', local_dir='pretrained')
hf_hub_download('ASLP-lab/DiffRhythm-vae', filename='vae_model.pt', local_dir='pretrained')
print('Done!')
"
```

The MuQ style encoder (~2 GB) is also downloaded automatically on first run. Set `TF_MODEL_NAME=diffrhythm-full` and `TF_GENERATION_SAMPLE_RATE=44100`.

### HeartMuLa Setup (Alternative -- Vocals)

HeartMuLa is an LLM-based music generation model (Llama 3.2 backbone) with lyrics/vocal support. It produces 48kHz audio. Available in 3B (~8-10GB VRAM) and 7B (~16-20GB VRAM) versions. Note: the open-source 3B version has weak prompt adherence and is mainly useful for vocal generation.

```bash
# Install heartlib
git clone https://github.com/HeartMuLa/heartlib.git
cd heartlib && pip install -e .
cd ..

# Download model checkpoints (3 repos required)
huggingface-cli download --local-dir ~/heartmula-ckpt HeartMuLa/HeartMuLaGen
huggingface-cli download --local-dir ~/heartmula-ckpt/HeartMuLa-oss-3B HeartMuLa/HeartMuLa-oss-3B-happy-new-year
huggingface-cli download --local-dir ~/heartmula-ckpt/HeartCodec-oss HeartMuLa/HeartCodec-oss-20260123
```

Set `TF_MODEL_NAME=heartmula` (3B) or `TF_MODEL_NAME=heartmula-7b` (7B), and `TF_GENERATION_SAMPLE_RATE=48000`. To use a custom checkpoint path, set `HEARTMULA_MODEL_PATH=/path/to/ckpt`.

### MusicGen Setup (Alternative -- Instrumental Only)

MusicGen Large is an alternative baseline model (~16GB VRAM, 3.3B parameters). Note: MusicGen is licensed under CC-BY-NC (non-commercial only), supports only 30s max duration, and does not support vocals. For GPUs with less than 16GB VRAM, use `facebook/musicgen-medium` instead. Install audiocraft separately because it pins specific torch versions:

```bash
pip install audiocraft --no-deps
```

Model weights are downloaded automatically from HuggingFace on first run.

### Vocals and Lyrics

The miner automatically handles vocal requests regardless of which backend is used:

- **Vocals with lyrics**: User provides lyrics text → miner passes them to the backend. For DiffRhythm, lyrics are auto-converted to LRC timestamp format.
- **Vocals without lyrics**: User requests vocals but provides no lyrics → miner uses a built-in lyrics generator (GPT-2 small, ~500MB VRAM) to create contextual lyrics from the prompt. Genre and mood are extracted from the prompt text to generate relevant lyrics.
- **Instrumental**: Default when vocals are not requested.

The lyrics generator loads alongside the music model on the miner's GPU. GPT-2 small requires ~500MB additional VRAM. It runs once during initialization and stays in memory for fast lyrics generation (~2-3s per request).

Backends that support vocals (like DiffRhythm) receive the generated lyrics via the `lyrics` parameter. Backends that don't support vocals (like MusicGen) simply ignore the parameter.

### Bringing Your Own Model

TuneForge is designed to be model-agnostic. If you have a custom music generation model, you can integrate it by implementing a backend class that follows the same interface as the existing backends in `tuneforge/generation/`. Your model just needs to accept a text prompt and duration, and return an audio array with a sample rate. If your model supports vocals, accept the `lyrics` keyword argument in your `generate()` method.

See the existing backends for reference:
- `tuneforge/generation/ace_step_backend.py`
- `tuneforge/generation/musicgen_backend.py`
- `tuneforge/generation/diffrhythm_backend.py`
- `tuneforge/generation/heartmula_backend.py`
- `tuneforge/generation/lyrics_generator.py` (lyrics generation + prompt analysis)

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
| `TF_AXON_EXTERNAL_PORT` | int | None | Public-facing port for Docker/NAT setups |
| `TF_AXON_EXTERNAL_IP` | str | None | Public IP for Docker/NAT setups |
| `TF_MODEL_NAME` | str | `ace-step-1.5` | Model to use for generation: `ace-step-1.5`, `facebook/musicgen-large`, `diffrhythm-full`, `heartmula`, `heartmula-7b` (see [Model Selection](#model-selection-guide)) |
| `TF_GENERATION_MAX_DURATION` | int | `180` | Maximum audio duration in seconds |
| `TF_GENERATION_SAMPLE_RATE` | int | `48000` | Audio sample rate in Hz |
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

### Minimal Example (ACE-Step 1.5)

```bash
TF_NETUID=234
TF_SUBTENSOR_NETWORK=test
TF_WALLET_NAME=my_wallet
TF_WALLET_HOTKEY=my_hotkey
TF_MODEL_NAME=ace-step-1.5
TF_GPU_DEVICE=cuda:0
TF_AXON_PORT=8091
TF_GENERATION_SAMPLE_RATE=48000
TF_GENERATION_MAX_DURATION=180
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

The `Dockerfile.miner` uses an NVIDIA CUDA 12.1.1 base image with Python 3.11.

---

## Model Selection Guide

The provided models are baselines to get you started. The real opportunity on TuneForge is to innovate -- fine-tune these models, train your own, or integrate other open-source or proprietary music generation systems.

### Baseline Models

| Model | `TF_MODEL_NAME` | VRAM | Speed | Sample Rate | Notes |
|-------|-----------------|------|-------|-------------|-------|
| **ACE-Step 1.5** | `ace-step-1.5` | ~4 GB | Sub-10s | 48 kHz stereo | **Recommended default.** Vocals+lyrics, 50+ languages, up to 10 min, MIT license. Requires repo clone |
| MusicGen Large | `facebook/musicgen-large` | ~16 GB | Moderate | 32 kHz mono | Instrumental only, no vocals, 30s max. CC-BY-NC license (non-commercial) |
| MusicGen Medium | `facebook/musicgen-medium` | ~8 GB | Faster | 32 kHz mono | Good for GPUs with <16GB VRAM. CC-BY-NC license |
| MusicGen Small | `facebook/musicgen-small` | ~4 GB | Fastest | 32 kHz mono | Good for testing or low-VRAM GPUs. CC-BY-NC license |
| DiffRhythm v1.2 (full) | `diffrhythm-full` | ~8-10 GB | Fast | 44.1 kHz stereo | Full-length songs up to 4m45s, vocals+lyrics. Fast generation, decent quality. Requires repo clone |
| HeartMuLa 3B | `heartmula` | ~8-10 GB | Moderate | 48 kHz | Vocals+lyrics. Open-source 3B has weak prompt adherence, mainly useful for vocal generation. Requires heartlib |
| HeartMuLa 7B | `heartmula-7b` | ~16-20 GB | Moderate | 48 kHz | Higher quality vocals than 3B. Requires heartlib + more VRAM |

### Custom Models

Any model that takes a text prompt and produces audio can be integrated. The scoring system evaluates the output audio, not the model architecture. To integrate a custom model:

1. Create a new backend class in `tuneforge/generation/` following the existing backend interfaces.
2. Register it in `tuneforge/generation/model_manager.py`.
3. Set `TF_MODEL_NAME` to your model identifier and `TF_GENERATION_SAMPLE_RATE` to match your model's output.

Speed accounts for only 2% of the total score. Quality and adherence signals collectively dominate. Focus on output quality over generation speed.

### Switching Between Baseline Models

Set the model in your `.env.miner` file:

```bash
# ACE-Step 1.5 (recommended default, ~4GB VRAM, 48kHz stereo, vocals+lyrics)
TF_MODEL_NAME=ace-step-1.5
TF_GENERATION_SAMPLE_RATE=48000
TF_GENERATION_MAX_DURATION=180

# Or MusicGen Large (instrumental only, ~16GB VRAM, CC-BY-NC license)
TF_MODEL_NAME=facebook/musicgen-large
TF_GENERATION_SAMPLE_RATE=32000

# Or DiffRhythm v1.2 full (up to 4m45s, ~8-10GB VRAM)
TF_MODEL_NAME=diffrhythm-full
TF_GENERATION_SAMPLE_RATE=44100

# Or HeartMuLa 3B (vocals, weak prompt adherence, ~8-10GB VRAM)
TF_MODEL_NAME=heartmula
TF_GENERATION_SAMPLE_RATE=48000

# Or HeartMuLa 7B (higher quality vocals, ~16-20GB VRAM)
TF_MODEL_NAME=heartmula-7b
TF_GENERATION_SAMPLE_RATE=48000

# Or MusicGen Medium (for GPUs with <16GB VRAM, CC-BY-NC license)
TF_MODEL_NAME=facebook/musicgen-medium
TF_GENERATION_SAMPLE_RATE=32000
```

### DiffRhythm Requirements

DiffRhythm requires the [DiffRhythm repository](https://github.com/ASLP-lab/DiffRhythm) cloned at `~/DiffRhythm` (or set `DIFFRHYTHM_PATH` to a custom location), plus the `espeak-ng` system package. See [Installation](#diffrhythm-v12-setup-alternative-baseline) above for setup instructions. Model weights (~4 GB) are downloaded automatically from HuggingFace on first run.

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

Understanding the scoring system helps you maximize rewards. The validator evaluates your audio across 16 scoring dimensions and applies 4 penalty multipliers. The scoring is model-agnostic -- it evaluates the audio itself, not what model produced it.

### Weight Distribution Summary

| Category | Scorers | Combined Weight |
|----------|---------|-----------------|
| Prompt adherence | CLAP (0.19), Attribute (0.11) | 30% |
| Composition | Musicality (0.09), Melody (0.06), Structural (0.06) | 21% |
| Naturalness/mix | Vocal Lyrics (0.08), Mix Separation (0.04), Timbral (0.03), Learned MOS (0.03) | 18% |
| Production/fidelity | Production (0.05), Neural Quality (0.05), Vocal (0.04), Audio Quality (0.02) | 16% |
| Other | Diversity (0.06), Speed (0.02) | 8% |
| Preference | Preference (0.07 base) | 0% bootstrap, 2--20% trained |

### The 16 Scoring Dimensions

| Scorer | Weight | What It Measures |
|--------|--------|-----------------|
| CLAP Adherence | 0.19 | Text-audio similarity using `laion/clap-htsat-fused`. Raw cosine similarity mapped from [0.05, 0.45] to [0, 1]. |
| Attribute Verification | 0.11 | Attribute-level verification (genre, mood, tempo, key, instruments) |
| Musicality | 0.09 | Pitch, harmony, rhythm, arrangement quality |
| Vocal Lyrics | 0.08 | Whisper-based lyrics intelligibility, vocal clarity, pitch accuracy, expressiveness, sibilance |
| Preference Model | 0.07 base | Perceptual quality scoring (0% during bootstrap, auto-scales 2--20% once trained) |
| Melody Coherence | 0.06 | Melodic intervals, contour, structure |
| Structural Completeness | 0.06 | Section detection, song form, compositional arc |
| Diversity | 0.06 | CLAP embedding diversity across your last 50 outputs (70% intra-miner, 30% population-level bonus) |
| Production Quality | 0.05 | Spectral balance, loudness (LUFS), dynamics |
| Neural Quality (MERT) | 0.05 | Learned music representations via `m-a-p/MERT-v1-95M` |
| Vocal Quality | 0.04 | Vocal presence, clarity, pitch accuracy (genre-aware) |
| Mix Separation | 0.04 | Spectral clarity, frequency masking, spatial depth |
| Timbral Naturalness | 0.03 | Spectral envelope, harmonic decay, transient quality |
| Learned MOS | 0.03 | Multi-resolution perceptual quality estimation |
| Audio Quality | 0.02 | Signal-level analysis (harmonic ratio, onsets, contrast) |
| Speed | 0.02 | Duration-relative generation speed (see below) |

### Speed Scoring

Speed is evaluated relative to the requested duration, not as an absolute value:

- **ratio** = generation_time / requested_duration
- ratio <= 1.0: score 1.0 (generated faster than real-time)
- ratio = 3.0: score 0.3
- ratio >= 6.0: score 0.0

The generation time is measured by the validator via `dendrite.process_time`, not self-reported by the miner.

### Final Score Calculation

The composite score from all 16 scorers is multiplied by four penalty factors:

```
final_score = composite * duration_penalty * artifact_penalty * fad_penalty * fingerprint_penalty
```

### Penalty Multipliers (4)

| Penalty | Trigger | Effect |
|---------|---------|--------|
| Duration | Off-target by more than 20% | Linear penalty reaching 0.0 at 50% deviation |
| Artifact | Clipping, loops, spectral discontinuities | Multiplier 0--1 applied to final score |
| FAD | Per-miner Frechet Audio Distance | Sigmoid curve, floor 0.5 |
| Fingerprint | AcoustID known-song match | Multiplier 0.0--1.0 (threshold 0.80) |

### Hard Penalties (score = 0)

| Condition | Threshold |
|-----------|-----------|
| Silence | RMS below 0.01 |
| Timeout | Round-trip exceeds 300 seconds |

### Multi-Scale Evaluation

The system adjusts scoring emphasis based on requested duration:

- **Short (<10s):** Higher weight on production, quality, and timbral naturalness.
- **Medium (10--30s):** Baseline weights.
- **Long (>=30s):** Higher weight on structure, melody, and composition. Bonuses for phrase coherence (+0.05) and compositional arc (+0.05).

### Genre-Aware Scoring

The system recognizes 9 genre families with per-genre quality targets:

- Vocal scorers return a neutral 0.5 for instrumental genres (ambient, electronic, classical-cinematic).
- When `vocals_requested=True`, the genre gate is overridden and vocal weights are boosted (2x vocal_lyrics, 1.5x vocal).

### Anti-Gaming Measures

- **Hardcoded weights:** Scoring weights are not configurable via env vars, ensuring all validators score identically.

### Optimization Tips

- **Prompt adherence (CLAP, 19%) is the single largest signal.** Your model must faithfully follow the text prompt.
- **Quality signals dominate.** Better models consistently score higher across all quality dimensions. This is why innovating beyond the baselines is the best path to higher rewards.
- **Diversity (6%):** Do not recycle outputs. The system tracks your last 50 CLAP embeddings and compares against the full population.
- **Speed (2%):** Generate faster than real-time if possible, but do not sacrifice quality. Speed is only 2% of the total score.
- **Duration accuracy matters.** Stay within 20% of the requested duration to avoid the duration penalty.
- **Organic requests** (where `is_organic=True`) do not affect your scoring or weight.
- **New miners** start with an EMA score of 0.0 and build up from their first scored round. The top 10 miners by EMA share 80% of total weight, creating a highly competitive ladder.
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

---

## Troubleshooting

### CUDA Out of Memory

MusicGen Large requires ~16 GB VRAM. If you run out of memory, try ACE-Step 1.5 (~4GB, recommended), DiffRhythm (~8-10GB), or MusicGen Medium (~8GB). For very low-VRAM GPUs, fall back to MusicGen Small:

```bash
TF_MODEL_NAME=facebook/musicgen-small
TF_MODEL_PRECISION=float16
TF_GENERATION_SAMPLE_RATE=32000
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

MusicGen models are downloaded automatically from HuggingFace on first startup. If the download fails, check your internet connection and try:

```bash
bash scripts/download_models.sh
```

For DiffRhythm, verify the repo is cloned at `~/DiffRhythm`, `espeak-ng` is installed, and pre-download weights:

```bash
cd ~/DiffRhythm && python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ASLP-lab/DiffRhythm-1_2-full', filename='cfm_model.pt', local_dir='pretrained')
hf_hub_download('ASLP-lab/DiffRhythm-vae', filename='vae_model.pt', local_dir='pretrained')
"
```

> **Vast.ai note:** Jupyter notebook runs on port 8080 by default on Vast.ai instances. Kill it before starting the miner: `kill $(lsof -ti:8080)` and disable autostart: `sed -i 's/autostart=true/autostart=false/' /etc/supervisor/conf.d/jupyter.conf`


### Low Scores

Check logs for per-signal scoring breakdowns. The most common causes of low scores are:

1. **Poor prompt adherence** -- the largest single signal at 19%.
2. **Duration mismatch** -- generating audio that is too short or too long relative to the request.
3. **Repeated outputs** -- diversity tracking covers 50 recent generations with population-level diversity bonus.
4. **Using a baseline model without improvements** -- the baselines are starting points. Fine-tuning or using a better model will yield higher scores.

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

# Also update ACE-Step if using it
cd ~/ace-step-repo
git pull origin main

# Also update DiffRhythm if using it
cd ~/DiffRhythm
git pull origin main
```

Then restart via your preferred method:

```bash
# PM2
pm2 restart tuneforge-miner-1

# Docker
docker compose up miner -d --build
```

---

## License

TuneForge is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.
