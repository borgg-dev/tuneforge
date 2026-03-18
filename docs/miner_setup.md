# TuneForge Miner Setup Guide

This guide covers everything you need to run a miner on the TuneForge subnet. A miner receives music generation challenges from validators, produces audio, and earns α (alpha) rewards based on quality scores.

**The subnet's purpose is to incentivize competition and model improvement through Bittensor.** MusicGen Large and Stable Audio Open 1.0 are provided as baseline starting models, but miners are strongly encouraged to bring their own models, fine-tune existing ones, or build entirely new generation pipelines. Any model that generates music from text prompts can be integrated. The scoring system rewards quality, not any specific model.

---

## What a Miner Does

A TuneForge miner performs the following:

- Receives `MusicGenerationSynapse` challenges from validators via the Bittensor axon.
- Generates audio using a configurable backend (MusicGen Large, Stable Audio Open, DiffRhythm, or your own custom model).
- Returns base64-encoded WAV audio along with metadata (`sample_rate`, `generation_time_ms`, `model_id`).
- Earns α (alpha) rewards proportional to quality scores assigned by validators.
- Handles organic requests from the SaaS API. Organic requests do not affect scoring.

---

## Hardware Requirements

### Recommended Setups by Model

| Setup | GPU | VRAM | CPU | RAM | Disk | Best For |
|-------|-----|------|-----|-----|------|----------|
| **Competitive (MusicGen Large)** | RTX 4090 / A100 | 24 GB | 8 cores | 32 GB | 50 GB SSD | Default baseline, highest quality from MusicGen family |
| **Mid-range (MusicGen Medium)** | RTX 3090 / A10 | 16 GB | 4 cores | 16 GB | 50 GB SSD | Good balance of quality and cost |
| **Budget (Stable Audio / DiffRhythm)** | RTX 3060 / T4 | 8 GB | 4 cores | 16 GB | 50 GB SSD | Lower VRAM models, still competitive on quality |
| **Entry (MusicGen Small)** | RTX 3060 | 6 GB | 4 cores | 16 GB | 30 GB SSD | Testing and development only |

**Network:** 50 Mbps up/down minimum. The miner must be reachable from the internet on its axon port.

### Baseline Model VRAM Usage

| Model | `TF_MODEL_NAME` | VRAM (fp16) | Speed (30s audio) | Sample Rate | Notes |
|-------|-----------------|-------------|--------------------| ------------|-------|
| **MusicGen Large** (default) | `facebook/musicgen-large` | ~16 GB | ~20-40s on 4090 | 32 kHz mono | Best baseline quality, 3.3B params |
| Stable Audio Open 1.0 | `stable_audio` | ~6 GB | ~10-20s on 4090 | 44.1 kHz stereo | High-fidelity stereo, gated (requires HF login) |
| DiffRhythm v1.2 (base) | `diffrhythm` | ~6-8 GB | ~2-3s on 4090 | 44.1 kHz stereo | 18x faster than MusicGen, vocal+lyrics support, up to 95s |
| DiffRhythm v1.2 (full) | `diffrhythm-full` | ~8-10 GB | ~5-10s on 4090 | 44.1 kHz stereo | Full-length songs up to 4m45s |
| MusicGen Medium | `facebook/musicgen-medium` | ~8 GB | ~10-20s on 4090 | 32 kHz mono | Reduced quality vs. Large |
| MusicGen Small | `facebook/musicgen-small` | ~4 GB | ~5-10s on 4090 | 32 kHz mono | Lowest quality, for testing only |

These are baseline models to get you started. The scoring system is model-agnostic -- it evaluates audio quality, prompt adherence, musicality, and many other signals. Miners who develop or integrate superior models will earn higher scores and more α (alpha).

### Disk Space Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| Python packages + PyTorch + CUDA | ~8 GB | Installed once |
| MusicGen Large model weights | ~7 GB | Downloaded on first run to HuggingFace cache |
| Stable Audio model weights | ~4 GB | Only if using Stable Audio |
| DiffRhythm model weights | ~4 GB | Only if using DiffRhythm |
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

### DiffRhythm v1.2 Setup (Alternative Baseline)

DiffRhythm is a latent diffusion model that generates full-length songs at 44.1kHz stereo. It is 18x faster than MusicGen, uses only 6-8GB VRAM, and supports vocals with lyrics. Two variants are available: base (up to 95s) and full (up to 4m45s).

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

# For full model (recommended):
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ASLP-lab/DiffRhythm-1_2-full', filename='cfm_model.pt', local_dir='pretrained')
hf_hub_download('ASLP-lab/DiffRhythm-vae', filename='vae_model.pt', local_dir='pretrained')
print('Done!')
"

# For base model:
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ASLP-lab/DiffRhythm-1_2', filename='cfm_model.pt', local_dir='pretrained')
hf_hub_download('ASLP-lab/DiffRhythm-vae', filename='vae_model.pt', local_dir='pretrained')
print('Done!')
"
```

The MuQ style encoder (~2 GB) is also downloaded automatically on first run. Set `TF_MODEL_NAME=diffrhythm` for the base model or `TF_MODEL_NAME=diffrhythm-full` for the full-length model, and `TF_GENERATION_SAMPLE_RATE=44100`.

### MusicGen Setup (Default Baseline)

MusicGen Large is the default baseline model (~16GB VRAM, 3.3B parameters). For GPUs with less than 16GB VRAM, use `facebook/musicgen-medium` instead. Install audiocraft separately because it pins specific torch versions:

```bash
pip install audiocraft --no-deps
```

Model weights are downloaded automatically from HuggingFace on first run.

### Stable Audio Open 1.0 Setup (Alternative Baseline)

Stable Audio Open 1.0 is the recommended alternative baseline (~6GB VRAM, 44.1kHz stereo output). It uses a diffusion-based architecture and requires the `diffusers` library:

```bash
pip install diffusers
```

**Important:** Stable Audio Open 1.0 is a **gated model** on HuggingFace. Before using it, you must:

1. Go to [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) and accept the license agreement.
2. Create a HuggingFace access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Log in on your miner machine:

```bash
pip install huggingface_hub
huggingface-cli login
```

Once authenticated, model weights are downloaded automatically on first run. Set `TF_MODEL_NAME=stable_audio` and `TF_GENERATION_SAMPLE_RATE=44100` in your `.env.miner` file.

Alternatively, you can set the `HF_TOKEN` environment variable in your `.env.miner` file instead of using `huggingface-cli login`:

```bash
HF_TOKEN=hf_your_token_here
```

### Bringing Your Own Model

TuneForge is designed to be model-agnostic. If you have a custom music generation model, you can integrate it by implementing a backend class that follows the same interface as the existing backends in `tuneforge/generation/`. Your model just needs to accept a text prompt and duration, and return an audio array with a sample rate.

See the existing backends for reference:
- `tuneforge/generation/musicgen_backend.py`
- `tuneforge/generation/stable_audio_backend.py`
- `tuneforge/generation/diffrhythm_backend.py`

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
| `HF_TOKEN` | str | None | HuggingFace access token (required for gated models like Stable Audio) |
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
| `TF_MODEL_NAME` | str | `facebook/musicgen-large` | Model to use for generation (see [Model Selection](#model-selection-guide)) |
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

### Minimal Example (MusicGen Large)

```bash
TF_NETUID=234
TF_SUBTENSOR_NETWORK=test
TF_WALLET_NAME=my_wallet
TF_WALLET_HOTKEY=my_hotkey
TF_MODEL_NAME=facebook/musicgen-large
TF_GPU_DEVICE=cuda:0
TF_AXON_PORT=8091
TF_GENERATION_SAMPLE_RATE=32000
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
| **MusicGen Large** | `facebook/musicgen-large` | ~16 GB | Moderate | 32 kHz mono | **Default baseline.** Autoregressive transformer (3.3B params) |
| **Stable Audio Open 1.0** | `stable_audio` | ~6 GB | Moderate | 44.1 kHz stereo | **Alternative baseline.** Diffusion-based, high-fidelity stereo. Gated model -- requires HuggingFace login |
| MusicGen Medium | `facebook/musicgen-medium` | ~8 GB | Faster | 32 kHz mono | Good for GPUs with <16GB VRAM |
| MusicGen Small | `facebook/musicgen-small` | ~4 GB | Fastest | 32 kHz mono | Good for testing or low-VRAM GPUs |
| DiffRhythm v1.2 (base) | `diffrhythm` | ~6-8 GB | Very fast | 44.1 kHz stereo | 18x faster, vocals+lyrics, up to 95s. Requires repo clone |
| DiffRhythm v1.2 (full) | `diffrhythm-full` | ~8-10 GB | Fast | 44.1 kHz stereo | Full-length songs up to 4m45s. Requires repo clone |

### Custom Models

Any model that takes a text prompt and produces audio can be integrated. The scoring system evaluates the output audio, not the model architecture. To integrate a custom model:

1. Create a new backend class in `tuneforge/generation/` following the existing backend interfaces.
2. Register it in `tuneforge/generation/model_manager.py`.
3. Set `TF_MODEL_NAME` to your model identifier and `TF_GENERATION_SAMPLE_RATE` to match your model's output.

Speed accounts for only 2% of the total score. Quality and adherence signals collectively dominate. Focus on output quality over generation speed.

### Switching Between Baseline Models

Set the model in your `.env.miner` file:

```bash
# MusicGen Large (default baseline, ~16GB VRAM)
TF_MODEL_NAME=facebook/musicgen-large
TF_GENERATION_SAMPLE_RATE=32000

# Or Stable Audio Open 1.0 (alternative baseline, ~6GB VRAM)
TF_MODEL_NAME=stable_audio
TF_GENERATION_SAMPLE_RATE=44100

# Or DiffRhythm v1.2 base (up to 95s, ~6-8GB VRAM, 18x faster)
TF_MODEL_NAME=diffrhythm
TF_GENERATION_SAMPLE_RATE=44100

# Or DiffRhythm v1.2 full (up to 4m45s, ~8-10GB VRAM)
TF_MODEL_NAME=diffrhythm-full
TF_GENERATION_SAMPLE_RATE=44100

# Or MusicGen Medium (for GPUs with <16GB VRAM)
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

MusicGen Large requires ~16 GB VRAM. If you run out of memory, try DiffRhythm (~6-8GB), Stable Audio Open (~6GB), or MusicGen Medium (~8GB). For very low-VRAM GPUs, fall back to MusicGen Small:

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

For **Stable Audio** (gated model), a 401 error means authentication is missing. Ensure you have:
1. Accepted the license at https://huggingface.co/stabilityai/stable-audio-open-1.0
2. Set `HF_TOKEN` in your `.env.miner` file, or logged in via `huggingface-cli login`

You can pre-download the model before starting the miner to avoid PM2 restart issues during large downloads:

```bash
HF_TOKEN=hf_your_token python3 -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/stable-audio-open-1.0')"
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
