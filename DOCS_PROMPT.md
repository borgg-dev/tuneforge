# Documentation Rewrite Prompt — TuneForge Subnet

Use this prompt to hire a team of expert agents to produce production-ready documentation for the TuneForge Bittensor subnet repository.

---

## PROMPT

You are assembling and leading a team of 4 expert agents to produce production-ready documentation for **TuneForge**, a Bittensor subnet for decentralized AI music generation. Each agent has a distinct role:

### Team Roles

1. **Technical Writer** — Owns the final prose. Writes clear, concise, developer-friendly documentation. Ensures consistent tone, structure, and formatting across all files. Targets an audience of crypto-native developers who may be new to audio/ML but understand Bittensor basics.

2. **Code Reviewer** — Reads every source file referenced below and verifies that every claim in the documentation matches the actual code. Catches outdated numbers, wrong defaults, missing parameters, and incorrect flow descriptions. Every statement must be traceable to a line of code.

3. **Architecture Analyst** — Maps the full system: data flow, component interactions, scoring pipeline, weight-setting lifecycle, organic routing, and anti-gaming mechanisms. Produces the diagrams, tables, and flow descriptions that the Technical Writer polishes.

4. **DevOps / Setup Specialist** — Owns the setup guides. Validates every installation command, Docker config, PM2 command, env variable, port number, and prerequisite. Ensures a developer can go from zero to running miner or validator by following the guide verbatim.

### Deliverables

Produce **4 markdown files**, each complete and self-contained:

---

### FILE 1: `README.md` (Main repository README)

This is the front door. It must be compelling, comprehensive, and professional. Structure:

1. **Hero section with branding** — The README MUST open with the TuneForge brand banner. Use GitHub's dark/light theme switching with a `<picture>` element:

```markdown
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/banner-dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="assets/brand/banner-light.png" />
    <img alt="TuneForge" src="assets/brand/banner-light.png" width="420" />
  </picture>

  <p><strong>Decentralized AI Music Generation on Bittensor</strong></p>

  <!-- badges row -->
  <a href="#"><img src="https://img.shields.io/badge/python-3.10--3.12-blue" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC--BY--NC--4.0-green" /></a>
  <a href="#"><img src="https://img.shields.io/badge/bittensor-subnet-purple" /></a>
</div>
```

The brand assets are located at:
- `assets/brand/banner-light.png` — Full logotype on white background (for light theme)
- `assets/brand/banner-dark.png` — Full logotype on transparent/dark background (for dark theme)
- `assets/brand/logomark.svg` — Icon-only logomark (tuning fork with resonance spark)

The banner shows the TuneForge tuning fork logomark in a violet gradient (#C4B5FD -> #8B5CF6 -> #6D28D9) alongside the wordmark where "tune" is light and "forge" is violet. Use `width="420"` to keep it tasteful, not oversized. Center-align the hero section. Place badges directly below the tagline. No emojis anywhere.
2. **What is TuneForge** — 2-3 paragraph overview explaining:
   - It's a Bittensor subnet (netuid to be assigned on mainnet, currently 234 on testnet) for decentralized AI music generation
   - Miners run music generation models (MusicGen, Stable Audio) and respond to challenges
   - Validators evaluate generated audio across 11 independent scoring dimensions and set on-chain weights
   - An organic query router enables a SaaS API layer on top of the subnet
3. **Architecture diagram** — ASCII or mermaid diagram showing: Bittensor network, Validator (scoring pipeline + weight setter + organic router), Miner (model manager + generation backends + axon), and the SaaS API layer. Show the data flow for both validation challenges and organic queries.
4. **Key Features** — Bullet list:
   - 11-signal composite scoring (CLAP adherence, audio quality, musicality, production quality, melody coherence, neural quality via MERT, preference model, structural completeness, vocal quality, diversity, speed)
   - Anti-gaming: hard penalties for silence/timeout, FAD penalty, fingerprint penalty, diversity tracking, EMA smoothing
   - Genre-aware evaluation (different quality targets per music style)
   - EMA leaderboard with steepening (only top performers get significant weight)
   - Multiple generation backends (MusicGen small/medium/large, Stable Audio)
   - Organic query router for production SaaS traffic (does NOT affect weights)
   - Crowd annotation system + preference model training pipeline
   - Fully configurable via environment variables (TF_ prefix)
5. **Scoring Overview** — A table showing all 11 scoring signals with their weights, a brief description of each, and what category they fall into (prompt adherence / music quality / other). Include the penalty system (silence, timeout, duration, artifacts, FAD, fingerprint).
6. **Reward Mechanism** — Explain the full flow:
   - Validator generates challenge → sends to K miners → scores responses → updates EMA leaderboard → steepening function → on-chain weight submission every 175 blocks
   - EMA formula: `ema_new = alpha * score + (1 - alpha) * ema_old` (alpha = 0.2)
   - Steepening: miners below baseline (0.35) get zero weight; above baseline, weight = ((ema - baseline) / (1 - baseline))^power (power = 2.0)
   - This means only consistently high-quality miners earn meaningful rewards
7. **Quick Start** — Minimal steps to get a miner or validator running. Link to detailed setup guides.
8. **Configuration Reference** — Complete table of ALL TF_ environment variables with type, default, and description. Group by category (network, wallet, generation, scoring weights, scoring thresholds, API, validation).
9. **Project Structure** — Full directory tree with one-line descriptions for every directory and key file.
10. **Roadmap / Areas of Improvement** — Current limitations and planned improvements:
    - Preference model training (currently bootstrap heuristic, trained model improves over time)
    - Mainnet deployment (currently testnet)
    - Additional generation backends
    - Vocal generation support (scoring pipeline is ready, awaiting vocal-capable models)
    - Advanced annotation UI and crowd quality controls
11. **Contributing** — Brief contribution guidelines.
12. **License** — Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Include a brief summary: free to share and adapt for non-commercial purposes with attribution. Link to the full license text at https://creativecommons.org/licenses/by-nc/4.0/

---

### FILE 2: `docs/miner_setup.md` (Complete Miner Guide)

Everything a miner operator needs. Structure:

1. **Overview** — What a miner does in TuneForge (receives challenges, generates audio, returns results, earns rewards based on quality scores).
2. **Hardware Requirements** — Detailed table:
   - MusicGen Small: ~4GB VRAM, good quality, fastest
   - MusicGen Medium: ~8GB VRAM, better quality, recommended baseline
   - MusicGen Large: ~16GB VRAM, best quality, slower
   - Stable Audio: ~6GB VRAM, different aesthetic
   - Minimum: 8GB VRAM (RTX 3070+), 16GB RAM, SSD
   - Recommended: 24GB VRAM (RTX 4090 / A100), 32GB RAM
3. **Prerequisites** — Python 3.10-3.12, CUDA 11.8+, Bittensor wallet (registered on subnet), git.
4. **Step-by-step Installation** — Every command, no gaps:
   - Clone repo
   - Create venv
   - Install dependencies (`pip install -e .`)
   - Download/cache models (first run auto-downloads, or manual)
   - Create wallet if needed (`btcli wallet create`)
   - Register on subnet (`btcli subnet register`)
5. **Configuration** — Full `.env.miner` reference with every variable, its purpose, and recommended values.
6. **Running the Miner** — Three methods with exact commands:
   - Direct Python
   - PM2 (with ecosystem.config.js)
   - Docker Compose
7. **Model Selection Guide** — How to choose between MusicGen small/medium/large and Stable Audio. Trade-offs: quality vs speed vs VRAM. Explain that speed is 5% of the score, so quality usually wins.
8. **Multi-GPU / Multi-Miner** — How to run multiple miners on different GPUs on the same machine.
9. **How Scoring Works (Miner Perspective)** — What the validator evaluates and practical tips:
   - Prompt adherence (30%) is the biggest factor — your model must follow the prompt
   - Audio quality, musicality, production quality matter — larger models score higher
   - Diversity (5%) — don't recycle outputs, the validator tracks your CLAP embeddings
   - Speed (5%) — under 5s is perfect, under 30s is fine, over 60s scores 0
   - Hard penalties: silence = 0, timeout = 0
   - Organic queries (is_organic=True) don't affect your score
10. **Monitoring & Logs** — Where logs go, what to look for, health checks.
11. **Troubleshooting** — Common issues: CUDA OOM, registration, axon port conflicts, model download failures, low scores.
12. **Upgrading** — How to update to new versions safely.

---

### FILE 3: `docs/validator_setup.md` (Complete Validator Guide)

Everything a validator operator needs. Structure:

1. **Overview** — What a validator does (generates challenges, sends to miners, scores responses across 11 dimensions, maintains EMA leaderboard, sets on-chain weights, optionally routes organic SaaS queries).
2. **Hardware Requirements** —
   - GPU recommended for CLAP + MERT scoring models (but can run on CPU, slower)
   - Minimum: 8GB RAM, modern CPU
   - Recommended: 16GB RAM, GPU with 8GB+ VRAM (for faster scoring)
   - Disk: SSD, 20GB+ free for model caches
3. **Prerequisites** — Same as miner plus sufficient TAO stake for validator permit.
4. **Step-by-step Installation** — Full commands.
5. **Configuration** — Full `.env.validator` reference. Highlight key tuning parameters:
   - `TF_VALIDATION_INTERVAL` (300s default — how often to run rounds)
   - `TF_EMA_ALPHA` (0.2 — EMA smoothing)
   - `TF_STEEPEN_BASELINE` (0.35 — minimum EMA to receive weight)
   - `TF_STEEPEN_POWER` (2.0 — reward curve steepness)
   - All 11 scoring weight variables
6. **Running the Validator** — Direct, PM2, Docker.
7. **The Scoring Pipeline (Deep Dive)** — Detailed explanation of each of the 11 scoring signals:
   - What it measures
   - How it's computed (high-level, no need to paste code)
   - Why it matters for music quality
   - Its weight and sub-metrics
   - Include the penalty system
   - Include anti-gaming measures (hardcoded weights, diversity tracking, FAD, fingerprint)
8. **EMA Leaderboard & Weight Setting** —
   - How EMA works, why alpha=0.2
   - Steepening function with example calculations
   - Weight submission cadence (every 175 blocks)
   - What happens to underperforming miners (below baseline → 0 weight)
9. **Organic Query Router** — How the validator can serve SaaS API requests:
   - Miner selection algorithm (EMA-weighted, load-balanced, with failure penalties)
   - Organic queries don't affect weights
   - API endpoints
10. **Challenge Generation** — How prompts are generated (100k+ unique combinations from genre/mood/tempo/key/instruments vocabulary).
11. **Preference Model** — Bootstrap heuristic vs trained model. How to train and deploy.
12. **Monitoring** — Logs, health endpoints, what to watch.
13. **Troubleshooting** — Common issues.

---

### FILE 4: `docs/setup.md` (General Setup & Architecture Reference)

A comprehensive technical reference. Structure:

1. **System Architecture** — Full architecture description with diagrams:
   - Validation flow (challenge → dendrite → miner → score → leaderboard → weights)
   - Organic flow (API request → router → miner selection → dendrite → response)
   - Scoring pipeline (audio decode → 11 scorers → penalties → composite → EMA)
2. **Protocol** — MusicGenerationSynapse fields (request and response), PingSynapse, HealthReportSynapse.
3. **Network Configuration** — Netuid, subtensor network, axon ports, API ports.
4. **Docker Deployment** — Full docker-compose.yml walkthrough, all services, volumes, ports.
5. **PM2 Deployment** — ecosystem.config.js walkthrough.
6. **Environment Variables (Complete Reference)** — Every single TF_ variable, grouped and described.
7. **Database & Persistence** — What the subnet stores (in-memory leaderboard), vs what the SaaS layer stores (PostgreSQL, Redis).
8. **Security** — Validator stake filtering, hotkey auth, API key auth.
9. **Annotation System** — Crowd annotation for preference learning: A/B comparisons, majority vote aggregation, CLAP embedding training, model upload/download cycle.
10. **Development** — Running tests, adding new scorers, modifying weights.

---

## CRITICAL INSTRUCTIONS

### Accuracy

The existing documentation is **outdated and wrong** in several places. Here are known inaccuracies that MUST be corrected:

- The README says scoring is 4 dimensions (Audio Quality 30%, Prompt Adherence 40%, Musicality 20%, Novelty 10%). This is **WRONG**. The actual system has **11 scoring signals** with these weights:
  - CLAP Adherence: 30%
  - Musicality: 10%
  - Neural Quality (MERT): 10%
  - Production Quality: 8%
  - Melody Coherence: 7%
  - Structural Completeness: 7%
  - Audio Quality: 6%
  - Preference Model: 6%
  - Vocal Quality: 6%
  - Diversity: 5%
  - Speed: 5%
  - Attribute Verification: 0% (opt-in)

- The validator_setup.md says 5 scoring signals with wrong weights (CLAP 35%, Audio Quality 25%, Preference 20%, Diversity 10%, Speed 10%). This is **WRONG**. Use the actual weights above.

- The validator_setup.md says steepen baseline is 0.6 and power is 3.0. The actual defaults in code are **baseline 0.35, power 2.0**.

- The README config table shows old weight variables (TF_AUDIO_QUALITY_WEIGHT, TF_PROMPT_ADHERENCE_WEIGHT, TF_MUSICALITY_WEIGHT, TF_NOVELTY_WEIGHT). These are outdated. The actual variables are in `tuneforge/config/scoring_config.py` and follow the pattern `TF_WEIGHT_CLAP`, `TF_WEIGHT_AUDIO_QUALITY`, etc.

### Source of Truth

The **code is the source of truth**. Key files to verify every claim against:

| What | File |
|------|------|
| Scoring weights & thresholds | `tuneforge/config/scoring_config.py` |
| Composite scoring logic | `tuneforge/rewards/reward.py` |
| EMA leaderboard | `tuneforge/rewards/leaderboard.py` |
| Weight setting | `tuneforge/rewards/weight_setter.py` |
| All settings & defaults | `tuneforge/settings.py` |
| Protocol definitions | `tuneforge/base/protocol.py` |
| Miner core logic | `tuneforge/core/miner.py` |
| Validator core logic | `tuneforge/core/validator.py` |
| Challenge generation | `tuneforge/validation/prompt_generator.py` |
| Organic router | `tuneforge/api/organic_router.py` |
| Each scorer | `tuneforge/scoring/*.py` |
| Model backends | `tuneforge/generation/*.py` |
| Entry points | `neurons/miner.py`, `neurons/validator.py` |
| Docker config | `docker-compose.yml`, `Dockerfile.miner`, `Dockerfile.validator` |
| PM2 config | `ecosystem.config.js` |
| Env examples | `.env.miner.example`, `.env.validator.example` |

### Style Guidelines

- **No emojis** in any documentation file
- Use professional, concise technical prose
- Use tables for reference data (config vars, scoring weights, hardware requirements)
- Use code blocks for all commands, config examples, and code snippets
- Use mermaid or ASCII diagrams for architecture flows
- Every section should be useful — no filler, no marketing fluff
- Target audience: developers familiar with Bittensor who want to run a miner or validator
- Use consistent heading hierarchy (# for title, ## for sections, ### for subsections)
- Link between documents where relevant (e.g., README links to detailed setup guides)
- All file paths should be relative to the repository root

### Penalties Table Format

Include this exact penalty information:

| Penalty | Trigger | Effect |
|---------|---------|--------|
| Silence | Audio RMS below threshold (0.01) | Hard zero — final score = 0.0 |
| Timeout | Generation exceeds timeout (120s default) | Hard zero — final score = 0.0 |
| Duration | Audio duration off-target by >20% | Linear penalty multiplier (1.0 → 0.0) |
| Artifacts | Spectral discontinuities, clipping, loops | Multiplier (0.0 - 1.0) on final score |

### Anti-Gaming Section

Explain these mechanisms clearly:
- **Hardcoded weights**: All scoring weights are hardcoded constants, not configurable via environment variables, ensuring validator consensus
- **Diversity tracking**: CLAP embeddings of each miner's last 50 outputs tracked; recycling audio scores low on diversity
- **Hard penalties**: Silence and timeout result in a complete zero score
- **FAD penalty**: Per-miner Frechet Audio Distance with sigmoid curve and floor of 0.5
- **Fingerprint penalty**: AcoustID known-song matching penalizes plagiarism
- **EMA smoothing**: Single good/bad rounds don't swing weights dramatically (alpha = 0.2)

### Process

1. The **Code Reviewer** reads all source files listed above and extracts the ground truth for every number, parameter, flow, and behavior.
2. The **Architecture Analyst** maps the system and produces flow descriptions and diagrams.
3. The **Technical Writer** drafts all 4 files using the verified information.
4. The **DevOps Specialist** validates all setup commands, Docker configs, and deployment instructions.
5. The **Code Reviewer** does a final pass to ensure zero inaccuracies.

### Output

Return all 4 files as complete, ready-to-commit markdown. Each file should be self-contained and production-ready. No placeholders, no TODOs, no "insert here" markers.
