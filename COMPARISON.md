# Subnet Framework Comparator

This repository is part of a three-way benchmark comparing Bittensor subnet development frameworks.

| Repo | Framework | Link |
|---|---|---|
| **tuneforge-chi** | Chi alone | https://github.com/borgg-dev/tuneforge-chi |
| **tuneforge-nexus** | Chi + Nexus | https://github.com/borgg-dev/tuneforge-nexus |
| **tuneforge** | Production (hand-built) | https://github.com/borgg-dev/tuneforge |

---

## The Frameworks

- **Chi**: https://github.com/unconst/Chi
  A vibe-codable Bittensor subnet template — minimal scaffold + knowledge base,
  designed for AI-assisted ideation and rapid prototyping.

- **Nexus**: https://github.com/bittensor-church/nexus-subnet-template
  A production infrastructure layer — commit-reveal weights, chain sync,
  async coordination, scoring pipeline, localnet, CI/CD.

---

## Reusable Comparator Prompt

Use this prompt with any AI assistant to reproduce or update this comparison
(e.g. when a new Nexus release candidate drops):

```
I want to benchmark two Bittensor subnet development frameworks — Chi and Nexus —
against an existing production subnet, by actually building the same subnet idea
with each framework and comparing the three resulting codebases side by side.

**The frameworks:**
- Chi: https://github.com/unconst/Chi
  A vibe-codable Bittensor subnet template — minimal scaffold + knowledge base,
  designed for AI-assisted ideation and rapid prototyping.
- Nexus: https://github.com/bittensor-church/nexus-subnet-template
  A production infrastructure layer — commit-reveal weights, chain sync,
  async coordination, scoring pipeline, localnet, CI/CD.

**The reference subnet (the idea to replicate):**
- TuneForge: https://github.com/borgg-dev/tuneforge
  Decentralized music generation on Bittensor — miners generate audio from
  text prompts, validators score output across multiple quality dimensions
  and set on-chain weights.

**What I want you to build:**

1. **tuneforge-chi** — Reimplement the TuneForge idea using Chi alone.
   Faithfully represent what an AI agent would produce by cloning Chi and
   prompting it with the subnet concept + @knowledge base. This means:
   - Monolithic style (validator.py, miner.py, scoring.py — single files)
   - Minimal scoring (only what Chi's knowledge base would suggest: 2-3 metrics)
   - Direct weight setting — no commit-reveal, no retry
   - Basic Docker setup
   - No localnet, no CI, no tests

2. **tuneforge-nexus** — Reimplement the TuneForge idea using Chi + Nexus.
   Use Chi for mechanism design and Nexus for infrastructure. This means:
   - Modular folder structure (validator/, miner/, neurons/, localnet/)
   - Commit-reveal weight setting with retry
   - Async validator loop with chain sync
   - Scoring pipeline with per-scorer error isolation and structured logging
   - Tiered weight distribution (top-10 = 80%, rest = 20%, quadratic power law)
   - Localnet setup (docker-compose + setup.sh)
   - CI workflow (lint, test, docker build)
   - Unit and scoring tests
   - Multi-backend generation registry
   - But domain scoring depth still limited vs the production subnet

3. **Three-way comparison table** covering:
   - File count and structure
   - Number of scoring metrics
   - Anti-gaming measures
   - Weight setting mechanism
   - Async/concurrency model
   - Localnet availability
   - Test coverage
   - CI/CD
   - Generation backends
   - Any domain-specific features (e.g. preference learning, SaaS integration)
   - Overall completeness estimate (%)

**Output:** Three actual code repositories on disk, each git-initialized and
ready to push. The comparison should be honest — show exactly what each
framework gives you and what it leaves for you to build yourself.
```

---

## Updating for a New Nexus Release Candidate

Swap the Nexus URL in the prompt above:

```
- Nexus (vX.Y): https://github.com/bittensor-church/nexus-subnet-template/tree/<tag>
```

Re-run the prompt. The new tuneforge-nexus output will reflect what changed,
and the gap to production TuneForge will show whether Nexus is closing it.

---

## Results Summary (this run)

| Dimension | Chi | Chi + Nexus | TuneForge |
|---|:---:|:---:|:---:|
| Files | 8 | 22 | ~60 |
| Scoring metrics | 3 | 5 | 16 + penalties |
| Anti-gaming | None | None | FAD + fingerprint + diversity |
| Weight setting | Direct | Commit-reveal + retry | Commit-reveal + retry |
| Async | No | Yes | Yes |
| Localnet | No | Yes | Yes |
| Tests | None | 9 | Full pytest suite |
| CI/CD | No | Yes (no workflow scope) | Yes |
| Generation backends | 1 | 1 + registry | 4 |
| Preference learning | No | No | Yes (Bradley-Terry) |
| SaaS / organic | No | No | Yes (FastAPI) |
| Genre-aware scoring | No | No | Yes |
| **Completeness** | ~15% | ~45% | ~95% |
