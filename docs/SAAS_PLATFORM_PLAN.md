# TuneForge SaaS Platform — Master Plan

## Mission

Build **tuneforge.io**, a production SaaS music generation platform competing with Suno.com, powered by the TuneForge Bittensor subnet. Leverage decentralized miner compute to deliver high-quality AI music generation at lower cost, with a polished consumer-facing web experience and a developer API.

---

## Context: What Already Exists

The TuneForge subnet is operational with:

- **Bittensor subnet integration**: Miners run MusicGen/Stable Audio, validators score outputs across 8+ dimensions (CLAP, MERT, musicality, production quality, melody coherence, structural completeness, vocal quality, diversity), EMA leaderboard, on-chain weight setting.
- **REST API** (FastAPI): `POST /api/v1/generate`, `GET /api/v1/tracks`, `GET /health` — with bearer-token auth, sliding-window rate limiting, SQLite database, local/S3 storage.
- **Subnet API** (`TuneForgeAPI`): Programmatic access to query top miners via dendrite and return best audio.
- **Protocol**: `MusicGenerationSynapse` carries prompt, genre, mood, tempo, duration, key signature, instruments, reference audio, seed → returns base64 WAV + metadata.
- **Storage**: Local filesystem or S3 with presigned URLs.
- **Database**: Single `tracks` table (SQLite + async SQLAlchemy).

### What's Missing

- No frontend / web UI
- No user accounts, profiles, or generation history
- No payment/billing integration (Stripe)
- No tiered access (free/pro/premier)
- No credit system
- No API key self-service management
- No real-time generation status (WebSocket/SSE)
- No social features (share, like, public library)
- No admin dashboard
- No production deployment infrastructure (CDN, managed DB, monitoring)

---

## Competitive Reference: Suno.com

| Feature | Suno Free | Suno Pro ($10/mo) | Suno Premier ($30/mo) |
|---|---|---|---|
| Credits | 50/day (~10 songs) | 2,500/month | 10,000/month |
| Model access | v4.5 | v5 | v5 + Studio |
| Commercial use | No | Yes | Yes |
| Studio (DAW) | No | No | Yes |
| API access | No | Limited | Full |

TuneForge's advantage: decentralized compute = lower marginal cost per generation, no single-vendor lock-in, community-driven model improvement through mining incentives.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    tuneforge.io (Frontend)               │
│         Next.js / React — Vercel or self-hosted          │
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Landing  │ │ Generate │ │  Library │ │  Dashboard │  │
│  │  + Auth   │ │  Studio  │ │ + Social │ │  + Billing │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTPS / WSS
┌──────────────────────▼──────────────────────────────────┐
│              TuneForge API Gateway                       │
│         FastAPI — extended from existing API              │
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Auth     │ │  Credits │ │ Generate │ │  Webhooks  │  │
│  │ (JWT/     │ │  + Tier  │ │ + Queue  │ │  + SSE     │  │
│  │  OAuth)   │ │  Mgmt    │ │  Router  │ │            │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Bittensor Subnet Layer                       │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  Organic Query Router                              │   │
│  │  - Select top miners by incentive + availability   │   │
│  │  - Fan-out to N miners, return best result         │   │
│  │  - Retry logic + fallback                          │   │
│  │  - Latency-aware routing                           │   │
│  └───────────────────────────────────────────────────┘   │
│                          │                                │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│    │ Miner 1 │ │ Miner 2 │ │ Miner 3 │ │ Miner N │      │
│    │ MusicGen│ │ Stable  │ │ MusicGen│ │  ...    │      │
│    │ Medium  │ │ Audio   │ │ Large   │ │         │      │
│    └─────────┘ └─────────┘ └─────────┘ └─────────┘      │
└─────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Data Layer                                   │
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ PostgreSQL│ │   S3 /   │ │  Redis   │ │ Stripe     │  │
│  │ (Users,  │ │  CDN     │ │ (Cache,  │ │ (Payments) │  │
│  │  Tracks, │ │ (Audio)  │ │  Queue,  │ │            │  │
│  │  Credits)│ │          │ │  Rate)   │ │            │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Workstreams

### WORKSTREAM 1: Backend API Evolution

**Goal**: Extend the existing FastAPI backend to support a full SaaS platform.

#### 1.1 — Database Migration (SQLite → PostgreSQL)

- Migrate from SQLite to PostgreSQL (async via `asyncpg` + SQLAlchemy).
- Design schema for new tables:
  - **users**: id, email, password_hash, display_name, avatar_url, plan_tier, stripe_customer_id, created_at, updated_at
  - **api_keys**: id, user_id, key_hash, name, permissions, rate_limit_override, last_used_at, created_at, revoked_at
  - **credits**: id, user_id, balance, monthly_allowance, top_up_balance, last_reset_at
  - **credit_transactions**: id, user_id, amount, type (grant/spend/topup/refund), reference_id, created_at
  - **generations**: id, user_id, request_id, prompt, params_json, status (queued/processing/completed/failed), credits_spent, created_at, completed_at
  - **tracks**: extend existing table with user_id, generation_id, is_public, likes_count, plays_count, download_count
  - **likes**: user_id, track_id, created_at
  - **playlists**: id, user_id, name, is_public, created_at
  - **playlist_tracks**: playlist_id, track_id, position
- Alembic migration setup for schema versioning.

#### 1.2 — Authentication & User Management

- **Registration/Login**: Email + password (bcrypt), Google OAuth, optionally GitHub OAuth.
- **JWT tokens**: Access token (15min) + refresh token (7 days), stored in httpOnly cookies for web, Bearer header for API.
- **Email verification**: Confirmation link on signup.
- **Password reset**: Token-based reset flow.
- **User profile**: Display name, avatar, bio, public profile page.
- Endpoints:
  - `POST /api/v1/auth/register`
  - `POST /api/v1/auth/login`
  - `POST /api/v1/auth/refresh`
  - `POST /api/v1/auth/logout`
  - `GET/PUT /api/v1/auth/me`
  - `POST /api/v1/auth/forgot-password`
  - `POST /api/v1/auth/reset-password`
  - `GET /api/v1/auth/google/callback` (OAuth)

#### 1.3 — Credit System & Tier Management

- **Tier definitions**:

  | Tier | Price | Monthly Credits | Credit Cost/Song | Commercial Use | Max Duration | Concurrent Jobs | Priority |
  |---|---|---|---|---|---|---|---|
  | Free | $0 | 50/day | 5 | No | 30s | 1 | Low |
  | Pro | $9/mo | 2,500/mo | 5 | Yes | 60s | 3 | Medium |
  | Premier | $29/mo | 10,000/mo | 5 | Yes | 120s | 5 | High |
  | API | Pay-as-you-go | Top-up | 5 | Yes | 120s | 10 | High |

- Credit deduction: deduct on generation start, refund on failure.
- Monthly credit reset (subscription credits don't roll over).
- Top-up credits persist but require active subscription.
- Endpoints:
  - `GET /api/v1/credits/balance`
  - `GET /api/v1/credits/history`
  - `POST /api/v1/credits/topup` (Stripe checkout)

#### 1.4 — Stripe Payment Integration

- **Stripe Checkout** for subscriptions (Pro/Premier).
- **Stripe Billing Portal** for plan changes, cancellation, invoice history.
- **Stripe Webhooks** for:
  - `checkout.session.completed` → activate subscription
  - `invoice.paid` → reset monthly credits
  - `customer.subscription.updated` → tier change
  - `customer.subscription.deleted` → downgrade to Free
  - `payment_intent.succeeded` → credit top-up
- One-time credit top-up packs ($5 = 500 credits, $20 = 2,500 credits, $50 = 7,500 credits).
- Endpoints:
  - `POST /api/v1/billing/checkout` (create Stripe session)
  - `POST /api/v1/billing/portal` (billing portal link)
  - `POST /api/v1/billing/webhook` (Stripe webhook handler)
  - `GET /api/v1/billing/subscription`

#### 1.5 — Generation Pipeline Enhancement

- **Job queue**: Redis-backed queue (or in-memory for MVP) for generation requests.
- **Real-time status**: SSE (Server-Sent Events) endpoint for generation progress.
  - States: `queued → routing → generating → scoring → completed/failed`
- **Organic query router**: Intelligent miner selection for API-originated requests:
  - Select top-K miners by incentive weight (proven quality).
  - Filter by availability (ping check or recent health report).
  - Fan-out generation request to N miners in parallel.
  - Return best result (by latency or quick quality check).
  - Retry with different miners on failure.
  - Track per-miner success rates for routing optimization.
- **Priority queue**: Higher-tier users get priority routing.
- **Format conversion**: Support mp3, wav, flac, ogg output (already partially exists via pydub).
- Endpoints:
  - `POST /api/v1/generate` (existing, enhanced with credits + queue)
  - `GET /api/v1/generate/{request_id}/status` (SSE stream)
  - `GET /api/v1/generate/{request_id}` (poll status + result)

#### 1.6 — API Key Self-Service

- Users can create/revoke API keys from dashboard.
- Each key has a name, permissions scope, optional rate limit override.
- Keys are hashed in DB (show full key only on creation).
- Endpoints:
  - `GET /api/v1/keys`
  - `POST /api/v1/keys`
  - `DELETE /api/v1/keys/{key_id}`

#### 1.7 — Social & Discovery Features

- **Public library**: Browse publicly shared tracks.
- **Like/save** tracks.
- **User profiles**: Public page with user's shared tracks.
- **Playlists**: Create and share playlists.
- **Play count** tracking.
- **Download**: Allowed for track owner (respects tier commercial rights).
- Endpoints:
  - `GET /api/v1/library` (public tracks, paginated, filterable)
  - `POST /api/v1/tracks/{id}/like`
  - `DELETE /api/v1/tracks/{id}/like`
  - `GET /api/v1/users/{id}/tracks`
  - `POST /api/v1/playlists`
  - `GET /api/v1/playlists/{id}`
  - `PUT /api/v1/playlists/{id}/tracks`

---

### WORKSTREAM 2: Frontend (tuneforge.io)

**Goal**: Build a **premium, enterprise-grade** web application at tuneforge.io that stands shoulder-to-shoulder with the best consumer music platforms (SoundCloud, Spotify, Apple Music) in terms of visual quality, interaction design, and polish. This is NOT a generic SaaS dashboard — it is a creative tool and music platform that must feel premium, immersive, and delightful to use.

**Stack**: Next.js 14+ (App Router), TypeScript, Tailwind CSS, Framer Motion, custom components built on Radix UI primitives.

---

#### CRITICAL: Frontend Design Philosophy

**This section is the highest-priority design brief for any agent working on the frontend. Read it fully before writing a single line of code.**

The TuneForge frontend must be **indistinguishable from a VC-backed, Series-B startup product**. It should feel like a music platform first and a tech product second. Every pixel matters. Every interaction should feel intentional. The bar is SoundCloud's player experience, Spotify's visual polish, Linear's interaction design, and Vercel's typography.

**What we are NOT building:**
- No generic admin-panel aesthetic (grey backgrounds, basic tables, default shadcn look)
- No cookie-cutter SaaS templates (bland hero sections, stock illustrations, generic gradients)
- No "AI product" cliches (rainbow gradients, floating particles everywhere, robot imagery)
- No cluttered dashboards with every metric visible at once
- No default component library styling — every component must be intentionally styled

**What we ARE building:**
- A dark-first, cinematic interface that feels like a professional music studio
- A platform where the music and audio are the hero — the UI serves the content
- Fluid, physicality-driven animations that feel like native apps (not web)
- Typography that commands attention — large, confident, editorial
- Whitespace used deliberately to create breathing room and hierarchy
- Micro-interactions on every interactive element (hover, focus, press, loading states)

---

#### 2.0 — Design System & Visual Foundation

Before building any page, establish a comprehensive design system. This is the foundation everything else is built on.

##### Color Palette
- **Primary background**: Deep, rich black — not flat #000 but a warm near-black (e.g., #0A0A0B) with very subtle blue or purple undertone.
- **Surface layers**: Use 3-4 elevation levels with subtle brightness steps (#111113, #18181B, #1F1F23) — like layers of depth, not flat cards.
- **Accent color**: A single bold signature color — electric violet (#7C3AED), hot coral, or emerald — used sparingly for CTAs, active states, and the playback progress bar. This becomes the brand color.
- **Text hierarchy**: Pure white (#FAFAFA) for headings, muted silver (#A1A1AA) for body text, dim grey (#52525B) for tertiary/disabled text. Never use grey text on grey backgrounds — contrast must always pass WCAG AA.
- **Gradients**: Subtle, dark-to-darker gradients for section backgrounds. One signature gradient (accent color to transparent) used for hero elements and the audio waveform glow.
- **Borders**: Extremely subtle — 1px with ~8% white opacity. Cards are defined by elevation (background color shift), not by visible borders.

##### Typography
- **Display / Hero**: A modern geometric sans-serif — Inter, Satoshi, or General Sans at large sizes (48-72px). Letter-spacing tightened (-0.02em to -0.04em). Bold weight.
- **Headings**: Same family, semibold, 24-36px. Clean and confident.
- **Body**: 15-16px, regular weight, generous line-height (1.6). Comfortable reading.
- **Mono / Technical**: JetBrains Mono or similar for API docs, code snippets, track metadata (BPM, key, duration).
- **Key rule**: Text should feel editorial and considered. Large headings, generous spacing, no cramped layouts.

##### Spacing & Layout
- **Grid**: 12-column grid with generous gutters (24-32px).
- **Content max-width**: 1280px for main content, full-bleed for hero sections and the audio player.
- **Vertical rhythm**: Consistent spacing scale (4, 8, 12, 16, 24, 32, 48, 64, 96px). Sections breathe with 96-128px vertical padding.
- **Cards**: Large, generous padding (24-32px). Rounded corners (12-16px). No thin, cramped cards.

##### Motion & Animation (Framer Motion)
- **Page transitions**: Smooth fade + subtle upward slide (200-300ms, ease-out).
- **Element entrances**: Staggered fade-in for lists and grids (each item delayed 50ms).
- **Hover states**: Every interactive element responds — subtle scale (1.02), background brightness shift, or border glow. Never a bare cursor:pointer with no visual feedback.
- **Loading states**: Skeleton screens with a subtle shimmer animation (not spinners). The shimmer should use the accent color at low opacity.
- **Audio waveform**: Smooth, real-time animation. The waveform should glow with the accent color as it plays — this is the signature visual.
- **Transitions**: Use spring physics (Framer Motion spring) for elements that move. No linear easing — everything should feel organic.
- **Scroll-triggered**: Key sections on the landing page animate in on scroll (IntersectionObserver + Framer Motion).

##### Iconography
- **Icon set**: Lucide icons (consistent line weight, 1.5px stroke).
- **Custom icons**: For music-specific actions (generate, waveform, BPM, key signature), create or source custom icons that feel premium — not generic.
- **Icon sizing**: Generous — 20-24px in navigation, 16-18px inline. Never tiny 12px icons.

##### Audio Player Component (Global)
This is the most important UI component on the platform. It must be best-in-class.
- **Position**: Fixed bottom bar (like Spotify/SoundCloud), persistent across all pages.
- **Waveform**: Real-time animated waveform visualization (using WaveSurfer.js or custom WebAudio API canvas). NOT a simple progress bar. The waveform shows the actual audio shape.
- **Waveform styling**: Played portion in accent color with a soft glow, unplayed portion in dim grey. Smooth cursor/playhead animation.
- **Controls**: Play/pause (large, centered), skip forward/back 10s, volume slider with mute toggle, download button, share button, like button.
- **Track info**: Album art (generated or genre-based placeholder), track title (the prompt, truncated), creator name, duration, genre tag.
- **Expand**: Click to expand into a full-screen immersive player view with large waveform, full metadata, and visual effects.
- **Responsiveness**: On mobile, the bottom bar collapses to mini-player (art + title + play/pause), swipe up for full player.

---

#### 2.1 — Landing Page

This is the first impression. It must be **cinematic, confident, and premium**. Think Apple product pages, Linear's landing page, or Stripe's homepage — not a generic SaaS template.

##### Hero Section
- **Full-viewport height**, dark background with a subtle animated gradient (accent color breathing slowly).
- **Headline**: Large (56-72px), bold, tight letter-spacing. Example: "Create Music That Moves" or "The Sound of the Future". One line, maximum two. No sub-clauses.
- **Subheadline**: One sentence, muted silver text, 18-20px. Explains the value prop clearly.
- **CTA**: One primary button ("Start Creating — It's Free"), large (48px height), accent color, subtle hover glow. Optionally a secondary ghost button ("See How It Works").
- **Visual element**: An animated waveform or audio visualizer graphic behind/beside the text. NOT a stock image. NOT an illustration of a robot. Something abstract, musical, and alive.
- **Social proof**: Below the fold — "Trusted by X creators" or logos of music publications / partners, if available.

##### Feature Sections (scroll-triggered)
- **Section 1 — Quality**: "Studio-Grade AI Music" — showcase the scoring pipeline (8 quality dimensions). Visual: animated radar chart or quality bars. Include an inline audio player with a sample track.
- **Section 2 — Speed**: "From Prompt to Track in Seconds" — emphasize generation speed. Visual: animated timeline showing prompt → audio.
- **Section 3 — Decentralized**: "Powered by a Global Network" — explain the Bittensor miner network. Visual: animated globe or network graph showing distributed compute nodes. Keep it simple and elegant, not overly technical.
- **Each section**: Full-width, generous vertical padding (128px+), scroll-triggered fade-in animation, one key visual, one headline, one paragraph.

##### Pricing Section
- Embedded on the landing page (anchor link from nav).
- Three cards side by side (Free / Pro / Premier).
- **Pro card highlighted**: Slightly elevated, accent border or glow, "Most Popular" badge.
- **Feature comparison**: Clean checklist below each price. Use checkmarks and X marks, not a dense table.
- **CTA per card**: "Get Started Free", "Upgrade to Pro", "Go Premier".
- **Annual toggle**: Switch between monthly/annual pricing with smooth number animation.

##### FAQ Section
- Accordion-style (click to expand), smooth height animation.
- Clean, minimal — 6-8 questions max.

##### Footer
- Dark, minimal. Logo, navigation links (Product, Pricing, API Docs, Blog, Legal), social icons (Twitter/X, Discord, GitHub), copyright.
- **Newsletter signup**: Simple email input + submit button. Subtle, not aggressive.

---

#### 2.2 — Authentication Pages

Auth pages are an opportunity to reinforce the brand, not a throwaway.

- **Layout**: Split screen — left side has a large, atmospheric visual (blurred waveform, abstract audio art) with a brand quote or tagline overlaid. Right side has the form.
- **Forms**: Centered, generous width (400-480px). Large input fields (48px height), clear labels, smooth focus states (accent color border glow).
- **Social login**: Google OAuth button, prominently placed above the email form with a divider ("or continue with email").
- **Transitions**: Smooth animated transitions between sign-up, sign-in, and forgot-password states (Framer Motion AnimatePresence).
- **Error states**: Inline validation with smooth red highlight, clear error messages. Never a generic alert box.
- **Loading**: Button shows a subtle loading spinner on submit, disables to prevent double-click.

---

#### 2.3 — Generation Studio (Core Feature)

This is the heart of the product. It must feel like a **professional creative tool** — powerful but approachable. Think Logic Pro meets a modern web app.

##### Layout
- **Two-panel layout**: Left panel (prompt + parameters, ~40% width), right panel (results + history, ~60% width).
- **Left panel**: Scrollable if needed. Sticky generate button at the bottom.
- **Right panel**: Shows the latest generation result prominently, with a scrollable history below.
- **On mobile**: Stacked — prompt on top, results below. Bottom sheet for parameters.

##### Prompt Input
- **Large textarea**: 3-4 lines visible, auto-expanding. Placeholder text that cycles through creative examples (fading in/out): "A melancholic piano ballad with rain sounds...", "Upbeat electronic dance track with heavy bass...", "Acoustic folk song with fingerpicked guitar..."
- **Character count**: Subtle counter in the corner.
- **Prompt suggestions**: Below the textarea, 3-4 clickable pill buttons with curated prompt starters (genre-based). These rotate periodically.

##### Parameter Controls
- **Collapsible section** below the prompt, labeled "Fine-tune" or "Advanced". Collapsed by default for simplicity, but one click expands smoothly.
- **Genre selector**: Custom dropdown with genre icons and visual previews. NOT a plain `<select>`. Each option shows the genre name + a small colored tag. Support multi-genre (e.g., "Jazz + Electronic").
- **Mood selector**: Horizontal scrollable row of mood chips (tag pills). Multi-select. Visual: each mood has a distinct subtle color tint. Selected chips glow with accent color.
- **Tempo**: Custom slider with BPM number display. Labeled zones (Slow / Medium / Fast / Very Fast) with soft color bands.
- **Duration**: Slider with seconds display. Show tier limit clearly (e.g., "Max 30s on Free plan — Upgrade for longer tracks").
- **Key signature**: Compact dropdown (C Major, A Minor, etc.).
- **Instruments**: Tag-style multi-select with search. Popular instruments shown as pills, search box for the full list.
- **Every control**: Has a tooltip explaining what it does. Smooth transitions when opening/closing sections.

##### Generate Button
- **Large, prominent**: Full-width of the left panel, accent color, 48-56px height.
- **Credit cost displayed on the button**: "Generate — 5 credits".
- **States**: Default (vibrant), hover (glow), loading (pulsing animation + "Generating..."), disabled (greyed, "Insufficient credits").
- **Keyboard shortcut**: Cmd/Ctrl+Enter to generate. Show hint next to button.

##### Real-Time Progress
- When generation starts, the right panel shows a **cinematic loading state**:
  - Animated waveform building up progressively.
  - Stage indicator: "Routing to network..." → "Generating audio..." → "Finalizing..." with smooth transitions between stages.
  - Estimated time remaining (based on historical averages).
  - NOT a boring progress bar. This should feel alive and exciting — the user is watching their music come to life.

##### Result Display
- When generation completes, the result appears with a smooth entrance animation (fade + slide up).
- **Large waveform player**: Full width of the right panel. Accent-colored waveform on dark background. This is the hero element.
- **Track metadata**: Prompt used, genre, mood, BPM, duration, key — displayed in a clean metadata row below the waveform using mono font.
- **Action buttons row**: Play (large), Download (dropdown: mp3/wav/flac), Share to Library, Regenerate (with same params), Save to Collection. Buttons are icon + label, well-spaced.
- **Quality indicator**: Optional — show a small quality badge based on the quick quality gate score (e.g., a 5-star micro-rating or a simple "High Quality" tag).

##### Generation History
- Below the current result, a scrollable list of past generations.
- Each entry: Mini waveform thumbnail, prompt (truncated), genre tag, duration, timestamp, play button.
- Click to load into the main player.
- Subtle dividers between entries. Hover state highlights the row.

---

#### 2.4 — My Library / Dashboard

The user's personal space. Clean, organized, content-focused.

##### Layout
- **Top bar**: Usage summary — credits remaining (large number, accent color if low), plan name, "Upgrade" button if on Free.
- **Tab navigation**: My Tracks / Favorites / Playlists — horizontal tabs with animated underline indicator.
- **View toggle**: Grid view (album art cards) / List view (table-like rows). Remember user preference.

##### Track Cards (Grid View)
- **Large cards**: 280-320px wide. Generated artwork or genre-specific placeholder (abstract art per genre, NOT generic grey).
- **On hover**: Subtle scale up (1.02), play button overlay fades in (centered, circular, semi-transparent dark background).
- **Card content**: Title (prompt, truncated to 2 lines), genre tag (colored pill), duration, date created.
- **Quick actions on hover**: Play, download, share, delete — icon buttons that appear at the bottom of the card.

##### Track Rows (List View)
- Clean table layout — play button, waveform mini-thumbnail, title, genre, duration, date, actions.
- Hover: Subtle row highlight, action icons appear.
- Currently playing track: Accent-colored left border or glow.

##### Usage Stats
- **Credit balance**: Prominent number with progress ring or bar showing usage this period.
- **Generation count**: This week/month, with a small sparkline chart.
- **Plan info**: Current tier, renewal date, link to billing.
- Keep it minimal — 3-4 key numbers, not a dense analytics dashboard.

---

#### 2.5 — Public Library / Explore

The discovery experience. Should feel like browsing a premium music platform.

##### Layout
- **Hero banner**: Rotating featured track or curated collection with large artwork, play button, and prompt. Auto-playing a 5-second preview on hover (optional).
- **Genre carousel**: Horizontal scrollable row of genre cards (large, visual, each genre has a distinct color/artwork style). Click to filter.
- **Trending section**: "Trending Now" — top tracks by plays/likes this week. Horizontal scrollable carousel.
- **Latest section**: "Fresh Tracks" — newest public tracks. Grid layout, paginated or infinite scroll.

##### Track Cards (in Explore)
- Same card design as My Library grid, but with:
  - Creator name/avatar (small, below the title).
  - Like count + play count (small, muted).
  - Like button (heart icon, animates on click with a subtle pop + accent color fill).

##### Filters & Search
- **Search bar**: Prominent, top of page. Large input with search icon. Instant results as you type (debounced).
- **Filter pills**: Below search — Genre, Mood, Duration range, Tempo range. Each is a dropdown or slider that applies filters instantly (no "Apply" button — real-time filtering).
- **Sort**: Dropdown — "Trending", "Newest", "Most Liked", "Most Played".

##### User Profiles
- **Public profile page**: `/u/{username}` — header with avatar, display name, bio, join date, total tracks, total likes received.
- **Track grid**: All of the user's public tracks in grid layout.
- **Follow button**: Future feature placeholder (design the button now, wire it later).

---

#### 2.6 — Pricing & Billing

##### Pricing Page (`/pricing`)
- **Clean, full-page layout** with the three tier cards as the centerpiece.
- **Annual/monthly toggle**: Smooth animated switch. Annual shows per-month price with "Save 20%" badge.
- **Tier cards**: Side-by-side, equal height. Pro card visually elevated (accent border, slight scale, "Most Popular" ribbon).
- **Feature matrix below**: Detailed comparison table — clean rows with check/cross icons. Alternating row backgrounds for readability.
- **FAQ**: Below the matrix, accordion-style.
- **Enterprise CTA**: At the bottom — "Need more? Contact us for custom plans." — simple form or email link.

##### Billing Dashboard (`/settings/billing`)
- **Current plan card**: Plan name, price, next billing date, accent-colored badge.
- **Credit balance**: Large number + usage bar.
- **Quick actions**: "Change Plan", "Buy Credits", "Manage Payment Method" — clean button row.
- **Invoice history**: Clean table — date, amount, status (paid/pending), download PDF link.
- **All payment actions**: Open Stripe Checkout or Stripe Billing Portal in a new tab (not embedded).

---

#### 2.7 — API Documentation (`/docs`)

Developer docs should feel premium too — think Stripe's API docs.

- **Split layout**: Left sidebar (navigation tree), right content area.
- **Sidebar**: Collapsible sections — Getting Started, Authentication, Endpoints, Webhooks, SDKs, Rate Limits.
- **Code examples**: Syntax-highlighted (dark theme), with tabs for Python / JavaScript / cURL. Copy button with "Copied!" feedback.
- **Interactive "Try It"**: For each endpoint, a collapsible panel where developers can input parameters and see a live request/response (hitting the real API with their key).
- **API key management**: Embedded section — create/revoke keys inline. Show key only once on creation (modal with copy button and warning).
- **Rate limit display**: Visual bar showing current usage vs limit.

---

#### 2.8 — Admin Dashboard (Internal, `/admin`)

- **Separate layout**: Sidebar navigation (Users, Tracks, Analytics, System, Moderation).
- **Users**: Searchable table, click to expand user details + override tier.
- **Analytics**: Key charts — generations/day (line chart), revenue/month (bar chart), top genres (pie chart), miner response times (histogram). Use a charting library like Recharts or Tremor.
- **System health**: Miner count, avg latency, error rate, uptime — live-updating cards.
- **Content moderation**: Queue of flagged tracks with play button, approve/remove actions.
- **Design**: Can be more utilitarian than the consumer-facing pages, but still consistent with the design system. Dark theme, same typography, same component patterns.

---

#### 2.9 — Responsive Design & Mobile Experience

The entire platform must work flawlessly on mobile. Not as an afterthought — as a first-class experience.

- **Mobile navigation**: Bottom tab bar (Home/Explore, Create, Library, Profile) — NOT a hamburger menu for primary navigation. Hamburger only for secondary items (Settings, Billing, API Docs).
- **Generation studio on mobile**: Full-screen prompt input, bottom sheet for parameters, full-screen result view with waveform.
- **Global player on mobile**: Mini-player bar at bottom (above the tab bar). Swipe up for full-screen player.
- **Touch targets**: Minimum 44x44px for all interactive elements.
- **Breakpoints**: 640px (mobile), 768px (tablet), 1024px (desktop), 1280px (wide desktop).
- **Performance**: Lazy-load images, virtualize long lists, code-split per route. Target Lighthouse score > 90 on mobile.

---

#### 2.10 — Implementation Standards for Frontend Agents

**Every agent working on the frontend MUST follow these rules:**

1. **No default component styling.** Every shadcn/Radix component must be restyled to match the design system. If a component looks like it came from a template, it's wrong.
2. **Dark mode is the only mode.** There is no light mode. Do not build theme toggling. The entire UI is dark.
3. **Every interactive element needs 4 states:** default, hover, active/pressed, disabled. No exceptions.
4. **Every list/grid needs 3 states:** loading (skeleton shimmer), empty (illustrated empty state with CTA), populated.
5. **No bare text.** All text must use the typography scale. No arbitrary font sizes or colors.
6. **Animations are required, not optional.** Page transitions, element entrances, hover effects, loading states — all must be animated with Framer Motion. But keep them subtle and fast (150-300ms). No slow, dramatic animations that block the user.
7. **Audio is the hero.** The waveform player, track cards, and generation results should be the most visually striking elements on any page. Everything else supports them.
8. **Test on mobile.** Every component must be tested at 375px width. If it breaks, fix it before moving on.
9. **Performance budget:** No page should load more than 200KB of JavaScript on initial load (code-split aggressively). Images must be optimized (WebP, lazy-loaded). Fonts subset to used characters.
10. **Accessibility:** All interactive elements keyboard-navigable, all images have alt text, focus indicators visible, screen reader labels on icon-only buttons.

---

### WORKSTREAM 3: Organic Query Integration

**Goal**: Bridge the SaaS API layer with the Bittensor subnet so that paying user requests ("organic queries") flow through miners, while keeping the validator's synthetic challenge loop separate.

#### 3.1 — Organic Query Router

This is the critical component that connects paying users to miner compute:

- **Miner selection strategy**:
  - Use metagraph incentive weights as primary quality signal.
  - Maintain a local availability cache (updated via PingSynapse or recent health reports).
  - Latency-aware: prefer miners with lower average generation time.
  - Geographic routing (future): prefer miners with lower network latency to API server.
- **Request flow**:
  1. User submits generation request via API.
  2. Credits are reserved (deducted from balance).
  3. Router selects top-K miners (K=3 for redundancy).
  4. MusicGenerationSynapse sent to K miners in parallel via dendrite.
  5. First valid response wins (fastest + passes basic quality check).
  6. Audio stored to S3, metadata to DB, credits finalized.
  7. On total failure: credits refunded, user notified.
- **Quality gate**: Quick sanity checks on organic responses (non-silent, correct duration, no clipping) — lighter than full validator scoring.
- **Separation from validation**: Organic queries do NOT affect miner scores or weights. The validator's synthetic challenge loop remains the sole weight-setting mechanism. This prevents gaming through organic traffic manipulation.

#### 3.2 — Miner Incentive Alignment

- Miners are already incentivized to produce high-quality audio through the validation scoring pipeline.
- Organic traffic provides additional income opportunity (future: revenue sharing via subnet mechanism).
- Consider: validators could slightly preference miners that successfully serve organic queries (optional, requires careful design to prevent gaming).

---

### WORKSTREAM 4: Infrastructure & Deployment

#### 4.1 — Production Database

- PostgreSQL (managed: AWS RDS, Supabase, or Neon).
- Redis (managed: AWS ElastiCache, Upstash, or Railway).
- Alembic migrations in CI/CD.

#### 4.2 — Storage & CDN

- S3 for audio file storage (already supported).
- CloudFront or Cloudflare CDN for audio delivery.
- Presigned URLs for private tracks, public URLs for shared tracks.

#### 4.3 — Deployment

- **API**: Docker container on AWS ECS, Railway, or Fly.io. Must run on a machine with Bittensor wallet access (for dendrite queries to miners).
- **Frontend**: Vercel (Next.js) or Cloudflare Pages.
- **Domain**: tuneforge.io with SSL.
- **Environment**: Staging + Production.

#### 4.4 — Monitoring & Observability

- **Logging**: Structured JSON logs → CloudWatch or Datadog.
- **Metrics**: Prometheus + Grafana (generation latency, success rate, credit usage, revenue).
- **Alerting**: PagerDuty or Slack alerts for downtime, high error rates, low miner availability.
- **Uptime**: External monitoring (UptimeRobot or Betterstack).

#### 4.5 — Security

- HTTPS everywhere.
- CORS restricted to tuneforge.io in production.
- Rate limiting at API gateway level (Redis-backed).
- Input validation and sanitization (already present in FastAPI models).
- Stripe webhook signature verification.
- API key hashing (bcrypt or SHA-256).
- Content Security Policy headers on frontend.
- Regular dependency audits (Dependabot / Snyk).

---

## Implementation Priority (Phased Rollout)

### Phase 1 — MVP (Weeks 1–3)
> Get a working end-to-end product live with free tier.

1. Database migration (SQLite → PostgreSQL) + new schema.
2. User auth (email + password, JWT).
3. Credit system (free tier: 50 credits/day).
4. Organic query router (select top miners, fan-out, return best).
5. SSE generation status.
6. Frontend: Landing page, auth pages, generation studio, my library.
7. S3 storage + CDN for audio.
8. Deploy API + frontend to staging.

### Phase 2 — Monetization (Weeks 4–5)
> Enable paid tiers and billing.

1. Stripe integration (subscriptions + one-time top-ups).
2. Tier enforcement (duration limits, concurrency, priority).
3. Frontend: Pricing page, billing dashboard, plan management.
4. API key self-service.
5. Google OAuth.

### Phase 3 — Social & Growth (Weeks 6–7)
> Public library and discovery features to drive engagement.

1. Public track sharing (toggle visibility).
2. Like/play count tracking.
3. Public library browse + search.
4. User public profiles.
5. Playlists.
6. Frontend: Explore page, user profiles, playlist UI.

### Phase 4 — Polish & Scale (Week 8+)
> Production hardening, monitoring, and optimization.

1. Admin dashboard.
2. Monitoring + alerting (Prometheus, Grafana).
3. API documentation page (interactive Swagger + code examples).
4. Performance optimization (caching, query optimization).
5. Analytics integration (Mixpanel or PostHog).
6. SEO optimization for landing page.

---

## Key Design Decisions to Make

1. **Frontend hosting**: Vercel (easiest for Next.js) vs self-hosted (more control)?
2. **Database provider**: Supabase (Postgres + auth built-in) vs standalone Postgres + custom auth?
3. **Audio CDN**: CloudFront vs Cloudflare R2+CDN (cheaper)?
4. **Queue system**: Redis queues vs simple in-memory for MVP?
5. **Miner selection for organic queries**: Pure incentive-weight-based vs hybrid (weight + latency + availability)?
6. **Revenue sharing with miners**: Future mechanism for sharing organic revenue back to miners via subnet?

---

## Success Metrics

- **Launch**: Working generation flow, end-to-end, with free tier.
- **Quality**: Average generation quality score ≥ 0.6 (from validator scoring pipeline).
- **Latency**: Median generation time < 30 seconds for 15s tracks.
- **Reliability**: > 99% generation success rate (at least 1 of K miners responds).
- **Growth**: Track signups, generations/day, conversion to paid tiers.
- **Revenue**: MRR from Pro + Premier subscriptions + credit top-ups.
