# Upper East Side 3+ BR Family Rental Autonomous Real Estate Agent

A **100% free, zero-spam, background-running real estate agent** engineered specifically for **Upper East Side (Carnegie Hill, Yorkville, Lenox Hill) 3+ bedroom family rentals**.

---

## Core Design Principles

1. **Strict Match Criteria**:
   - **Neighborhoods**: Upper East Side, Carnegie Hill, Yorkville, Lenox Hill.
   - **Size**: 3+ Bedrooms (`MIN_BEDROOMS=3`).
   - **Layout & Natural Light (Non-Negotiable)**:
     - High natural light exposure (high floors, corner units, or unobstructed South/East/West exposures).
     - **Open-concept kitchen** with a **real center island or breakfast bar / peninsula** that flows directly into the main living/dining room.
   - **Strict Disqualifiers**:
     - Closed galley kitchens or separate enclosed kitchens.
     - Windowed pass-throughs / wall cut-outs without an actual island or breakfast bar flow.
     - Dark lower floors (1st–3rd floor, courtyard/airshaft facing) or dark interior hallways.

2. **StreetEasy-Exclusive Sourcing + Calibrated 100-Point Ranking Rubric (`25 / 25 / 25 / 25`)**:
   - **Source**: Scrapes **StreetEasy exclusively** (filtering out out-of-neighborhood `?infeed=1` ads and cross-referencing verified StreetEasy `washer_dryer` and `doorman` filters).
   - **Stage-1 Deterministic Filter** (`src/filters.py`) immediately drops non-UES listings, `< 3 BR` units, and unambiguous galley/low-floor units at zero API cost.
   - **Stage-2 Gemini 3.6 Flash Vision Evaluation** (`src/evaluator.py`) inspects high-resolution (`960px`) StreetEasy gallery photos, floorplan slides, and NYC floor height signals across an equal **100-point rubric**:
     - **Kitchen & Island Flow**: `0–25 pts`
     - **Natural Light & Exposure**: `0–25 pts`
     - **Family Layout & Size**: `0–25 pts`
     - **Learned Preference Fit**: `0–25 pts`
   - **Top-10 Active Calibration -> New-Listings-Only Mode**: Sends the Top 10 best currently active StreetEasy listings once to calibrate your ratings, baselines 100% of active StreetEasy inventory as `seen`, and thereafter alerts **only** on brand-new StreetEasy listings meeting 100% of active + learned criteria.

3. **Active Training Loop (`[ YES 👍 ]` / `[ NO 👎 ]` + Voice/Text Memory Injection)**:
   - Every Telegram alert includes two inline buttons: **`[ YES 👍 ]`** and **`[ NO 👎 ]`**.
   - Clicking **`[ NO 👎 ]`** offers 1-tap quick reasons plus **text message or voice note** capture (e.g., *"Pass-through isn't open enough"* or *"Bedrooms look too small"*), transcribed via Gemini multimodal audio.
   - All feedback is persisted in your database (**Supabase Free Tier PostgreSQL** or **Local JSON store**) and dynamically injected into the `ACTIVE TRAINING MEMORY` section of every subsequent Gemini prompt.

---

## Quick Start

### 1. Initial Setup (Already Initialized in `.venv`)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure `.env` (100% Free Keys)
Edit `.env` and add:
- **`GEMINI_API_KEY`**: Get a free key from [Google AI Studio](https://aistudio.google.com/apikey).
- **`TELEGRAM_BOT_TOKEN`**: Message `@BotFather` on Telegram and run `/newbot`.
- **`TELEGRAM_ALLOWED_CHAT_ID`**: Message `@userinfobot` on Telegram to get your numeric chat ID.
- **Storage**:
  - Leave `STORAGE_BACKEND=local` to use the zero-config persistent store at `data/agent_memory.json`.
  - Or set `STORAGE_BACKEND=supabase`, run [`sql/schema.sql`](sql/schema.sql) in your free [Supabase](https://supabase.com) SQL Editor, and set `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`.

### 3. Test the Pipeline Immediately (`--demo --dry-run`)
```bash
.venv/bin/python main.py --demo --dry-run
```

### 4. Run the Continuous Background Service
- **Foreground / Terminal**:
  ```bash
  .venv/bin/python main.py
  ```
- **macOS Background System Daemon (`launchd`)**:
  ```bash
  mkdir -p logs
  cp deploy/com.ues.apartment-agent.plist ~/Library/LaunchAgents/
  launchctl load ~/Library/LaunchAgents/com.ues.apartment-agent.plist
  ```
- **Render Free Web Service**:
  Push this repository to GitHub and connect it to Render as a Blueprint (`render.yaml`).

---

## Telegram Bot Commands
- `/actions` — Interactive menu of all agent capabilities and 1-tap shortcuts.
- `/calibrate` — Send the Top 10 best active StreetEasy listings to rate and baseline current inventory as seen.
- `/criteria` — View and adjust active search filters (beds, baths, max rent, sqft, island, light, standing rules).
- `/scan` — Trigger an immediate scan on StreetEasy for brand-new qualifying apartments.
- `/top` — View the Top 5 highest-ranked StreetEasy apartments evaluated so far.
- `/memory` — Inspect the live Active Training Memory log injected into Gemini prompts.
- `/status` — View calibration state, StreetEasy scan stats, and learned feedback counts.
- `/test_alert` — Send a sample Carnegie Hill 3BR card with `[ YES 👍 ]` and `[ NO 👎 ]` buttons.
