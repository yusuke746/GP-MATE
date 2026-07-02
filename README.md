# GP-MATE

GP-MATE is a GPT-powered multi-agent trading EA for XAU/USD (GOLD) on MT5.
The system prioritizes capital protection and uses a staged workflow for safe operation.

## Overview

- Symbol: XAU/USD (XM symbol auto-detected, currently GOLD#)
- Timeframes: H4 for trend, H1 for entries
- Architecture: multi-agent analysis (technical, sentiment, debate, trader)
- Risk-first policy: fail-safe HOLD on uncertainty or failures

## Project Structure

- `data/`: MT5 integration, market/news data access
- `agents/`: LLM agents for analysis, debate, and final decision
- `backtest/`: time-capsule validation and bug-detection flows
- `analysis/`: performance metrics and reporting scripts
- `scripts/`: operation scripts (`check_connection`, `run_manual`, `run_scheduler`)

## Setup

1. Install dependencies.
2. Copy `.env.example` to `.env`.
3. Fill required values in `.env`:
   - MT5 credentials (`MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH`)
   - API keys (`OPENAI_API_KEY`, optional `NEWS_API_KEY`, `FRED_API_KEY`)

## Run Flow (Safe 3-Step)

1. Connection check (no trading):
   - `python scripts/check_connection.py`
2. Manual single run (with order confirmation):
   - `python scripts/run_manual.py`
3. Automated schedule run:
   - `python scripts/run_scheduler.py`

## Tests

- Run all tests:
  - `python -m pytest tests -q`

## Security Notes

- Never commit `.env`, logs, CSVs, or state files.
- `.gitignore` is configured to block sensitive files.
- Verify `git status` before every commit.

## Disclaimer

This software is for research and automation support only.
Trading involves financial risk. Validate on demo accounts first, then move to live trading at your own responsibility.
