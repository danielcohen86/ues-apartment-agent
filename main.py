#!/usr/bin/env python3
"""
Entrypoint for the Zero-Cost Upper East Side 3+ BR Family Rental Real Estate Agent.

Usage:
  # Run continuous background daemon (Telegram bot + scheduled scraper)
  python main.py

  # Run a single scan pass and exit
  python main.py --once

  # Run a dry-run demo with realistic UES candidate listings (tests Stage-1 filter,
  # Gemini/Fallback evaluation, active feedback memory injection, and alert formatting)
  python main.py --demo --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys

from src.config import AgentConfig
from src.runner import UESRentalAgentRunner
from src.scrapers import MockUESScraper


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upper East Side 3+ BR Family Rental Autonomous Real Estate Agent"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scrape & evaluation pass and exit.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use curated UES seed listings (Carnegie Hill, Yorkville, Lenox Hill) to test the pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print formatted high-conviction Telegram alerts to stdout instead of sending via Telegram API.",
    )
    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="Print the current Active Training Memory block that gets injected into Gemini prompts and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    configure_logging(verbose=args.verbose)
    config = AgentConfig.from_env()
    if args.demo:
        config = dataclasses.replace(config, enable_demo_mode=True)

    scrapers = [MockUESScraper()] if config.enable_demo_mode else None
    runner = UESRentalAgentRunner(config=config, scrapers=scrapers)

    if args.show_memory:
        print(runner.storage.format_feedback_for_prompt())
        return 0

    if args.once or args.dry_run:
        await runner.run_scan_cycle(dry_run=args.dry_run)
        return 0

    await runner.run_forever()
    return 0


def main() -> None:
    args = parse_args()
    try:
        exit_code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        exit_code = 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
