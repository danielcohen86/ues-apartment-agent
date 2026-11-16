"""Background daemon runner orchestrating scrapers, pre-filtering, Gemini ranking, and Telegram alerts."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass

from src.bot import TelegramAgentBot, format_telegram_alert_html
from src.config import AgentConfig
from src.evaluator import GeminiListingEvaluator
from src.filters import prefilter_listing
from src.models import EvaluatedListing, Listing
from src.scrapers import (
    BaseScraper,
    CraigslistUESScraper,
    GeminiSearchUESScraper,
    MockUESScraper,
    NYBitsUESScraper,
    RentHopUESScraper,
    StreetEasyUESScraper,
)
from src.storage import StorageBackend, create_storage_backend

logger = logging.getLogger(__name__)


@dataclass
class ScanCycleReport:
    """Metrics produced by a single scrape-filter-evaluate-rank-notify cycle."""

    scraped_count: int = 0
    already_seen_count: int = 0
    prefilter_rejected_count: int = 0
    gemini_evaluated_count: int = 0
    below_threshold_count: int = 0
    high_conviction_alerts_sent: int = 0


class UESRentalAgentRunner:
    """Continuous background service coordinating scrapers, Gemini evaluator, storage, and Telegram bot."""

    def __init__(
        self,
        config: AgentConfig,
        storage: StorageBackend | None = None,
        scrapers: list[BaseScraper] | None = None,
    ) -> None:
        self.config = config
        self.storage = storage or create_storage_backend(config)
        self.evaluator = GeminiListingEvaluator(config=config, storage=self.storage)

        if scrapers is not None:
            self.scrapers = scrapers
        elif config.enable_demo_mode:
            self.scrapers = [MockUESScraper()]
        else:
            self.scrapers = [
                GeminiSearchUESScraper(
                    api_key=config.gemini_api_key, model=config.gemini_model
                ),
                CraigslistUESScraper(),
                StreetEasyUESScraper(),
                RentHopUESScraper(),
                NYBitsUESScraper(),
            ]

        self.bot = TelegramAgentBot(
            config=config,
            storage=self.storage,
            evaluator=self.evaluator,
            on_manual_scan=self._manual_scan_From_telegram,
        )
        self._stop_event = asyncio.Event()

    async def _manual_scan_From_telegram(self) -> dict[str, int]:
        report = await self.run_scan_cycle()
        return {
            "scraped_count": report.scraped_count,
            "already_seen_count": report.already_seen_count,
            "prefilter_rejected_count": report.prefilter_rejected_count,
            "gemini_evaluated_count": report.gemini_evaluated_count,
            "below_threshold_count": report.below_threshold_count,
            "high_conviction_alerts_sent": report.high_conviction_alerts_sent,
        }

    async def collect_candidates(self) -> list[Listing]:
        """Run all configured scrapers concurrently and deduplicate by listing_id."""
        tasks = [scraper.fetch_listings() for scraper in self.scrapers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: dict[str, Listing] = {}
        for scraper, res in zip(self.scrapers, results):
            if isinstance(res, Exception):
                logger.error("[%s] Scraper raised exception: %s", scraper.source_name, res)
                continue
            for item in res:
                merged[item.listing_id] = item

        return list(merged.values())

    async def run_scan_cycle(self, dry_run: bool = False) -> ScanCycleReport:
        """
        Execute a complete scan cycle:
        1. Scrape candidates across all sources.
        2. Filter against the user's live SearchCriteriaProfile (Stage 1).
        3. Evaluate & score via Gemini with dynamic feedback memory injection (Stage 2).
        4. Select 100% of apartments that meet all active + learned criteria, rank them #1 to #N
           by their 100-point rubric score, and send them in ranked order.
        """
        report = ScanCycleReport()
        criteria = self.storage.get_criteria_profile()
        candidates = await self.collect_candidates()
        report.scraped_count = len(candidates)
        logger.info("Scan cycle started: %d total candidate listings collected", len(candidates))

        qualifying_batch: list[EvaluatedListing] = []

        for listing in candidates:
            if self.storage.is_listing_seen(listing.listing_id):
                report.already_seen_count += 1
                continue

            # Stage 1: Fast zero-cost deterministic pre-filter using live criteria profile
            verdict = prefilter_listing(
                listing=listing,
                min_bedrooms=criteria.min_bedrooms,
                max_monthly_rent=criteria.max_monthly_rent,
                criteria=criteria,
            )
            if not verdict.passed:
                report.prefilter_rejected_count += 1
                logger.info(
                    "[Stage-1 Rejected] %s (%s, %s BR, $%d): %s",
                    listing.display_title,
                    listing.neighborhood,
                    listing.bedrooms,
                    listing.price,
                    verdict.reason,
                )
                continue

            # Stage 2: Gemini multimodal evaluation + 100-Point Ranking Rubric
            evaluation = self.evaluator.evaluate_listing(listing)
            report.gemini_evaluated_count += 1

            meets_criteria = evaluation.passes_qualifying_criteria(
                require_open_island=criteria.require_open_island_kitchen,
                require_high_light=criteria.require_high_natural_light,
            )
            evaluated_record = EvaluatedListing(
                listing=listing,
                evaluation=evaluation,
                alerted=False,
            )
            self.storage.save_evaluated_listing(evaluated_record)

            if not meets_criteria:
                report.below_threshold_count += 1
                logger.info(
                    "[Did Not Meet All Criteria - Score %d/100] %s (%s): %s",
                    evaluation.overall_score,
                    listing.display_title,
                    listing.neighborhood,
                    evaluation.reasoning_summary,
                )
                continue

            qualifying_batch.append(evaluated_record)

        # Rank 100% of qualifying apartments from #1 (highest rubric score) to #N
        qualifying_batch.sort(
            key=lambda item: (
                item.evaluation.overall_score,
                item.evaluation.kitchen_points,
                item.evaluation.light_points,
            ),
            reverse=True,
        )

        total_qualifying = len(qualifying_batch)
        for rank_idx, evaluated_record in enumerate(qualifying_batch, start=1):
            logger.info(
                "🏆 [QUALIFYING MATCH — RANK #%d/%d (%d/100 pts)] %s (%s) - $%d/mo",
                rank_idx,
                total_qualifying,
                evaluated_record.evaluation.overall_score,
                evaluated_record.listing.display_title,
                evaluated_record.listing.neighborhood,
                evaluated_record.listing.price,
            )
            if dry_run:
                print("\n" + "=" * 78)
                print(
                    format_telegram_alert_html(
                        evaluated_record,
                        rank_index=rank_idx,
                        total_qualified=total_qualifying,
                    )
                )
                print("INLINE BUTTONS: [ YES 👍 ]   [ NO 👎 ]")
                print("=" * 78 + "\n")
                report.high_conviction_alerts_sent += 1
            else:
                await self.bot.send_high_conviction_alert(
                    evaluated_record,
                    rank_index=rank_idx,
                    total_qualified=total_qualifying,
                )
                report.high_conviction_alerts_sent += 1

        logger.info(
            "Scan cycle finished: scraped=%d, seen=%d, prefilter_rejected=%d, "
            "gemini_evaluated=%d, disqualified=%d, qualifying_ranked_alerts=%d",
            report.scraped_count,
            report.already_seen_count,
            report.prefilter_rejected_count,
            report.gemini_evaluated_count,
            report.below_threshold_count,
            report.high_conviction_alerts_sent,
        )
        return report

    def request_shutdown(self) -> None:
        """Signal the background daemon loop to exit cleanly."""
        logger.info("Shutdown signal received; stopping background agent...")
        self._stop_event.set()

    async def _handle_health_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Minimal HTTP health-check responder with strict security headers for Render Free Web Tier."""
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            parts = request_line.decode("utf-8", errors="ignore").strip().split(" ")
            method = parts[0].upper() if parts else ""
            path = parts[1] if len(parts) > 1 else "/"

            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not line or line in (b"\r\n", b"\n"):
                    break

            security_headers = (
                "X-Content-Type-Options: nosniff\r\n"
                "X-Frame-Options: DENY\r\n"
                "Content-Security-Policy: default-src 'none'; frame-ancestors 'none'\r\n"
                "Permissions-Policy: camera=(), microphone=(), geolocation=()\r\n"
                "Cache-Control: no-store\r\n"
                "Connection: close\r\n"
            )

            if method != "GET":
                body = b'{"error":"Method Not Allowed"}'
                resp = (
                    "HTTP/1.1 405 Method Not Allowed\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"{security_headers}\r\n"
                ).encode("utf-8") + body
                writer.write(resp)
                await writer.drain()
                return

            if path in ("/", "/health", "/healthz"):
                body = b'{"status":"ok","agent":"ues-family-rental-agent"}'
                resp = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"{security_headers}\r\n"
                ).encode("utf-8") + body
            else:
                body = b'{"error":"Not Found"}'
                resp = (
                    "HTTP/1.1 404 Not Found\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"{security_headers}\r\n"
                ).encode("utf-8") + body

            writer.write(resp)
            await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def run_forever(self) -> None:
        """
        Start the Telegram Bot listener (for [ YES 👍 ] / [ NO 👎 ] + /actions + /criteria),
        optional Render Free Tier health-check server, and periodic scraper loop.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_shutdown)
            except NotImplementedError:
                pass

        health_server: asyncio.AbstractServer | None = None
        port_str = os.getenv("PORT", "").strip()
        if port_str.isdigit():
            bind_host = os.getenv("HEALTH_BIND_HOST", "127.0.0.1").strip()
            port = int(port_str)
            health_server = await asyncio.start_server(
                self._handle_health_client, host=bind_host, port=port
            )
            logger.info("Started health-check HTTP server on %s:%d/healthz", bind_host, port)

        if self.bot.app is not None:
            logger.info("Starting Telegram Bot polling for inline feedback & training loop...")
            await self.bot.app.initialize()
            await self.bot.register_bot_commands_menu()
            await self.bot.app.start()
            if self.bot.app.updater:
                await self.bot.app.updater.start_polling(drop_pending_updates=True)
        else:
            logger.warning(
                "TELEGRAM_BOT_TOKEN not set; running in headless background scan mode."
            )

        interval_seconds = self.config.scan_interval_minutes * 60
        try:
            while not self._stop_event.is_set():
                try:
                    await self.run_scan_cycle()
                except Exception as exc:
                    logger.error("Error in scan cycle: %s", exc, exc_info=True)

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=float(interval_seconds)
                    )
                except asyncio.TimeoutError:
                    continue
        finally:
            if health_server is not None:
                health_server.close()
                await health_server.wait_closed()
            if self.bot.app is not None:
                logger.info("Stopping Telegram Bot polling...")
                if self.bot.app.updater and self.bot.app.updater.running:
                    await self.bot.app.updater.stop()
                await self.bot.app.stop()
                await self.bot.app.shutdown()
