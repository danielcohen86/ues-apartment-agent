"""Background daemon runner orchestrating StreetEasy-exclusive scraping, pre-filtering, Gemini Vision ranking, and Telegram alerts."""

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
    MockUESScraper,
    StreetEasyUESScraper,
)
from src.scrapers.base import canonicalize_nyc_address_key, infer_nyc_floor_from_unit
from src.storage import StorageBackend, create_storage_backend

logger = logging.getLogger(__name__)

MAX_CALIBRATION_GEMINI_EVALS = 24


def _calibration_candidate_priority(listing: Listing) -> tuple[int, ...]:
    """
    Pre-rank Stage-1 qualifying StreetEasy listings before full Gemini Vision inspection
    so calibration evaluates the highest-potential UES family homes first while still
    marking 100% of active StreetEasy inventory as seen.
    """
    floor_num, _ = infer_nyc_floor_from_unit(listing.unit, listing.floor_info)
    eff_floor = floor_num if floor_num is not None else 8
    am_text = f"{' '.join(listing.amenities)} {listing.description}".lower()
    has_wd = 1 if ("washer" in am_text or "w/d" in am_text) else 0
    is_luxury = 1 if ("condo" in am_text or "new development" in am_text or "doorman" in am_text) else 0
    has_media = len(listing.floorplan_urls) * 2 + min(4, len(listing.image_urls))
    prime_hood = 1 if listing.neighborhood in ("Carnegie Hill", "Lenox Hill", "Yorkville") else 0
    return (
        1 if eff_floor >= 10 else (0 if eff_floor >= 6 else -1),
        has_wd,
        int(round(listing.bathrooms * 10)),
        is_luxury,
        listing.sqft or 0,
        eff_floor,
        prime_hood,
        has_media,
    )


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
    """Continuous background service coordinating StreetEasy scraper, Gemini Vision evaluator, storage, and Telegram bot."""

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
            # StreetEasy-exclusive apartment sourcing (no Craigslist / secondary portals)
            self.scrapers = [StreetEasyUESScraper()]

        self.bot = TelegramAgentBot(
            config=config,
            storage=self.storage,
            evaluator=self.evaluator,
            on_manual_scan=self._manual_scan_From_telegram,
            on_calibrate=self._calibrate_from_telegram,
        )
        self._scan_lock = asyncio.Lock()
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

    async def _calibrate_from_telegram(self) -> dict[str, int]:
        return await self.run_calibration_batch()

    async def collect_candidates(self) -> list[Listing]:
        """Sync live criteria into StreetEasy scraper, fetch active listings, and deduplicate."""
        criteria = self.storage.get_criteria_profile()
        for scraper in self.scrapers:
            if isinstance(scraper, StreetEasyUESScraper):
                scraper.update_search_parameters(
                    min_bedrooms=criteria.min_bedrooms,
                    min_bathrooms=criteria.min_bathrooms,
                    max_monthly_rent=criteria.max_monthly_rent,
                )

        tasks = [scraper.fetch_listings() for scraper in self.scrapers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: dict[str, Listing] = {}
        seen_batch_addr_keys: set[str] = set()
        for scraper, res in zip(self.scrapers, results):
            if isinstance(res, Exception):
                logger.error("[%s] Scraper raised exception: %s", scraper.source_name, res)
                continue
            for item in res:
                addr_key = canonicalize_nyc_address_key(item.address, item.unit, item.url)
                if item.listing_id in merged or (addr_key and addr_key in seen_batch_addr_keys):
                    continue
                merged[item.listing_id] = item
                if addr_key:
                    seen_batch_addr_keys.add(addr_key)

        return list(merged.values())

    async def run_calibration_batch(
        self, limit: int | None = None, dry_run: bool = False
    ) -> dict[str, int]:
        """
        StreetEasy-Exclusive Calibration Workflow (Startup v2 or on-demand /calibrate):
        1. Purges any legacy non-StreetEasy (e.g. Craigslist) rows from storage.
        2. Scrapes all currently active StreetEasy UES 3+ BR inventory.
        3. Evaluates top active StreetEasy candidates via Gemini 3.6 Vision (high-res 960px photos + floorplans)
           and baselines 100% of all currently active StreetEasy apartments as SEEN so zero existing
           apartments ever trickle in later.
        4. Sends ONLY the Top `limit` (default 10) highest-scoring active StreetEasy listings to Telegram
           for [ YES 👍 ] / [ NO 👎 ] calibration rating, then switches to New-StreetEasy-Listings-Only mode.
        """
        async with self._scan_lock:
            purged_count = self.storage.purge_non_streeteasy_listings()
            if purged_count > 0:
                logger.info("Purged %d legacy non-StreetEasy rows prior to StreetEasy calibration.", purged_count)

            criteria = self.storage.get_criteria_profile()
            batch_limit = limit or criteria.calibration_batch_size

            if not dry_run:
                await self.bot.send_info_message(
                    f"🎯 <b>Starting StreetEasy-Only Top-{batch_limit} Calibration</b>\n"
                    "Scanning all currently active <b>StreetEasy</b> UES 3+ BR listings, inspecting high-res "
                    "gallery photos &amp; floorplans with Gemini Vision, baselining all existing StreetEasy inventory "
                    f"as seen, and selecting the <b>{batch_limit} highest-scoring active apartments</b> for you to rate..."
                )

            candidates = await self.collect_candidates()
            seen_ids, seen_addr_keys = self.storage.get_seen_identifiers()
            newly_evaluated = 0
            baselined_count = 0

            unseen_stage1_passed: list[Listing] = []
            for listing in candidates:
                addr_key = canonicalize_nyc_address_key(listing.address, listing.unit, listing.url)
                if listing.listing_id in seen_ids or (addr_key and addr_key in seen_addr_keys):
                    continue

                seen_ids.add(listing.listing_id)
                if addr_key:
                    seen_addr_keys.add(addr_key)

                verdict = prefilter_listing(
                    listing=listing,
                    min_bedrooms=criteria.min_bedrooms,
                    max_monthly_rent=criteria.max_monthly_rent,
                    criteria=criteria,
                )
                if not verdict.passed:
                    self.storage.save_baseline_seen_listing(listing, verdict.reason)
                    baselined_count += 1
                    continue

                unseen_stage1_passed.append(listing)

            # Prioritize strongest StreetEasy candidates for full Gemini Vision inspection
            # while marking 100% of remaining active StreetEasy listings as baseline-seen.
            unseen_stage1_passed.sort(key=_calibration_candidate_priority, reverse=True)
            eval_cap = max(batch_limit * 2, MAX_CALIBRATION_GEMINI_EVALS)
            to_evaluate = unseen_stage1_passed[:eval_cap]
            to_baseline_overflow = unseen_stage1_passed[eval_cap:]

            for overflow_listing in to_baseline_overflow:
                self.storage.save_baseline_seen_listing(
                    overflow_listing,
                    "Baselined active StreetEasy inventory during initial calibration",
                )
                baselined_count += 1

            for listing in to_evaluate:
                if newly_evaluated > 0 and self.config.gemini_api_key:
                    await asyncio.sleep(2.0)

                evaluation = await asyncio.to_thread(self.evaluator.evaluate_listing, listing)
                newly_evaluated += 1
                self.storage.save_evaluated_listing(
                    EvaluatedListing(listing=listing, evaluation=evaluation, alerted=False)
                )

            # Build Top-N calibration batch strictly from StreetEasy evaluated inventory
            all_evaluated = self.storage.get_top_evaluated_listings(limit=80)
            voted_ids = {f.listing_id for f in self.storage.get_all_feedback(limit=200)}

            unalerted_pool = [
                item
                for item in all_evaluated
                if not item.alerted and item.listing.listing_id not in voted_ids
            ]
            if len(unalerted_pool) < batch_limit:
                extra_unvoted = [
                    item
                    for item in all_evaluated
                    if item.alerted and item.listing.listing_id not in voted_ids
                ]
                pool = unalerted_pool + extra_unvoted
            else:
                pool = unalerted_pool

            pool.sort(
                key=lambda item: (
                    item.evaluation.passes_qualifying_criteria(
                        require_open_island=criteria.require_open_island_kitchen,
                        require_high_light=criteria.require_high_natural_light,
                    ),
                    item.evaluation.overall_score,
                    item.evaluation.kitchen_points,
                    item.evaluation.light_points,
                ),
                reverse=True,
            )
            calibration_batch = pool[:batch_limit]
            total_to_send = len(calibration_batch)

            sent_count = 0
            for rank_idx, evaluated_record in enumerate(calibration_batch, start=1):
                if dry_run:
                    self.storage.mark_listing_alerted(evaluated_record.listing.listing_id)
                    sent_count += 1
                else:
                    await self.bot.send_high_conviction_alert(
                        evaluated_record,
                        rank_index=rank_idx,
                        total_qualified=total_to_send,
                        is_calibration=True,
                    )
                    sent_count += 1

            # Persist StreetEasy v2 calibration flags so future restarts & scans only send NEW StreetEasy options
            latest_criteria = self.storage.get_criteria_profile()
            latest_criteria.calibration_completed = True
            latest_criteria.streeteasy_calibration_v2_completed = True
            self.storage.save_criteria_profile(latest_criteria)

            total_seen_now = len(seen_ids)
            if not dry_run:
                if sent_count > 0:
                    await self.bot.send_info_message(
                        f"✅ <b>StreetEasy Calibration Complete — Top {sent_count} Active Listings Sent</b>\n"
                        f"• 🏢 Active StreetEasy Inventory Baselined as Seen: <b>{total_seen_now} apartments</b>\n"
                        "• 👍/👎 Rate these cards at your pace using <b>[ YES 👍 ]</b> or <b>[ NO 👎 ]</b> to calibrate my scoring &amp; learned rules.\n"
                        "• 🔒 <b>StreetEasy New-Listings-Only Mode Active:</b> Going forward, all existing inventory is skipped and I will <b>only</b> send you brand-new StreetEasy listings!"
                    )
                else:
                    await self.bot.send_info_message(
                        "ℹ️ <b>StreetEasy Calibration Check:</b> No unrated active StreetEasy listings were available to send, "
                        "and New-Listings-Only mode is active for upcoming scans."
                    )

            logger.info(
                "StreetEasy calibration batch finished: scraped=%d, newly_evaluated=%d, baselined=%d, calibration_sent=%d",
                len(candidates),
                newly_evaluated,
                baselined_count,
                sent_count,
            )
            return {
                "scraped_count": len(candidates),
                "newly_evaluated": newly_evaluated,
                "baselined_total": total_seen_now,
                "calibration_sent": sent_count,
            }

    async def run_scan_cycle(self, dry_run: bool = False) -> ScanCycleReport:
        """
        Execute a complete StreetEasy scan cycle (New-Listings-Only Mode):
        1. Scrape active StreetEasy UES candidates.
        2. Skip any listing already seen by ID or canonical NYC address+unit key.
        3. Filter brand-new listings against the user's live SearchCriteriaProfile (Stage 1).
        4. Evaluate & score via Gemini Vision with dynamic feedback memory injection (Stage 2).
        5. Rank qualifying new apartments #1 to #N (capped at max_alerts_per_scan) and send in ranked order.
        """
        async with self._scan_lock:
            report = ScanCycleReport()
            criteria = self.storage.get_criteria_profile()
            candidates = await self.collect_candidates()
            seen_ids, seen_addr_keys = self.storage.get_seen_identifiers()
            report.scraped_count = len(candidates)
            logger.info("StreetEasy scan cycle started: %d active UES listings checked", len(candidates))

            qualifying_batch: list[EvaluatedListing] = []

            for listing in candidates:
                addr_key = canonicalize_nyc_address_key(listing.address, listing.unit, listing.url)
                if listing.listing_id in seen_ids or (addr_key and addr_key in seen_addr_keys):
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

                if report.gemini_evaluated_count > 0 and self.config.gemini_api_key:
                    await asyncio.sleep(2.0)

                # Stage 2: Gemini multimodal vision evaluation + 100-Point Ranking Rubric
                evaluation = await asyncio.to_thread(self.evaluator.evaluate_listing, listing)
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
                seen_ids.add(listing.listing_id)
                if addr_key:
                    seen_addr_keys.add(addr_key)

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

            # Rank qualifying brand-new StreetEasy apartments from #1 (highest rubric score) to #N
            qualifying_batch.sort(
                key=lambda item: (
                    item.evaluation.overall_score,
                    item.evaluation.kitchen_points,
                    item.evaluation.light_points,
                ),
                reverse=True,
            )
            capped_batch = qualifying_batch[: criteria.max_alerts_per_scan]

            total_qualifying = len(capped_batch)
            for rank_idx, evaluated_record in enumerate(capped_batch, start=1):
                logger.info(
                    "🏆 [QUALIFYING STREETEASY MATCH — RANK #%d/%d (%d/100 pts)] %s (%s) - $%d/mo",
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
                body = b'{"status":"ok","agent":"ues-family-rental-agent","source":"StreetEasy"}'
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
        optional Render Free Tier health-check server, and periodic StreetEasy scraper loop.
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
                    criteria = self.storage.get_criteria_profile()
                    if not (
                        criteria.calibration_completed
                        and criteria.streeteasy_calibration_v2_completed
                    ):
                        logger.info(
                            "StreetEasy v2 calibration pending — purging legacy sources and running Top-%d StreetEasy calibration batch...",
                            criteria.calibration_batch_size,
                        )
                        await self.run_calibration_batch(limit=criteria.calibration_batch_size)
                    else:
                        await self.run_scan_cycle()
                except Exception as exc:
                    logger.error("Error in scan/calibration cycle: %s", exc, exc_info=True)

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
