"""Base scraper interface and polite async HTTP client for NYC rental portals."""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod

import httpx

from src.models import Listing

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}


def parse_int_price(raw: str | int | float | None) -> int:
    """Extract numeric monthly rent from strings like '$12,500/mo'."""
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw)
    digits = re.sub(r"[^\d]", "", str(raw))
    return int(digits) if digits else 0


def parse_float_count(raw: str | int | float | None, default: float = 0.0) -> float:
    """Extract bedroom or bathroom count from strings like '3.5 Baths' or '3 Beds'."""
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    match = re.search(r"(\d+(?:\.\d+)?)", str(raw))
    return float(match.group(1)) if match else default


def infer_ues_subneighborhood(address: str, raw_neighborhood: str = "") -> str:
    """Normalize Upper East Side sub-neighborhoods (Carnegie Hill, Yorkville, Lenox Hill)."""
    combined = f"{raw_neighborhood} {address}".lower()
    if "carnegie hill" in combined:
        return "Carnegie Hill"
    if "yorkville" in combined:
        return "Yorkville"
    if "lenox hill" in combined:
        return "Lenox Hill"

    # Infer from East street number if available (e.g., "East 91st" -> Carnegie Hill)
    street_match = re.search(r"\b(?:east|e\.?)\s*(\d{2})(?:st|nd|rd|th)?\b", combined)
    if street_match:
        st_num = int(street_match.group(1))
        if 59 <= st_num <= 77:
            return "Lenox Hill"
        if 79 <= st_num <= 96 and any(
            ave in combined for ave in ("york", "first", "1st", "second", "2nd", "east end")
        ):
            return "Yorkville"
        if 86 <= st_num <= 96 and any(
            ave in combined for ave in ("fifth", "5th", "madison", "park", "lexington")
        ):
            return "Carnegie Hill"

    return "Upper East Side"


class BaseScraper(ABC):
    """Abstract base class for Upper East Side rental scrapers."""

    source_name: str = "Base"

    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def fetch_html(self, url: str) -> str | None:
        """Fetch a webpage politely with timeout and backoff."""
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            for attempt in range(2):
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return resp.text
                    logger.warning(
                        "[%s] HTTP %s for %s (attempt %d)",
                        self.source_name,
                        resp.status_code,
                        url,
                        attempt + 1,
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s] Network error fetching %s: %s",
                        self.source_name,
                        url,
                        exc,
                    )
                await asyncio.sleep(1.5 * (attempt + 1))
        return None

    @abstractmethod
    async def fetch_listings(self) -> list[Listing]:
        """Scrape and return normalized 3+ BR Upper East Side listings."""
