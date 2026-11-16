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


def _ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def infer_nyc_floor_from_unit(unit: str, context_text: str = "") -> tuple[int | None, str]:
    """
    Infer floor number and human-readable floor label from NYC unit designators
    (e.g. '#27F6' -> 27th Floor, '#2102' -> 21st Floor, '#804' -> 8th Floor, '#PHB' -> Penthouse).
    """
    combined = f"{unit} {context_text}".lower()
    raw = re.sub(
        r"^(?:#|apt\.?\s*|apartment\s*|unit\s*|residence\s*|res\.?\s*|fl\.?\s*|floor\s*)+",
        "",
        (unit or "").strip(),
        flags=re.I,
    ).strip()

    if not raw:
        return None, ""

    if re.match(r"^(?:ph[-a-z0-9]*|penthouse\b)", raw, re.I) or "penthouse" in combined:
        return 30, "Penthouse (Top Floor)"

    if re.match(r"^(?:th[-a-z0-9]*|townhouse\b)", raw, re.I) or "townhouse" in combined:
        return None, "Townhouse / Multi-Floor"

    if re.match(r"^(?:g\d*|garden|bsmt|basement|cellar|parlor)$", raw, re.I):
        if "parlor" in raw.lower():
            return 2, "Parlor Floor"
        return 1, "Garden / Ground Floor"

    # Multi-floor duplex unit: #10/11J -> 10th/11th Floor Duplex
    m_duplex = re.match(r"^0?(\d{1,2})/0?(\d{1,2})[A-Za-z0-9-]*$", raw)
    if m_duplex:
        f1, f2 = int(m_duplex.group(1)), int(m_duplex.group(2))
        top_fl = max(f1, f2)
        if 1 <= top_fl <= 90:
            return top_fl, f"{_ordinal(f1)}/{_ordinal(f2)} Floor Duplex"

    # Standard NYC alphanumeric floor+line: #27B, #27F6, #19B, #4C, #12-A
    m_alpha = re.match(r"^0?(\d{1,2})[-]?[A-Za-z][A-Za-z0-9]*$", raw)
    if m_alpha:
        fl = int(m_alpha.group(1))
        if 1 <= fl <= 90:
            return fl, f"{_ordinal(fl)} Floor"

    # 4-digit high-rise unit: #2102 -> 21st Floor, #1405 -> 14th Floor
    m_four = re.match(r"^(\d{2})(\d{2})[A-Za-z]?$", raw)
    if m_four:
        fl = int(m_four.group(1))
        line_num = int(m_four.group(2))
        if 10 <= fl <= 85 and 0 <= line_num <= 40:
            return fl, f"{_ordinal(fl)} Floor"

    # 3-digit unit: #804 -> 8th Floor, #201 -> 2nd Floor
    m_three = re.match(r"^([1-9])(\d{2})[A-Za-z]?$", raw)
    if m_three:
        fl = int(m_three.group(1))
        line_num = int(m_three.group(2))
        if 0 <= line_num <= 40:
            return fl, f"{_ordinal(fl)} Floor"

    # Whole-floor unit: Floor 14 or #14
    m_plain = re.match(r"^(\d{1,2})$", raw)
    if m_plain:
        fl = int(m_plain.group(1))
        if 1 <= fl <= 85:
            return fl, f"{_ordinal(fl)} Floor"

    return None, ""


def canonicalize_nyc_address_key(address: str, unit: str = "", url: str = "") -> str:
    """
    Build a canonical NYC building+unit fingerprint so identical apartments across scans
    or portals (e.g. '200 East 79th Street #14B' vs '200 E. 79 St Apt 14B') resolve to
    the exact same key and are never re-sent as 'new'.
    """
    addr = (address or "").lower().strip()
    # Strip borough/city/state/zip noise
    addr = re.sub(
        r"\b(?:new\s+york|manhattan|nyc|ny|10021|10028|10128|10065|10075|10029|10044)\b",
        " ",
        addr,
    )
    # Extract embedded unit if present in address (e.g., "200 East 79th St #14B" or "Apt 14B")
    embedded_unit = ""
    emb_match = re.search(
        r"(?:,\s*|\s+)(?:#|apt\.?\s*|apartment\s*|unit\s*|residence\s*)([a-z0-9-]+)\s*$",
        addr,
        re.I,
    )
    if emb_match:
        embedded_unit = emb_match.group(1)
        addr = addr[: emb_match.start()].strip()

    # Normalize directional & street tokens
    addr = re.sub(r"\be\.?(?=\s|\d)", "east ", addr)
    addr = re.sub(r"\bw\.?(?=\s|\d)", "west ", addr)
    addr = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", addr)
    addr = re.sub(r"\b(?:st|str)\.?\b", "street", addr)
    addr = re.sub(r"\b(?:ave|av)\.?\b", "avenue", addr)
    addr = re.sub(r"\bblvd\.?\b", "boulevard", addr)
    addr = re.sub(r"\bpl\.?\b", "place", addr)

    unit_raw = (unit or embedded_unit or "").lower().strip()
    unit_raw = re.sub(
        r"^(?:#|apt\.?\s*|apartment\s*|unit\s*|residence\s*|res\.?\s*)+",
        "",
        unit_raw,
    )
    unit_clean = re.sub(r"[^a-z0-9]+", "", unit_raw)
    addr_clean = re.sub(r"[^a-z0-9]+", "-", addr).strip("-")

    # Fallback to StreetEasy building/unit URL path if address was generic
    if (not addr_clean or addr_clean in ("upper-east-side", "carnegie-hill", "yorkville", "lenox-hill")) and url:
        se_match = re.search(r"streeteasy\.com/building/([a-z0-9_-]+)/([a-z0-9_-]+)", url.lower())
        if se_match:
            bldg_slug = se_match.group(1).replace("-new_york", "").replace("_", "-")
            u_slug = re.sub(r"[^a-z0-9]+", "", se_match.group(2))
            return f"{bldg_slug}-{u_slug}".strip("-")

    if not addr_clean:
        return ""
    return f"{addr_clean}-{unit_clean}" if unit_clean else addr_clean


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
