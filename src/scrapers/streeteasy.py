"""StreetEasy Upper East Side (Carnegie Hill, Yorkville, Lenox Hill) 3+ BR 3-provider scraper + live link verifier."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

from src.models import Listing
from src.scrapers.base import (
    BaseScraper,
    infer_nyc_floor_from_unit,
    infer_ues_subneighborhood,
    parse_float_count,
    parse_int_price,
)

logger = logging.getLogger(__name__)

VALID_UES_CARD_NEIGHBORHOODS = {
    "upper east side",
    "ues",
    "lenox hill",
    "carnegie hill",
    "yorkville",
}

MOCK_DEMO_URL_SUBSTRINGS = (
    "1185-park-avenue-new_york/15b",
    "200-east-79-street-new_york/18a",
    "315-east-68-street-new_york/8d",
    "155-west-72-street-new_york/12a",
)

CARD_TITLE_LINK_RE = re.compile(
    r"(?<!!)\[([^\[\]\n]+)\]\((https://streeteasy\.com/(?:building/[^/)\s?]+/[^)\s]+|rental/\d+[^)\s]*))\)"
)

BUILDING_TYPE_PATTERN = (
    r"(?:Condo|Co-op|Condop|Rental unit|Townhouse|House|Multi-family|Loft|"
    r"New development|Mixed-use Building|Building)"
)

# Anchored at \s*\Z so it matches both r.jina.ai (newline before link) and markdown.new (space before link)
CARD_TYPE_NEIGH_RE = re.compile(
    rf"({BUILDING_TYPE_PATTERN})\s+in\s+([A-Za-z][A-Za-z \-]*?)\s*\Z",
    re.I,
)

INLINE_TYPE_NEIGH_RE = re.compile(
    rf"({BUILDING_TYPE_PATTERN})\s+in\s+([A-Za-z][A-Za-z \-]*)",
    re.I,
)

ZILLOW_IMAGE_RE = re.compile(
    r"!\[(?:Image\s+\d+:\s*)?([^\]]*)\]\((https://photos\.zillowstatic\.com/fp/([a-zA-Z0-9_-]+))-[^)\s.]+\.(?:webp|jpg|jpeg|png)\)"
)

ZILLOW_HASH_FROM_URL_RE = re.compile(
    r"https://photos\.zillowstatic\.com/fp/([a-zA-Z0-9_-]+)-[^)\s.\"]+\.(?:webp|jpg|jpeg|png)",
    re.I,
)

DETAIL_PHOTO_RE = re.compile(
    r"!\[photo\s+\d+[^\]]*\]\(https://photos\.zillowstatic\.com/fp/([a-zA-Z0-9_-]+)-[^)\s.]+\.(?:webp|jpg|jpeg|png)\)",
    re.I,
)

DETAIL_FLOORPLAN_RE = re.compile(
    r"!\[floor\s*plan\s+\d+[^\]]*\]\(https://photos\.zillowstatic\.com/fp/([a-zA-Z0-9_-]+)-[^)\s.]+\.(?:webp|jpg|jpeg|png)\)",
    re.I,
)


def is_known_mock_streeteasy_url(url: str) -> bool:
    """Return True if the URL is one of the synthetic demo sample links."""
    u = (url or "").lower()
    return any(tok in u for tok in MOCK_DEMO_URL_SUBSTRINGS)


def split_streeteasy_title_address_and_unit(title: str, canonical_url: str = "") -> tuple[str, str]:
    """Split '400 East 67th Street #27B', '340 East 64th Street #10/11J', or '127 East 69th Street RESIDENCE-4'."""
    cleaned = " ".join((title or "").split()).strip()
    m = re.match(
        r"^(.*?)(?:\s+#|\s+Apt\.?\s+|\s+Unit\s+|\s+Residence(?:[-\s]+#?|\s+)|\s+Penthouse[-\s]+|,\s*#?)([A-Za-z0-9/-]+)$",
        cleaned,
        re.I,
    )
    if m:
        addr = m.group(1).strip().rstrip(",")
        raw_u = m.group(2).strip().lstrip("#")
        if "penthouse" in cleaned.lower() and not raw_u.upper().startswith("PH"):
            raw_u = f"PH{raw_u}"
        return addr, f"#{raw_u}"

    if canonical_url:
        u_match = re.search(r"streeteasy\.com/building/[^/]+/([^/?#]+)", canonical_url)
        if u_match:
            u_slug = u_match.group(1).strip().upper()
            if u_slug and u_slug != "RENTAL" and len(u_slug) <= 12:
                return cleaned, f"#{u_slug}"

    return cleaned, ""


def is_strict_ues_street_bounds(address: str, card_neighborhood: str) -> bool:
    """Reject East Harlem (E 98th+) or Midtown (< E 59th) sponsored spillover cards."""
    neigh_clean = card_neighborhood.strip().lower()
    if neigh_clean and not any(v in neigh_clean for v in VALID_UES_CARD_NEIGHBORHOODS):
        return False

    st_match = re.search(
        r"\b(?:east|e\.?)\s*(\d{1,3})(?:st|nd|rd|th)?\s+(?:street|st)\b", address, re.I
    )
    if st_match:
        st_num = int(st_match.group(1))
        if st_num >= 98 or st_num <= 58:
            return False

    if re.search(r"\b(?:west|w\.?)\s*\d+", address, re.I):
        return False

    return True


def parse_streeteasy_detail_markdown(detail_md: str) -> dict[str, Any]:
    """
    Validate a StreetEasy detail page's markdown and extract full broker description,
    amenities, 1500px floorplans, and full high-res 960px gallery photos.
    """
    if not detail_md or len(detail_md) < 1200:
        return {"reachable": False, "valid": True}

    lower_head = detail_md[:3500].lower()
    if (
        "we can't find that page" in lower_head
        or "page not found" in lower_head
        or "404 not found" in lower_head
    ):
        return {"reachable": True, "valid": False, "reason": "404 page not found on StreetEasy"}

    before_history = detail_md.split("## Property history")[0]
    top_listing_header = before_history.split("## About")[0]
    if re.search(
        r"\b(?:no longer available|delisted|off[- ]market|rented)\b", top_listing_header, re.I
    ):
        return {
            "reachable": True,
            "valid": False,
            "reason": "Listing is no longer available on StreetEasy",
        }

    floorplan_urls: list[str] = []
    seen_fp: set[str] = set()
    for fp_m in DETAIL_FLOORPLAN_RE.finditer(before_history):
        fp_hash = fp_m.group(1)
        if fp_hash not in seen_fp:
            seen_fp.add(fp_hash)
            floorplan_urls.append(
                f"https://photos.zillowstatic.com/fp/{fp_hash}-se_extra_large_1500_800.webp"
            )

    photo_urls: list[str] = []
    seen_ph: set[str] = set()
    for ph_m in DETAIL_PHOTO_RE.finditer(before_history):
        ph_hash = ph_m.group(1)
        if ph_hash not in seen_ph and ph_hash not in seen_fp:
            seen_ph.add(ph_hash)
            photo_urls.append(f"https://photos.zillowstatic.com/fp/{ph_hash}-cc_ft_960.webp")
            if len(photo_urls) >= 6:
                break

    about_text = ""
    m_about = re.search(
        r"##\s+About\s*\n+(.*?)(?=\n##\s+(?:Policies|Home features|Building amenities|About the building|Property history)|\Z)",
        before_history,
        re.S,
    )
    if m_about:
        about_text = " ".join(m_about.group(1).split()).strip()

    features_list: list[str] = []
    for sec_title in ("Home features", "Building amenities"):
        m_sec = re.search(
            rf"##\s+{sec_title}\s*\n+(.*?)(?=\n##\s+|\Z)",
            before_history,
            re.S,
        )
        if m_sec:
            for line in m_sec.group(1).splitlines():
                cleaned_line = (
                    re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line).strip().lstrip("-*").strip()
                )
                if (
                    cleaned_line
                    and len(cleaned_line) <= 60
                    and not cleaned_line.lower().startswith(
                        ("services and", "wellness and", "shared outdoor", "no info")
                    )
                ):
                    features_list.append(cleaned_line)

    building_meta = ""
    m_bldg = re.search(
        r"##\s+About the building\s*\n+(.*?)(?=\n##\s+|\Z)",
        before_history,
        re.S,
    )
    if m_bldg:
        stories_m = re.search(r"(\d+\s+stories)", m_bldg.group(1), re.I)
        built_m = re.search(r"(\d{4}\s+built)", m_bldg.group(1), re.I)
        parts = [p.group(1) for p in (stories_m, built_m) if p]
        if parts:
            building_meta = ", ".join(parts)

    sqft_m = re.search(r"\b([\d,]+)\s*ft²", top_listing_header)
    sqft = parse_int_price(sqft_m.group(1)) if sqft_m else None

    return {
        "reachable": True,
        "valid": True,
        "floorplan_urls": floorplan_urls,
        "photo_urls": photo_urls,
        "about_text": about_text,
        "features": features_list,
        "building_meta": building_meta,
        "sqft": sqft,
    }


def parse_microlink_detail_html(main_html: str) -> dict[str, Any]:
    """Parse StreetEasy detail-page <main> DOM HTML returned by api.microlink.io."""
    if not main_html or len(main_html) < 800:
        return {"reachable": False, "valid": True}

    soup = BeautifulSoup(main_html, "html.parser")
    full_text = soup.get_text("\n", strip=True)
    lower_head = full_text[:2500].lower()
    if "we can't find that page" in lower_head or "page not found" in lower_head:
        return {"reachable": True, "valid": False, "reason": "404 page not found on StreetEasy"}

    before_hist_text = full_text.split("Property history")[0]
    top_header_text = before_hist_text.split("About")[0]
    if re.search(
        r"\b(?:no longer available|delisted|off[- ]market|rented)\b", top_header_text, re.I
    ):
        return {
            "reachable": True,
            "valid": False,
            "reason": "Listing is no longer available on StreetEasy",
        }

    floorplan_urls: list[str] = []
    photo_urls: list[str] = []
    seen_fp: set[str] = set()
    seen_ph: set[str] = set()

    for img in soup.find_all("img"):
        src = str(img.get("src") or "")
        alt = str(img.get("alt") or "").lower()
        hm = ZILLOW_HASH_FROM_URL_RE.search(src)
        if not hm:
            continue
        h = hm.group(1)
        if "floor" in alt and "plan" in alt:
            if h not in seen_fp:
                seen_fp.add(h)
                floorplan_urls.append(
                    f"https://photos.zillowstatic.com/fp/{h}-se_extra_large_1500_800.webp"
                )
        elif alt.startswith("photo ") or "image" in alt:
            if h not in seen_ph and h not in seen_fp:
                seen_ph.add(h)
                photo_urls.append(f"https://photos.zillowstatic.com/fp/{h}-cc_ft_960.webp")

    about_text = ""
    for h2 in soup.find_all(["h2", "h3"]):
        if h2.get_text(" ", strip=True).lower() == "about":
            sec_parent = h2.find_parent(["section", "div"])
            if sec_parent:
                raw_sec = sec_parent.get_text(" ", strip=True)
                about_text = re.sub(r"^About\s+", "", raw_sec, flags=re.I).strip()[:1600]
            break

    features_list: list[str] = []
    for h2 in soup.find_all(["h2", "h3"]):
        label = h2.get_text(" ", strip=True).lower()
        if label in ("home features", "building amenities"):
            sec_parent = h2.find_parent(["section", "div"])
            if sec_parent:
                for li in sec_parent.find_all("li"):
                    t = li.get_text(" ", strip=True)
                    if t and len(t) <= 60 and t not in features_list:
                        features_list.append(t)

    sqft_m = re.search(r"\b([\d,]+)\s*ft²", top_header_text)
    sqft = parse_int_price(sqft_m.group(1)) if sqft_m else None

    return {
        "reachable": True,
        "valid": True,
        "floorplan_urls": floorplan_urls,
        "photo_urls": photo_urls[:6],
        "about_text": about_text,
        "features": features_list,
        "building_meta": "",
        "sqft": sqft,
    }


class StreetEasyUESScraper(BaseScraper):
    """
    Scrapes live StreetEasy 3+ BR rental inventory for Upper East Side,
    Carnegie Hill, Yorkville, and Lenox Hill using a 3-provider cloud-IP-resilient architecture:
      1. r.jina.ai (with X-Target-Selector: main, 38s timeout, optional JINA_API_KEY, and 429 backoff)
      2. api.microlink.io (headless browser DOM <main> HTML extractor)
      3. markdown.new (Cloudflare Browser Rendering API)
    """

    source_name = "StreetEasy"
    _provider_cooldowns: dict[str, float] = {}

    def __init__(
        self,
        min_bedrooms: float = 3.0,
        min_bathrooms: float = 2.0,
        max_monthly_rent: int = 25000,
        max_pages: int = 5,
        timeout_seconds: float = 40.0,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.min_bedrooms = min_bedrooms
        self.min_bathrooms = min_bathrooms
        self.max_monthly_rent = max_monthly_rent
        self.max_pages = max_pages

    @classmethod
    def _is_provider_available(cls, name: str) -> bool:
        return time.monotonic() >= cls._provider_cooldowns.get(name, 0.0)

    @classmethod
    def _mark_provider_cooldown(cls, name: str, seconds: float) -> None:
        cls._provider_cooldowns[name] = time.monotonic() + seconds
        logger.info("[StreetEasy] Provider '%s' placed on %.0fs rate-limit cooldown", name, seconds)

    def update_search_parameters(
        self,
        min_bedrooms: float = 3.0,
        min_bathrooms: float = 2.0,
        max_monthly_rent: int = 25000,
    ) -> None:
        """Sync scraper URL filters with the user's live SearchCriteriaProfile."""
        self.min_bedrooms = max(1.0, min_bedrooms)
        self.min_bathrooms = max(1.0, min_bathrooms)
        self.max_monthly_rent = max(2000, max_monthly_rent)

    def _build_raw_streeteasy_search_url(
        self, page: int = 1, extra_filter: str = ""
    ) -> str:
        beds_int = max(1, int(self.min_bedrooms))
        filters = [f"beds>={beds_int}"]
        if self.min_bathrooms >= 1.5:
            baths_str = (
                str(int(self.min_bathrooms))
                if float(self.min_bathrooms).is_integer()
                else f"{self.min_bathrooms:g}"
            )
            filters.append(f"baths>={baths_str}")
        if self.max_monthly_rent > 0:
            filters.append(f"price:-{int(self.max_monthly_rent)}")
        if extra_filter:
            filters.append(extra_filter)

        encoded_filters = quote("|".join(filters), safe=":-,")
        return f"https://streeteasy.com/for-rent/ues/{encoded_filters}?sort_by=listed_desc&page={page}"

    async def _fetch_via_jina(
        self, client: httpx.AsyncClient, target_se_url: str, is_detail: bool = False
    ) -> str | None:
        if not self._is_provider_available("jina"):
            return None

        ua_headers: dict[str, str] = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/plain, text/markdown, */*;q=0.8",
            "X-Target-Selector": "main",
        }
        jina_key = os.getenv("JINA_API_KEY", "").strip()
        if jina_key:
            ua_headers["Authorization"] = f"Bearer {jina_key}"

        jina_url = f"https://r.jina.ai/{target_se_url}"
        attempts = [{}] if is_detail else [{}, {"X-No-Cache": "true"}]

        for idx, extra_h in enumerate(attempts):
            try:
                resp = await client.get(
                    jina_url,
                    headers={**ua_headers, **extra_h},
                    timeout=38.0,
                )
                if resp.status_code == 200 and len(resp.text) > 1200:
                    if "Access to this page has been denied" not in resp.text:
                        return resp.text
                    if is_detail:
                        return None
                elif resp.status_code in (429, 451):
                    # On shared Render NAT IPs, Jina's 60s window may transiently trip; cooldown briefly
                    self._mark_provider_cooldown("jina", 45.0)
                    return None
            except Exception as exc:
                logger.debug("[StreetEasy] r.jina.ai attempt %d error on %s: %s", idx + 1, target_se_url, exc)
            if idx + 1 < len(attempts):
                await asyncio.sleep(1.5)
        return None

    async def _fetch_via_microlink_html(
        self, client: httpx.AsyncClient, target_se_url: str
    ) -> str | None:
        if not self._is_provider_available("microlink"):
            return None

        try:
            resp = await client.get(
                "https://api.microlink.io/",
                params={
                    "url": target_se_url,
                    "data.mainHtml.selector": "main",
                    "data.mainHtml.type": "html",
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30.0,
            )
            if resp.status_code == 200:
                payload = resp.json()
                main_html = str((payload.get("data") or {}).get("mainHtml") or "")
                if (
                    len(main_html) > 1200
                    and "Access to this page has been denied" not in main_html
                ):
                    return main_html
            elif resp.status_code == 429:
                self._mark_provider_cooldown("microlink", 1800.0)
        except Exception as exc:
            logger.debug("[StreetEasy] Microlink error on %s: %s", target_se_url, exc)
        return None

    async def _fetch_via_markdown_new(
        self, client: httpx.AsyncClient, target_se_url: str
    ) -> str | None:
        if not self._is_provider_available("markdown_new"):
            return None

        ua_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
        }
        try:
            resp = await client.post(
                "https://markdown.new/",
                json={"url": target_se_url, "retain_images": True},
                headers=ua_headers,
                timeout=20.0,
            )
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "").lower()
                md_text = str(resp.json().get("content") or "") if "application/json" in ct else resp.text
                if len(md_text) > 1200 and "Access to this page has been denied" not in md_text:
                    return md_text
            elif resp.status_code == 429:
                self._mark_provider_cooldown("markdown_new", 1800.0)
                return None
        except Exception as exc:
            logger.debug("[StreetEasy] markdown.new POST error on %s: %s", target_se_url, exc)

        if "?" not in target_se_url and self._is_provider_available("markdown_new"):
            try:
                resp_get = await client.get(
                    f"https://markdown.new/{target_se_url}",
                    params={"retain_images": "true"},
                    headers=ua_headers,
                    timeout=20.0,
                )
                if (
                    resp_get.status_code == 200
                    and len(resp_get.text) > 1200
                    and "Access to this page has been denied" not in resp_get.text
                ):
                    return resp_get.text
                if resp_get.status_code == 429:
                    self._mark_provider_cooldown("markdown_new", 1800.0)
            except Exception as exc:
                logger.debug("[StreetEasy] markdown.new GET error on %s: %s", target_se_url, exc)

        return None

    async def _fetch_search_page_listings(
        self, client: httpx.AsyncClient, page: int
    ) -> tuple[list[Listing], int | None, bool]:
        """
        Fetch and parse a single StreetEasy search page across 3 independent cloud providers:
        1. r.jina.ai (20 RPM rolling window, rich 4-slide card previews)
        2. api.microlink.io (Vercel/Lambda headless browser DOM <main> HTML)
        3. markdown.new (Cloudflare Browser Rendering)
        Returns (listings, total_expected, fetched_ok).
        """
        se_url = self._build_raw_streeteasy_search_url(page=page)

        # 1. Try r.jina.ai
        md_jina = await self._fetch_via_jina(client, se_url, is_detail=False)
        if md_jina:
            items, tot = self.parse_jina_markdown(md_jina)
            if items:
                return items, tot, True

        # 2. Try api.microlink.io DOM <main> HTML
        html_ml = await self._fetch_via_microlink_html(client, se_url)
        if html_ml:
            items, tot = self.parse_microlink_search_html(html_ml)
            if items:
                return items, tot, True

        # 3. Try markdown.new
        md_new = await self._fetch_via_markdown_new(client, se_url)
        if md_new:
            items, tot = self.parse_jina_markdown(md_new)
            if items:
                return items, tot, True

        # 4. Last-resort retry on r.jina.ai after waiting 12s for shared-IP 60s window reset
        if page == 1:
            logger.info("[StreetEasy] Waiting 12s for Jina shared-IP window reset on page 1...")
            await asyncio.sleep(12.0)
            self._provider_cooldowns.pop("jina", None)
            md_retry = await self._fetch_via_jina(client, se_url, is_detail=False)
            if md_retry:
                items, tot = self.parse_jina_markdown(md_retry)
                if items:
                    return items, tot, True

        logger.warning("[StreetEasy] All 3 providers failed on search page %d (%s)", page, se_url)
        return [], None, False

    async def enrich_and_verify_listing(
        self, listing: Listing, client: httpx.AsyncClient | None = None
    ) -> bool:
        """
        Verify that `listing.url` resolves to a real, currently active StreetEasy listing
        (never a 404 broken link or off-market page), and enrich `listing` with the full
        broker description, complete amenity list, 1500px floorplan, and high-res gallery photos.
        """
        if not listing.url.startswith("https://streeteasy.com/") or is_known_mock_streeteasy_url(
            listing.url
        ):
            logger.warning("[StreetEasy] Rejected invalid or mock URL: %s", listing.url)
            return False

        owns_client = client is None
        http_client = client or httpx.AsyncClient(timeout=35.0, follow_redirects=True)
        try:
            parsed: dict[str, Any] | None = None

            # Try markdown.new first if not rate-limited, then Microlink DOM, then Jina
            md_detail = await self._fetch_via_markdown_new(http_client, listing.url)
            if md_detail:
                parsed = parse_streeteasy_detail_markdown(md_detail)

            if not parsed or not parsed.get("reachable"):
                ml_html = await self._fetch_via_microlink_html(http_client, listing.url)
                if ml_html:
                    parsed = parse_microlink_detail_html(ml_html)

            if not parsed or not parsed.get("reachable"):
                jina_md = await self._fetch_via_jina(http_client, listing.url, is_detail=True)
                if jina_md:
                    parsed = parse_streeteasy_detail_markdown(jina_md)

            if not parsed or not parsed.get("reachable"):
                # Listing was just discovered on StreetEasy's live search page; keep search-card data
                return True

            if not parsed.get("valid"):
                logger.info(
                    "[StreetEasy] Dropping dead/off-market link %s (%s)",
                    listing.url,
                    parsed.get("reason"),
                )
                return False

            if parsed.get("floorplan_urls"):
                existing_fp = [
                    u for u in listing.floorplan_urls if u not in parsed["floorplan_urls"]
                ]
                listing.floorplan_urls = list(parsed["floorplan_urls"]) + existing_fp

            if parsed.get("photo_urls"):
                listing.image_urls = list(parsed["photo_urls"])

            if parsed.get("sqft") and not listing.sqft:
                listing.sqft = int(parsed["sqft"])

            features = parsed.get("features") or []
            for feat in features:
                if feat not in listing.amenities:
                    listing.amenities.append(feat)

            extra_desc: list[str] = []
            if parsed.get("building_meta"):
                extra_desc.append(f"Building: {parsed['building_meta']}")
            if parsed.get("about_text"):
                extra_desc.append(f"Broker Description: {parsed['about_text']}")
            if features:
                extra_desc.append(f"Verified StreetEasy Features: {', '.join(features[:18])}")

            if extra_desc:
                listing.description = f"{listing.description}\n\n" + "\n".join(extra_desc)

            return True
        finally:
            if owns_client:
                await http_client.aclose()

    async def fetch_listings(self) -> list[Listing]:
        """Fetch all currently active UES StreetEasy listings matching bedroom/bath/rent filters."""
        discovered: dict[str, Listing] = {}
        total_expected: int | None = None
        consecutive_failures = 0

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds, follow_redirects=True
        ) as client:
            for page in range(1, self.max_pages + 1):
                if page > 1:
                    await asyncio.sleep(1.0)
                page_listings, page_total, ok = await self._fetch_search_page_listings(
                    client, page=page
                )
                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures >= 2:
                        break
                    continue

                consecutive_failures = 0
                if page_total is not None and total_expected is None:
                    total_expected = page_total

                before_count = len(discovered)
                for item in page_listings:
                    if item.listing_id not in discovered:
                        discovered[item.listing_id] = item

                added_on_page = len(discovered) - before_count
                logger.info(
                    "[StreetEasy] Page %d yielded %d organic UES cards (%d new, %d/%s total)",
                    page,
                    len(page_listings),
                    added_on_page,
                    len(discovered),
                    total_expected if total_expected is not None else "?",
                )
                if added_on_page == 0:
                    break
                if total_expected is not None and len(discovered) >= total_expected:
                    break

        logger.info(
            "[StreetEasy] Completed live UES scan: %d active StreetEasy listings discovered",
            len(discovered),
        )
        return list(discovered.values())

    def parse_microlink_search_html(self, main_html: str) -> tuple[list[Listing], int | None]:
        """Parse StreetEasy search <main> DOM HTML returned by api.microlink.io into Listing objects."""
        soup = BeautifulSoup(main_html, "html.parser")
        h1 = soup.find("h1")
        h1_text = h1.get_text(" ", strip=True) if h1 else ""
        tot_m = re.search(r"^(\d+)\s+.*?\bApartments?\s+for\s+Rent", h1_text, re.I)
        total_expected = int(tot_m.group(1)) if tot_m else None

        listings: dict[str, Listing] = {}
        unit_href_re = re.compile(
            r"^https://streeteasy\.com/(?:building/[^/?#]+/[^?#]+|rental/\d+)"
        )

        for li in soup.find_all("li"):
            matching_links = [
                a for a in li.find_all("a", href=True) if unit_href_re.match(str(a["href"]))
            ]
            if not matching_links:
                continue

            raw_url = str(matching_links[0]["href"]).strip()
            if "infeed=1" in raw_url.lower() or is_known_mock_streeteasy_url(raw_url):
                continue

            li_text = li.get_text(" ", strip=True)
            if re.search(r"\bSponsored\b", li_text) or "Near All Upper East Side" in li_text:
                continue

            title_a = None
            for a in matching_links:
                txt = a.get_text(" ", strip=True)
                if txt and not txt.startswith(
                    ("Featured", "Enhanced", "3D tour", "Video", "Live", "Open:", "Sponsored", "Image ")
                ):
                    title_a = a
                    break
            if title_a is None:
                continue

            raw_title = title_a.get_text(" ", strip=True)
            canonical_url = raw_url.split("?")[0].split("#")[0].rstrip("/")
            pre_title = li_text.split(raw_title, 1)[0]

            building_type = "Apartment"
            card_neigh_raw = ""
            tm = INLINE_TYPE_NEIGH_RE.search(pre_title)
            if tm:
                building_type = tm.group(1).strip()
                card_neigh_raw = tm.group(2).strip()

            address, unit = split_streeteasy_title_address_and_unit(raw_title, canonical_url)
            if not address or not is_strict_ues_street_bounds(address, card_neigh_raw):
                continue

            neighborhood = infer_ues_subneighborhood(address, card_neigh_raw or "Upper East Side")

            base_rent_m = re.search(r"\$([\d,]+)\s*base\s*rent", li_text, re.I)
            if base_rent_m:
                price = parse_int_price(base_rent_m.group(1))
            else:
                any_price_m = re.search(r"\$([\d,]+)", li_text)
                price = parse_int_price(any_price_m.group(1)) if any_price_m else 0
            if price <= 0:
                continue

            beds_m = re.search(r"(\d+(?:\.\d+)?)\s*beds?\b", li_text, re.I)
            beds = float(beds_m.group(1)) if beds_m else 0.0
            if beds < self.min_bedrooms:
                continue

            full_half_m = re.search(
                r"(\d+)\s*full(?:,\s*(\d+)\s*half)?\s*baths?\b", li_text, re.I
            )
            if full_half_m:
                baths = float(full_half_m.group(1)) + (
                    0.5 * float(full_half_m.group(2)) if full_half_m.group(2) else 0.0
                )
            else:
                baths_m = re.search(r"(\d+(?:\.\d+)?)\s*baths?\b", li_text, re.I)
                baths = float(baths_m.group(1)) if baths_m else 1.0

            sqft_m = re.search(r"([\d,]+)\s*ft²", li_text)
            sqft = parse_int_price(sqft_m.group(1)) if sqft_m else None

            photo_urls: list[str] = []
            floorplan_urls: list[str] = []
            seen_hashes: set[str] = set()
            for img in li.find_all("img"):
                src = str(img.get("src") or "")
                alt = str(img.get("alt") or "")
                hm = ZILLOW_HASH_FROM_URL_RE.search(src)
                if not hm:
                    continue
                h = hm.group(1)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                hi_res_url = f"https://photos.zillowstatic.com/fp/{h}-cc_ft_960.webp"
                pos_m = re.search(r"image\s+(\d+)\s+of\s+(\d+)", alt, re.I)
                if pos_m and int(pos_m.group(1)) == int(pos_m.group(2)) and int(pos_m.group(2)) >= 5:
                    floorplan_urls.append(hi_res_url)
                else:
                    photo_urls.append(hi_res_url)

            _, floor_label = infer_nyc_floor_from_unit(unit, f"{building_type} {raw_title}")
            amenities: list[str] = []
            if building_type.lower() in ("condo", "new development", "condop"):
                amenities.append(building_type.title())
            if "new development" in pre_title.lower() and "New Development" not in amenities:
                amenities.append("New Development")
            if "3d tour" in li_text.lower():
                amenities.append("3D Tour")

            path_slug = canonical_url.split("streeteasy.com/")[-1]
            path_slug = re.sub(r"^(?:building|rental)/", "", path_slug)
            clean_id_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", path_slug).strip("-").lower()
            listing_id = f"se:{clean_id_slug}"

            desc_parts = [
                f"StreetEasy Verified {building_type} in {neighborhood}: {address} {unit}".strip(),
                f"${price:,}/mo ({beds:g} BR / {baths:g} BA"
                + (f", {sqft:,} sq ft)" if sqft else ")"),
            ]
            if floor_label:
                desc_parts.append(f"Inferred Floor Level: {floor_label}")

            listings[listing_id] = Listing(
                listing_id=listing_id,
                source=self.source_name,
                url=canonical_url,
                address=address,
                unit=unit,
                neighborhood=neighborhood,
                price=price,
                bedrooms=beds,
                bathrooms=baths,
                sqft=sqft,
                floor_info=floor_label,
                description=" | ".join(desc_parts),
                amenities=amenities,
                image_urls=photo_urls,
                floorplan_urls=floorplan_urls,
            )

        return list(listings.values()), total_expected

    def parse_jina_markdown(self, markdown_text: str) -> tuple[list[Listing], int | None]:
        """Parse StreetEasy search results markdown (from r.jina.ai or markdown.new) into Listing objects."""
        total_match = re.search(
            r"^#\s+(\d+)\s+.*?\bApartments?\s+for\s+Rent",
            markdown_text,
            re.I | re.M,
        )
        total_expected = int(total_match.group(1)) if total_match else None

        matches = list(CARD_TITLE_LINK_RE.finditer(markdown_text))
        listings: dict[str, Listing] = {}

        for idx, match in enumerate(matches):
            raw_title = match.group(1).strip()
            raw_url = match.group(2).strip()

            if "infeed=1" in raw_url.lower() or is_known_mock_streeteasy_url(raw_url):
                continue

            prefix_start = matches[idx - 1].end() if idx > 0 else max(0, match.start() - 260)
            raw_prefix = markdown_text[max(prefix_start, match.start() - 320) : match.start()]
            prefix_text = re.split(r"\)Image\s+\d+\s+of\s+\d+", raw_prefix)[-1]

            if "Sponsored" in prefix_text or "Near All Upper East Side" in prefix_text:
                continue

            card_end = (
                matches[idx + 1].start()
                if idx + 1 < len(matches)
                else min(len(markdown_text), match.end() + 2200)
            )
            card_body = markdown_text[match.end() : card_end]
            own_card_segment = re.split(r"\)Image\s+\d+\s+of\s+\d+", card_body, maxsplit=1)[0]
            if re.search(r"\bSponsored\b", own_card_segment):
                continue

            building_type = "Apartment"
            card_neigh_raw = ""
            tm = CARD_TYPE_NEIGH_RE.search(prefix_text)
            if tm:
                building_type = tm.group(1).strip()
                card_neigh_raw = tm.group(2).strip()

            canonical_url = raw_url.split("?")[0].split("#")[0].rstrip("/")
            address, unit = split_streeteasy_title_address_and_unit(raw_title, canonical_url)
            if not address or not is_strict_ues_street_bounds(address, card_neigh_raw):
                continue

            neighborhood = infer_ues_subneighborhood(address, card_neigh_raw or "Upper East Side")

            base_rent_m = re.search(r"\$([\d,]+)\s*base\s*rent", own_card_segment, re.I)
            if base_rent_m:
                price = parse_int_price(base_rent_m.group(1))
            else:
                any_price_m = re.search(r"\$([\d,]+)", own_card_segment)
                price = parse_int_price(any_price_m.group(1)) if any_price_m else 0

            if price <= 0:
                continue

            beds_m = re.search(r"(\d+(?:\.\d+)?)\s*beds?\b", own_card_segment, re.I)
            beds = float(beds_m.group(1)) if beds_m else 0.0
            if beds < self.min_bedrooms:
                continue

            full_half_m = re.search(
                r"(\d+)\s*full(?:,\s*(\d+)\s*half)?\s*baths?\b", own_card_segment, re.I
            )
            if full_half_m:
                baths = float(full_half_m.group(1)) + (
                    0.5 * float(full_half_m.group(2)) if full_half_m.group(2) else 0.0
                )
            else:
                baths_m = re.search(r"(\d+(?:\.\d+)?)\s*baths?\b", own_card_segment, re.I)
                baths = float(baths_m.group(1)) if baths_m else 1.0

            sqft_m = re.search(r"([\d,]+)\s*ft²", own_card_segment)
            sqft = parse_int_price(sqft_m.group(1)) if sqft_m else None

            broker_m = re.search(r"Listing by\s+([^\n\[]+)", own_card_segment)
            broker_name = broker_m.group(1).strip() if broker_m else ""

            net_eff_m = re.search(r"\*\*\$([\d,]+)\*\*\s*net effective", own_card_segment, re.I)
            free_mo_m = re.search(
                r"(\d+(?:\.\d+)?\s*months?\s*free[^\n\[]*)", own_card_segment, re.I
            )

            photo_urls: list[str] = []
            floorplan_urls: list[str] = []
            seen_hashes: set[str] = set()
            total_gallery_photos = 0

            for img_m in ZILLOW_IMAGE_RE.finditer(own_card_segment):
                alt_text = img_m.group(1)
                img_hash = img_m.group(3)
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)
                hi_res_url = f"https://photos.zillowstatic.com/fp/{img_hash}-cc_ft_960.webp"

                pos_m = re.search(r"image\s+(\d+)\s+of\s+(\d+)", alt_text, re.I)
                if pos_m:
                    idx_num = int(pos_m.group(1))
                    tot_num = int(pos_m.group(2))
                    total_gallery_photos = max(total_gallery_photos, tot_num)
                    if idx_num == tot_num and tot_num >= 5:
                        floorplan_urls.append(hi_res_url)
                        continue
                photo_urls.append(hi_res_url)

            _, floor_label = infer_nyc_floor_from_unit(unit, f"{building_type} {raw_title}")

            amenities: list[str] = []
            if building_type.lower() in ("condo", "new development", "condop"):
                amenities.append(building_type.title())
            if "new development" in prefix_text.lower() and "New Development" not in amenities:
                amenities.append("New Development")
            if "no fee" in own_card_segment.lower():
                amenities.append("No Fee")
            if "3d tour" in own_card_segment.lower():
                amenities.append("3D Tour")

            path_slug = canonical_url.split("streeteasy.com/")[-1]
            path_slug = re.sub(r"^(?:building|rental)/", "", path_slug)
            clean_id_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", path_slug).strip("-").lower()
            listing_id = f"se:{clean_id_slug}"

            desc_parts = [
                f"StreetEasy Verified {building_type} in {neighborhood}: {address} {unit}".strip(),
                f"${price:,}/mo ({beds:g} BR / {baths:g} BA"
                + (f", {sqft:,} sq ft)" if sqft else ")"),
            ]
            if floor_label:
                desc_parts.append(f"Inferred Floor Level: {floor_label}")
            if net_eff_m:
                desc_parts.append(f"Net Effective Rent: ${net_eff_m.group(1)}/mo")
            if free_mo_m:
                desc_parts.append(f"Concession: {free_mo_m.group(1).strip()}")
            if broker_name:
                desc_parts.append(f"Listed by {broker_name}")
            if total_gallery_photos:
                desc_parts.append(f"StreetEasy Gallery: {total_gallery_photos} total photos")

            listing = Listing(
                listing_id=listing_id,
                source=self.source_name,
                url=canonical_url,
                address=address,
                unit=unit,
                neighborhood=neighborhood,
                price=price,
                bedrooms=beds,
                bathrooms=baths,
                sqft=sqft,
                floor_info=floor_label,
                description=" | ".join(desc_parts),
                amenities=amenities,
                image_urls=photo_urls,
                floorplan_urls=floorplan_urls,
            )
            listings[listing_id] = listing

        return list(listings.values()), total_expected

    def parse_search_html(self, html: str) -> list[Listing]:
        """Parse raw StreetEasy HTML via __NEXT_DATA__, JSON-LD, or DOM cards (offline test helper)."""
        soup = BeautifulSoup(html, "html.parser")
        results: list[Listing] = []

        next_script = soup.find("script", id="__NEXT_DATA__")
        if next_script and next_script.string:
            try:
                payload = json.loads(next_script.string)
                results.extend(self._extract_from_next_data(payload))
            except Exception as exc:
                logger.debug("[StreetEasy] __NEXT_DATA__ parse error: %s", exc)

        for ld_script in soup.find_all("script", type="application/ld+json"):
            if not ld_script.string:
                continue
            try:
                ld_data = json.loads(ld_script.string)
                results.extend(self._extract_from_json_ld(ld_data))
            except Exception:
                continue

        if not results:
            results.extend(self._extract_from_dom_cards(soup))

        return results

    def _extract_from_next_data(self, data: dict[str, Any]) -> list[Listing]:
        listings: list[Listing] = []
        raw_str = json.dumps(data)

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                if (
                    ("bedrooms" in node or "bedCount" in node)
                    and ("price" in node or "monthlyRent" in node)
                    and ("address" in node or "streetAddress" in node or "title" in node)
                ):
                    raw_id = str(node.get("id") or node.get("listingId") or "")
                    address = str(
                        node.get("streetAddress")
                        or node.get("address")
                        or node.get("title")
                        or ""
                    ).strip()
                    beds = parse_float_count(node.get("bedrooms") or node.get("bedCount"), 0.0)
                    price = parse_int_price(node.get("price") or node.get("monthlyRent"))
                    if raw_id and address and beds >= self.min_bedrooms and price > 0:
                        unit = str(node.get("unit") or "")
                        url_path = str(
                            node.get("url") or node.get("permalink") or f"/rental/{raw_id}"
                        )
                        full_url = (
                            url_path
                            if url_path.startswith("http")
                            else f"https://streeteasy.com{url_path}"
                        )
                        neigh = infer_ues_subneighborhood(
                            address, str(node.get("neighborhood") or "Upper East Side")
                        )
                        _, inferred_floor = infer_nyc_floor_from_unit(unit)
                        floor_info = str(
                            node.get("floor") or node.get("floorInfo") or inferred_floor
                        )
                        listings.append(
                            Listing(
                                listing_id=f"se:{raw_id}",
                                source=self.source_name,
                                url=full_url,
                                address=address,
                                unit=unit,
                                neighborhood=neigh,
                                price=price,
                                bedrooms=beds,
                                bathrooms=parse_float_count(
                                    node.get("bathrooms") or node.get("bathCount"), 2.0
                                ),
                                sqft=parse_int_price(node.get("sqft")) or None,
                                floor_info=floor_info,
                                description=str(
                                    node.get("description") or node.get("summary") or ""
                                ),
                                amenities=[
                                    str(a)
                                    for a in (node.get("amenities") or [])
                                    if isinstance(a, str)
                                ],
                                image_urls=[
                                    str(img)
                                    for img in (node.get("images") or node.get("photoUrls") or [])
                                    if isinstance(img, str) and img.startswith("http")
                                ],
                                floorplan_urls=[
                                    str(fp)
                                    for fp in (
                                        node.get("floorplans") or node.get("floorPlanUrls") or []
                                    )
                                    if isinstance(fp, str) and fp.startswith("http")
                                ],
                            )
                        )
                for val in node.values():
                    _walk(val)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        if len(raw_str) < 5_000_000:
            _walk(data)
        return listings

    def _extract_from_json_ld(self, ld_data: Any) -> list[Listing]:
        listings: list[Listing] = []
        items = ld_data if isinstance(ld_data, list) else [ld_data]
        for entry in items:
            if not isinstance(entry, dict):
                continue
            if entry.get("@type") in (
                "Apartment",
                "SingleFamilyResidence",
                "Product",
                "Residence",
            ):
                url = str(entry.get("url") or "")
                address_obj = entry.get("address") or {}
                street = (
                    address_obj.get("streetAddress")
                    if isinstance(address_obj, dict)
                    else str(entry.get("name") or "")
                )
                if not street or not url:
                    continue
                beds = parse_float_count(
                    entry.get("numberOfBedrooms") or entry.get("numberOfRooms"), 3.0
                )
                offers = entry.get("offers") or {}
                price = parse_int_price(offers.get("price") if isinstance(offers, dict) else 0)
                if beds >= self.min_bedrooms and price > 0:
                    slug = url.rstrip("/").split("/")[-1]
                    listings.append(
                        Listing(
                            listing_id=f"se:{slug}",
                            source=self.source_name,
                            url=url,
                            address=str(street),
                            neighborhood=infer_ues_subneighborhood(
                                str(street), "Upper East Side"
                            ),
                            price=price,
                            bedrooms=beds,
                            description=str(entry.get("description") or ""),
                        )
                    )
        return listings

    def _extract_from_dom_cards(self, soup: BeautifulSoup) -> list[Listing]:
        listings: list[Listing] = []
        for card in soup.select(
            "article, div[data-testid='listing-card'], li.searchCardList--listItem"
        ):
            link_el = card.select_one("a[href*='/building/'], a[href*='/rental/']")
            if not link_el:
                continue
            href = str(link_el.get("href") or "").strip()
            if not href or "infeed=1" in href:
                continue
            full_url = href if href.startswith("http") else f"https://streeteasy.com{href}"
            title_text = link_el.get_text(" ", strip=True)
            card_text = card.get_text(" ", strip=True)
            price = parse_int_price(card_text)
            beds_match = re.search(r"(\d+)\s*(?:bed|br|bd)", card_text, re.I)
            baths_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:bath|ba)", card_text, re.I)
            beds = float(beds_match.group(1)) if beds_match else 3.0
            baths = float(baths_match.group(1)) if baths_match else 2.0
            if price <= 0 or beds < self.min_bedrooms:
                continue

            address, unit = split_streeteasy_title_address_and_unit(title_text, full_url)
            _, floor_label = infer_nyc_floor_from_unit(unit)
            slug = re.sub(r"[^a-zA-Z0-9_-]", "_", full_url.rstrip("/").split("/")[-1])
            listings.append(
                Listing(
                    listing_id=f"se:{slug}",
                    source=self.source_name,
                    url=full_url,
                    address=address,
                    unit=unit,
                    neighborhood=infer_ues_subneighborhood(address, card_text),
                    price=price,
                    bedrooms=beds,
                    bathrooms=baths,
                    floor_info=floor_label,
                    description=card_text,
                )
            )
        return listings
