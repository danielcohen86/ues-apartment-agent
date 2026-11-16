"""NYBits Upper East Side 3+ BR no-fee & family rental scraper."""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

from src.models import Listing
from src.scrapers.base import BaseScraper, infer_ues_subneighborhood, parse_int_price

logger = logging.getLogger(__name__)


class NYBitsUESScraper(BaseScraper):
    """Scrapes NYBits Upper East Side 3+ BR apartment listings."""

    source_name = "NYBits"

    SEARCH_URL = (
        "https://www.nybits.com/search/?_a=search&nh=upper_east_side&bd=3_or_more"
    )

    async def fetch_listings(self) -> list[Listing]:
        html = await self.fetch_html(self.SEARCH_URL)
        if not html:
            return []
        listings = self.parse_search_html(html)
        logger.info("[NYBits] Discovered %d candidate 3+ BR UES listings", len(listings))
        return listings

    def parse_search_html(self, html: str) -> list[Listing]:
        soup = BeautifulSoup(html, "html.parser")
        results: dict[str, Listing] = {}

        for row in soup.select("tr, div.srch_row, li.listing"):
            link_el = row.select_one("a[href*='apartment'], a[href*='nybits.com']")
            if not link_el:
                continue
            href = str(link_el.get("href") or "").strip()
            if not href:
                continue
            full_url = href if href.startswith("http") else f"https://www.nybits.com{href}"
            text = row.get_text(" ", strip=True)
            price = parse_int_price(text)
            beds_match = re.search(r"(\d+)\s*(?:br|bed)", text, re.I)
            beds = float(beds_match.group(1)) if beds_match else 3.0
            if price <= 0 or beds < 3:
                continue

            slug = re.sub(r"\W+", "_", full_url.rstrip("/").split("/")[-1])
            title = link_el.get_text(" ", strip=True) or "Upper East Side 3BR"
            results[f"nyb:{slug}"] = Listing(
                listing_id=f"nyb:{slug}",
                source=self.source_name,
                url=full_url,
                address=title,
                neighborhood=infer_ues_subneighborhood(title, text),
                price=price,
                bedrooms=beds,
                bathrooms=2.0,
                description=text,
            )

        return list(results.values())
