"""RentHop Upper East Side (Carnegie Hill, Yorkville, Lenox Hill) 3+ BR rental scraper."""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

from src.models import Listing
from src.scrapers.base import (
    BaseScraper,
    infer_ues_subneighborhood,
    parse_float_count,
    parse_int_price,
)

logger = logging.getLogger(__name__)


class RentHopUESScraper(BaseScraper):
    """Scrapes RentHop Upper East Side 3+ bedroom apartment listings."""

    source_name = "RentHop"

    SEARCH_URL = (
        "https://www.renthop.com/search/nyc?"
        "min_price=0&max_price=50000&bedrooms%5B%5D=3&bedrooms%5B%5D=4"
        "&neighborhoods_str=10&sort=hopscore&page=1"
    )

    async def fetch_listings(self) -> list[Listing]:
        html = await self.fetch_html(self.SEARCH_URL)
        if not html:
            return []
        listings = self.parse_search_html(html)
        logger.info("[RentHop] Discovered %d candidate 3+ BR UES listings", len(listings))
        return listings

    def parse_search_html(self, html: str) -> list[Listing]:
        soup = BeautifulSoup(html, "html.parser")
        results: dict[str, Listing] = {}

        for card in soup.select("div.search-listing, div[id^='listing-']"):
            link_el = card.select_one("a[id^='listing-'][href*='/listings/'], a.listing-title-link")
            if not link_el:
                continue
            href = str(link_el.get("href") or "").strip()
            if not href:
                continue
            full_url = href if href.startswith("http") else f"https://www.renthop.com{href}"
            title = link_el.get_text(" ", strip=True)
            card_text = card.get_text(" ", strip=True)

            # Extract listing ID from URL
            id_match = re.search(r"/listings/[^/]+/[^/]+/(\d+)", full_url) or re.search(
                r"(\d{6,})", full_url
            )
            raw_id = id_match.group(1) if id_match else re.sub(r"\W+", "_", href[-24:])

            price_el = card.select_one("[id*='price'], .font-size-130, .listing-price")
            price = parse_int_price(price_el.get_text() if price_el else card_text)

            beds_match = re.search(r"(\d+)\s*(?:Bed|BR)", card_text, re.I)
            baths_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:Bath|BA)", card_text, re.I)
            beds = parse_float_count(beds_match.group(1) if beds_match else 3.0, 3.0)
            baths = parse_float_count(baths_match.group(1) if baths_match else 2.0, 2.0)

            if beds < 3 or price <= 0:
                continue

            images: list[str] = []
            for img in card.select("img"):
                src = str(img.get("src") or img.get("data-src") or "").strip()
                if src.startswith("http") and "renthop" in src:
                    images.append(src)

            neigh = infer_ues_subneighborhood(title, card_text)
            results[f"rh:{raw_id}"] = Listing(
                listing_id=f"rh:{raw_id}",
                source=self.source_name,
                url=full_url,
                address=title,
                neighborhood=neigh,
                price=price,
                bedrooms=beds,
                bathrooms=baths,
                description=card_text,
                image_urls=images,
            )

        return list(results.values())
