"""Craigslist NYC Upper East Side / Carnegie Hill / Yorkville / Lenox Hill 3+ BR scraper."""

from __future__ import annotations

import asyncio
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


class CraigslistUESScraper(BaseScraper):
    """
    Scrapes live Manhattan 3+ BR apartment listings in Upper East Side, Carnegie Hill,
    Yorkville, and Lenox Hill from Craigslist NYC (returns HTTP 200 without Cloudflare blocks).
    Fetches individual detail pages to extract full descriptions, bath counts, and photos.
    """

    source_name = "Craigslist-NYC"

    SEARCH_URLS = (
        "https://newyork.craigslist.org/search/mnh/apa?min_bedrooms=3&query=upper+east+side",
        "https://newyork.craigslist.org/search/mnh/apa?min_bedrooms=3&query=carnegie+hill",
        "https://newyork.craigslist.org/search/mnh/apa?min_bedrooms=3&query=yorkville",
        "https://newyork.craigslist.org/search/mnh/apa?min_bedrooms=3&query=lenox+hill",
    )

    def __init__(self, max_detail_fetches: int = 15, timeout_seconds: float = 20.0) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.max_detail_fetches = max_detail_fetches

    async def fetch_listings(self) -> list[Listing]:
        stubs: dict[str, dict[str, str | int]] = {}
        for search_url in self.SEARCH_URLS:
            html = await self.fetch_html(search_url)
            if not html:
                continue
            for stub in self._parse_search_stubs(html):
                stubs[str(stub["listing_id"])] = stub

        selected_stubs = list(stubs.values())[: self.max_detail_fetches]
        listings: list[Listing] = []
        for stub in selected_stubs:
            detail_html = await self.fetch_html(str(stub["url"]))
            if detail_html:
                parsed = self._parse_detail_html(stub, detail_html)
                if parsed:
                    listings.append(parsed)
            await asyncio.sleep(0.4)

        logger.info(
            "[Craigslist-NYC] Discovered %d live 3+ BR UES listings with full details",
            len(listings),
        )
        return listings

    def _parse_search_stubs(self, html: str) -> list[dict[str, str | int]]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict[str, str | int]] = []
        for item in soup.select("li.cl-static-search-result"):
            link_el = item.select_one("a[href]")
            if not link_el:
                continue
            href = str(link_el.get("href") or "").strip()
            if not href.startswith("http"):
                continue
            title = str(item.get("title") or link_el.get_text(" ", strip=True)).strip()
            price_el = item.select_one(".price")
            price = parse_int_price(price_el.get_text(strip=True) if price_el else 0)
            if price <= 0:
                continue

            slug = href.rstrip("/").split("/")[-1].replace(".html", "")
            results.append(
                {
                    "listing_id": f"cl:{slug}",
                    "url": href,
                    "title": title,
                    "price": price,
                }
            )
        return results

    def _parse_detail_html(
        self, stub: dict[str, str | int], html: str
    ) -> Listing | None:
        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.select_one("#titletextonly")
        title = (
            title_el.get_text(" ", strip=True)
            if title_el
            else str(stub.get("title") or "Upper East Side 3BR")
        )

        body_el = soup.select_one("#postingbody")
        description = body_el.get_text("\n", strip=True) if body_el else title
        description = re.sub(r"QR Code Link to This Post\s*", "", description).strip()

        attr_spans = [
            s.get_text(" ", strip=True) for s in soup.select(".attrgroup span")
        ]
        attrs_joined = " | ".join(attr_spans)

        # Extract bedrooms and bathrooms from attributes like "3BR / 2.5Ba"
        beds = 3.0
        baths = 2.0
        sqft: int | None = None
        for span_text in attr_spans:
            br_match = re.search(r"(\d+)\s*BR", span_text, re.I)
            ba_match = re.search(r"(\d+(?:\.\d+)?)\s*Ba", span_text, re.I)
            ft_match = re.search(r"(\d{3,5})\s*ft2", span_text, re.I)
            if br_match:
                beds = parse_float_count(br_match.group(1), 3.0)
            if ba_match:
                baths = parse_float_count(ba_match.group(1), 2.0)
            if ft_match:
                sqft = int(ft_match.group(1))

        if beds < 3:
            return None

        # Extract address or cross streets if available
        map_addr_el = soup.select_one("div.mapaddress")
        address = (
            map_addr_el.get_text(" ", strip=True)
            if map_addr_el
            else title
        )

        # Extract images
        images: list[str] = []
        for thumb in soup.select("#thumbs a[href]"):
            img_href = str(thumb.get("href") or "").strip()
            if img_href.startswith("http"):
                images.append(img_href)
        if not images:
            first_img = soup.select_one(".slide.first img[src]")
            if first_img and str(first_img.get("src", "")).startswith("http"):
                images.append(str(first_img["src"]))

        # Extract floor/exposure hints from title + description
        floor_hints: list[str] = []
        combined = f"{title} {description}"
        floor_m = re.search(
            r"\b((?:high|\d{1,2}(?:st|nd|rd|th))\s+floor|penthouse|corner\s+unit|south(?:ern)?\s+exposure|sun[- ]?(?:flooded|drenched))\b",
            combined,
            re.I,
        )
        if floor_m:
            floor_hints.append(floor_m.group(0))

        neigh = infer_ues_subneighborhood(f"{address} {title}", description)

        return Listing(
            listing_id=str(stub["listing_id"]),
            source=self.source_name,
            url=str(stub["url"]),
            address=address,
            neighborhood=neigh,
            price=int(stub["price"]),
            bedrooms=beds,
            bathrooms=baths,
            sqft=sqft,
            floor_info=", ".join(floor_hints),
            description=f"{title}\n\n{description}\n\nAttributes: {attrs_joined}",
            amenities=attr_spans,
            image_urls=images[:6],
        )
