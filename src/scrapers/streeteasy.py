"""StreetEasy Upper East Side (Carnegie Hill, Yorkville, Lenox Hill) 3+ BR scraper."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from src.models import Listing
from src.scrapers.base import (
    BaseScraper,
    infer_ues_subneighborhood,
    parse_float_count,
    parse_int_price,
)

logger = logging.getLogger(__name__)


class StreetEasyUESScraper(BaseScraper):
    """
    Scrapes StreetEasy 3+ BR rental search pages for Upper East Side,
    Carnegie Hill, Yorkville, and Lenox Hill.
    """

    source_name = "StreetEasy"

    SEARCH_URLS = (
        "https://streeteasy.com/for-rent/upper-east-side/beds>=3",
        "https://streeteasy.com/for-rent/carnegie-hill/beds>=3",
        "https://streeteasy.com/for-rent/yorkville/beds>=3",
        "https://streeteasy.com/for-rent/lenox-hill/beds>=3",
    )

    async def fetch_listings(self) -> list[Listing]:
        discovered: dict[str, Listing] = {}
        for url in self.SEARCH_URLS:
            html = await self.fetch_html(url)
            if not html:
                continue
            for item in self.parse_search_html(html):
                discovered[item.listing_id] = item
        logger.info("[StreetEasy] Discovered %d candidate 3+ BR UES listings", len(discovered))
        return list(discovered.values())

    def parse_search_html(self, html: str) -> list[Listing]:
        """Parse StreetEasy HTML via __NEXT_DATA__ JSON state, JSON-LD, or DOM cards."""
        soup = BeautifulSoup(html, "html.parser")
        results: list[Listing] = []

        # 1. Try __NEXT_DATA__ hydration payload if present
        next_script = soup.find("script", id="__NEXT_DATA__")
        if next_script and next_script.string:
            try:
                payload = json.loads(next_script.string)
                results.extend(self._extract_from_next_data(payload))
            except Exception as exc:
                logger.debug("[StreetEasy] __NEXT_DATA__ parse error: %s", exc)

        # 2. Try JSON-LD structured data blocks
        for ld_script in soup.find_all("script", type="application/ld+json"):
            if not ld_script.string:
                continue
            try:
                ld_data = json.loads(ld_script.string)
                results.extend(self._extract_from_json_ld(ld_data))
            except Exception:
                continue

        # 3. Try DOM listing cards fallback
        if not results:
            results.extend(self._extract_from_dom_cards(soup))

        return results

    def _extract_from_next_data(self, data: dict[str, Any]) -> list[Listing]:
        """Extract listings from Next.js Apollo/Redux state."""
        listings: list[Listing] = []
        raw_str = json.dumps(data)
        # Search recursively for objects with id, price, bedrooms, address
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
                    if raw_id and address and beds >= 3 and price > 0:
                        unit = str(node.get("unit") or "")
                        url_path = str(node.get("url") or node.get("permalink") or f"/rental/{raw_id}")
                        full_url = (
                            url_path
                            if url_path.startswith("http")
                            else f"https://streeteasy.com{url_path}"
                        )
                        neigh = infer_ues_subneighborhood(
                            address, str(node.get("neighborhood") or "Upper East Side")
                        )
                        desc = str(node.get("description") or node.get("summary") or "")
                        floor_info = str(node.get("floor") or node.get("floorInfo") or "")
                        images = [
                            str(img)
                            for img in (node.get("images") or node.get("photoUrls") or [])
                            if isinstance(img, str) and img.startswith("http")
                        ]
                        floorplans = [
                            str(fp)
                            for fp in (node.get("floorplans") or node.get("floorPlanUrls") or [])
                            if isinstance(fp, str) and fp.startswith("http")
                        ]
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
                                description=desc,
                                amenities=[
                                    str(a) for a in (node.get("amenities") or []) if isinstance(a, str)
                                ],
                                image_urls=images,
                                floorplan_urls=floorplans,
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
            if entry.get("@type") in ("Apartment", "SingleFamilyResidence", "Product", "Residence"):
                url = str(entry.get("url") or "")
                address_obj = entry.get("address") or {}
                street = (
                    address_obj.get("streetAddress")
                    if isinstance(address_obj, dict)
                    else str(entry.get("name") or "")
                )
                if not street or not url:
                    continue
                beds = parse_float_count(entry.get("numberOfBedrooms") or entry.get("numberOfRooms"), 3.0)
                offers = entry.get("offers") or {}
                price = parse_int_price(offers.get("price") if isinstance(offers, dict) else 0)
                if beds >= 3 and price > 0:
                    slug = url.rstrip("/").split("/")[-1]
                    listings.append(
                        Listing(
                            listing_id=f"se:{slug}",
                            source=self.source_name,
                            url=url,
                            address=str(street),
                            neighborhood=infer_ues_subneighborhood(str(street), "Upper East Side"),
                            price=price,
                            bedrooms=beds,
                            description=str(entry.get("description") or ""),
                        )
                    )
        return listings

    def _extract_from_dom_cards(self, soup: BeautifulSoup) -> list[Listing]:
        listings: list[Listing] = []
        for card in soup.select("article, div[data-testid='listing-card'], li.searchCardList--listItem"):
            link_el = card.select_one("a[href*='/building/'], a[href*='/rental/']")
            if not link_el:
                continue
            href = str(link_el.get("href") or "").strip()
            if not href:
                continue
            full_url = href if href.startswith("http") else f"https://streeteasy.com{href}"
            title_text = link_el.get_text(" ", strip=True)
            card_text = card.get_text(" ", strip=True)
            price = parse_int_price(card_text)
            beds_match = re.search(r"(\d+)\s*(?:bed|br|bd)", card_text, re.I)
            baths_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:bath|ba)", card_text, re.I)
            beds = float(beds_match.group(1)) if beds_match else 3.0
            baths = float(baths_match.group(1)) if baths_match else 2.0
            if price <= 0 or beds < 3:
                continue

            slug = re.sub(r"[^a-zA-Z0-9_-]", "_", full_url.rstrip("/").split("/")[-1])
            listings.append(
                Listing(
                    listing_id=f"se:{slug}",
                    source=self.source_name,
                    url=full_url,
                    address=title_text,
                    neighborhood=infer_ues_subneighborhood(title_text, card_text),
                    price=price,
                    bedrooms=beds,
                    bathrooms=baths,
                    description=card_text,
                )
            )
        return listings
