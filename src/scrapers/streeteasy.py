"""StreetEasy Upper East Side (Carnegie Hill, Yorkville, Lenox Hill) 3+ BR live scraper."""

from __future__ import annotations

import asyncio
import json
import logging
import re
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

CARD_TITLE_LINK_RE = re.compile(
    r"(?<!!)\[([^\[\]\n]+)\]\((https://streeteasy\.com/(?:building/[^/)\s?]+/[^)\s]+|rental/\d+[^)\s]*))\)"
)

CARD_TYPE_NEIGH_RE = re.compile(
    r"(Condo|Co-op|Condop|Rental unit|Townhouse|House|Multi-family|Loft|New development|Building)\s+in\s+([A-Za-z \-]+)\s*$",
    re.I | re.M,
)

ZILLOW_IMAGE_RE = re.compile(
    r"!\[Image\s+\d+:\s*([^\]]*)\]\((https://photos\.zillowstatic\.com/fp/([a-zA-Z0-9_-]+))-[^)\s.]+\.(?:webp|jpg|jpeg|png)\)"
)


def split_streeteasy_title_address_and_unit(title: str, canonical_url: str = "") -> tuple[str, str]:
    """Split '400 East 67th Street #27B' or '340 East 64th Street #10/11J' into address and unit."""
    cleaned = " ".join((title or "").split()).strip()
    m = re.match(
        r"^(.*?)(?:\s+#|\s+Apt\.?\s+|\s+Unit\s+|\s+Residence\s+|,\s*#?)([A-Za-z0-9/-]+)$",
        cleaned,
        re.I,
    )
    if m:
        addr = m.group(1).strip().rstrip(",")
        raw_u = m.group(2).strip().lstrip("#")
        return addr, f"#{raw_u}"

    # Fallback: derive unit from /building/<slug>/<unit>
    if canonical_url:
        u_match = re.search(r"streeteasy\.com/building/[^/]+/([^/?#]+)", canonical_url)
        if u_match:
            u_slug = u_match.group(1).strip().upper()
            if u_slug and len(u_slug) <= 10:
                return cleaned, f"#{u_slug}"

    return cleaned, ""


def is_strict_ues_street_bounds(address: str, card_neighborhood: str) -> bool:
    """Reject East Harlem (E 98th+) or Midtown (< E 59th) sponsored spillover cards."""
    neigh_clean = card_neighborhood.strip().lower()
    if neigh_clean and not any(v in neigh_clean for v in VALID_UES_CARD_NEIGHBORHOODS):
        return False

    st_match = re.search(r"\b(?:east|e\.?)\s*(\d{1,3})(?:st|nd|rd|th)?\s+(?:street|st)\b", address, re.I)
    if st_match:
        st_num = int(st_match.group(1))
        if st_num >= 98 or st_num <= 58:
            return False

    if re.search(r"\b(?:west|w\.?)\s*\d+", address, re.I):
        return False

    return True


class StreetEasyUESScraper(BaseScraper):
    """
    Scrapes live StreetEasy 3+ BR rental inventory for Upper East Side,
    Carnegie Hill, Yorkville, and Lenox Hill via Jina Reader markdown + high-res 960px gallery photos,
    plus StreetEasy amenity filter cross-referencing (in-unit W/D, doorman).
    """

    source_name = "StreetEasy"

    def __init__(
        self,
        min_bedrooms: float = 3.0,
        min_bathrooms: float = 2.0,
        max_monthly_rent: int = 25000,
        max_pages: int = 6,
        timeout_seconds: float = 28.0,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.min_bedrooms = min_bedrooms
        self.min_bathrooms = min_bathrooms
        self.max_monthly_rent = max_monthly_rent
        self.max_pages = max_pages

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

    def _build_streeteasy_search_url(
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
        se_url = f"https://streeteasy.com/for-rent/ues/{encoded_filters}?sort_by=listed_desc&page={page}"
        return f"https://r.jina.ai/{se_url}"

    async def _fetch_jina_markdown(self, client: httpx.AsyncClient, jina_url: str) -> str | None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/plain, text/markdown, */*;q=0.8",
        }
        for attempt in range(2):
            try:
                resp = await client.get(jina_url, headers=headers)
                if resp.status_code == 200 and len(resp.text) > 1500:
                    if "Access to this page has been denied" in resp.text:
                        logger.warning("[StreetEasy] PerimeterX block detected on %s", jina_url)
                        return None
                    return resp.text
                if resp.status_code == 429:
                    await asyncio.sleep(3.5 * (attempt + 1))
                    continue
            except Exception as exc:
                logger.debug("[StreetEasy] Fetch error on %s: %s", jina_url, exc)
            await asyncio.sleep(1.2 * (attempt + 1))
        return None

    async def _collect_verified_filter_urls(
        self, client: httpx.AsyncClient, extra_filter: str, pages: int = 2
    ) -> set[str]:
        """Fetch StreetEasy amenity-filtered search pages to verify exact database attributes (e.g., W/D, Doorman)."""
        matched_urls: set[str] = set()
        for p in range(1, pages + 1):
            await asyncio.sleep(0.85)
            md = await self._fetch_jina_markdown(
                client, self._build_streeteasy_search_url(page=p, extra_filter=extra_filter)
            )
            if not md:
                break
            page_items, total_expected = self.parse_jina_markdown(md)
            if not page_items:
                break
            for item in page_items:
                matched_urls.add(item.url)
            if total_expected is not None and len(matched_urls) >= total_expected:
                break
        return matched_urls

    async def fetch_listings(self) -> list[Listing]:
        """Fetch all currently active UES StreetEasy listings matching bedroom/bath/rent filters."""
        discovered: dict[str, Listing] = {}
        total_expected: int | None = None

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds, follow_redirects=True
        ) as client:
            for page in range(1, self.max_pages + 1):
                if page > 1:
                    await asyncio.sleep(0.85)
                jina_url = self._build_streeteasy_search_url(page=page)
                md = await self._fetch_jina_markdown(client, jina_url)
                if not md:
                    break

                page_listings, page_total = self.parse_jina_markdown(md)
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

            # Cross-reference StreetEasy verified In-Unit Washer/Dryer & Doorman filters
            if discovered:
                wd_urls = await self._collect_verified_filter_urls(
                    client, "amenities:washer_dryer", pages=3
                )
                dm_urls = await self._collect_verified_filter_urls(
                    client, "amenities:doorman", pages=3
                )
                for item in discovered.values():
                    extra_notes: list[str] = []
                    if item.url in wd_urls:
                        if "In-Unit Washer/Dryer" not in item.amenities:
                            item.amenities.append("In-Unit Washer/Dryer")
                        extra_notes.append("Verified In-Unit Washer/Dryer on StreetEasy")
                    if item.url in dm_urls:
                        if "Doorman" not in item.amenities:
                            item.amenities.append("Doorman")
                        if "Elevator" not in item.amenities:
                            item.amenities.append("Elevator")
                        extra_notes.append("Verified Full-Service Doorman Building")
                    if extra_notes:
                        item.description = f"{item.description} | {' • '.join(extra_notes)}".strip(" |")

        logger.info(
            "[StreetEasy] Completed live UES scan: %d active StreetEasy listings ready",
            len(discovered),
        )
        return list(discovered.values())

    def parse_jina_markdown(self, markdown_text: str) -> tuple[list[Listing], int | None]:
        """Parse Jina Reader markdown of a StreetEasy search results page into normalized Listing objects."""
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

            # Reject sponsored out-of-neighborhood in-feed ads (?infeed=1)
            if "infeed=1" in raw_url.lower():
                continue

            # Prefix window immediately above the [Address #Unit] link contains building type & neighborhood
            prefix_start = matches[idx - 1].end() if idx > 0 else max(0, match.start() - 260)
            prefix_text = markdown_text[max(prefix_start, match.start() - 220) : match.start()]
            if "Sponsored" in prefix_text or "Near All Upper East Side" in prefix_text:
                continue

            card_end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(markdown_text), match.end() + 2200)
            card_body = markdown_text[match.end() : card_end]

            # Extract building type & neighborhood (e.g. "Condo in Lenox Hill" or "Rental unit in Carnegie Hill")
            building_type = "Apartment"
            card_neigh_raw = ""
            type_matches = list(CARD_TYPE_NEIGH_RE.finditer(prefix_text))
            if type_matches:
                last_tm = type_matches[-1]
                building_type = last_tm.group(1).strip()
                card_neigh_raw = last_tm.group(2).strip()

            canonical_url = raw_url.split("?")[0].split("#")[0].rstrip("/")
            address, unit = split_streeteasy_title_address_and_unit(raw_title, canonical_url)
            if not address or not is_strict_ues_street_bounds(address, card_neigh_raw):
                continue

            neighborhood = infer_ues_subneighborhood(address, card_neigh_raw or "Upper East Side")

            # Parse base rent + optional net-effective / concession details
            base_rent_m = re.search(r"\$([\d,]+)\s*base\s*rent", card_body, re.I)
            if base_rent_m:
                price = parse_int_price(base_rent_m.group(1))
            else:
                any_price_m = re.search(r"\$([\d,]+)", card_body)
                price = parse_int_price(any_price_m.group(1)) if any_price_m else 0

            if price <= 0:
                continue

            # Parse beds, baths, sqft
            beds_m = re.search(r"(\d+(?:\.\d+)?)\s*beds?\b", card_body, re.I)
            beds = float(beds_m.group(1)) if beds_m else 0.0
            if beds < self.min_bedrooms:
                continue

            full_half_m = re.search(r"(\d+)\s*full(?:,\s*(\d+)\s*half)?\s*baths?\b", card_body, re.I)
            if full_half_m:
                baths = float(full_half_m.group(1)) + (
                    0.5 * float(full_half_m.group(2)) if full_half_m.group(2) else 0.0
                )
            else:
                baths_m = re.search(r"(\d+(?:\.\d+)?)\s*baths?\b", card_body, re.I)
                baths = float(baths_m.group(1)) if baths_m else 1.0

            sqft_m = re.search(r"([\d,]+)\s*ft²", card_body)
            sqft = parse_int_price(sqft_m.group(1)) if sqft_m else None

            # Extract brokerage & concessions
            broker_m = re.search(r"Listing by\s+([^\n\[]+)", card_body)
            broker_name = broker_m.group(1).strip() if broker_m else ""

            net_eff_m = re.search(r"\*\*\$([\d,]+)\*\*\s*net effective", card_body, re.I)
            free_mo_m = re.search(r"(\d+(?:\.\d+)?\s*months?\s*free[^\n\[]*)", card_body, re.I)

            # Extract and upgrade Zillow/StreetEasy CDN images to crisp 960px (-cc_ft_960.webp)
            # Note: StreetEasy card carousels include image 1, 2, 3 AND image N of N (often the floorplan!).
            photo_urls: list[str] = []
            floorplan_urls: list[str] = []
            seen_hashes: set[str] = set()
            total_gallery_photos = 0

            for img_m in ZILLOW_IMAGE_RE.finditer(card_body):
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

            # Infer NYC floor height from unit designator
            _, floor_label = infer_nyc_floor_from_unit(unit, f"{building_type} {raw_title}")

            amenities: list[str] = []
            if building_type.lower() in ("condo", "new development", "condop"):
                amenities.append(building_type.title())
            if "no fee" in card_body.lower():
                amenities.append("No Fee")
            if "3d tour" in card_body.lower():
                amenities.append("3D Tour")

            # Deterministic StreetEasy ID from canonical URL path
            path_slug = canonical_url.split("streeteasy.com/")[-1]
            path_slug = re.sub(r"^(?:building|rental)/", "", path_slug)
            clean_id_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", path_slug).strip("-").lower()
            listing_id = f"se:{clean_id_slug}"

            # Build structured summary description for Gemini + user transparency
            desc_parts = [
                f"StreetEasy Verified {building_type} in {neighborhood}: {address} {unit}".strip(),
                f"${price:,}/mo ({beds:g} BR / {baths:g} BA" + (f", {sqft:,} sq ft)" if sqft else ")"),
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
                        url_path = str(node.get("url") or node.get("permalink") or f"/rental/{raw_id}")
                        full_url = (
                            url_path
                            if url_path.startswith("http")
                            else f"https://streeteasy.com{url_path}"
                        )
                        neigh = infer_ues_subneighborhood(
                            address, str(node.get("neighborhood") or "Upper East Side")
                        )
                        _, inferred_floor = infer_nyc_floor_from_unit(unit)
                        floor_info = str(node.get("floor") or node.get("floorInfo") or inferred_floor)
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
                                description=str(node.get("description") or node.get("summary") or ""),
                                amenities=[
                                    str(a) for a in (node.get("amenities") or []) if isinstance(a, str)
                                ],
                                image_urls=[
                                    str(img)
                                    for img in (node.get("images") or node.get("photoUrls") or [])
                                    if isinstance(img, str) and img.startswith("http")
                                ],
                                floorplan_urls=[
                                    str(fp)
                                    for fp in (node.get("floorplans") or node.get("floorPlanUrls") or [])
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
                if beds >= self.min_bedrooms and price > 0:
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
