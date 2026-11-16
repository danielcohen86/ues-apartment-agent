"""
Gemini Google-Search-Grounded Scraper for StreetEasy, Compass, Corcoran, Elliman, and RentHop.

Why this exists:
StreetEasy and RentHop block datacenter IPs (like Render) with Cloudflare HTTP 403 Forbidden.
By using Gemini 2.5 Flash's built-in Google Search Grounding (`google_search` tool), we query
Google's live index of StreetEasy, Compass, Corcoran, Douglas Elliman, and RentHop at $0 cost,
completely bypassing Cloudflare 403 blocks.
"""

from __future__ import annotations

import json
import logging
import re

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.models import Listing
from src.scrapers.base import BaseScraper, infer_ues_subneighborhood

logger = logging.getLogger(__name__)


class GroundedListingItem(BaseModel):
    """Schema for a live rental listing discovered via Google Search Grounding."""

    source: str = Field(..., description="Portal name, e.g. 'StreetEasy', 'Compass', 'RentHop', 'Corcoran'")
    url: str = Field(..., description="Direct listing URL")
    address: str = Field(..., description="Street address in Upper East Side / Carnegie Hill / Yorkville / Lenox Hill")
    unit: str = Field(default="", description="Apartment unit number if shown, e.g. '#14A', '#11B', 'PH'")
    neighborhood: str = Field(
        default="Upper East Side",
        description="Upper East Side, Carnegie Hill, Yorkville, or Lenox Hill",
    )
    price: int = Field(..., description="Monthly rent in USD (integer)")
    bedrooms: float = Field(..., description="Number of bedrooms (must be >= 3)")
    bathrooms: float = Field(default=2.0, description="Number of bathrooms")
    sqft: int | None = Field(default=None, description="Square footage if mentioned")
    floor_info: str = Field(
        default="",
        description="Floor number, corner status, or light/exposure notes (e.g. '14th floor, South/East corner')",
    )
    description: str = Field(
        ...,
        description="Detailed description of the kitchen (island, breakfast bar, open concept) and natural light/exposures from the search result",
    )
    amenities: list[str] = Field(default_factory=list)


class GroundedListingBatch(BaseModel):
    """Batch of live UES 3+ BR rental listings returned from Google Search."""

    listings: list[GroundedListingItem] = Field(default_factory=list)


class GeminiSearchUESScraper(BaseScraper):
    """
    Uses Gemini 2.5 Flash with Google Search Grounding to find currently active
    3+ BR Upper East Side / Carnegie Hill / Yorkville / Lenox Hill rentals across
    StreetEasy, Compass, RentHop, Corcoran, and Brown Harris Stevens.
    """

    source_name = "Gemini-GoogleSearch-UES"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self._client = genai.Client(api_key=api_key) if api_key else None

    async def fetch_listings(self) -> list[Listing]:
        if not self._client:
            logger.info("[GeminiSearchUESScraper] Skipped (GEMINI_API_KEY not configured)")
            return []

        prompt = """Use Google Search to find currently active 3+ bedroom (3 bed, 4 bed) apartment rentals in Manhattan's Upper East Side, Carnegie Hill, Yorkville, and Lenox Hill on StreetEasy, Compass, RentHop, Corcoran, or Douglas Elliman.

Focus especially on listings that mention:
- Open kitchen, center island, waterfall island, or breakfast bar flowing into the living room
- High floor, corner unit, sunny, sun-flooded, or South/East/West exposure
- Located between East 59th Street and East 96th Street (Upper East Side, Carnegie Hill, Yorkville, Lenox Hill)

Return up to 12 real, distinct rental listings with their actual street address, unit number, monthly rent price, bedroom/bathroom count, URL, floor/exposure info, and detailed description of the kitchen and natural light.
Format your response ONLY as a valid JSON object matching this exact structure:
{
  "listings": [
    {
      "source": "StreetEasy",
      "url": "https://...",
      "address": "200 East 79th Street",
      "unit": "#14B",
      "neighborhood": "Upper East Side",
      "price": 13500,
      "bedrooms": 3,
      "bathrooms": 3,
      "sqft": 1950,
      "floor_info": "14th Floor, Corner South/East Exposure",
      "description": "Full details of kitchen island, layout, and light...",
      "amenities": ["Doorman", "Elevator", "Washer/Dryer"]
    }
  ]
}"""

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            raw_text = (response.text or "").strip()
            if not raw_text:
                return []

            # Extract JSON block from response text
            json_str = self._extract_json_object(raw_text)
            if not json_str:
                logger.warning("[GeminiSearchUESScraper] Could not extract JSON block from grounded search")
                return []

            batch = GroundedListingBatch.model_validate_json(json_str)
            results: dict[str, Listing] = {}
            for item in batch.listings:
                if item.bedrooms < 3 or item.price <= 0 or not item.address:
                    continue
                slug_base = f"{item.address}-{item.unit}".lower()
                slug = re.sub(r"[^a-z0-9]+", "-", slug_base).strip("-")
                listing_id = f"gsearch:{slug}"
                neigh = infer_ues_subneighborhood(item.address, item.neighborhood)
                results[listing_id] = Listing(
                    listing_id=listing_id,
                    source=item.source or "StreetEasy/Web",
                    url=item.url if item.url.startswith("http") else "https://streeteasy.com/for-rent/upper-east-side/beds>=3",
                    address=item.address,
                    unit=item.unit,
                    neighborhood=neigh,
                    price=item.price,
                    bedrooms=item.bedrooms,
                    bathrooms=item.bathrooms,
                    sqft=item.sqft,
                    floor_info=item.floor_info,
                    description=item.description,
                    amenities=item.amenities,
                )

            logger.info(
                "[GeminiSearchUESScraper] Discovered %d live UES 3+ BR listings via Google Search Grounding",
                len(results),
            )
            return list(results.values())
        except Exception as exc:
            logger.error("[GeminiSearchUESScraper] Search grounding error: %s", exc)
            return []

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if fence_match:
            return fence_match.group(1)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                candidate = text[start : end + 1]
                json.loads(candidate)
                return candidate
            except Exception:
                return None
        return None
