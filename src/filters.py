"""Deterministic Stage-1 pre-filter to eliminate non-matches before calling Gemini."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models import Listing

# Valid Upper East Side neighborhoods and sub-neighborhood aliases
ALLOWED_UES_NEIGHBORHOODS = {
    "upper east side",
    "ues",
    "carnegie hill",
    "yorkville",
    "lenox hill",
    "upper east side / carnegie hill",
    "upper east side / yorkville",
    "upper east side / lenox hill",
}

# Unambiguous closed/galley/windowed-pass-through-only disqualifier phrases
HARD_KITCHEN_DISQUALIFIER_PATTERNS = [
    re.compile(r"\bclosed\s+(?:galley\s+)?kitchen\b", re.I),
    re.compile(r"\bseparate\s+(?:windowed\s+)?(?:galley\s+)?kitchen\b", re.I),
    re.compile(r"\bclassic\s+galley\s+kitchen\b", re.I),
    re.compile(r"\bwindowed\s+galley\s+kitchen\b", re.I),
    re.compile(r"\benclosed\s+kitchen\b", re.I),
    re.compile(r"\bpass[- ]through\s+kitchen\s+without\s+island\b", re.I),
]

# Unambiguous low/dark floor disqualifier patterns
HARD_LIGHT_DISQUALIFIER_PATTERNS = [
    re.compile(r"\b(?:basement|cellar|garden\s+level|first\s+floor|1st\s+floor|ground\s+floor)\b", re.I),
    re.compile(r"\b(?:interior\s+courtyard\s+facing|airshaft\s+views?)\b", re.I),
]

# Low unit number pattern for apartment numbers like #1A, #1B, #2A, #2B, #G1 (unless duplex/townhouse)
LOW_FLOOR_UNIT_RE = re.compile(r"^#?(?:0?[12][A-Z]?|G\d*|BSMT|1ST|2ND)$", re.I)


@dataclass(frozen=True)
class PreFilterVerdict:
    """Result of deterministic Stage-1 pre-filtering."""

    passed: bool
    reason: str = ""


def is_valid_ues_location(listing: Listing) -> bool:
    """Verify the listing is located in Upper East Side, Carnegie Hill, Yorkville, or Lenox Hill."""
    neigh_lower = listing.neighborhood.strip().lower()
    if any(alias in neigh_lower for alias in ALLOWED_UES_NEIGHBORHOODS):
        return True

    # Also check if East 59th to East 96th Street address is explicitly in UES range
    combined = f"{listing.address} {listing.description}".lower()
    if any(alias in combined for alias in ("upper east side", "carnegie hill", "yorkville", "lenox hill")):
        return True

    return False


def prefilter_listing(
    listing: Listing,
    min_bedrooms: float = 3.0,
    max_monthly_rent: int | None = None,
) -> PreFilterVerdict:
    """
    Fast, zero-cost deterministic filter before invoking Gemini API.

    Checks:
    1. Neighborhood must be Upper East Side, Carnegie Hill, Yorkville, or Lenox Hill.
    2. Bedrooms must be >= 3.
    3. Optional rent ceiling check.
    4. Unambiguous hard disqualifiers in description/unit (closed galley kitchen, ground/1st/2nd floor).
    """
    if not is_valid_ues_location(listing):
        return PreFilterVerdict(
            passed=False,
            reason=f"Out-of-bounds neighborhood: '{listing.neighborhood}' (Requires UES / Carnegie Hill / Yorkville / Lenox Hill).",
        )

    if listing.bedrooms < min_bedrooms:
        return PreFilterVerdict(
            passed=False,
            reason=f"Insufficient bedrooms: {listing.bedrooms} BR (Requires {min_bedrooms}+ BR).",
        )

    if max_monthly_rent is not None and listing.price > max_monthly_rent:
        return PreFilterVerdict(
            passed=False,
            reason=f"Price ${listing.price:,}/mo exceeds max rent ceiling ${max_monthly_rent:,}/mo.",
        )

    combined_text = f"{listing.floor_info} {listing.description} {' '.join(listing.amenities)}"

    # Check hard kitchen disqualifiers (unless description explicitly says converted to open island kitchen)
    for pattern in HARD_KITCHEN_DISQUALIFIER_PATTERNS:
        match = pattern.search(combined_text)
        if match:
            return PreFilterVerdict(
                passed=False,
                reason=f"Hard kitchen disqualifier matched: '{match.group(0)}'.",
            )

    # Check hard light/floor disqualifiers
    for pattern in HARD_LIGHT_DISQUALIFIER_PATTERNS:
        match = pattern.search(combined_text)
        if match and "parlor" not in combined_text.lower() and "triplex" not in combined_text.lower():
            return PreFilterVerdict(
                passed=False,
                reason=f"Hard light/floor disqualifier matched: '{match.group(0)}'.",
            )

    unit_clean = listing.unit.strip()
    if unit_clean and LOW_FLOOR_UNIT_RE.match(unit_clean):
        if "townhouse" not in combined_text.lower() and "duplex" not in combined_text.lower():
            return PreFilterVerdict(
                passed=False,
                reason=f"Low-floor unit designator '{unit_clean}' rejected by high-natural-light policy.",
            )

    return PreFilterVerdict(passed=True, reason="Passed deterministic Stage-1 pre-filter.")
