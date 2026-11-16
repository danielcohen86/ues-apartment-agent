"""Deterministic Stage-1 pre-filter enforcing user's live SearchCriteriaProfile."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models import Listing, SearchCriteriaProfile

# Default Upper East Side neighborhoods and sub-neighborhood aliases
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
    re.compile(
        r"\b(?:basement|cellar|garden\s+level|first\s+floor|1st\s+floor|ground\s+floor)\b",
        re.I,
    ),
    re.compile(r"\b(?:interior\s+courtyard\s+facing|airshaft\s+views?)\b", re.I),
]

# Low unit number pattern for apartment numbers like #1A, #1B, #2A, #2B, #G1 (unless duplex/townhouse)
LOW_FLOOR_UNIT_RE = re.compile(r"^#?(?:0?[12][A-Z]?|G\d*|BSMT|1ST|2ND)$", re.I)


@dataclass(frozen=True)
class PreFilterVerdict:
    """Result of deterministic Stage-1 pre-filtering."""

    passed: bool
    reason: str = ""


def is_valid_ues_location(
    listing: Listing, allowed_neighborhoods: list[str] | None = None
) -> bool:
    """Verify the listing is located in the user's allowed neighborhoods."""
    target_set = set(ALLOWED_UES_NEIGHBORHOODS)
    if allowed_neighborhoods:
        for n in allowed_neighborhoods:
            target_set.add(n.strip().lower())

    neigh_lower = listing.neighborhood.strip().lower()
    if any(alias in neigh_lower for alias in target_set):
        return True

    combined = f"{listing.address} {listing.description}".lower()
    if any(alias in combined for alias in target_set):
        return True

    return False


def prefilter_listing(
    listing: Listing,
    min_bedrooms: float = 3.0,
    max_monthly_rent: int | None = None,
    criteria: SearchCriteriaProfile | None = None,
) -> PreFilterVerdict:
    """
    Fast, zero-cost deterministic filter before invoking Gemini API.
    Enforces the user's live SearchCriteriaProfile (bedrooms, bathrooms, max rent, sqft, location,
    and hard kitchen/light disqualifiers).
    """
    eff_min_beds = criteria.min_bedrooms if criteria else min_bedrooms
    eff_min_baths = criteria.min_bathrooms if criteria else 1.0
    eff_max_rent = criteria.max_monthly_rent if criteria else max_monthly_rent
    eff_min_sqft = criteria.min_sqft if criteria else None
    eff_hoods = criteria.neighborhoods if criteria else None
    req_island = criteria.require_open_island_kitchen if criteria else True
    req_light = criteria.require_high_natural_light if criteria else True

    if not is_valid_ues_location(listing, allowed_neighborhoods=eff_hoods):
        hood_str = ", ".join(eff_hoods) if eff_hoods else "UES / Carnegie Hill / Yorkville / Lenox Hill"
        return PreFilterVerdict(
            passed=False,
            reason=f"Out-of-bounds neighborhood: '{listing.neighborhood}' (Requires {hood_str}).",
        )

    if listing.bedrooms < eff_min_beds:
        return PreFilterVerdict(
            passed=False,
            reason=f"Insufficient bedrooms: {listing.bedrooms} BR (Requires {eff_min_beds:g}+ BR).",
        )

    if listing.bathrooms < eff_min_baths:
        return PreFilterVerdict(
            passed=False,
            reason=f"Insufficient bathrooms: {listing.bathrooms} BA (Requires {eff_min_baths:g}+ BA).",
        )

    if eff_max_rent is not None and listing.price > eff_max_rent:
        return PreFilterVerdict(
            passed=False,
            reason=f"Price ${listing.price:,}/mo exceeds max rent ceiling ${eff_max_rent:,}/mo.",
        )

    if eff_min_sqft is not None and listing.sqft is not None and listing.sqft < eff_min_sqft:
        return PreFilterVerdict(
            passed=False,
            reason=f"Sqft {listing.sqft:,} is below minimum {eff_min_sqft:,} ft².",
        )

    combined_text = f"{listing.floor_info} {listing.description} {' '.join(listing.amenities)}"

    if req_island:
        for pattern in HARD_KITCHEN_DISQUALIFIER_PATTERNS:
            match = pattern.search(combined_text)
            if match:
                return PreFilterVerdict(
                    passed=False,
                    reason=f"Hard kitchen disqualifier matched: '{match.group(0)}'.",
                )

    if req_light:
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
