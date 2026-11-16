"""Data models and Pydantic schemas for listings, dynamic criteria, Gemini evaluations, and feedback memory."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class SearchCriteriaProfile(BaseModel):
    """
    Live, user-adjustable search criteria profile persisted in the database.
    Can be viewed and modified anytime via Telegram (/criteria and /set).
    """

    neighborhoods: list[str] = Field(
        default_factory=lambda: [
            "Upper East Side",
            "Carnegie Hill",
            "Yorkville",
            "Lenox Hill",
        ],
        description="Allowed neighborhoods",
    )
    min_bedrooms: float = Field(default=3.0, ge=1.0, description="Minimum number of bedrooms")
    min_bathrooms: float = Field(default=2.0, ge=1.0, description="Minimum number of bathrooms")
    max_monthly_rent: int = Field(default=25000, ge=1000, description="Maximum monthly rent in USD")
    min_sqft: int | None = Field(default=None, description="Optional minimum square footage")
    require_open_island_kitchen: bool = Field(
        default=True,
        description="Require open-concept kitchen with real island or breakfast bar flowing into living area",
    )
    require_high_natural_light: bool = Field(
        default=True,
        description="Require high floor, corner exposure, or bright open natural light exposure",
    )
    custom_rules: list[str] = Field(
        default_factory=list,
        description="Explicit user-added standing rules (e.g. 'Must have in-unit washer/dryer')",
    )
    calibration_completed: bool = Field(
        default=False,
        description="True once the initial Top-10 calibration batch of active listings has been sent and baseline inventory marked seen.",
    )
    streeteasy_calibration_v2_completed: bool = Field(
        default=False,
        description="True once the StreetEasy-exclusive v2 calibration batch has purged legacy sources and sent Top-10 StreetEasy listings.",
    )
    calibration_batch_size: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Number of best active listings to send during initial calibration.",
    )
    max_alerts_per_scan: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum alerts sent in a single scan cycle so notifications are never overwhelming.",
    )
    updated_at: str = Field(default_factory=utc_now_iso)


class Listing(BaseModel):
    """Normalized rental listing scraped from StreetEasy, RentHop, NYBits, Craigslist, or Search."""

    listing_id: str = Field(..., description="Unique deterministic ID (e.g. 'se:123456')")
    source: str = Field(..., description="Source portal name (e.g. 'StreetEasy', 'Craigslist-NYC')")
    url: str = Field(..., description="Canonical listing URL")
    address: str = Field(..., description="Building street address")
    unit: str = Field(default="", description="Apartment unit number (e.g. '#14A', 'PH-B')")
    neighborhood: str = Field(
        ...,
        description="Neighborhood name (Upper East Side, Carnegie Hill, Yorkville, Lenox Hill)",
    )
    price: int = Field(..., ge=0, description="Monthly rent in USD")
    bedrooms: float = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(default=1.0, ge=0, description="Number of bathrooms")
    sqft: int | None = Field(default=None, description="Approximate square footage if listed")
    floor_info: str = Field(
        default="",
        description="Floor level or exposure metadata (e.g. '15th Floor, Corner South/East')",
    )
    description: str = Field(default="", description="Full broker listing description text")
    amenities: list[str] = Field(default_factory=list, description="Building & unit amenities")
    image_urls: list[str] = Field(default_factory=list, description="Listing photo URLs")
    floorplan_urls: list[str] = Field(default_factory=list, description="Floorplan image URLs")
    first_seen_at: str = Field(default_factory=utc_now_iso)

    @property
    def display_title(self) -> str:
        """Formatted address + unit title."""
        if self.unit:
            unit_clean = self.unit if self.unit.startswith("#") else f"#{self.unit}"
            return f"{self.address} {unit_clean}"
        return self.address


class EvaluationResult(BaseModel):
    """
    Two-Part Evaluation Schema:
    1. Binary Eligibility Gate: Does the listing meet 100% of active + learned criteria?
    2. Explicit 100-Point Ranking Rubric: Ranks all qualifying listings from #1 to #N.
    """

    # Part 1: Strict Criteria Gate (100% of qualifying apartments are shown & ranked)
    meets_all_criteria: bool = Field(
        default=True,
        description="True if and only if the apartment satisfies ALL active criteria AND all learned user feedback rules.",
    )
    has_open_island_or_breakfast_bar: bool = Field(
        ...,
        description="True ONLY if the kitchen is open-concept with a real island or breakfast bar/peninsula flowing into the main living space.",
    )
    has_high_natural_light: bool = Field(
        ...,
        description="True ONLY if the unit has high natural light exposure (high floor, corner unit, or bright open exposures).",
    )
    is_target_ues_location: bool = Field(
        ...,
        description="True ONLY if located in the user's target neighborhoods (Upper East Side, Carnegie Hill, Yorkville, Lenox Hill).",
    )
    passes_learned_feedback_rules: bool = Field(
        default=True,
        description="True ONLY if the apartment does NOT violate any of the user's learned [NO 👎] rules or custom standing criteria.",
    )
    disqualifiers_detected: list[str] = Field(
        default_factory=list,
        description="Any strict disqualifiers found (e.g., 'Closed galley kitchen', 'Pass-through without island', 'Low/dark floor', or violates a learned user NO rule).",
    )

    # Part 2: Explicit 100-Point Ranking Rubric (25 / 25 / 25 / 25 split)
    overall_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Total ranking score (0-100), equal to kitchen_points + light_points + layout_points + learned_preference_points.",
    )
    kitchen_points: int = Field(
        default=0,
        ge=0,
        le=25,
        description="0-25 pts: Kitchen & Island Flow — scale of center island/breakfast bar and openness into living/dining space.",
    )
    light_points: int = Field(
        default=0,
        ge=0,
        le=25,
        description="0-25 pts: Natural Light & Exposure — floor height, corner exposures, South/East/West sunlight, and window scale.",
    )
    layout_points: int = Field(
        default=0,
        ge=0,
        le=25,
        description="0-25 pts: Family Layout & Size — bedroom proportions, bathroom count, square footage, closets, and in-unit laundry.",
    )
    learned_preference_points: int = Field(
        default=0,
        ge=0,
        le=25,
        description="0-25 pts: Learned Preference Fit — calibrated toward user's prior [YES 👍] votes, micro-location, and value.",
    )

    @field_validator(
        "kitchen_points",
        "light_points",
        "layout_points",
        "learned_preference_points",
        mode="before",
    )
    @classmethod
    def _clamp_rubric_points(cls, v: Any) -> int:
        """Clamp rubric sub-scores to [0, 25] for backwards compatibility with legacy stored rows."""
        if v is None:
            return 0
        try:
            return max(0, min(25, int(round(float(v)))))
        except (TypeError, ValueError):
            return 0

    # Legacy 0-100 sub-scores preserved for backwards compatibility
    light_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Normalized 0-100 natural light score.",
    )
    kitchen_layout_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Normalized 0-100 kitchen openness score.",
    )

    light_analysis: str = Field(
        ...,
        description="Concise 1-2 sentence assessment of floor height, exposures, window scale, and natural light.",
    )
    kitchen_analysis: str = Field(
        ...,
        description="Concise 1-2 sentence assessment of the kitchen layout, island/breakfast bar seating, and flow into the living/dining space.",
    )
    feedback_alignment_notes: str = Field(
        default="",
        description="How this property aligns with or violates prior user YES/NO training feedback.",
    )
    pros: list[str] = Field(default_factory=list, description="Top 2-4 standout strengths")
    cons: list[str] = Field(default_factory=list, description="Notable caveats or trade-offs")
    reasoning_summary: str = Field(
        ...,
        description="Executive summary explaining why this property qualifies and how it ranks.",
    )

    def passes_qualifying_criteria(
        self,
        require_open_island: bool = True,
        require_high_light: bool = True,
    ) -> bool:
        """
        Return True if the apartment meets 100% of the user's active criteria
        (both current criteria settings and learned feedback rules).
        100% of qualifying apartments are shown to the user and ranked by overall_score.
        """
        if require_open_island and not self.has_open_island_or_breakfast_bar:
            return False
        if require_high_light and not self.has_high_natural_light:
            return False
        return (
            self.is_target_ues_location is True
            and self.passes_learned_feedback_rules is True
            and len(self.disqualifiers_detected) == 0
        )

    def passes_high_conviction_policy(self, threshold: int = 85) -> bool:
        """Backwards-compatible check for threshold + qualifying criteria."""
        if not self.passes_qualifying_criteria():
            return False
        return self.overall_score >= threshold


class EvaluatedListing(BaseModel):
    """Combined listing record and evaluation stored in the database."""

    listing: Listing
    evaluation: EvaluationResult
    alerted: bool = False
    telegram_message_id: int | None = None
    evaluated_at: str = Field(default_factory=utc_now_iso)


class FeedbackEntry(BaseModel):
    """Active training feedback item recorded from Telegram [ YES 👍 ] / [ NO 👎 ] + reason."""

    id: str = Field(..., description="Feedback record ID")
    listing_id: str = Field(..., description="Associated listing_id")
    vote: Literal["YES", "NO"] = Field(..., description="User vote")
    reason: str = Field(
        default="",
        description="User's text or voice-transcribed explanation (especially for NO votes)",
    )
    feedback_source: Literal["inline_button", "text_reply", "voice_note"] = "inline_button"
    listing_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of listing attributes (address, neighborhood, floor, kitchen, score)",
    )
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class PendingFeedbackPrompt(BaseModel):
    """Tracks an active prompt waiting for the user's text or voice explanation after clicking NO."""

    chat_id: int
    listing_id: str
    prompt_message_id: int | None = None
    created_at: str = Field(default_factory=utc_now_iso)
