"""Gemini AI evaluation pipeline with Two-Stage Eligibility Gate + Dynamic 100-Point Ranking Rubric."""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import urlparse

import httpx
from google import genai
from google.genai import types

from src.config import AgentConfig
from src.models import EvaluationResult, Listing
from src.storage import StorageBackend

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4MB per image safety cap
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}


SYSTEM_PROMPT_TEMPLATE = """You are an elite Upper East Side (NYC) family real estate advisor.
Your job has TWO distinct stages for every candidate 3+ bedroom rental:

================================================================================
STAGE 1: STRICT binary ELIGIBILITY GATE (Show 100% of Qualifying Apartments)
================================================================================
First, determine if the apartment satisfies 100% of the family's mandatory criteria (both original criteria and learned criteria from their feedback log):

1. LOCATION (`is_target_ues_location`):
   Must be in Upper East Side, Carnegie Hill, Yorkville, or Lenox Hill.
2. SIZE:
   Must be 3+ bedrooms.
3. OPEN ISLAND / BREAKFAST BAR KITCHEN (`has_open_island_or_breakfast_bar`):
   - MUST have an open-concept kitchen with a real center island or breakfast bar / peninsula that flows into the main living/dining room.
   - STRICT DISQUALIFIERS: Closed galley kitchens, separate enclosed kitchens, or small windowed pass-through cut-outs without an island/breakfast bar flow.
4. NATURAL LIGHT (`has_high_natural_light`):
   - Must have bright natural light exposure: high floor (note: in NYC unit numbers like #12A, #15B, #18C indicate floors 12, 15, 18), corner unit, or open sunny South/East/West exposures.
   - STRICT DISQUALIFIERS: Dark lower floors (floors 1-3, basement/garden unless sunny parlor triplex), dark interior courtyard/airshaft views, or dark interior hallways isolating the living space.
5. LEARNED USER FEEDBACK RULES (`passes_learned_feedback_rules`):
   - Must NOT violate any of the user's learned `[NO 👎]` rejection rules or standing criteria listed in the Active Training Memory below.

If ALL 5 checks pass and `disqualifiers_detected` is empty, set `meets_all_criteria = true`.
Every single apartment with `meets_all_criteria = true` will be shown to the user!

================================================================================
STAGE 2: EXPLICIT 100-POINT RANKING RUBRIC (Rank Qualifying Apartments #1 to #N)
================================================================================
Score each apartment using this transparent 100-point additive rubric so all qualifying apartments can be ranked from best (#1) to lowest:

1. `kitchen_points` (0 to 25 pts — Kitchen & Island Flow):
   - 22–25 pts: Showpiece chef's open kitchen with large center/waterfall island, counter seating, and seamless great-room flow.
   - 16–21 pts: Verified open kitchen with breakfast bar / peninsula or standard island opening into living/dining area.
   - 0–15 pts: Closed, galley, or non-open pass-through (fails Stage 1).

2. `light_points` (0 to 25 pts — Natural Light & Exposure):
   - 22–25 pts: High floor (10th+ floor or PH) AND corner exposure or open South/East/West views with oversized windows.
   - 16–21 pts: Solid high/mid-high floor (6th–9th floor) or bright open exposure with good natural light.
   - 0–15 pts: Low floor or dark exposure (fails Stage 1).

3. `layout_points` (0 to 25 pts — Family Layout & Size):
   - 22–25 pts: Generous family bedroom dimensions (king primary + spacious kids' bedrooms), 2.5–3+ bathrooms, strong square footage, great closets, and in-unit washer/dryer.
   - 16–21 pts: Solid 3+ BR / 2+ BA family layout meeting active size criteria.
   - 0–15 pts: Cramped/awkward layout or undersized bedrooms.

4. `learned_preference_points` (0 to 25 pts — Learned Preference Fit):
   - Calibrated toward the user's prior `[YES 👍]` votes, preferred micro-neighborhood/blocks, building quality, and price-to-quality value (if no `[YES 👍]` votes exist yet, score 16–22 pts based on Carnegie Hill / UES family appeal and rent value).

Set `overall_score = kitchen_points + light_points + layout_points + learned_preference_points`.

================================================================================
ACTIVE TRAINING MEMORY (DYNAMIC USER FEEDBACK LOG)
================================================================================
Treat every user `[NO 👎]` rejection reason below as a HARD rule for `passes_learned_feedback_rules`.
Treat every user `[YES 👍]` approval below as a benchmark that boosts `learned_preference_points`.

{feedback_memory_block}
"""


def build_listing_evaluation_prompt(
    listing: Listing,
    feedback_memory_block: str,
    min_score: int = 85,
) -> str:
    """Construct the full evaluation prompt combining system instructions, feedback log, and listing data."""
    system_block = SYSTEM_PROMPT_TEMPLATE.format(
        feedback_memory_block=feedback_memory_block,
    )
    amenities_str = ", ".join(listing.amenities) if listing.amenities else "None listed"
    sqft_str = f"{listing.sqft:,} sq ft" if listing.sqft else "Not specified"

    user_block = f"""
================================================================================
CANDIDATE LISTING TO EVALUATE & RANK
================================================================================
- Listing ID: {listing.listing_id}
- Source: {listing.source}
- Title / Address: {listing.display_title}
- Unit Number: {listing.unit or 'Not specified'}
- Neighborhood: {listing.neighborhood}
- Monthly Rent: ${listing.price:,}/month
- Size: {listing.bedrooms} Bedrooms | {listing.bathrooms} Bathrooms | {sqft_str}
- Floor / Exposure Info: {listing.floor_info or 'Infer from unit number or description'}
- Amenities: {amenities_str}
- Floorplan URLs Present: {len(listing.floorplan_urls)}
- Photo URLs Present: {len(listing.image_urls)}
- Listing URL: {listing.url}

FULL DESCRIPTION:
\"\"\"
{listing.description}
\"\"\"

1. Check Stage 1 (Eligibility Gate: Location, 3+ BR, Open Kitchen w/ Island or Breakfast Bar, High Natural Light, and Learned Feedback Rules).
2. Compute Stage 2 (100-Point Ranking Rubric: kitchen_points [0-25] + light_points [0-25] + layout_points [0-25] + learned_preference_points [0-25]).
Return your assessment matching the required JSON schema.
"""
    return f"{system_block}\n{user_block}"


class GeminiListingEvaluator:
    """Evaluates and ranks candidate listings using Google Gemini API with dynamic feedback memory."""

    def __init__(self, config: AgentConfig, storage: StorageBackend) -> None:
        self.config = config
        self.storage = storage
        self._client: genai.Client | None = None
        if config.gemini_api_key:
            self._client = genai.Client(api_key=config.gemini_api_key)

    def _fetch_media_parts(self, listing: Listing) -> list[types.Part]:
        """Download up to max_images_per_listing floorplans and photos for multimodal inspection."""
        parts: list[types.Part] = []
        candidate_urls = list(listing.floorplan_urls[:2]) + list(listing.image_urls)
        candidate_urls = candidate_urls[: self.config.max_images_per_listing]

        if not candidate_urls:
            return parts

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            )
        }
        with httpx.Client(headers=headers, timeout=10.0, follow_redirects=True) as http_client:
            for url in candidate_urls:
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https"):
                    continue
                try:
                    resp = http_client.get(url)
                    if resp.status_code != 200:
                        continue
                    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if content_type not in ALLOWED_IMAGE_MIMES:
                        continue
                    if len(resp.content) > MAX_IMAGE_BYTES:
                        continue
                    parts.append(
                        types.Part.from_bytes(data=resp.content, mime_type=content_type)
                    )
                except Exception as exc:
                    logger.debug("Skipping image fetch for %s: %s", url, exc)
        return parts

    def evaluate_listing(self, listing: Listing) -> EvaluationResult:
        """
        Run the Gemini evaluation pipeline on a candidate listing, dynamically injecting
        the full user feedback log from storage.
        """
        feedback_memory_block = self.storage.format_feedback_for_prompt()
        prompt_text = build_listing_evaluation_prompt(
            listing=listing,
            feedback_memory_block=feedback_memory_block,
            min_score=self.config.min_conviction_score,
        )

        if self._client is None:
            logger.info(
                "GEMINI_API_KEY not configured; using deterministic rubric evaluator for %s",
                listing.listing_id,
            )
            return self._heuristic_fallback_evaluate(listing, feedback_memory_block)

        try:
            contents: list[types.Part | str] = [prompt_text]
            media_parts = self._fetch_media_parts(listing)
            if media_parts:
                contents.extend(media_parts)

            response = None
            for attempt in range(2):
                try:
                    response = self._client.models.generate_content(
                        model=self.config.gemini_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            response_mime_type="application/json",
                            response_schema=EvaluationResult,
                        ),
                    )
                    break
                except Exception as api_exc:
                    err_str = str(api_exc)
                    if attempt == 0 and ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str):
                        logger.info(
                            "Gemini free-tier RPM limit hit for %s; pausing 6s before retry...",
                            listing.listing_id,
                        )
                        time.sleep(6.0)
                        continue
                    raise

            if not response or not response.text:
                raise RuntimeError("Empty response from Gemini API")

            result = EvaluationResult.model_validate_json(response.text)
            return self._normalize_rubric_result(result)
        except Exception as exc:
            logger.error(
                "Gemini evaluation failed for %s (%s); running deterministic fallback.",
                listing.listing_id,
                exc,
            )
            return self._heuristic_fallback_evaluate(listing, feedback_memory_block)

    def transcribe_voice_feedback(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """
        Transcribe a Telegram voice note reason using Gemini 2.5 Flash multimodal audio support.
        100% free on the Gemini API tier.
        """
        if not self._client:
            return "[Voice note received — set GEMINI_API_KEY to enable automatic voice transcription]"

        try:
            prompt = (
                "Transcribe this user's real estate feedback voice note accurately and concisely. "
                "Return ONLY the transcribed reason explaining why they disliked or liked the apartment "
                "(e.g., 'Pass-through isn't open enough' or 'Bedrooms look too small')."
            )
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            response = self._client.models.generate_content(
                model=self.config.gemini_model,
                contents=[prompt, audio_part],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            text = (response.text or "").strip().strip('"')
            return text or "Voice feedback recorded (empty transcription)."
        except Exception as exc:
            logger.error("Failed to transcribe voice note via Gemini: %s", exc)
            return "Voice feedback received (transcription unavailable)."

    @staticmethod
    def _normalize_rubric_result(result: EvaluationResult) -> EvaluationResult:
        """Synchronize additive rubric points, legacy 0-100 sub-scores, and meets_all_criteria flag."""
        if (
            result.kitchen_points > 0
            or result.light_points > 0
            or result.layout_points > 0
            or result.learned_preference_points > 0
        ):
            computed_total = (
                result.kitchen_points
                + result.light_points
                + result.layout_points
                + result.learned_preference_points
            )
            result.overall_score = max(0, min(100, computed_total))
            if result.kitchen_layout_score == 0:
                result.kitchen_layout_score = round((result.kitchen_points / 25.0) * 100)
            if result.light_score == 0:
                result.light_score = round((result.light_points / 25.0) * 100)

        qualifies = (
            result.has_open_island_or_breakfast_bar
            and result.has_high_natural_light
            and result.is_target_ues_location
            and result.passes_learned_feedback_rules
            and len(result.disqualifiers_detected) == 0
        )
        result.meets_all_criteria = qualifies
        return result

    def _heuristic_fallback_evaluate(
        self, listing: Listing, feedback_memory_block: str
    ) -> EvaluationResult:
        """
        Deterministic rubric evaluator used when GEMINI_API_KEY is unset.
        Applies Stage-1 Eligibility Gate + Stage-2 100-Point Ranking Rubric (25/25/25/25).
        """
        text = f"{listing.display_title} {listing.floor_info} {listing.description} {' '.join(listing.amenities)}".lower()
        disqualifiers: list[str] = []
        pros: list[str] = []
        cons: list[str] = []

        # 1. Kitchen & Island Flow evaluation (0 - 25 pts)
        has_open_kitchen = any(
            kw in text
            for kw in (
                "open-concept chef's kitchen",
                "open-concept kitchen",
                "open kitchen",
                "chef's island",
                "center island",
                "waterfall island",
                "breakfast bar",
                "island kitchen",
            )
        )
        has_island_or_bar = any(
            kw in text for kw in ("island", "breakfast bar", "peninsula")
        )
        has_pass_through_only = "pass-through" in text and "island" not in text
        has_galley_or_closed = any(
            kw in text for kw in ("galley", "closed kitchen", "separate kitchen", "enclosed kitchen")
        )

        if has_galley_or_closed:
            disqualifiers.append("Closed or galley kitchen layout")
            kitchen_pts = 7
            kitchen_score = 28
            has_open_island = False
            kitchen_analysis = "Disqualified: Listing describes a closed or galley kitchen layout."
        elif has_pass_through_only:
            disqualifiers.append("Windowed/cut-out pass-through without a true open island")
            kitchen_pts = 13
            kitchen_score = 52
            has_open_island = False
            kitchen_analysis = (
                "Disqualified: Features a pass-through cut-out rather than a true open island flowing into the living room."
            )
        elif has_open_kitchen and has_island_or_bar:
            kitchen_pts = 24 if "waterfall" in text or "chef" in text else 21
            kitchen_score = round((kitchen_pts / 25.0) * 100)
            has_open_island = True
            kitchen_analysis = (
                "Open-concept kitchen with a real island/breakfast bar flowing directly into the main living and dining area."
            )
            pros.append("Open-concept kitchen with center island / breakfast bar flow")
        else:
            disqualifiers.append("Unverified open island or breakfast bar kitchen layout")
            kitchen_pts = 14
            kitchen_score = 56
            has_open_island = False
            kitchen_analysis = "Does not clearly confirm an open kitchen with a real island or breakfast bar."

        # 2. Natural Light & Exposure evaluation (0 - 25 pts)
        high_light_signals = [
            "high floor",
            "corner unit",
            "corner",
            "south exposure",
            "southern exposure",
            "sun-flooded",
            "sun-drenched",
            "panoramic",
            "penthouse",
            "oversized windows",
            "floor-to-ceiling",
        ]
        matched_light = [s for s in high_light_signals if s in text]
        floor_match = re.search(r"\b(\d{1,2})(?:th|st|nd|rd)\s+floor\b", text)
        floor_num = int(floor_match.group(1)) if floor_match else None
        if floor_num is None and listing.unit:
            unit_m = re.match(r"^#?(\d{1,2})[A-Za-z]$", listing.unit.strip())
            if unit_m:
                floor_num = int(unit_m.group(1))

        if floor_num is not None and floor_num <= 4:
            disqualifiers.append(f"Low floor ({floor_num}th floor) with limited natural light")
            light_pts = 8
            light_score = 32
            has_high_light = False
            light_analysis = f"Disqualified: Located on a lower floor ({floor_num}th floor)."
        elif len(matched_light) >= 2 or (floor_num is not None and floor_num >= 6):
            light_pts = 24 if (floor_num and floor_num >= 12) else 21
            light_score = round((light_pts / 25.0) * 100)
            has_high_light = True
            light_analysis = (
                f"Strong natural light indicators ({', '.join(matched_light[:3]) or 'high floor'}) "
                f"with {listing.floor_info or f'floor {floor_num} open exposures'}."
            )
            pros.append(f"High natural light ({listing.floor_info or 'elevated open exposures'})")
        else:
            disqualifiers.append("Insufficient proof of high natural light / high floor")
            light_pts = 14
            light_score = 56
            has_high_light = False
            light_analysis = "Modest or unverified natural light exposure."

        # 3. Family Layout & Size Points (0 - 25 pts)
        layout_pts = 18
        if listing.bedrooms >= 4:
            layout_pts += 3
        if listing.bathrooms >= 3:
            layout_pts += 2
        if listing.sqft and listing.sqft >= 2000:
            layout_pts += 2
        if "washer" in text or "w/d" in text or "laundry in unit" in text:
            layout_pts += 1
        layout_pts = min(25, layout_pts)

        # 4. Learned Preference Fit (0 - 25 pts) & Disqualifier Check
        passes_learned = True
        learned_pts = 21
        feedback_alignment = "Matches all active training feedback criteria."
        for entry in self.storage.get_all_feedback(limit=50):
            if entry.vote == "NO" and entry.reason.strip():
                reason_lower = entry.reason.lower()
                if "small" in reason_lower and ("bedroom" in reason_lower or "room" in reason_lower) and (
                    "compact bedrooms" in text or (listing.sqft is not None and listing.sqft < 1400)
                ):
                    passes_learned = False
                    learned_pts = 5
                    disqualifiers.append(f"Violates learned rule: '{entry.reason}'")
                    feedback_alignment = f"Disqualified by learned user rule: '{entry.reason}'"
                elif "pass-through" in reason_lower and "pass-through" in text:
                    passes_learned = False
                    learned_pts = 5
                    disqualifiers.append(f"Violates learned rule: '{entry.reason}'")
                    feedback_alignment = f"Disqualified by learned user rule: '{entry.reason}'"

        qualifies = (
            has_open_island
            and has_high_light
            and passes_learned
            and len(disqualifiers) == 0
        )

        total_score = max(0, min(100, kitchen_pts + light_pts + layout_pts + learned_pts))
        if not qualifies:
            cons.extend(disqualifiers)
            summary = (
                f"Did not meet all criteria ({total_score}/100): "
                f"{'; '.join(disqualifiers) or 'missing open island or high light'}."
            )
        else:
            pros.append(f"{int(listing.bedrooms)}BR family layout in {listing.neighborhood}")
            summary = (
                f"Meets 100% of criteria (Rank Score {total_score}/100 — "
                f"Kitchen {kitchen_pts}/25, Light {light_pts}/25, Layout {layout_pts}/25, Fit {learned_pts}/25)."
            )

        return EvaluationResult(
            meets_all_criteria=qualifies,
            has_open_island_or_breakfast_bar=has_open_island,
            has_high_natural_light=has_high_light,
            is_target_ues_location=True,
            passes_learned_feedback_rules=passes_learned,
            disqualifiers_detected=disqualifiers,
            overall_score=total_score,
            kitchen_points=kitchen_pts,
            light_points=light_pts,
            layout_points=layout_pts,
            learned_preference_points=learned_pts,
            light_score=light_score,
            kitchen_layout_score=kitchen_score,
            light_analysis=light_analysis,
            kitchen_analysis=kitchen_analysis,
            feedback_alignment_notes=feedback_alignment,
            pros=pros,
            cons=cons,
            reasoning_summary=summary,
        )
