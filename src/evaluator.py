"""Gemini AI multimodal vision evaluation pipeline with Two-Stage Eligibility Gate + Calibrated 100-Point Rubric."""

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
from src.scrapers.base import infer_nyc_floor_from_unit
from src.storage import StorageBackend

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB per image safety cap
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}

DEFAULT_MODEL_FALLBACK_CHAIN = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)


SYSTEM_PROMPT_TEMPLATE = """You are an elite Upper East Side (NYC) family real estate advisor evaluating live StreetEasy 3+ bedroom listings.
Up to 4 high-resolution (960px) StreetEasy gallery photos and/or floorplan diagrams are attached for visual inspection.

Inspect EVERY attached photo and floorplan carefully alongside the structured StreetEasy metadata, then execute TWO stages:

================================================================================
STAGE 1: STRICT BINARY ELIGIBILITY GATE (Show 100% of Qualifying Apartments)
================================================================================
Determine whether the apartment satisfies ALL mandatory criteria (original + learned feedback rules):

1. LOCATION (`is_target_ues_location`):
   Must be in Upper East Side, Carnegie Hill, Yorkville, or Lenox Hill.
2. SIZE:
   Must meet active bedroom/bathroom minimums (3+ bedrooms).
3. OPEN ISLAND / BREAKFAST BAR KITCHEN (`has_open_island_or_breakfast_bar`):
   - Inspect the attached photos AND floorplan diagram (on NYC floorplans, an island is a freestanding counter rectangle in an open kitchen; a breakfast bar/peninsula is an open counter facing the living/dining room).
   - Set `true` when photos, floorplan, or text show an open-concept kitchen with a center island, waterfall island, peninsula, or breakfast bar flowing into the living/dining room.
   - STRICT DISQUALIFIERS (`false` + add to `disqualifiers_detected`): Closed kitchen, narrow walled galley kitchen, separate enclosed kitchen room, or small cut-out pass-through window without open island/counter flow.
   - NOTE ON SINGLE/LEAD PREVIEW PHOTOS: Non-featured StreetEasy search cards often preload only the hero living-room/view photo (while the kitchen is slide #4–6 on StreetEasy). If the kitchen angle is not shown in the preview photos AND zero closed-kitchen signs exist:
     * If the visible architecture/building is a modern Condo, New Development, luxury high-rise, or renovated open-concept great room, set `has_open_island_or_breakfast_bar = true` (keep `disqualifiers_detected` empty so strong matches are not blindly dropped!) and note in `kitchen_analysis` that the open great room is visible while the island itself should be verified on the StreetEasy gallery link.
     * Only set `has_open_island_or_breakfast_bar = false` when photos/floorplan/building era indicate a closed/galley/separated pre-war kitchen layout.
4. NATURAL LIGHT (`has_high_natural_light`):
   - Use BOTH visual sunlight/window evidence in photos AND NYC vertical floor conventions (e.g., `#27B` or `#27F6` = 27th floor, `#19B` = 19th floor, `#2102` = 21st floor, `PH` = Penthouse).
   - Set `true` for high floors (7th+ floor), Penthouses, sunny corner exposures, or visibly bright rooms with oversized/floor-to-ceiling windows and open sky/city/park/river views.
   - STRICT DISQUALIFIERS (`false` + add to `disqualifiers_detected`): Low floors (1st–4th floor, unless a bright multi-floor townhouse/parlor residence), dark airshaft/interior courtyard views, or visibly dim rooms.
5. LEARNED USER FEEDBACK RULES (`passes_learned_feedback_rules`):
   - Must NOT violate any of the user's learned `[NO 👎]` rejection rules or standing custom rules in Active Training Memory below.

If ALL 5 checks pass and `disqualifiers_detected` is empty, set `meets_all_criteria = true`.

================================================================================
STAGE 2: CALIBRATED 100-POINT RANKING RUBRIC (Use Full Spread — Differentiate Clearly!)
================================================================================
Score every apartment across four 0–25 point pillars (`overall_score` = sum of all 4 pillars, 0–100).
Calibrate scores honestly across the full spectrum so standout homes score 86–97, strong matches score 76–85, mixed/unconfirmed homes score 64–75, and weak/disqualified homes score <64:

1. `kitchen_points` (0 to 25 pts — Kitchen & Island Flow):
   - 23–25 pts: Showpiece open chef's kitchen visually confirmed in photos/floorplan (or text) with a large center/waterfall island, counter seating, upscale appliances/stone, and seamless great-room flow.
   - 20–22 pts: Visually confirmed open-concept kitchen with a standard island, peninsula, or breakfast bar opening directly into the living/dining area.
   - 17–19 pts: Kitchen angle not captured in the 1–3 card preview photos, but visible modern/luxury open-concept great room strongly aligns with an open kitchen layout (unconfirmed island).
   - 0–15 pts: Closed kitchen, enclosed galley kitchen, traditional separated pre-war kitchen, or small pass-through cut-out without island/bar flow (fails Stage 1).

2. `light_points` (0 to 25 pts — Natural Light & Exposure):
   - 23–25 pts: Tower/high floor (15th+ floor or Penthouse) and/or sun-drenched corner exposure with oversized or floor-to-ceiling windows and open sky/city/river/park views.
   - 19–22 pts: Elevated floor (8th–14th floor) or visibly bright, sunlit open exposure in photos with generous windows.
   - 15–18 pts: Mid-level floor (5th–7th floor) or pleasant daytime light without open views.
   - 0–14 pts: Low floor (1st–4th floor) or dark/obstructed interior exposure (fails Stage 1).

3. `layout_points` (0 to 25 pts — Family Layout & Size):
   - 23–25 pts: Exceptional family scale — 4+ BR, OR 3 BR with 3+ BA, OR 2,000+ sq ft, gracious dining/living expanse, generous bedroom dimensions, and In-Unit Washer/Dryer.
   - 19–22 pts: Strong family layout — 3 BR with 2.5+ BA, OR 1,650–1,999 sq ft, OR spacious full-service 3 BR / 2 BA with verified In-Unit Washer/Dryer and good proportions.
   - 15–18 pts: Standard 3 BR / 2 BA meeting minimum criteria with modest room sizes or unconfirmed in-unit laundry.
   - 0–14 pts: Cramped secondary bedrooms, choppy hallway-heavy layout, or undersized living area.

4. `learned_preference_points` (0 to 25 pts — Learned Preference Fit):
   - 22–25 pts: Strong match to prior `[YES 👍]` votes (or, before votes exist: prime Carnegie Hill / Lenox Hill / prime Yorkville block, luxury Condo / New Development / Full-Service Doorman building, crisp modern finishes, and strong rent-to-quality value).
   - 18–21 pts: Good UES location, full-service building, fair market value for the asking rent.
   - 13–17 pts: Less convenient fringe block, older walk-up/dated finishes, or weak value near top of budget.
   - 0–12 pts: Violates one or more learned `[NO 👎]` rejection rules or standing custom rules (fails Stage 1).

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
    amenities_str = ", ".join(listing.amenities) if listing.amenities else "None explicitly tagged"
    sqft_str = f"{listing.sqft:,} sq ft" if listing.sqft else "Not listed on card (estimate from photos/floorplan)"
    _, inferred_fl = infer_nyc_floor_from_unit(listing.unit, listing.floor_info)
    floor_display = listing.floor_info or inferred_fl or "Infer from unit number & window views in photos"

    user_block = f"""
================================================================================
CANDIDATE STREETEASY LISTING TO EVALUATE & RANK
================================================================================
- Listing ID: {listing.listing_id}
- Source: {listing.source}
- Building Address & Unit: {listing.display_title}
- Unit Designator: {listing.unit or 'Not specified'}
- UES Sub-Neighborhood: {listing.neighborhood}
- Monthly Rent: ${listing.price:,}/month
- Size: {listing.bedrooms:g} Bedrooms | {listing.bathrooms:g} Bathrooms | {sqft_str}
- Floor / Exposure Signal: {floor_display}
- Verified StreetEasy Amenities & Tags: {amenities_str}
- Attached Floorplan/Final-Slide Images: {len(listing.floorplan_urls)}
- Attached High-Res Gallery Photos: {len(listing.image_urls)}
- Canonical StreetEasy URL: {listing.url}

LISTING METADATA & DESCRIPTION:
\"\"\"
{listing.description}
\"\"\"

1. Visually inspect all attached high-res photos & floorplan slides alongside the NYC unit floor height and verified StreetEasy tags.
2. Complete Stage 1 (Eligibility Gate) and Stage 2 (Calibrated 100-Point Rubric: kitchen_points [0-25] + light_points [0-25] + layout_points [0-25] + learned_preference_points [0-25]).
Return your assessment matching the required JSON schema.
"""
    return f"{system_block}\n{user_block}"


class GeminiListingEvaluator:
    """Evaluates and ranks candidate StreetEasy listings using Gemini multimodal vision with multi-model fallback."""

    def __init__(self, config: AgentConfig, storage: StorageBackend) -> None:
        self.config = config
        self.storage = storage
        self._client: genai.Client | None = None
        if config.gemini_api_key:
            self._client = genai.Client(api_key=config.gemini_api_key)

    def _candidate_models(self) -> list[str]:
        """Return ordered list of live Gemini multimodal models with automatic fallback."""
        models: list[str] = []
        primary = (self.config.gemini_model or "").strip()
        if primary and not primary.startswith(("gemini-2.5", "gemini-2.0", "gemini-1.5")):
            models.append(primary)
        for m in DEFAULT_MODEL_FALLBACK_CHAIN:
            if m not in models:
                models.append(m)
        return models

    def _fetch_media_parts(self, listing: Listing) -> list[types.Part]:
        """Download up to max_images_per_listing high-res StreetEasy floorplans and photos for Gemini Vision."""
        parts: list[types.Part] = []
        candidate_urls = list(listing.floorplan_urls[:1]) + list(listing.image_urls)
        if len(listing.floorplan_urls) > 1:
            candidate_urls.extend(listing.floorplan_urls[1:2])
        candidate_urls = candidate_urls[: self.config.max_images_per_listing]

        if not candidate_urls:
            return parts

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "image/webp,image/jpeg,image/png,*/*;q=0.8",
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
                    if len(resp.content) > MAX_IMAGE_BYTES or len(resp.content) < 500:
                        continue
                    parts.append(
                        types.Part.from_bytes(data=resp.content, mime_type=content_type)
                    )
                except Exception as exc:
                    logger.debug("Skipping image fetch for %s: %s", url, exc)
        return parts

    def evaluate_listing(self, listing: Listing) -> EvaluationResult:
        """
        Run the Gemini multimodal vision evaluation pipeline on a candidate listing,
        dynamically injecting the user's criteria profile + historical feedback log.
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
            last_exc: Exception | None = None
            models_to_try = self._candidate_models()

            for model_idx, model_name in enumerate(models_to_try):
                try:
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            response_mime_type="application/json",
                            response_schema=EvaluationResult,
                        ),
                    )
                    if response and response.text:
                        break
                except Exception as api_exc:
                    last_exc = api_exc
                    err_str = str(api_exc)
                    is_transient_or_model_err = any(
                        code in err_str
                        for code in (
                            "404",
                            "NOT_FOUND",
                            "429",
                            "RESOURCE_EXHAUSTED",
                            "503",
                            "UNAVAILABLE",
                            "500",
                        )
                    )
                    if is_transient_or_model_err and model_idx + 1 < len(models_to_try):
                        next_model = models_to_try[model_idx + 1]
                        logger.info(
                            "Gemini model '%s' returned transient/availability error for %s; switching to '%s'...",
                            model_name,
                            listing.listing_id,
                            next_model,
                        )
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                            time.sleep(2.0)
                        continue
                    raise

            if not response or not response.text:
                raise RuntimeError(f"Empty response from Gemini API ({last_exc})")

            result = EvaluationResult.model_validate_json(response.text)
            return self._normalize_rubric_result(result)
        except Exception as exc:
            logger.error(
                "All Gemini models failed for %s (%s); running deterministic fallback.",
                listing.listing_id,
                exc,
            )
            return self._heuristic_fallback_evaluate(listing, feedback_memory_block)

    def transcribe_voice_feedback(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        """
        Transcribe a Telegram voice note reason using live Gemini Flash multimodal audio support.
        """
        if not self._client:
            return "[Voice note received — set GEMINI_API_KEY to enable automatic voice transcription]"

        prompt = (
            "Transcribe this user's real estate feedback voice note accurately and concisely. "
            "Return ONLY the transcribed reason explaining why they disliked or liked the apartment "
            "(e.g., 'Pass-through isn't open enough' or 'Bedrooms look too small')."
        )
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        for model_name in self._candidate_models():
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=[prompt, audio_part],
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                text = (response.text or "").strip().strip('"')
                if text:
                    return text
            except Exception as exc:
                logger.debug("Voice transcription attempt on %s failed: %s", model_name, exc)
                continue

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
        Deterministic rubric evaluator used when GEMINI_API_KEY is unset (e.g., offline unit tests).
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
            kitchen_pts = 24 if ("waterfall" in text or "chef" in text) else 21
            kitchen_score = round((kitchen_pts / 25.0) * 100)
            has_open_island = True
            kitchen_analysis = (
                "Open-concept kitchen with a real island/breakfast bar flowing directly into the main living and dining area."
            )
            pros.append("Open-concept kitchen with center island / breakfast bar flow")
        else:
            disqualifiers.append("Unverified open island or breakfast bar kitchen layout")
            kitchen_pts = 15
            kitchen_score = 60
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
        if floor_num is None:
            inferred_num, _ = infer_nyc_floor_from_unit(listing.unit, listing.floor_info)
            floor_num = inferred_num

        if floor_num is not None and floor_num <= 4:
            disqualifiers.append(f"Low floor ({floor_num}th floor) with limited natural light")
            light_pts = 8
            light_score = 32
            has_high_light = False
            light_analysis = f"Disqualified: Located on a lower floor ({floor_num}th floor)."
        elif len(matched_light) >= 2 or (floor_num is not None and floor_num >= 7):
            light_pts = 24 if (floor_num and floor_num >= 14) else 21
            light_score = round((light_pts / 25.0) * 100)
            has_high_light = True
            light_analysis = (
                f"Strong natural light indicators ({', '.join(matched_light[:3]) or 'elevated floor'}) "
                f"with {listing.floor_info or f'floor {floor_num} open exposures'}."
            )
            pros.append(f"High natural light ({listing.floor_info or f'Floor {floor_num} open exposures'})")
        else:
            disqualifiers.append("Insufficient proof of high natural light / high floor")
            light_pts = 15
            light_score = 60
            has_high_light = False
            light_analysis = "Modest or unverified natural light exposure."

        # 3. Family Layout & Size Points (0 - 25 pts)
        layout_pts = 18
        if listing.bedrooms >= 4:
            layout_pts += 3
        if listing.bathrooms >= 3:
            layout_pts += 2
        elif listing.bathrooms >= 2.5:
            layout_pts += 1
        if listing.sqft and listing.sqft >= 2000:
            layout_pts += 2
        elif listing.sqft and listing.sqft >= 1700:
            layout_pts += 1
        if "washer" in text or "w/d" in text or "laundry in unit" in text:
            layout_pts += 2
        layout_pts = min(25, layout_pts)

        # 4. Learned Preference Fit (0 - 25 pts) & Disqualifier Check
        passes_learned = True
        learned_pts = 21
        if listing.neighborhood in ("Carnegie Hill", "Lenox Hill"):
            learned_pts += 1
        if "condo" in text or "doorman" in text:
            learned_pts += 1
        learned_pts = min(25, learned_pts)
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
