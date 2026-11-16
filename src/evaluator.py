"""Gemini AI evaluation pipeline with dynamic feedback memory injection & multimodal vision."""

from __future__ import annotations

import logging
import re
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


SYSTEM_PROMPT_TEMPLATE = """You are an elite, hyper-selective Upper East Side (NYC) family real estate advisor.
Your sole mission is to evaluate 3+ bedroom rental listings and filter out 95%+ of noise so the family ONLY hears about exceptional, high-conviction homes (Score >= {min_score}/100).

================================================================================
1. NON-NEGOTIABLE CORE CRITERIA
================================================================================
- LOCATION: Must be in Upper East Side, Carnegie Hill, Yorkville, or Lenox Hill.
- SIZE: 3+ real bedrooms suitable for a family.
- NATURAL LIGHT (CRITICAL):
  * Must have abundant, bright natural light exposure: high floor (typically 6th floor or higher, penthouse, or sun-drenched top floors), corner unit, or open South/East/West exposures.
  * Inspect both listing text AND attached photos/floorplans for window scale, sky visibility, and sun exposure.
- KITCHEN LAYOUT (CRITICAL):
  * MUST have a genuinely open-concept kitchen with a REAL center island or breakfast bar / peninsula that flows directly into the main living/dining room (great-room feel).
  * Inspect photos and floorplan carefully: verify the kitchen is NOT walled off and has actual island or breakfast bar counter seating/flow facing the living space.

================================================================================
2. STRICT DISQUALIFIERS (AUTOMATIC SCORE < 60 & ADD TO disqualifiers_detected)
================================================================================
- Closed galley kitchens or traditional enclosed/separate kitchens.
- Windowed pass-throughs or small wall cut-outs WITHOUT a true open island or breakfast bar flowing into the living room.
- Dark lower floors (e.g., floors 1-3, rear courtyard/airshaft views, blocked light).
- Long dark interior hallways or chopped-up layouts that isolate the kitchen from the living space.
- Any attribute that repeats a flaw the user previously rejected in the Active Training Memory below.

================================================================================
3. ACTIVE TRAINING MEMORY (DYNAMIC USER FEEDBACK LOG)
================================================================================
Below is the user's real-time feedback history from prior Telegram alerts.
Treat every user rejection reason as a HARD learned rule: if a candidate listing exhibits a similar flaw to a rejected property, penalize it heavily (score < 70) and list the conflict in `disqualifiers_detected`.

{feedback_memory_block}

================================================================================
4. SCORING CALIBRATION (LOW-NOISE / HIGH-CONVICTION POLICY)
================================================================================
- 90 to 100: Extraordinary match. Unmistakable open island/breakfast bar kitchen flowing into a sun-flooded high-floor or corner living room in UES/Carnegie Hill/Yorkville/Lenox Hill.
- 85 to 89: Strong high-conviction match. Verified open island/peninsula kitchen + bright high-floor or open exposure with zero disqualifiers.
- 70 to 84: Borderline or "maybe" (e.g., uncertain floor height, pass-through that is only semi-open, or modest light). DO NOT pass these — score them honestly below 85 so they are suppressed.
- 0 to 69: Clear rejection (galley/closed kitchen, pass-through without island, low floor, or violates user feedback memory).
"""


def build_listing_evaluation_prompt(
    listing: Listing,
    feedback_memory_block: str,
    min_score: int = 85,
) -> str:
    """Construct the full evaluation prompt combining system instructions, feedback log, and listing data."""
    system_block = SYSTEM_PROMPT_TEMPLATE.format(
        min_score=min_score,
        feedback_memory_block=feedback_memory_block,
    )
    amenities_str = ", ".join(listing.amenities) if listing.amenities else "None listed"
    sqft_str = f"{listing.sqft:,} sq ft" if listing.sqft else "Not specified"

    user_block = f"""
================================================================================
CANDIDATE LISTING TO EVALUATE
================================================================================
- Listing ID: {listing.listing_id}
- Source: {listing.source}
- Title / Address: {listing.display_title}
- Neighborhood: {listing.neighborhood}
- Monthly Rent: ${listing.price:,}/month
- Size: {listing.bedrooms} Bedrooms | {listing.bathrooms} Bathrooms | {sqft_str}
- Floor / Exposure Info: {listing.floor_info or 'Not explicitly stated in header'}
- Amenities: {amenities_str}
- Floorplan URLs Present: {len(listing.floorplan_urls)}
- Photo URLs Present: {len(listing.image_urls)}
- Listing URL: {listing.url}

FULL DESCRIPTION:
\"\"\"
{listing.description}
\"\"\"

Evaluate this candidate strictly against all criteria and the Active Training Memory.
Return your assessment matching the required JSON schema.
"""
    return f"{system_block}\n{user_block}"


class GeminiListingEvaluator:
    """Evaluates candidate listings using Google Gemini API (Free Tier) with dynamic feedback memory."""

    def __init__(self, config: AgentConfig, storage: StorageBackend) -> None:
        self.config = config
        self.storage = storage
        self._client: genai.Client | None = None
        if config.gemini_api_key:
            self._client = genai.Client(api_key=config.gemini_api_key)

    def _fetch_media_parts(self, listing: Listing) -> list[types.Part]:
        """Download up to max_images_per_listing floorplans and photos for multimodal inspection."""
        parts: list[types.Part] = []
        # Prioritize floorplans first (crucial for verifying open island vs closed galley layout), then photos
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
                "GEMINI_API_KEY not configured; using deterministic fallback evaluator for %s",
                listing.listing_id,
            )
            return self._heuristic_fallback_evaluate(listing, feedback_memory_block)

        try:
            contents: list[types.Part | str] = [prompt_text]
            media_parts = self._fetch_media_parts(listing)
            if media_parts:
                contents.extend(media_parts)

            response = self._client.models.generate_content(
                model=self.config.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=EvaluationResult,
                ),
            )
            if not response.text:
                raise RuntimeError("Empty response from Gemini API")

            result = EvaluationResult.model_validate_json(response.text)
            return self._enforce_post_evaluation_guardrails(result)
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
    def _enforce_post_evaluation_guardrails(result: EvaluationResult) -> EvaluationResult:
        """Ensure score consistency: if any disqualifier is present or island/light is False, cap overall_score."""
        if (
            not result.has_open_island_or_breakfast_bar
            or not result.has_high_natural_light
            or not result.is_target_ues_location
            or len(result.disqualifiers_detected) > 0
        ):
            result.overall_score = min(result.overall_score, 68)
        return result

    def _heuristic_fallback_evaluate(
        self, listing: Listing, feedback_memory_block: str
    ) -> EvaluationResult:
        """
        Deterministic rule-based evaluator used when GEMINI_API_KEY is unset (e.g., local dry-run/testing).
        Inspects kitchen layout signals, natural light signals, and learned user rejection reasons.
        """
        text = f"{listing.display_title} {listing.floor_info} {listing.description} {' '.join(listing.amenities)}".lower()
        disqualifiers: list[str] = []
        pros: list[str] = []
        cons: list[str] = []

        # 1. Kitchen evaluation
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
            kw in text
            for kw in (
                "island",
                "breakfast bar",
                "peninsula",
            )
        )
        has_pass_through_only = "pass-through" in text and "island" not in text
        has_galley_or_closed = any(
            kw in text for kw in ("galley", "closed kitchen", "separate kitchen", "enclosed kitchen")
        )

        if has_galley_or_closed:
            disqualifiers.append("Closed or galley kitchen layout")
            kitchen_score = 35
            has_open_island = False
            kitchen_analysis = "Disqualified: Listing describes a closed or galley kitchen layout."
        elif has_pass_through_only:
            disqualifiers.append("Windowed/cut-out pass-through without a true open island")
            kitchen_score = 55
            has_open_island = False
            kitchen_analysis = (
                "Disqualified: Features a pass-through cut-out rather than a true open island flowing into the living room."
            )
        elif has_open_kitchen and has_island_or_bar:
            kitchen_score = 94
            has_open_island = True
            kitchen_analysis = (
                "Open-concept kitchen with a real island/breakfast bar flowing directly into the main living and dining area."
            )
            pros.append("Open-concept kitchen with center island / breakfast bar flow")
        else:
            disqualifiers.append("Unverified open island or breakfast bar kitchen layout")
            kitchen_score = 62
            has_open_island = False
            kitchen_analysis = "Borderline: Does not clearly confirm an open kitchen with a real island or breakfast bar."

        # 2. Natural Light & Floor Height evaluation
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

        if floor_num is not None and floor_num <= 4:
            disqualifiers.append(f"Low floor ({floor_num}th floor) with limited natural light")
            light_score = 45
            has_high_light = False
            light_analysis = f"Disqualified: Located on a lower floor ({floor_num}th floor)."
        elif len(matched_light) >= 2 or (floor_num is not None and floor_num >= 8):
            light_score = 92
            has_high_light = True
            light_analysis = (
                f"Exceptional natural light indicators ({', '.join(matched_light[:3])}) "
                f"with {listing.floor_info or 'elevated open exposures'}."
            )
            pros.append(f"High natural light ({listing.floor_info or 'open sunny exposures'})")
        else:
            disqualifiers.append("Insufficient proof of high natural light / high floor")
            light_score = 65
            has_high_light = False
            light_analysis = "Modest or unverified natural light exposure."

        # 3. Check Active Training Memory for user rejection patterns
        feedback_alignment = "Checked against active training feedback memory."
        for entry in self.storage.get_all_feedback(limit=50):
            if entry.vote == "NO" and entry.reason.strip():
                reason_lower = entry.reason.lower()
                if "small" in reason_lower and "bedroom" in reason_lower and (
                    "compact bedrooms" in text or (listing.sqft is not None and listing.sqft < 1400)
                ):
                    disqualifiers.append(
                        f"Matches user feedback rejection: '{entry.reason}'"
                    )
                    feedback_alignment = f"Penalized due to prior user feedback: '{entry.reason}'"
                elif "pass-through" in reason_lower and "pass-through" in text:
                    disqualifiers.append(
                        f"Matches user feedback rejection: '{entry.reason}'"
                    )
                    feedback_alignment = f"Penalized due to prior user feedback: '{entry.reason}'"

        if has_open_island and has_high_light and not disqualifiers:
            overall = round((kitchen_score * 0.55) + (light_score * 0.45))
            pros.append(f"{int(listing.bedrooms)}BR family layout in {listing.neighborhood}")
            summary = (
                f"High-conviction {listing.neighborhood} {int(listing.bedrooms)}BR match ({overall}/100): "
                "verified open island/breakfast bar kitchen flowing into living space with bright high-floor exposure."
            )
        else:
            overall = min(65, round((kitchen_score + light_score) / 2))
            cons.extend(disqualifiers)
            summary = (
                f"Filtered out ({overall}/100): Did not clear the 85/100 conviction bar due to: "
                f"{'; '.join(disqualifiers) or 'insufficient light/layout conviction'}."
            )

        return EvaluationResult(
            overall_score=overall,
            light_score=light_score,
            kitchen_layout_score=kitchen_score,
            has_open_island_or_breakfast_bar=has_open_island,
            has_high_natural_light=has_high_light,
            is_target_ues_location=True,
            disqualifiers_detected=disqualifiers,
            light_analysis=light_analysis,
            kitchen_analysis=kitchen_analysis,
            feedback_alignment_notes=feedback_alignment,
            pros=pros,
            cons=cons,
            reasoning_summary=summary,
        )
