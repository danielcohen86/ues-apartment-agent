"""Automated tests for pre-filtering, 85/100 conviction threshold, active feedback loop, and Telegram UI."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bot import build_feedback_keyboard, format_telegram_alert_html
from src.config import AgentConfig
from src.evaluator import GeminiListingEvaluator, build_listing_evaluation_prompt
from src.filters import prefilter_listing
from src.models import EvaluatedListing, EvaluationResult, Listing
from src.runner import UESRentalAgentRunner
from src.scrapers.mock_listings import MockUESScraper
from src.storage import LocalJSONStorage


def make_test_config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        gemini_api_key="",
        gemini_model="gemini-2.5-flash",
        telegram_bot_token="",
        telegram_allowed_chat_id=111222333,
        storage_backend="local",
        local_storage_path=tmp_path / "test_memory.json",
        min_conviction_score=85,
        min_bedrooms=3.0,
        enable_demo_mode=True,
    )


def test_stage1_prefilter_enforces_location_size_and_hard_disqualifiers() -> None:
    # 1. Valid Carnegie Hill 3BR high floor
    good = Listing(
        listing_id="se:1",
        source="StreetEasy",
        url="https://streeteasy.com/rental/1",
        address="1185 Park Avenue",
        unit="#15B",
        neighborhood="Carnegie Hill",
        price=13500,
        bedrooms=3.0,
        bathrooms=3.0,
        floor_info="15th Floor, South/East Corner",
        description="Sun-flooded corner 3BR with open-concept chef's kitchen and center island.",
    )
    assert prefilter_listing(good).passed is True

    # 2. Out-of-bounds neighborhood (Upper West Side)
    uws = good.model_copy(update={"listing_id": "se:2", "neighborhood": "Upper West Side", "address": "150 W 72nd St"})
    verdict_uws = prefilter_listing(uws)
    assert verdict_uws.passed is False
    assert "Out-of-bounds neighborhood" in verdict_uws.reason

    # 3. Under 3 bedrooms
    two_bed = good.model_copy(update={"listing_id": "se:3", "bedrooms": 2.0})
    verdict_beds = prefilter_listing(two_bed)
    assert verdict_beds.passed is False
    assert "Insufficient bedrooms" in verdict_beds.reason

    # 4. Closed galley kitchen disqualifier
    galley = good.model_copy(
        update={
            "listing_id": "se:4",
            "description": "Spacious 3BR with a classic closed galley kitchen and formal dining room.",
        }
    )
    verdict_galley = prefilter_listing(galley)
    assert verdict_galley.passed is False
    assert "galley kitchen" in verdict_galley.reason.lower()

    # 5. Low floor unit (#2A) disqualifier
    low_floor = good.model_copy(update={"listing_id": "se:5", "unit": "#2A"})
    verdict_low = prefilter_listing(low_floor)
    assert verdict_low.passed is False
    assert "Low-floor unit" in verdict_low.reason


def test_conviction_threshold_85_and_mandatory_flags() -> None:
    high_match = EvaluationResult(
        overall_score=91,
        light_score=90,
        kitchen_layout_score=92,
        has_open_island_or_breakfast_bar=True,
        has_high_natural_light=True,
        is_target_ues_location=True,
        disqualifiers_detected=[],
        light_analysis="15th floor corner South/East exposure.",
        kitchen_analysis="Open chef's kitchen with 9ft waterfall island flowing into living space.",
        reasoning_summary="Exceptional match.",
    )
    assert high_match.passes_high_conviction_policy(threshold=85) is True

    # Borderline 83/100 must be suppressed
    borderline = high_match.model_copy(update={"overall_score": 83})
    assert borderline.passes_high_conviction_policy(threshold=85) is False

    # High score but missing open island must be suppressed
    no_island = high_match.model_copy(update={"has_open_island_or_breakfast_bar": False})
    assert no_island.passes_high_conviction_policy(threshold=85) is False

    # Any disqualifier present must be suppressed
    with_disq = high_match.model_copy(
        update={"disqualifiers_detected": ["Windowed pass-through without island"]}
    )
    assert with_disq.passes_high_conviction_policy(threshold=85) is False


def test_feedback_memory_persistence_and_dynamic_prompt_injection(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    storage = LocalJSONStorage(config.local_storage_path)
    evaluator = GeminiListingEvaluator(config=config, storage=storage)

    sample = Listing(
        listing_id="se:test-777",
        source="StreetEasy",
        url="https://streeteasy.com/rental/777",
        address="300 East 75th Street",
        unit="#10B",
        neighborhood="Lenox Hill",
        price=11500,
        bedrooms=3.0,
        bathrooms=2.5,
        floor_info="10th Floor",
        description="High floor 3BR with open kitchen and island.",
    )
    ev = evaluator.evaluate_listing(sample)
    storage.save_evaluated_listing(EvaluatedListing(listing=sample, evaluation=ev, alerted=True))

    # Simulate user clicking [ NO 👎 ] and replying with a text reason
    storage.record_feedback(
        listing_id="se:test-777",
        vote="NO",
        reason="Pass-through isn't open enough and bedrooms look too small",
        feedback_source="text_reply",
    )

    feedback_block = storage.format_feedback_for_prompt()
    assert "USER REJECTIONS (NO 👎)" in feedback_block
    assert "300 East 75th Street #10B" in feedback_block
    assert "Pass-through isn't open enough and bedrooms look too small" in feedback_block

    # Verify the full Gemini prompt dynamically injects this exact feedback block
    full_prompt = build_listing_evaluation_prompt(
        listing=sample,
        feedback_memory_block=feedback_block,
        min_score=85,
    )
    assert "Pass-through isn't open enough and bedrooms look too small" in full_prompt


def test_telegram_alert_html_escaping_and_inline_buttons() -> None:
    listing = Listing(
        listing_id="se:999",
        source="StreetEasy",
        url="https://streeteasy.com/rental/999",
        address="1185 Park Ave <script>alert(1)</script>",
        unit="#15B",
        neighborhood="Carnegie Hill",
        price=13500,
        bedrooms=3.0,
        bathrooms=3.0,
        sqft=2150,
        floor_info="15th Floor South & East",
        description="Open island kitchen.",
    )
    ev = EvaluationResult(
        overall_score=93,
        light_score=92,
        kitchen_layout_score=94,
        has_open_island_or_breakfast_bar=True,
        has_high_natural_light=True,
        is_target_ues_location=True,
        disqualifiers_detected=[],
        light_analysis="Bright South & East exposure.",
        kitchen_analysis="Waterfall island flowing into living space.",
        pros=["9ft island", "High floor corner"],
        cons=[],
        reasoning_summary="Top tier family layout.",
    )
    html_msg = format_telegram_alert_html(EvaluatedListing(listing=listing, evaluation=ev))
    # Verify XSS payload is safely HTML-escaped
    assert "<script>" not in html_msg
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_msg
    assert "HIGH-CONVICTION UES MATCH (93/100)" in html_msg

    # Verify the 2 inline buttons: [ YES 👍 ] and [ NO 👎 ]
    kb = build_feedback_keyboard("se:999")
    buttons = kb.inline_keyboard[0]
    assert len(buttons) == 2
    assert buttons[0].text == "YES 👍"
    assert buttons[0].callback_data == "vote|YES|se:999"
    assert buttons[1].text == "NO 👎"
    assert buttons[1].callback_data == "vote|NO|se:999"


@pytest.mark.asyncio
async def test_end_to_end_demo_scan_cycle_and_deduplication(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    storage = LocalJSONStorage(config.local_storage_path)
    runner = UESRentalAgentRunner(
        config=config,
        storage=storage,
        scrapers=[MockUESScraper()],
    )

    # First pass: 5 seed listings:
    # - 2 rejected by Stage-1 pre-filter (2nd floor closed galley in Yorkville + Upper West Side out-of-bounds)
    # - 1 suppressed at Stage-2 (Lenox Hill pass-through without island, score 55 < 85)
    # - 2 high-conviction matches alerted (Carnegie Hill 15th floor 93/100 & Lenox Hill 18th floor 93/100)
    report1 = await runner.run_scan_cycle(dry_run=True)
    assert report1.scraped_count == 5
    assert report1.prefilter_rejected_count == 2
    assert report1.gemini_evaluated_count == 3
    assert report1.below_threshold_count == 1
    assert report1.high_conviction_alerts_sent == 2

    # Second pass: all 3 evaluated listings are already seen -> 0 duplicate alerts
    report2 = await runner.run_scan_cycle(dry_run=True)
    assert report2.already_seen_count == 3
    assert report2.high_conviction_alerts_sent == 0
