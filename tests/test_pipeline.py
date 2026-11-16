"""Automated tests for pre-filtering, qualifying criteria gate, 100-pt ranking rubric, /criteria editor, and /actions."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bot import (
    TelegramAgentBot,
    build_actions_menu_keyboard,
    build_criteria_editor_keyboard,
    build_feedback_keyboard,
    format_telegram_alert_html,
)
from src.config import AgentConfig
from src.evaluator import GeminiListingEvaluator, build_listing_evaluation_prompt
from src.filters import prefilter_listing
from src.models import EvaluatedListing, EvaluationResult, Listing, SearchCriteriaProfile
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

    # Out-of-bounds neighborhood (Upper West Side)
    uws = good.model_copy(update={"listing_id": "se:2", "neighborhood": "Upper West Side", "address": "150 W 72nd St"})
    verdict_uws = prefilter_listing(uws)
    assert verdict_uws.passed is False
    assert "Out-of-bounds neighborhood" in verdict_uws.reason

    # Under 3 bedrooms
    two_bed = good.model_copy(update={"listing_id": "se:3", "bedrooms": 2.0})
    verdict_beds = prefilter_listing(two_bed)
    assert verdict_beds.passed is False
    assert "Insufficient bedrooms" in verdict_beds.reason

    # Closed galley kitchen disqualifier
    galley = good.model_copy(
        update={
            "listing_id": "se:4",
            "description": "Spacious 3BR with a classic closed galley kitchen and formal dining room.",
        }
    )
    verdict_galley = prefilter_listing(galley)
    assert verdict_galley.passed is False
    assert "galley kitchen" in verdict_galley.reason.lower()

    # Low floor unit (#2A) disqualifier
    low_floor = good.model_copy(update={"listing_id": "se:5", "unit": "#2A"})
    verdict_low = prefilter_listing(low_floor)
    assert verdict_low.passed is False
    assert "Low-floor unit" in verdict_low.reason


def test_qualifying_criteria_gate_and_100_point_ranking_rubric() -> None:
    # A qualifying apartment scoring 79/100 (e.g. 20/25 kitchen + 20/25 light + 20/25 layout + 19/25 fit)
    # still passes passes_qualifying_criteria() so 100% of qualifying matches are shown and ranked!
    qualifying_79 = EvaluationResult(
        meets_all_criteria=True,
        overall_score=79,
        kitchen_points=20,
        light_points=20,
        layout_points=20,
        learned_preference_points=19,
        has_open_island_or_breakfast_bar=True,
        has_high_natural_light=True,
        is_target_ues_location=True,
        passes_learned_feedback_rules=True,
        disqualifiers_detected=[],
        light_analysis="9th floor open East exposure.",
        kitchen_analysis="Open kitchen with breakfast bar flowing into living room.",
        reasoning_summary="Meets 100% of criteria (Rank Score 79/100).",
    )
    assert qualifying_79.passes_qualifying_criteria() is True

    # Legacy stored >25 rubric values are safely clamped to 25 for backwards compatibility
    legacy_row = qualifying_79.model_dump()
    legacy_row["kitchen_points"] = 33
    legacy_row["light_points"] = 33
    parsed_legacy = EvaluationResult.model_validate(legacy_row)
    assert parsed_legacy.kitchen_points == 25
    assert parsed_legacy.light_points == 25

    # Missing open island must fail qualifying criteria
    no_island = qualifying_79.model_copy(update={"has_open_island_or_breakfast_bar": False})
    assert no_island.passes_qualifying_criteria() is False

    # Violating a learned user feedback rule must fail qualifying criteria
    violates_learned = qualifying_79.model_copy(
        update={
            "passes_learned_feedback_rules": False,
            "disqualifiers_detected": ["Violates learned rule: Bedrooms too small"],
        }
    )
    assert violates_learned.passes_qualifying_criteria() is False


def test_dynamic_criteria_profile_and_actions_menu(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    storage = LocalJSONStorage(config.local_storage_path)
    evaluator = GeminiListingEvaluator(config=config, storage=storage)
    bot = TelegramAgentBot(config=config, storage=storage, evaluator=evaluator)

    # Check default criteria profile
    profile = storage.get_criteria_profile()
    assert profile.min_bedrooms == 3.0
    assert profile.min_bathrooms == 2.0
    assert profile.max_monthly_rent == 25000

    # Update criteria profile (e.g. min_bathrooms=2.5, max_monthly_rent=22000, custom rule)
    profile.min_bathrooms = 2.5
    profile.max_monthly_rent = 22000
    profile.custom_rules.append("Must have in-unit washer/dryer")
    storage.save_criteria_profile(profile)

    reloaded = storage.get_criteria_profile()
    assert reloaded.min_bathrooms == 2.5
    assert reloaded.max_monthly_rent == 22000
    assert "Must have in-unit washer/dryer" in reloaded.custom_rules

    # Verify /criteria HTML and editor buttons
    criteria_html, _ = bot._format_criteria_html()
    assert "2.5+ BA" in criteria_html
    assert "$22,000/month" in criteria_html
    assert "Must have in-unit washer/dryer" in criteria_html
    assert "0–25 pts" in criteria_html
    assert "Learned Preference Fit" in criteria_html

    crit_kb = build_criteria_editor_keyboard(reloaded)
    assert len(crit_kb.inline_keyboard) == 6

    # Verify /actions menu
    actions_html = bot._format_actions_text()
    assert "/criteria" in actions_html
    assert "/scan" in actions_html
    assert "/top" in actions_html
    actions_kb = build_actions_menu_keyboard()
    assert len(actions_kb.inline_keyboard) == 3


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
    assert "CURRENT ACTIVE SEARCH CRITERIA" in feedback_block
    assert "USER REJECTIONS (NO 👎)" in feedback_block
    assert "300 East 75th Street #10B" in feedback_block
    assert "Pass-through isn't open enough and bedrooms look too small" in feedback_block

    full_prompt = build_listing_evaluation_prompt(
        listing=sample,
        feedback_memory_block=feedback_block,
    )
    assert "Pass-through isn't open enough and bedrooms look too small" in full_prompt


def test_telegram_alert_html_escaping_and_ranked_header() -> None:
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
        meets_all_criteria=True,
        overall_score=93,
        kitchen_points=24,
        light_points=24,
        layout_points=23,
        learned_preference_points=22,
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
    html_msg = format_telegram_alert_html(
        EvaluatedListing(listing=listing, evaluation=ev),
        rank_index=1,
        total_qualified=3,
    )
    assert "<script>" not in html_msg
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_msg
    assert "RANK #1 OF 3 QUALIFYING MATCHES — 93/100 PTS" in html_msg
    assert "Kitchen &amp; Island Flow: <b>24/25</b>" in html_msg
    assert "Natural Light &amp; Exposure: <b>24/25</b>" in html_msg
    assert "Family Layout &amp; Size: <b>23/25</b>" in html_msg
    assert "Learned Preference Fit: <b>22/25</b>" in html_msg

    kb = build_feedback_keyboard("se:999")
    buttons = kb.inline_keyboard[0]
    assert len(buttons) == 2
    assert buttons[0].text == "YES 👍"
    assert buttons[1].text == "NO 👎"


@pytest.mark.asyncio
async def test_end_to_end_demo_scan_cycle_and_ranked_delivery(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    storage = LocalJSONStorage(config.local_storage_path)
    runner = UESRentalAgentRunner(
        config=config,
        storage=storage,
        scrapers=[MockUESScraper()],
    )

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


@pytest.mark.asyncio
async def test_top_10_calibration_batch_then_new_listings_only(tmp_path: Path) -> None:
    from src.scrapers.base import BaseScraper, canonicalize_nyc_address_key

    # Verify NYC street & unit canonicalizer collapses StreetEasy / web formatting variations
    assert canonicalize_nyc_address_key("200 East 79th Street", "#14B") == canonicalize_nyc_address_key(
        "200 E. 79 St., Manhattan", "Apt 14B"
    )
    assert canonicalize_nyc_address_key("1185 Park Avenue", "Residence 15B") == canonicalize_nyc_address_key(
        "1185 Park Ave, New York, NY 10128", "#15B"
    )

    class StaticInventoryScraper(BaseScraper):
        source_name = "Test-StreetEasy"

        def __init__(self, items: list[Listing]) -> None:
            super().__init__()
            self.items = items

        async def fetch_listings(self) -> list[Listing]:
            return list(self.items)

    # Create 15 currently active UES 3BR listings
    active_inventory = [
        Listing(
            listing_id=f"se:active-{i}",
            source="StreetEasy",
            url=f"https://streeteasy.com/rental/{1000 + i}",
            address=f"{100 + i} East 82nd Street",
            unit=f"#{10 + (i % 8)}A",
            neighborhood="Upper East Side",
            price=12000 + i * 250,
            bedrooms=3.0,
            bathrooms=2.5,
            sqft=1850 + i * 20,
            floor_info="High Floor South Exposure",
            description="Sun-flooded corner 3BR with open-concept chef's kitchen and waterfall island.",
        )
        for i in range(15)
    ]

    config = make_test_config(tmp_path)
    storage = LocalJSONStorage(config.local_storage_path)
    scraper = StaticInventoryScraper(active_inventory)
    runner = UESRentalAgentRunner(config=config, storage=storage, scrapers=[scraper])

    assert storage.get_criteria_profile().calibration_completed is False

    # 1. Calibration sends ONLY the Top 10 best active listings (not all 15) and baselines all 15 as seen
    cal_report = await runner.run_calibration_batch(limit=10, dry_run=True)
    assert cal_report["scraped_count"] == 15
    assert cal_report["calibration_sent"] == 10
    assert cal_report["baselined_total"] == 15
    assert storage.get_criteria_profile().calibration_completed is True

    # 2. Next scan has the 15 existing apartments (with slightly altered address formatting on #0) + 1 brand-new listing
    reformatted_existing_0 = active_inventory[0].model_copy(
        update={
            "listing_id": "gsearch:variant-id",
            "address": "100 E. 82 St, Manhattan",
            "unit": "Apt 10A",
        }
    )
    brand_new_listing = Listing(
        listing_id="se:brand-new-999",
        source="StreetEasy",
        url="https://streeteasy.com/rental/9999",
        address="1150 Fifth Avenue",
        unit="#16C",
        neighborhood="Carnegie Hill",
        price=16500,
        bedrooms=3.0,
        bathrooms=3.0,
        sqft=2200,
        floor_info="16th Floor Corner",
        description="Brand new to market! Sun-drenched 3BR with open kitchen and center island.",
    )
    scraper.items = [reformatted_existing_0, *active_inventory[1:], brand_new_listing]

    scan_report = await runner.run_scan_cycle(dry_run=True)
    assert scan_report.already_seen_count == 15
    assert scan_report.gemini_evaluated_count == 1
    assert scan_report.high_conviction_alerts_sent == 1

