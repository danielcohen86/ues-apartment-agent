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
    assert storage.get_criteria_profile().streeteasy_calibration_v2_completed is True


def test_streeteasy_jina_markdown_parser_and_sponsored_filter() -> None:
    from src.scrapers.base import infer_nyc_floor_from_unit
    from src.scrapers.streeteasy import StreetEasyUESScraper, split_streeteasy_title_address_and_unit

    assert infer_nyc_floor_from_unit("#27F6") == (27, "27th Floor")
    assert infer_nyc_floor_from_unit("#2102") == (21, "21st Floor")
    assert infer_nyc_floor_from_unit("#PHB") == (30, "Penthouse (Top Floor)")
    assert infer_nyc_floor_from_unit("#10/11J") == (11, "10th/11th Floor Duplex")
    assert split_streeteasy_title_address_and_unit(
        "340 East 64th Street #10/11J", "https://streeteasy.com/building/st-tropez/1011j"
    ) == ("340 East 64th Street", "#10/11J")

    sample_md = """
# 60 All Upper East Side, Manhattan NY Apartments for Rent under $25,000

*   
Condo in Lenox Hill

[400 East 67th Street #27B](https://streeteasy.com/building/the-laurel/27b?featured=1)  Save Listing actions    $20,500 base rent  **$19,200** net effective base rent

1 month free·12-month lease

    *   3 beds
    *   3.5 baths
    *   1,964 ft²

Listing by Douglas Elliman

[![Image 6: 400 East 67th Street image 20 of 20](https://photos.zillowstatic.com/fp/aaaa1111bbbb2222cccc3333dddd4444-p_e.webp)![Image 7: 400 East 67th Street image 1 of 20](https://photos.zillowstatic.com/fp/11112222333344445555666677778888-p_e.webp)![Image 8: 400 East 67th Street image 2 of 20](https://photos.zillowstatic.com/fp/99998888777766665555444433332222-p_e.webp)previous next 3D tour](https://streeteasy.com/building/the-laurel/27b?featured=1)Image 1 of 20

*   
Rental unit in East Harlem · Sponsored

[205 East 110th Street #9A](https://streeteasy.com/building/205-east-110th-street/9a?infeed=1)  Save Listing actions    $8,900 base rent

    *   3 beds
    *   2 baths
    *   1,200 ft²
"""
    scraper = StreetEasyUESScraper(min_bedrooms=3.0, min_bathrooms=2.0, max_monthly_rent=25000)
    listings, total_expected = scraper.parse_jina_markdown(sample_md)

    assert total_expected == 60
    assert len(listings) == 1
    item = listings[0]
    assert item.source == "StreetEasy"
    assert item.listing_id == "se:the-laurel-27b"
    assert item.url == "https://streeteasy.com/building/the-laurel/27b"
    assert item.address == "400 East 67th Street"
    assert item.unit == "#27B"
    assert item.neighborhood == "Lenox Hill"
    assert item.price == 20500
    assert item.bedrooms == 3.0
    assert item.bathrooms == 3.5
    assert item.sqft == 1964
    assert item.floor_info == "27th Floor"
    assert item.floorplan_urls == [
        "https://photos.zillowstatic.com/fp/aaaa1111bbbb2222cccc3333dddd4444-cc_ft_960.webp"
    ]
    assert len(item.image_urls) == 2
    assert all(u.endswith("-cc_ft_960.webp") for u in item.image_urls)


@pytest.mark.asyncio
async def test_streeteasy_only_runner_purges_legacy_craigslist_and_upgrades_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.scrapers.streeteasy import StreetEasyUESScraper

    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("ENABLE_DEMO_MODE", "false")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", "test_prod.json")
    env_cfg = AgentConfig.from_env()
    assert env_cfg.gemini_model == "gemini-3.6-flash"

    prod_cfg = AgentConfig(
        gemini_api_key="",
        gemini_model="gemini-3.6-flash",
        storage_backend="local",
        local_storage_path=tmp_path / "purge_test.json",
        enable_demo_mode=False,
    )
    storage = LocalJSONStorage(prod_cfg.local_storage_path)
    prod_runner = UESRentalAgentRunner(config=prod_cfg, storage=storage)
    assert len(prod_runner.scrapers) == 1
    assert isinstance(prod_runner.scrapers[0], StreetEasyUESScraper)

    # Seed a legacy Craigslist row from old v1 calibration
    legacy_cl = Listing(
        listing_id="cl:legacy-1",
        source="Craigslist-NYC",
        url="https://newyork.craigslist.org/mnh/apa/d/1.html",
        address="300 East 85th Street",
        unit="#4A",
        neighborhood="Yorkville",
        price=9500,
        bedrooms=3.0,
        bathrooms=2.0,
        description="Craigslist post",
    )
    storage.save_evaluated_listing(
        EvaluatedListing(
            listing=legacy_cl,
            evaluation=prod_runner.evaluator.evaluate_listing(legacy_cl),
            alerted=True,
        )
    )
    assert storage.is_listing_seen("cl:legacy-1") is True
    assert storage.get_top_evaluated_listings(limit=10) == []
    purged = storage.purge_non_streeteasy_listings()
    assert purged == 1
    assert storage.is_listing_seen("cl:legacy-1") is False

    # Seed synthetic /test_alert mock row with source='StreetEasy' and broken mock URL
    fake_mock = Listing(
        listing_id="se:carnegie-hill-1185-park-15b",
        source="StreetEasy",
        url="https://streeteasy.com/building/1185-park-avenue-new_york/15b",
        address="1185 Park Avenue",
        unit="#15B",
        neighborhood="Carnegie Hill",
        price=13500,
        bedrooms=3.0,
        bathrooms=3.0,
        description="Synthetic demo card",
    )
    storage.save_evaluated_listing(
        EvaluatedListing(
            listing=fake_mock,
            evaluation=prod_runner.evaluator.evaluate_listing(fake_mock),
            alerted=True,
        )
    )
    assert storage.get_top_evaluated_listings(limit=10) == []
    assert storage.purge_non_streeteasy_listings() == 1


def test_markdown_new_format_and_detail_page_link_verification() -> None:
    from src.scrapers.streeteasy import StreetEasyUESScraper, parse_streeteasy_detail_markdown

    md_new_sample = """
# 61 All Upper East Side, Manhattan NY Apartments for Rent under $25,000

- New development

  Rental unit in Astoria [11-14 35th Avenue #918](https://streeteasy.com/building/nova-astoria/918?infeed=1)SaveListing actions$7,800base rent
  - 3 beds
  - 2.5 baths
  - \\- ft²
  Listing by EXR [Sponsored](https://streeteasy.com/building/nova-astoria/918?infeed=1)Image 1 of 32
- Rental unit in Lenox Hill [356 East 78th Street #26B](https://streeteasy.com/building/maison-78/26b)SaveListing actions$22,400base rent
  - 3 beds
  - 3 baths
  - 1,850 ft²
  Listing by Greystar [![356 East 78th Street image 1 of 25](https://photos.zillowstatic.com/fp/abcd1234efgh5678ijkl9012mnop3456-p_e.webp)](https://streeteasy.com/building/maison-78/26b)Image 1 of 25
"""
    scraper = StreetEasyUESScraper()
    items, tot = scraper.parse_jina_markdown(md_new_sample)
    assert tot == 61
    # Astoria ?infeed=1 Sponsored card is dropped, and Sponsored badge does NOT bleed into the next Lenox Hill card
    assert len(items) == 1
    assert items[0].address == "356 East 78th Street"
    assert items[0].unit == "#26B"
    assert items[0].floor_info == "26th Floor"
    assert len(items[0].image_urls) == 1

    # 404 broken page must be flagged valid=False
    broken_404 = "Title: Converted Content\n\nHmmm...\nWe can't find that page\n" + ("x" * 1500)
    res_404 = parse_streeteasy_detail_markdown(broken_404)
    assert res_404["reachable"] is True
    assert res_404["valid"] is False

    # Valid detail page extracts floorplan, photos, About text, and Home features
    valid_detail = (
        "400 East 67th Street #27B\n1,964 ft²\n3 beds\n3.5 baths\n"
        "![photo 1](https://photos.zillowstatic.com/fp/11112222333344445555666677778888-se_large_800_400.webp)\n"
        "![floor plan 1](https://photos.zillowstatic.com/fp/ffff0000eeee1111dddd2222cccc3333-se_extra_large_1500_800.webp)\n"
        "## About\nSun-drenched corner 3BR condo with open chef's island kitchen.\n"
        "## Home features\n- Central air\n- Washer/dryer\n"
        "## Building amenities\n- Doorman\n- Children's playroom\n"
        "## Property history\n| 7/11/2019 | $16,000 | No longer available |\n"
        + (" " * 1200)
    )
    res_ok = parse_streeteasy_detail_markdown(valid_detail)
    assert res_ok["valid"] is True
    assert res_ok["sqft"] == 1964
    assert len(res_ok["floorplan_urls"]) == 1
    assert len(res_ok["photo_urls"]) == 1
    assert "open chef's island kitchen" in res_ok["about_text"]
    assert "Washer/dryer" in res_ok["features"]


def test_microlink_dom_search_and_detail_parsers() -> None:
    from src.scrapers.streeteasy import (
        StreetEasyUESScraper,
        parse_microlink_detail_html,
        split_streeteasy_title_address_and_unit,
    )

    assert split_streeteasy_title_address_and_unit(
        "127 East 69th Street RESIDENCE-4",
        "https://streeteasy.com/building/127-east-69-street-new_york/residence4",
    ) == ("127 East 69th Street", "#4")

    search_main_html = """
    <main>
      <h1>62 All Upper East Side, Manhattan NY Apartments for Rent under $25,000</h1>
      <ul>
        <li>
          <span>Mixed-use Building in Lenox Hill</span>
          <a href="https://streeteasy.com/building/maison-78/26b?featured=1">356 East 78th Street #26B</a>
          <span>$22,400 base rent</span>
          <ul><li>3 beds</li><li>3 baths</li><li>1,850 ft²</li></ul>
          <a href="https://streeteasy.com/building/maison-78/26b?featured=1">
            <img alt="356 East 78th Street image 1 of 25" src="https://photos.zillowstatic.com/fp/26a488872e028c7dc9004b338baaff16-p_e.webp" />
          </a>
        </li>
        <li>
          <span>Rental unit in Astoria</span>
          <a href="https://streeteasy.com/building/nova-astoria/918?infeed=1">11-14 35th Avenue #918</a>
          <span>$7,800 base rent • 3 beds • 2.5 baths • Sponsored</span>
        </li>
      </ul>
    </main>
    """
    scraper = StreetEasyUESScraper()
    items, tot = scraper.parse_microlink_search_html(search_main_html)
    assert tot == 62
    assert len(items) == 1
    assert items[0].listing_id == "se:maison-78-26b"
    assert items[0].address == "356 East 78th Street"
    assert items[0].unit == "#26B"
    assert items[0].price == 22400
    assert items[0].sqft == 1850
    assert items[0].image_urls == [
        "https://photos.zillowstatic.com/fp/26a488872e028c7dc9004b338baaff16-cc_ft_960.webp"
    ]

    detail_html = (
        "<main><div>400 East 67th Street #27B • 1,964 ft²</div>"
        "<img alt='photo 1' src='https://photos.zillowstatic.com/fp/29545326f53a92eb25c837bee017304e-se_large_800_400.webp'/>"
        "<img alt='floor plan 1' src='https://photos.zillowstatic.com/fp/c6468c60efdd4d1a59dda9451391e375-se_extra_large_1500_800.webp'/>"
        "<section><h2>About</h2><p>Corner 3BR in Lenox Hill with open island kitchen.</p></section>"
        "<section><h2>Home features</h2><ul><li>Washer/dryer</li><li>Central air</li></ul></section>"
        + (" " * 800)
        + "</main>"
    )
    det = parse_microlink_detail_html(detail_html)
    assert det["valid"] is True
    assert det["sqft"] == 1964
    assert len(det["floorplan_urls"]) == 1
    assert len(det["photo_urls"]) == 1
    assert "Washer/dryer" in det["features"]




