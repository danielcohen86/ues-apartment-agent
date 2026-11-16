"""Curated realistic Upper East Side 3+ BR listings for dry-run verification & demo testing."""

from __future__ import annotations

from src.models import Listing
from src.scrapers.base import BaseScraper


class MockUESScraper(BaseScraper):
    """
    Provides realistic Upper East Side candidate listings spanning:
    - High-conviction matches (Score >= 85: Carnegie Hill & Lenox Hill high-floor corner units with open island kitchens)
    - Borderline / Pass-through without island (Score ~55: suppressed by Low-Noise policy)
    - Hard disqualifiers (Closed galley kitchen on 2nd floor, or non-UES neighborhood: rejected by Stage-1 Pre-Filter)
    """

    source_name = "UES-DemoFeed"

    async def fetch_listings(self) -> list[Listing]:
        return [
            Listing(
                listing_id="se:carnegie-hill-1185-park-15b",
                source="StreetEasy",
                url="https://streeteasy.com/building/1185-park-avenue-new_york/15b",
                address="1185 Park Avenue",
                unit="#15B",
                neighborhood="Carnegie Hill",
                price=13500,
                bedrooms=3.0,
                bathrooms=3.0,
                sqft=2150,
                floor_info="15th Floor, Sun-Flooded South & East Corner Exposure",
                description=(
                    "Perched high on the 15th floor in prime Carnegie Hill, Residence 15B is a sun-flooded "
                    "corner 3-bedroom, 3-bathroom family home boasting open Southern and Eastern exposures "
                    "through oversized picture windows. The heart of the home is an expansive open-concept "
                    "chef's kitchen featuring a 9-foot waterfall Calacatta marble center island with breakfast "
                    "bar seating for four that flows seamlessly into the 28-foot corner living and dining great room. "
                    "Sub-Zero, Wolf, and Miele appliances, wide-plank white oak floors, vented washer/dryer in unit, "
                    "and generous bedroom proportions with custom closets."
                ),
                amenities=[
                    "Full-Time Doorman",
                    "Elevator",
                    "In-Unit Washer/Dryer",
                    "Open Chef's Island Kitchen",
                    "Children's Playroom",
                    "Fitness Center",
                ],
                image_urls=[],
                floorplan_urls=[],
            ),
            Listing(
                listing_id="se:lenox-hill-200-e79-18a",
                source="StreetEasy",
                url="https://streeteasy.com/building/200-east-79-street-new_york/18a",
                address="200 East 79th Street",
                unit="#18A",
                neighborhood="Lenox Hill",
                price=15900,
                bedrooms=4.0,
                bathrooms=3.5,
                sqft=2480,
                floor_info="18th Floor, High-Floor Corner South/West Exposure",
                description=(
                    "High-floor 4-bedroom, 3.5-bath family residence on the 18th floor with dramatic Southern "
                    "and Western city views and abundant all-day natural light. Designed with a true open-concept "
                    "kitchen and oversized center island / breakfast bar that opens directly into the sun-drenched "
                    "corner living and dining room. No dark hallways—gracious gallery leads directly into the great room."
                ),
                amenities=[
                    "24-Hour Doorman",
                    "Concierge",
                    "Open Island Kitchen",
                    "Washer/Dryer in Unit",
                    "Roof Terrace",
                ],
                image_urls=[],
                floorplan_urls=[],
            ),
            Listing(
                listing_id="se:lenox-hill-315-e68-8d",
                source="StreetEasy",
                url="https://streeteasy.com/building/315-east-68-street-new_york/8d",
                address="315 East 68th Street",
                unit="#8D",
                neighborhood="Lenox Hill",
                price=9800,
                bedrooms=3.0,
                bathrooms=2.0,
                sqft=1550,
                floor_info="8th Floor, North Exposure",
                description=(
                    "Spacious 3-bedroom, 2-bathroom apartment in Lenox Hill on the 8th floor. Features a "
                    "renovated pass-through kitchen with a small wall cut-out looking toward the dining area, "
                    "hardwood floors, and good closet space."
                ),
                amenities=["Doorman", "Elevator", "Laundry in Building"],
                image_urls=[],
                floorplan_urls=[],
            ),
            Listing(
                listing_id="rh:yorkville-425-e86-2c",
                source="RentHop",
                url="https://www.renthop.com/listings/425-east-86th-street/2c/99887766",
                address="425 East 86th Street",
                unit="#2C",
                neighborhood="Yorkville",
                price=8500,
                bedrooms=3.0,
                bathrooms=2.0,
                sqft=1450,
                floor_info="2nd Floor, Courtyard Facing",
                description=(
                    "Pre-war 3 bedroom in Yorkville on the 2nd floor with a classic closed galley kitchen, "
                    "long entry hallway, and quiet interior courtyard views."
                ),
                amenities=["Doorman", "Elevator"],
                image_urls=[],
                floorplan_urls=[],
            ),
            Listing(
                listing_id="se:uws-155-w72-12a",
                source="StreetEasy",
                url="https://streeteasy.com/building/155-west-72-street-new_york/12a",
                address="155 West 72nd Street",
                unit="#12A",
                neighborhood="Upper West Side",
                price=12500,
                bedrooms=3.0,
                bathrooms=2.5,
                sqft=1900,
                floor_info="12th Floor, South Exposure",
                description=(
                    "Sunny 12th floor 3-bedroom on the Upper West Side with an open island kitchen."
                ),
                amenities=["Doorman", "Elevator"],
                image_urls=[],
                floorplan_urls=[],
            ),
        ]
