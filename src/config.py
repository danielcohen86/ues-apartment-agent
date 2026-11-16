"""Configuration loader and security validation for the UES Rental Agent."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_safe_local_path(raw_path: str) -> Path:
    """Resolve local storage path and enforce that it stays inside PROJECT_ROOT/data."""
    sandbox_dir = (PROJECT_ROOT / "data").resolve()
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    # Strip any directory traversal components by using only the basename inside data/
    safe_filename = os.path.basename(raw_path.strip() or "agent_memory.json")
    if not safe_filename.endswith(".json"):
        safe_filename += ".json"
    resolved = (sandbox_dir / safe_filename).resolve()
    if not str(resolved).startswith(str(sandbox_dir) + os.sep):
        raise ValueError(f"Unsafe storage path rejected: {raw_path}")
    return resolved


@dataclass(frozen=True)
class AgentConfig:
    """Typed runtime configuration loaded strictly from environment variables."""

    # AI Engine (Gemini Free Tier)
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # Notification UI (Telegram Bot API)
    telegram_bot_token: str = ""
    telegram_allowed_chat_id: int | None = None

    # Storage & Memory ("local" or "supabase")
    storage_backend: str = "local"
    local_storage_path: Path = field(
        default_factory=lambda: _resolve_safe_local_path("agent_memory.json")
    )
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # Strict Match & Conviction Thresholds
    min_conviction_score: int = 85
    min_bedrooms: float = 3.0
    max_monthly_rent: int | None = None
    scan_interval_minutes: int = 30
    max_images_per_listing: int = 4
    enable_demo_mode: bool = False

    target_neighborhoods: Tuple[str, ...] = (
        "Upper East Side",
        "Carnegie Hill",
        "Yorkville",
        "Lenox Hill",
    )

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables without hardcoded secrets."""
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_api_key == "your_gemini_api_key_here":
            gemini_api_key = ""

        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if telegram_bot_token == "your_telegram_bot_token_here":
            telegram_bot_token = ""

        raw_chat_id = os.getenv("TELEGRAM_ALLOWED_CHAT_ID", "").strip()
        allowed_chat_id: int | None = None
        if raw_chat_id and raw_chat_id != "123456789":
            try:
                allowed_chat_id = int(raw_chat_id)
            except ValueError:
                logging.warning(
                    "Invalid TELEGRAM_ALLOWED_CHAT_ID format; must be a numeric integer."
                )

        storage_backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
        if storage_backend not in ("local", "supabase"):
            raise ValueError(
                f"Invalid STORAGE_BACKEND '{storage_backend}'. Must be 'local' or 'supabase'."
            )

        raw_local_path = os.getenv("LOCAL_STORAGE_PATH", "data/agent_memory.json")
        local_storage_path = _resolve_safe_local_path(raw_local_path)

        supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        if supabase_url == "https://your-project-id.supabase.co":
            supabase_url = ""

        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if supabase_key == "your_supabase_service_role_or_anon_key_here":
            supabase_key = ""

        # Enforce minimum 85/100 conviction policy (can be raised even higher via env)
        try:
            min_score = max(85, int(os.getenv("MIN_CONVICTION_SCORE", "85")))
        except ValueError:
            min_score = 85

        try:
            min_beds = max(3.0, float(os.getenv("MIN_BEDROOMS", "3")))
        except ValueError:
            min_beds = 3.0

        max_rent_raw = os.getenv("MAX_MONTHLY_RENT", "").strip()
        max_rent: int | None = None
        if max_rent_raw:
            try:
                parsed_rent = int(max_rent_raw)
                if parsed_rent > 0:
                    max_rent = parsed_rent
            except ValueError:
                pass

        try:
            scan_interval = max(5, int(os.getenv("SCAN_INTERVAL_MINUTES", "30")))
        except ValueError:
            scan_interval = 30

        try:
            max_images = max(1, min(8, int(os.getenv("MAX_IMAGES_PER_LISTING", "4"))))
        except ValueError:
            max_images = 4

        enable_demo = os.getenv("ENABLE_DEMO_MODE", "false").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        raw_neighborhoods = os.getenv(
            "TARGET_NEIGHBORHOODS",
            "Upper East Side,Carnegie Hill,Yorkville,Lenox Hill",
        )
        neighborhoods = tuple(
            n.strip() for n in raw_neighborhoods.split(",") if n.strip()
        )

        return cls(
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            telegram_bot_token=telegram_bot_token,
            telegram_allowed_chat_id=allowed_chat_id,
            storage_backend=storage_backend,
            local_storage_path=local_storage_path,
            supabase_url=supabase_url,
            supabase_service_role_key=supabase_key,
            min_conviction_score=min_score,
            min_bedrooms=min_beds,
            max_monthly_rent=max_rent,
            scan_interval_minutes=scan_interval,
            max_images_per_listing=max_images,
            enable_demo_mode=enable_demo,
            target_neighborhoods=neighborhoods,
        )
