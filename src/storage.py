"""Storage & active training memory persistence (Supabase PostgreSQL or Local JSON)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import httpx

from src.config import AgentConfig
from src.models import (
    EvaluatedListing,
    EvaluationResult,
    FeedbackEntry,
    Listing,
    PendingFeedbackPrompt,
    utc_now_iso,
)

logger = logging.getLogger(__name__)


def build_listing_snapshot(evaluated: EvaluatedListing) -> dict[str, Any]:
    """Create a compact snapshot of a listing and its evaluation for feedback memory context."""
    listing = evaluated.listing
    ev = evaluated.evaluation
    return {
        "address": listing.display_title,
        "neighborhood": listing.neighborhood,
        "price": listing.price,
        "bedrooms": listing.bedrooms,
        "bathrooms": listing.bathrooms,
        "floor_info": listing.floor_info,
        "overall_score": ev.overall_score,
        "light_score": ev.light_score,
        "kitchen_layout_score": ev.kitchen_layout_score,
        "kitchen_analysis": ev.kitchen_analysis,
        "light_analysis": ev.light_analysis,
        "url": listing.url,
    }


class StorageBackend(ABC):
    """Abstract interface for listing deduplication, score storage, and active training memory."""

    @abstractmethod
    def is_listing_seen(self, listing_id: str) -> bool:
        """Return True if this listing_id has already been evaluated."""

    @abstractmethod
    def save_evaluated_listing(self, evaluated: EvaluatedListing) -> None:
        """Persist an evaluated listing and its Gemini score breakdown."""

    @abstractmethod
    def mark_listing_alerted(
        self, listing_id: str, telegram_message_id: int | None = None
    ) -> None:
        """Mark a listing as having been sent to Telegram."""

    @abstractmethod
    def get_evaluated_listing(self, listing_id: str) -> EvaluatedListing | None:
        """Retrieve a previously evaluated listing by ID."""

    @abstractmethod
    def record_feedback(
        self,
        listing_id: str,
        vote: Literal["YES", "NO"],
        reason: str = "",
        feedback_source: Literal["inline_button", "text_reply", "voice_note"] = "inline_button",
    ) -> FeedbackEntry:
        """Insert or update user feedback ([ YES 👍 ] or [ NO 👎 ] + reason) in training memory."""

    @abstractmethod
    def get_all_feedback(self, limit: int = 100) -> list[FeedbackEntry]:
        """Return chronological feedback entries for dynamic prompt injection."""

    @abstractmethod
    def set_pending_feedback_prompt(
        self, chat_id: int, listing_id: str, prompt_message_id: int | None = None
    ) -> None:
        """Record that a Telegram chat clicked [ NO 👎 ] and is awaiting a text/voice reason."""

    @abstractmethod
    def get_pending_feedback_prompt(self, chat_id: int) -> PendingFeedbackPrompt | None:
        """Get the active pending feedback prompt for a Telegram chat, if any."""

    @abstractmethod
    def clear_pending_feedback_prompt(self, chat_id: int) -> None:
        """Clear the pending feedback prompt after the user replies with a reason."""

    @abstractmethod
    def get_stats(self) -> dict[str, int]:
        """Return summary statistics (total evaluated, total alerted, yes votes, no votes)."""

    def format_feedback_for_prompt(self, max_entries: int = 50) -> str:
        """
        Format the user's historical YES/NO feedback log for dynamic injection into Gemini prompts.
        Ensures Gemini continuously learns from every button click, text note, and voice note.
        """
        entries = self.get_all_feedback(limit=max_entries)
        if not entries:
            return (
                "No historical user feedback recorded yet. Apply the baseline strict criteria "
                "(85/100 conviction threshold, mandatory open island/breakfast bar kitchen flowing "
                "into main living area, high natural light exposure)."
            )

        no_entries = [e for e in entries if e.vote == "NO"]
        yes_entries = [e for e in entries if e.vote == "YES"]

        sections: list[str] = []

        if no_entries:
            lines = [
                "### USER REJECTIONS (NO 👎) — STRICTLY PENALIZE OR DISQUALIFY SIMILAR PROPERTIES:"
            ]
            for idx, item in enumerate(no_entries, start=1):
                snap = item.listing_snapshot or {}
                addr = snap.get("address", item.listing_id)
                neigh = snap.get("neighborhood", "UES")
                floor_info = snap.get("floor_info", "Unknown floor")
                kitchen_desc = snap.get("kitchen_analysis", "")
                reason_str = (
                    item.reason.strip()
                    if item.reason.strip()
                    else "User rejected this property as not meeting their high-conviction bar."
                )
                lines.append(
                    f"{idx}. REJECTED [{addr} ({neigh}, {floor_info})]: "
                    f"User Reason: \"{reason_str}\" | Prior Kitchen Note: {kitchen_desc}"
                )
            sections.append("\n".join(lines))

        if yes_entries:
            lines = [
                "### USER APPROVALS (YES 👍) — CALIBRATE HIGH SCORES TOWARD THESE ATTRIBUTES:"
            ]
            for idx, item in enumerate(yes_entries, start=1):
                snap = item.listing_snapshot or {}
                addr = snap.get("address", item.listing_id)
                neigh = snap.get("neighborhood", "UES")
                floor_info = snap.get("floor_info", "High floor")
                kitchen_desc = snap.get("kitchen_analysis", "")
                light_desc = snap.get("light_analysis", "")
                extra = f" | User Note: \"{item.reason}\"" if item.reason.strip() else ""
                lines.append(
                    f"{idx}. APPROVED [{addr} ({neigh}, {floor_info})]: "
                    f"Kitchen: {kitchen_desc} | Light: {light_desc}{extra}"
                )
            sections.append("\n".join(lines))

        return "\n\n".join(sections)


class LocalJSONStorage(StorageBackend):
    """Zero-cost local persistent JSON storage with atomic file writes."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._write_state(
                {
                    "listings": {},
                    "feedback": {},
                    "pending_prompts": {},
                }
            )

    def _read_state(self) -> dict[str, Any]:
        if not self.file_path.exists():
            return {"listings": {}, "feedback": {}, "pending_prompts": {}}
        try:
            raw = self.file_path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
            data.setdefault("listings", {})
            data.setdefault("feedback", {})
            data.setdefault("pending_prompts", {})
            return data
        except Exception as exc:
            logger.error("Failed reading local JSON store, initializing clean dict: %s", exc)
            return {"listings": {}, "feedback": {}, "pending_prompts": {}}

    def _write_state(self, state: dict[str, Any]) -> None:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.file_path.parent), prefix=".agent_mem_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.file_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def is_listing_seen(self, listing_id: str) -> bool:
        state = self._read_state()
        return listing_id in state["listings"]

    def save_evaluated_listing(self, evaluated: EvaluatedListing) -> None:
        state = self._read_state()
        state["listings"][evaluated.listing.listing_id] = evaluated.model_dump()
        self._write_state(state)

    def mark_listing_alerted(
        self, listing_id: str, telegram_message_id: int | None = None
    ) -> None:
        state = self._read_state()
        rec = state["listings"].get(listing_id)
        if rec:
            rec["alerted"] = True
            if telegram_message_id is not None:
                rec["telegram_message_id"] = telegram_message_id
            self._write_state(state)

    def get_evaluated_listing(self, listing_id: str) -> EvaluatedListing | None:
        state = self._read_state()
        rec = state["listings"].get(listing_id)
        if not rec:
            return None
        return EvaluatedListing.model_validate(rec)

    def record_feedback(
        self,
        listing_id: str,
        vote: Literal["YES", "NO"],
        reason: str = "",
        feedback_source: Literal["inline_button", "text_reply", "voice_note"] = "inline_button",
    ) -> FeedbackEntry:
        state = self._read_state()
        existing_raw = state["feedback"].get(listing_id)
        evaluated = self.get_evaluated_listing(listing_id)
        snapshot = build_listing_snapshot(evaluated) if evaluated else {"listing_id": listing_id}

        now = utc_now_iso()
        if existing_raw:
            entry = FeedbackEntry.model_validate(existing_raw)
            entry.vote = vote
            if reason.strip():
                entry.reason = reason.strip()
            entry.feedback_source = feedback_source
            if snapshot and not entry.listing_snapshot:
                entry.listing_snapshot = snapshot
            entry.updated_at = now
        else:
            entry = FeedbackEntry(
                id=str(uuid.uuid4()),
                listing_id=listing_id,
                vote=vote,
                reason=reason.strip(),
                feedback_source=feedback_source,
                listing_snapshot=snapshot,
                created_at=now,
                updated_at=now,
            )

        state["feedback"][listing_id] = entry.model_dump()
        self._write_state(state)
        return entry

    def get_all_feedback(self, limit: int = 100) -> list[FeedbackEntry]:
        state = self._read_state()
        items = [
            FeedbackEntry.model_validate(raw)
            for raw in state["feedback"].values()
        ]
        items.sort(key=lambda x: x.updated_at, reverse=True)
        return items[:limit]

    def set_pending_feedback_prompt(
        self, chat_id: int, listing_id: str, prompt_message_id: int | None = None
    ) -> None:
        state = self._read_state()
        prompt = PendingFeedbackPrompt(
            chat_id=chat_id,
            listing_id=listing_id,
            prompt_message_id=prompt_message_id,
        )
        state["pending_prompts"][str(chat_id)] = prompt.model_dump()
        self._write_state(state)

    def get_pending_feedback_prompt(self, chat_id: int) -> PendingFeedbackPrompt | None:
        state = self._read_state()
        raw = state["pending_prompts"].get(str(chat_id))
        if not raw:
            return None
        return PendingFeedbackPrompt.model_validate(raw)

    def clear_pending_feedback_prompt(self, chat_id: int) -> None:
        state = self._read_state()
        if str(chat_id) in state["pending_prompts"]:
            del state["pending_prompts"][str(chat_id)]
            self._write_state(state)

    def get_stats(self) -> dict[str, int]:
        state = self._read_state()
        listings = state["listings"].values()
        feedback = state["feedback"].values()
        return {
            "total_evaluated": len(listings),
            "total_alerted": sum(1 for item in listings if item.get("alerted")),
            "yes_votes": sum(1 for f in feedback if f.get("vote") == "YES"),
            "no_votes": sum(1 for f in feedback if f.get("vote") == "NO"),
            "feedback_with_reasons": sum(
                1 for f in feedback if str(f.get("reason", "")).strip()
            ),
        }


class SupabaseStorage(StorageBackend):
    """
    Zero-cost Supabase PostgreSQL storage via parameterized PostgREST HTTPS API.
    Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (see sql/schema.sql).
    """

    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required when STORAGE_BACKEND=supabase."
            )
        self.rest_url = f"{supabase_url.rstrip('/')}/rest/v1"
        self.headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self._client = httpx.Client(headers=self.headers, timeout=15.0)

    def is_listing_seen(self, listing_id: str) -> bool:
        resp = self._client.get(
            f"{self.rest_url}/listings",
            params={"listing_id": f"eq.{listing_id}", "select": "listing_id"},
        )
        resp.raise_for_status()
        return len(resp.json()) > 0

    def save_evaluated_listing(self, evaluated: EvaluatedListing) -> None:
        listing = evaluated.listing
        ev = evaluated.evaluation
        payload = {
            "listing_id": listing.listing_id,
            "source": listing.source,
            "url": listing.url,
            "address": listing.address,
            "unit": listing.unit,
            "neighborhood": listing.neighborhood,
            "price": listing.price,
            "bedrooms": listing.bedrooms,
            "bathrooms": listing.bathrooms,
            "sqft": listing.sqft,
            "floor_info": listing.floor_info,
            "description": listing.description,
            "amenities": listing.amenities,
            "image_urls": listing.image_urls,
            "floorplan_urls": listing.floorplan_urls,
            "overall_score": ev.overall_score,
            "light_score": ev.light_score,
            "kitchen_layout_score": ev.kitchen_layout_score,
            "has_open_island_or_breakfast_bar": ev.has_open_island_or_breakfast_bar,
            "has_high_natural_light": ev.has_high_natural_light,
            "disqualifiers_detected": ev.disqualifiers_detected,
            "evaluation_json": ev.model_dump(),
            "alerted": evaluated.alerted,
            "telegram_message_id": evaluated.telegram_message_id,
            "first_seen_at": listing.first_seen_at,
            "evaluated_at": evaluated.evaluated_at,
        }
        headers = {**self.headers, "Prefer": "resolution=merge-duplicates,return=representation"}
        resp = self._client.post(f"{self.rest_url}/listings", headers=headers, json=payload)
        resp.raise_for_status()

    def mark_listing_alerted(
        self, listing_id: str, telegram_message_id: int | None = None
    ) -> None:
        patch_data: dict[str, Any] = {"alerted": True}
        if telegram_message_id is not None:
            patch_data["telegram_message_id"] = telegram_message_id
        resp = self._client.patch(
            f"{self.rest_url}/listings",
            params={"listing_id": f"eq.{listing_id}"},
            json=patch_data,
        )
        resp.raise_for_status()

    def get_evaluated_listing(self, listing_id: str) -> EvaluatedListing | None:
        resp = self._client.get(
            f"{self.rest_url}/listings",
            params={"listing_id": f"eq.{listing_id}", "select": "*"},
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        row = rows[0]
        listing = Listing(
            listing_id=row["listing_id"],
            source=row["source"],
            url=row["url"],
            address=row["address"],
            unit=row.get("unit") or "",
            neighborhood=row["neighborhood"],
            price=int(row["price"]),
            bedrooms=float(row["bedrooms"]),
            bathrooms=float(row.get("bathrooms") or 1.0),
            sqft=row.get("sqft"),
            floor_info=row.get("floor_info") or "",
            description=row.get("description") or "",
            amenities=row.get("amenities") or [],
            image_urls=row.get("image_urls") or [],
            floorplan_urls=row.get("floorplan_urls") or [],
            first_seen_at=row.get("first_seen_at") or utc_now_iso(),
        )
        evaluation = EvaluationResult.model_validate(row["evaluation_json"])
        return EvaluatedListing(
            listing=listing,
            evaluation=evaluation,
            alerted=bool(row.get("alerted")),
            telegram_message_id=row.get("telegram_message_id"),
            evaluated_at=row.get("evaluated_at") or utc_now_iso(),
        )

    def record_feedback(
        self,
        listing_id: str,
        vote: Literal["YES", "NO"],
        reason: str = "",
        feedback_source: Literal["inline_button", "text_reply", "voice_note"] = "inline_button",
    ) -> FeedbackEntry:
        evaluated = self.get_evaluated_listing(listing_id)
        snapshot = build_listing_snapshot(evaluated) if evaluated else {"listing_id": listing_id}
        now = utc_now_iso()

        # Check if feedback already exists for this listing_id
        existing_resp = self._client.get(
            f"{self.rest_url}/feedback_memory",
            params={"listing_id": f"eq.{listing_id}", "select": "*"},
        )
        existing_resp.raise_for_status()
        existing_rows = existing_resp.json()

        if existing_rows:
            row = existing_rows[0]
            updated_reason = reason.strip() if reason.strip() else row.get("reason", "")
            patch_payload = {
                "vote": vote,
                "reason": updated_reason,
                "feedback_source": feedback_source,
                "updated_at": now,
            }
            resp = self._client.patch(
                f"{self.rest_url}/feedback_memory",
                params={"listing_id": f"eq.{listing_id}"},
                json=patch_payload,
            )
            resp.raise_for_status()
            updated_row = resp.json()[0]
            return FeedbackEntry(
                id=str(updated_row["id"]),
                listing_id=updated_row["listing_id"],
                vote=updated_row["vote"],
                reason=updated_row.get("reason") or "",
                feedback_source=updated_row.get("feedback_source") or "inline_button",
                listing_snapshot=updated_row.get("listing_snapshot") or snapshot,
                created_at=updated_row.get("created_at") or now,
                updated_at=updated_row.get("updated_at") or now,
            )

        insert_payload = {
            "listing_id": listing_id,
            "vote": vote,
            "reason": reason.strip(),
            "feedback_source": feedback_source,
            "listing_snapshot": snapshot,
            "created_at": now,
            "updated_at": now,
        }
        resp = self._client.post(f"{self.rest_url}/feedback_memory", json=insert_payload)
        resp.raise_for_status()
        created_row = resp.json()[0]
        return FeedbackEntry(
            id=str(created_row["id"]),
            listing_id=created_row["listing_id"],
            vote=created_row["vote"],
            reason=created_row.get("reason") or "",
            feedback_source=created_row.get("feedback_source") or "inline_button",
            listing_snapshot=created_row.get("listing_snapshot") or snapshot,
            created_at=created_row.get("created_at") or now,
            updated_at=created_row.get("updated_at") or now,
        )

    def get_all_feedback(self, limit: int = 100) -> list[FeedbackEntry]:
        resp = self._client.get(
            f"{self.rest_url}/feedback_memory",
            params={"select": "*", "order": "updated_at.desc", "limit": str(limit)},
        )
        resp.raise_for_status()
        return [
            FeedbackEntry(
                id=str(r["id"]),
                listing_id=r["listing_id"],
                vote=r["vote"],
                reason=r.get("reason") or "",
                feedback_source=r.get("feedback_source") or "inline_button",
                listing_snapshot=r.get("listing_snapshot") or {},
                created_at=r.get("created_at") or utc_now_iso(),
                updated_at=r.get("updated_at") or utc_now_iso(),
            )
            for r in resp.json()
        ]

    def set_pending_feedback_prompt(
        self, chat_id: int, listing_id: str, prompt_message_id: int | None = None
    ) -> None:
        headers = {**self.headers, "Prefer": "resolution=merge-duplicates,return=representation"}
        payload = {
            "chat_id": chat_id,
            "listing_id": listing_id,
            "prompt_message_id": prompt_message_id,
            "created_at": utc_now_iso(),
        }
        resp = self._client.post(
            f"{self.rest_url}/pending_feedback_prompts", headers=headers, json=payload
        )
        resp.raise_for_status()

    def get_pending_feedback_prompt(self, chat_id: int) -> PendingFeedbackPrompt | None:
        resp = self._client.get(
            f"{self.rest_url}/pending_feedback_prompts",
            params={"chat_id": f"eq.{chat_id}", "select": "*"},
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        r = rows[0]
        return PendingFeedbackPrompt(
            chat_id=int(r["chat_id"]),
            listing_id=r["listing_id"],
            prompt_message_id=r.get("prompt_message_id"),
            created_at=r.get("created_at") or utc_now_iso(),
        )

    def clear_pending_feedback_prompt(self, chat_id: int) -> None:
        resp = self._client.delete(
            f"{self.rest_url}/pending_feedback_prompts",
            params={"chat_id": f"eq.{chat_id}"},
        )
        resp.raise_for_status()

    def get_stats(self) -> dict[str, int]:
        listings_resp = self._client.get(
            f"{self.rest_url}/listings", params={"select": "listing_id,alerted"}
        )
        listings_resp.raise_for_status()
        listings_rows = listings_resp.json()

        fb_resp = self._client.get(
            f"{self.rest_url}/feedback_memory", params={"select": "vote,reason"}
        )
        fb_resp.raise_for_status()
        fb_rows = fb_resp.json()

        return {
            "total_evaluated": len(listings_rows),
            "total_alerted": sum(1 for r in listings_rows if r.get("alerted")),
            "yes_votes": sum(1 for r in fb_rows if r.get("vote") == "YES"),
            "no_votes": sum(1 for r in fb_rows if r.get("vote") == "NO"),
            "feedback_with_reasons": sum(
                1 for r in fb_rows if str(r.get("reason", "")).strip()
            ),
        }


def create_storage_backend(config: AgentConfig) -> StorageBackend:
    """Factory that initializes Supabase PostgreSQL or falls back to Local JSON storage."""
    if config.storage_backend == "supabase":
        if config.supabase_url and config.supabase_service_role_key:
            logger.info("Using Supabase PostgreSQL storage backend (%s)", config.supabase_url)
            return SupabaseStorage(
                supabase_url=config.supabase_url,
                supabase_key=config.supabase_service_role_key,
            )
        logger.warning(
            "STORAGE_BACKEND=supabase requested, but SUPABASE_URL/KEY missing. "
            "Falling back to LocalJSONStorage at %s",
            config.local_storage_path,
        )
    return LocalJSONStorage(config.local_storage_path)
