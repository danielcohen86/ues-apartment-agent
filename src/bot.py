"""Telegram Bot UI: Ranked Alerts, Active Training Loop, /criteria Editor, and /actions Menu."""

from __future__ import annotations

import html
import logging
import re
import time
from typing import Awaitable, Callable

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.config import AgentConfig
from src.evaluator import GeminiListingEvaluator
from src.models import EvaluatedListing, SearchCriteriaProfile
from src.scrapers.mock_listings import MockUESScraper
from src.storage import StorageBackend

logger = logging.getLogger(__name__)

QUICK_NO_REASONS: dict[str, str] = {
    "rooms_small": "Bedrooms / living rooms look too small for a family",
    "kitchen_closed": "Kitchen / pass-through isn't open enough or lacks a real island",
    "location_weak": "Location or specific block is not ideal",
    "light_weak": "Natural light or exposure doesn't look bright enough",
    "layout_choppy": "Choppy layout or too much wasted hallway space",
}

RENT_CYCLE_STEPS = [15000, 18000, 20000, 25000, 30000, 40000]
BEDS_CYCLE_STEPS = [3.0, 3.5, 4.0]
BATHS_CYCLE_STEPS = [1.5, 2.0, 2.5, 3.0]
SQFT_CYCLE_STEPS: list[int | None] = [None, 1500, 1800, 2000, 2400]


def build_feedback_keyboard(
    listing_id: str, selected_vote: str | None = None
) -> InlineKeyboardMarkup:
    """
    Build the 2 inline feedback buttons required on every Telegram alert:
    [ YES 👍 ] and [ NO 👎 ]
    """
    if selected_vote == "YES":
        yes_label = "✅ YES 👍 (Saved)"
        no_label = "NO 👎"
    elif selected_vote == "NO":
        yes_label = "YES 👍"
        no_label = "❌ NO 👎 (Tell me why)"
    else:
        yes_label = "YES 👍"
        no_label = "NO 👎"

    keyboard = [
        [
            InlineKeyboardButton(yes_label, callback_data=f"vote|YES|{listing_id}"),
            InlineKeyboardButton(no_label, callback_data=f"vote|NO|{listing_id}"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def build_quick_reasons_keyboard(listing_id: str) -> InlineKeyboardMarkup:
    """Build 1-tap quick rejection reason buttons shown after clicking [ NO 👎 ]."""
    keyboard = [
        [
            InlineKeyboardButton(
                "🛏️ Rooms too small", callback_data=f"qreason|rooms_small|{listing_id}"
            ),
            InlineKeyboardButton(
                "🍳 Kitchen not open", callback_data=f"qreason|kitchen_closed|{listing_id}"
            ),
        ],
        [
            InlineKeyboardButton(
                "📍 Location not great", callback_data=f"qreason|location_weak|{listing_id}"
            ),
            InlineKeyboardButton(
                "🌑 Not enough light", callback_data=f"qreason|light_weak|{listing_id}"
            ),
        ],
        [
            InlineKeyboardButton(
                "📐 Choppy layout / hallway", callback_data=f"qreason|layout_choppy|{listing_id}"
            ),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def build_actions_menu_keyboard() -> InlineKeyboardMarkup:
    """Build 1-tap interactive buttons for /actions capability menu."""
    keyboard = [
        [
            InlineKeyboardButton("🔍 Run Scan Now", callback_data="act|scan"),
            InlineKeyboardButton("🏆 Top 5 Ranked", callback_data="act|top"),
        ],
        [
            InlineKeyboardButton("⚙️ View / Edit Criteria", callback_data="act|criteria"),
            InlineKeyboardButton("🧠 Learned Memory", callback_data="act|memory"),
        ],
        [
            InlineKeyboardButton("📊 Agent Status", callback_data="act|status"),
            InlineKeyboardButton("🧪 Send Test Alert", callback_data="act|test_alert"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def build_criteria_editor_keyboard(profile: SearchCriteriaProfile) -> InlineKeyboardMarkup:
    """Build 1-tap buttons allowing the user to adjust search criteria directly on their phone."""
    sqft_label = f"{profile.min_sqft:,} ft²" if profile.min_sqft else "Any"
    island_label = "ON ✅" if profile.require_open_island_kitchen else "OFF ❌"
    light_label = "ON ✅" if profile.require_high_natural_light else "OFF ❌"

    keyboard = [
        [
            InlineKeyboardButton(
                f"🛏️ Min Beds: {profile.min_bedrooms:g}+ (Tap to change)",
                callback_data="crit|cycle_beds",
            ),
        ],
        [
            InlineKeyboardButton(
                f"🛁 Min Baths: {profile.min_bathrooms:g}+ (Tap to change)",
                callback_data="crit|cycle_baths",
            ),
        ],
        [
            InlineKeyboardButton(
                f"💰 Max Rent: ${profile.max_monthly_rent:,}/mo (Tap to change)",
                callback_data="crit|cycle_rent",
            ),
        ],
        [
            InlineKeyboardButton(
                f"📏 Min SqFt: {sqft_label} (Tap to change)",
                callback_data="crit|cycle_sqft",
            ),
        ],
        [
            InlineKeyboardButton(
                f"🍳 Island Kitchen Req: {island_label}",
                callback_data="crit|toggle_island",
            ),
            InlineKeyboardButton(
                f"☀️ High Light Req: {light_label}",
                callback_data="crit|toggle_light",
            ),
        ],
        [
            InlineKeyboardButton(
                "🔍 Run Scan With These Criteria",
                callback_data="act|scan",
            ),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def format_telegram_alert_html(
    evaluated: EvaluatedListing,
    rank_index: int | None = None,
    total_qualified: int | None = None,
) -> str:
    """
    Format a qualifying UES listing into a ranked HTML Telegram card with explicit rubric breakdown.
    All external/scraped fields are strictly HTML-escaped.
    """
    listing = evaluated.listing
    ev = evaluated.evaluation

    title_safe = html.escape(listing.display_title)
    neigh_safe = html.escape(listing.neighborhood)
    source_safe = html.escape(listing.source)
    floor_safe = html.escape(listing.floor_info or "High Floor / Open Exposure")
    light_safe = html.escape(ev.light_analysis)
    kitchen_safe = html.escape(ev.kitchen_analysis)
    summary_safe = html.escape(ev.reasoning_summary)
    url_safe = html.escape(listing.url, quote=True)

    sqft_part = f" • {listing.sqft:,} ft²" if listing.sqft else ""
    beds_str = (
        f"{int(listing.bedrooms)}"
        if listing.bedrooms.is_integer()
        else f"{listing.bedrooms}"
    )
    baths_str = (
        f"{int(listing.bathrooms)}"
        if listing.bathrooms.is_integer()
        else f"{listing.bathrooms}"
    )

    if rank_index is not None and total_qualified is not None and total_qualified > 0:
        header = (
            f"🏆 <b>RANK #{rank_index} OF {total_qualified} QUALIFYING MATCHES "
            f"— {ev.overall_score}/100 PTS</b>"
        )
    else:
        header = f"🏛️ <b>HIGH-CONVICTION UES MATCH ({ev.overall_score}/100)</b>"

    if ev.kitchen_points > 0 or ev.light_points > 0:
        rubric_line = (
            f"📊 <b>100-Pt Ranking Rubric:</b> Total <b>{ev.overall_score}/100</b>\n"
            f"  • Kitchen &amp; Island Flow: <b>{ev.kitchen_points}/35</b>\n"
            f"  • Natural Light &amp; Exposure: <b>{ev.light_points}/35</b>\n"
            f"  • Family Layout &amp; Size: <b>{ev.layout_points}/15</b>\n"
            f"  • Learned Preference Fit: <b>{ev.learned_preference_points}/15</b>"
        )
    else:
        rubric_line = (
            f"📊 <b>Scores:</b> Overall <b>{ev.overall_score}</b> | "
            f"Light <b>{ev.light_score}</b> | Kitchen <b>{ev.kitchen_layout_score}</b>"
        )

    pros_lines = "\n".join(f"  • {html.escape(p)}" for p in ev.pros[:4])

    lines = [
        header,
        f"📍 <b>{title_safe}</b> — <i>{neigh_safe}</i>",
        f"💰 <b>${listing.price:,}/mo</b> • 🛏️ <b>{beds_str} BR</b> / 🛁 <b>{baths_str} BA</b>{sqft_part}",
        f"☀️ <b>Floor &amp; Exposure:</b> {floor_safe}",
        "",
        rubric_line,
        "",
        f"🍳 <b>Open Island / Breakfast Bar Check:</b>\n{kitchen_safe}",
        "",
        f"🔆 <b>Natural Light Check:</b>\n{light_safe}",
    ]

    if pros_lines:
        lines.extend(["", f"✨ <b>Standout Highlights:</b>\n{pros_lines}"])

    lines.extend(
        [
            "",
            f"🧠 <b>Advisor Verdict:</b> {summary_safe}",
            "",
            f"🔗 <a href=\"{url_safe}\">View Full Listing on {source_safe}</a>",
            f"<code>ID: {html.escape(listing.listing_id)}</code>",
        ]
    )
    return "\n".join(lines)


class TelegramAgentBot:
    """Manages Telegram notifications, inline button callbacks, criteria editor, and voice/text training loop."""

    def __init__(
        self,
        config: AgentConfig,
        storage: StorageBackend,
        evaluator: GeminiListingEvaluator,
        on_manual_scan: Callable[[], Awaitable[dict[str, int] | int]] | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.evaluator = evaluator
        self.on_manual_scan = on_manual_scan
        self.app: Application | None = None

        if config.telegram_bot_token:
            self.app = (
                Application.builder()
                .token(config.telegram_bot_token)
                .build()
            )
            self._register_handlers(self.app)

    async def register_bot_commands_menu(self) -> None:
        """Register the native `/` command menu inside the Telegram mobile client."""
        if not self.app:
            return
        try:
            await self.app.bot.set_my_commands(
                [
                    BotCommand("actions", "List all agent capabilities & 1-tap actions"),
                    BotCommand("criteria", "View & adjust search criteria (beds, baths, rent, rules)"),
                    BotCommand("scan", "Run an immediate scan & rank qualifying apartments"),
                    BotCommand("top", "Show Top 5 highest-ranked apartments evaluated so far"),
                    BotCommand("memory", "View learned YES/NO feedback rules injected into Gemini"),
                    BotCommand("status", "Check agent health, stats & scan schedule"),
                    BotCommand("test_alert", "Send a sample UES alert card to test buttons"),
                ]
            )
        except Exception as exc:
            logger.debug("Could not register Telegram command menu: %s", exc)

    def _is_authorized(self, update: Update) -> bool:
        """Restrict bot interaction strictly to TELEGRAM_ALLOWED_CHAT_ID when configured."""
        if self.config.telegram_allowed_chat_id is None:
            return True
        chat = update.effective_chat
        if chat is None:
            return False
        if chat.id != self.config.telegram_allowed_chat_id:
            logger.warning(
                "Rejected unauthorized Telegram access attempt from chat_id=%s", chat.id
            )
            return False
        return True

    def _register_handlers(self, app: Application) -> None:
        app.add_handler(CommandHandler(["actions", "help"], self.cmd_actions))
        app.add_handler(CommandHandler(["start", "status"], self.cmd_status))
        app.add_handler(CommandHandler("criteria", self.cmd_criteria))
        app.add_handler(CommandHandler("set", self.cmd_set_criteria))
        app.add_handler(CommandHandler("add_rule", self.cmd_add_rule))
        app.add_handler(CommandHandler("remove_rule", self.cmd_remove_rule))
        app.add_handler(CommandHandler("memory", self.cmd_memory))
        app.add_handler(CommandHandler("scan", self.cmd_scan))
        app.add_handler(CommandHandler("top", self.cmd_top))
        app.add_handler(CommandHandler("test_alert", self.cmd_test_alert))
        app.add_handler(CallbackQueryHandler(self.handle_vote_callback, pattern=r"^vote\|"))
        app.add_handler(
            CallbackQueryHandler(self.handle_quick_reason_callback, pattern=r"^qreason\|")
        )
        app.add_handler(
            CallbackQueryHandler(self.handle_show_top_callback, pattern=r"^show_top$")
        )
        app.add_handler(
            CallbackQueryHandler(self.handle_action_menu_callback, pattern=r"^act\|")
        )
        app.add_handler(
            CallbackQueryHandler(self.handle_criteria_button_callback, pattern=r"^crit\|")
        )
        app.add_handler(
            MessageHandler(filters.VOICE & ~filters.COMMAND, self.handle_voice_feedback)
        )
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_feedback)
        )

    async def send_high_conviction_alert(
        self,
        evaluated: EvaluatedListing,
        target_chat_id: int | None = None,
        rank_index: int | None = None,
        total_qualified: int | None = None,
    ) -> int | None:
        """Send a ranked qualifying listing alert with [ YES 👍 ] and [ NO 👎 ] inline buttons."""
        chat_id = target_chat_id or self.config.telegram_allowed_chat_id
        if not self.app or chat_id is None:
            logger.info(
                "[Console Alert Preview - Score %d/100] %s (%s) - $%d/mo - %s",
                evaluated.evaluation.overall_score,
                evaluated.listing.display_title,
                evaluated.listing.neighborhood,
                evaluated.listing.price,
                evaluated.listing.url,
            )
            return None

        text_html = format_telegram_alert_html(
            evaluated, rank_index=rank_index, total_qualified=total_qualified
        )
        reply_markup = build_feedback_keyboard(evaluated.listing.listing_id)

        msg = await self.app.bot.send_message(
            chat_id=chat_id,
            text=text_html,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
            disable_web_page_preview=False,
        )
        self.storage.mark_listing_alerted(
            evaluated.listing.listing_id, telegram_message_id=msg.message_id
        )
        return msg.message_id

    async def cmd_actions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /actions and /help: full summary of all agent capabilities + 1-tap buttons."""
        if not self._is_authorized(update) or not update.effective_message:
            return
        text = self._format_actions_text()
        await update.effective_message.reply_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=build_actions_menu_keyboard(),
        )

    @staticmethod
    def _format_actions_text() -> str:
        return (
            "🤖 <b>UES Family Rental Agent — Capabilities &amp; Commands</b>\n\n"
            "<b>1. 🔍 Search &amp; Ranked Discovery</b>\n"
            "• /scan — Scan StreetEasy (via Google Search), Craigslist NYC, RentHop &amp; NYBits right now. "
            "Shows <b>100% of apartments meeting your criteria</b>, ranked #1 to #N.\n"
            "• /top — View the Top 5 highest-ranked apartments evaluated so far.\n"
            "• /test_alert — Send a sample Carnegie Hill 3BR card to test feedback buttons.\n\n"
            "<b>2. ⚙️ View &amp; Adjust Search Criteria</b>\n"
            "• /criteria — View all active filters (Min Beds, Min Baths, Max Rent, Min SqFt, Neighborhoods, "
            "Open Island Kitchen, High Light) with <b>1-tap buttons</b> to change them.\n"
            "• <code>/set max_rent 25000</code> — Set exact maximum monthly rent\n"
            "• <code>/set min_beds 3</code> — Set minimum bedrooms (e.g., 3 or 4)\n"
            "• <code>/set min_baths 2</code> — Set minimum bathrooms (e.g., 2, 2.5, 3)\n"
            "• <code>/set min_sqft 1800</code> — Set minimum sq ft (or <code>0</code> for any)\n"
            "• <code>/set neighborhoods UES, Carnegie Hill, Lenox Hill</code>\n\n"
            "<b>3. 🧠 Active Learning &amp; Memory</b>\n"
            "• Tap <b>[ YES 👍 ]</b> or <b>[ NO 👎 ]</b> on any alert to train the ranking &amp; eligibility model.\n"
            "• Send any <b>text message</b> or <b>voice note 🎙️</b> anytime to add a new standing rule "
            "(e.g., <i>\"Must have in-unit washer/dryer\"</i> or <i>\"No buildings east of 1st Ave\"</i>).\n"
            "• /memory — View all learned YES/NO feedback &amp; rules injected into Gemini.\n"
            "• <code>/add_rule &lt;rule text&gt;</code> — Explicitly add a custom standing rule.\n"
            "• <code>/remove_rule &lt;#&gt;</code> — Delete a learned rule or custom rule by number.\n\n"
            "<b>4. 📊 System Health</b>\n"
            "• /status — View database stats, vote counts, and scan schedule."
        )

    def _format_criteria_html(self) -> tuple[str, SearchCriteriaProfile]:
        profile = self.storage.get_criteria_profile()
        feedback_entries = self.storage.get_all_feedback(limit=30)
        no_entries = [e for e in feedback_entries if e.vote == "NO"]
        yes_entries = [e for e in feedback_entries if e.vote == "YES"]

        sqft_str = f"{profile.min_sqft:,} ft²" if profile.min_sqft else "Any (Not restricted)"
        island_str = "Mandatory ✅" if profile.require_open_island_kitchen else "Preferred (Optional)"
        light_str = "Mandatory ✅" if profile.require_high_natural_light else "Preferred (Optional)"

        lines = [
            "⚙️ <b>Current Active Search Criteria</b>",
            "",
            "<b>Hard Eligibility Filters (100% of matches shown):</b>",
            f"• 📍 <b>Neighborhoods:</b> {html.escape(', '.join(profile.neighborhoods))}",
            f"• 🛏️ <b>Minimum Bedrooms:</b> {profile.min_bedrooms:g}+ BR",
            f"• 🛁 <b>Minimum Bathrooms:</b> {profile.min_bathrooms:g}+ BA",
            f"• 💰 <b>Maximum Rent:</b> ${profile.max_monthly_rent:,}/month",
            f"• 📏 <b>Minimum SqFt:</b> {sqft_str}",
            f"• 🍳 <b>Open Island / Breakfast Bar Kitchen:</b> {island_str}",
            f"• ☀️ <b>High Natural Light (High Floor / Corner / Open Exp):</b> {light_str}",
            "",
            "<b>100-Point Ranking Rubric (Orders Qualifying Matches #1 to #N):</b>",
            "• Kitchen &amp; Island Flow: <b>35 pts</b>",
            "• Natural Light &amp; Exposure: <b>35 pts</b>",
            "• Family Layout &amp; Size: <b>15 pts</b>",
            f"• Learned Preference Alignment ({len(yes_entries)} YES 👍 benchmarks): <b>15 pts</b>",
        ]

        if profile.custom_rules:
            lines.append("\n<b>Explicit Standing Rules:</b>")
            for idx, rule in enumerate(profile.custom_rules, start=1):
                lines.append(f"  C{idx}. <i>{html.escape(rule)}</i>")

        if no_entries:
            lines.append("\n<b>Learned Disqualifiers from Your [ NO 👎 ] Feedback:</b>")
            for idx, item in enumerate(no_entries[:12], start=1):
                addr = item.listing_snapshot.get("address", item.listing_id)
                reason = item.reason or "Rejected via inline button"
                lines.append(
                    f"  #{idx}. <b>{html.escape(str(addr))}</b>: <i>\"{html.escape(reason)}\"</i>"
                )
            lines.append("<i>(Use <code>/remove_rule #</code> to remove a learned rule)</i>")
        else:
            lines.append("\n<b>Learned Disqualifiers:</b> None yet (tap NO 👎 on any alert to teach me).")

        lines.extend(
            [
                "",
                "💡 <b>Tap any button below to adjust, or send:</b>",
                "<code>/set max_rent 25000</code> | <code>/set min_beds 3</code> | <code>/set min_baths 2</code>",
            ]
        )
        return "\n".join(lines), profile

    async def cmd_criteria(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /criteria command: show all active & learned criteria with 1-tap editor buttons."""
        if not self._is_authorized(update) or not update.effective_message:
            return
        text, profile = self._format_criteria_html()
        await update.effective_message.reply_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=build_criteria_editor_keyboard(profile),
        )

    async def cmd_set_criteria(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle `/set <field> <value>` to set exact criteria values:
          /set max_rent 25000
          /set min_beds 3
          /set min_baths 2.5
          /set min_sqft 1800
          /set neighborhoods Upper East Side, Carnegie Hill, Lenox Hill
        """
        if not self._is_authorized(update) or not update.effective_message:
            return

        args = context.args or []
        if len(args) < 2:
            await update.effective_message.reply_text(
                "⚙️ <b>Usage Examples:</b>\n"
                "• <code>/set max_rent 25000</code>\n"
                "• <code>/set min_beds 3</code>\n"
                "• <code>/set min_baths 2</code>\n"
                "• <code>/set min_sqft 1800</code> (or <code>0</code> for no minimum)\n"
                "• <code>/set neighborhoods Upper East Side, Carnegie Hill, Lenox Hill</code>",
                parse_mode=ParseMode.HTML,
            )
            return

        key = args[0].strip().lower()
        val_raw = " ".join(args[1:]).strip()
        profile = self.storage.get_criteria_profile()

        try:
            if key in ("max_rent", "rent", "price", "max_price"):
                cleaned = int(re.sub(r"[^\d]", "", val_raw))
                profile.max_monthly_rent = max(1000, cleaned)
            elif key in ("min_beds", "beds", "bedrooms"):
                profile.min_bedrooms = max(1.0, float(val_raw))
            elif key in ("min_baths", "baths", "bathrooms"):
                profile.min_bathrooms = max(1.0, float(val_raw))
            elif key in ("min_sqft", "sqft"):
                cleaned = int(re.sub(r"[^\d]", "", val_raw) or "0")
                profile.min_sqft = cleaned if cleaned > 0 else None
            elif key in ("neighborhoods", "neighborhood", "hoods"):
                hoods = [h.strip() for h in val_raw.split(",") if h.strip()]
                if hoods:
                    profile.neighborhoods = hoods
            else:
                await update.effective_message.reply_text(
                    f"Unknown setting <code>{html.escape(key)}</code>. "
                    "Supported keys: <code>max_rent</code>, <code>min_beds</code>, "
                    "<code>min_baths</code>, <code>min_sqft</code>, <code>neighborhoods</code>.",
                    parse_mode=ParseMode.HTML,
                )
                return
        except ValueError:
            await update.effective_message.reply_text(
                f"Invalid value <code>{html.escape(val_raw)}</code> for <code>{html.escape(key)}</code>.",
                parse_mode=ParseMode.HTML,
            )
            return

        self.storage.save_criteria_profile(profile)
        text, updated_profile = self._format_criteria_html()
        await update.effective_message.reply_text(
            f"✅ <b>Criteria Updated!</b>\n\n{text}",
            parse_mode=ParseMode.HTML,
            reply_markup=build_criteria_editor_keyboard(updated_profile),
        )

    async def cmd_add_rule(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle `/add_rule <text>` to add an explicit standing rule to the criteria profile."""
        if not self._is_authorized(update) or not update.effective_message:
            return
        rule_text = " ".join(context.args or []).strip()
        if not rule_text:
            await update.effective_message.reply_text(
                "Usage: <code>/add_rule Must have in-unit washer/dryer</code>",
                parse_mode=ParseMode.HTML,
            )
            return
        profile = self.storage.get_criteria_profile()
        profile.custom_rules.append(rule_text)
        self.storage.save_criteria_profile(profile)
        await update.effective_message.reply_text(
            f"✅ Added standing rule: <i>\"{html.escape(rule_text)}\"</i>\n"
            "Send /criteria to view all active rules.",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_remove_rule(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle `/remove_rule <number>` to delete a learned NO feedback rule or custom rule."""
        if not self._is_authorized(update) or not update.effective_message:
            return
        arg = " ".join(context.args or []).strip().lstrip("#")
        if not arg:
            await update.effective_message.reply_text(
                "Usage: <code>/remove_rule 1</code> (removes learned rule #1 shown in /criteria)",
                parse_mode=ParseMode.HTML,
            )
            return

        if arg.upper().startswith("C") and arg[1:].isdigit():
            c_idx = int(arg[1:]) - 1
            profile = self.storage.get_criteria_profile()
            if 0 <= c_idx < len(profile.custom_rules):
                removed = profile.custom_rules.pop(c_idx)
                self.storage.save_criteria_profile(profile)
                await update.effective_message.reply_text(
                    f"🗑️ Removed custom rule: <i>\"{html.escape(removed)}\"</i>",
                    parse_mode=ParseMode.HTML,
                )
                return

        if arg.isdigit():
            idx = int(arg) - 1
            no_entries = [
                e for e in self.storage.get_all_feedback(limit=50) if e.vote == "NO"
            ]
            if 0 <= idx < len(no_entries):
                target = no_entries[idx]
                self.storage.delete_feedback_entry(target.listing_id)
                await update.effective_message.reply_text(
                    f"🗑️ Removed learned rule #{idx + 1}: <i>\"{html.escape(target.reason or target.listing_id)}\"</i>",
                    parse_mode=ParseMode.HTML,
                )
                return

        await update.effective_message.reply_text(
            "Could not find a rule with that number. Check /criteria to see current rule numbers.",
            parse_mode=ParseMode.HTML,
        )

    async def handle_criteria_button_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle 1-tap inline buttons in /criteria editor."""
        if not self._is_authorized(update) or not update.callback_query:
            return
        query = update.callback_query
        action = (query.data or "").split("|", 1)[-1]
        profile = self.storage.get_criteria_profile()

        if action == "cycle_beds":
            curr_idx = (
                BEDS_CYCLE_STEPS.index(profile.min_bedrooms)
                if profile.min_bedrooms in BEDS_CYCLE_STEPS
                else 0
            )
            profile.min_bedrooms = BEDS_CYCLE_STEPS[(curr_idx + 1) % len(BEDS_CYCLE_STEPS)]
        elif action == "cycle_baths":
            curr_idx = (
                BATHS_CYCLE_STEPS.index(profile.min_bathrooms)
                if profile.min_bathrooms in BATHS_CYCLE_STEPS
                else 1
            )
            profile.min_bathrooms = BATHS_CYCLE_STEPS[(curr_idx + 1) % len(BATHS_CYCLE_STEPS)]
        elif action == "cycle_rent":
            curr_idx = (
                RENT_CYCLE_STEPS.index(profile.max_monthly_rent)
                if profile.max_monthly_rent in RENT_CYCLE_STEPS
                else 3
            )
            profile.max_monthly_rent = RENT_CYCLE_STEPS[(curr_idx + 1) % len(RENT_CYCLE_STEPS)]
        elif action == "cycle_sqft":
            curr_idx = (
                SQFT_CYCLE_STEPS.index(profile.min_sqft)
                if profile.min_sqft in SQFT_CYCLE_STEPS
                else 0
            )
            profile.min_sqft = SQFT_CYCLE_STEPS[(curr_idx + 1) % len(SQFT_CYCLE_STEPS)]
        elif action == "toggle_island":
            profile.require_open_island_kitchen = not profile.require_open_island_kitchen
        elif action == "toggle_light":
            profile.require_high_natural_light = not profile.require_high_natural_light

        self.storage.save_criteria_profile(profile)
        await query.answer("Criteria updated!")
        text, updated_profile = self._format_criteria_html()
        if query.message:
            try:
                await query.edit_message_text(
                    text=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=build_criteria_editor_keyboard(updated_profile),
                )
            except Exception as exc:
                logger.debug("Could not edit criteria message: %s", exc)

    async def handle_action_menu_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle 1-tap buttons from /actions menu."""
        if not self._is_authorized(update) or not update.callback_query or not update.effective_chat:
            return
        query = update.callback_query
        action = (query.data or "").split("|", 1)[-1]
        await query.answer()

        if action == "scan":
            await self._execute_scan_and_reply(update.effective_chat.id)
        elif action == "top":
            await self._send_top_evaluated(update.effective_chat.id)
        elif action == "criteria":
            text, profile = self._format_criteria_html()
            if self.app:
                await self.app.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=build_criteria_editor_keyboard(profile),
                )
        elif action == "memory":
            memory_block = self.storage.format_feedback_for_prompt(max_entries=25)
            if self.app:
                await self.app.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=(
                        "🧠 <b>Active Training Memory (Injected into Gemini Prompts):</b>\n\n"
                        f"<pre>{html.escape(memory_block)}</pre>"
                    ),
                    parse_mode=ParseMode.HTML,
                )
        elif action == "status":
            await self.cmd_status(update, context)
        elif action == "test_alert":
            await self.cmd_test_alert(update, context)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start and /status commands."""
        if not self._is_authorized(update) or not update.effective_chat:
            return
        stats = self.storage.get_stats()
        criteria = self.storage.get_criteria_profile()
        hoods = ", ".join(criteria.neighborhoods)
        msg = (
            "🏛️ <b>UES Family Rental Agent — Active</b>\n\n"
            f"• <b>Target Areas:</b> {html.escape(hoods)}\n"
            f"• <b>Min Size:</b> {criteria.min_bedrooms:g}+ BR / {criteria.min_bathrooms:g}+ BA\n"
            f"• <b>Max Rent:</b> ${criteria.max_monthly_rent:,}/month\n"
            "• <b>Selection Policy:</b> Shows <b>100% of apartments meeting your active + learned criteria</b>, ranked #1 to #N\n"
            f"• <b>Scan Interval:</b> Every {self.config.scan_interval_minutes} mins\n\n"
            "📈 <b>Memory &amp; Scan Stats:</b>\n"
            f"• Listings Evaluated: <b>{stats['total_evaluated']}</b>\n"
            f"• Qualifying Alerts Sent: <b>{stats['total_alerted']}</b>\n"
            f"• Training Votes: <b>{stats['yes_votes']} YES 👍</b> | <b>{stats['no_votes']} NO 👎</b> "
            f"(<b>{stats['feedback_with_reasons']}</b> detailed reasons learned)\n\n"
            "💡 Send <b>/actions</b> for all capabilities or <b>/criteria</b> to view/edit your filters."
        )
        if self.app:
            await self.app.bot.send_message(
                chat_id=update.effective_chat.id,
                text=msg,
                parse_mode=ParseMode.HTML,
                reply_markup=build_actions_menu_keyboard(),
            )

    async def cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Display the active feedback training log currently injected into Gemini prompts."""
        if not self._is_authorized(update) or not update.effective_message:
            return
        memory_block = self.storage.format_feedback_for_prompt(max_entries=25)
        msg = (
            "🧠 <b>Active Training Memory (Injected into Gemini Prompts):</b>\n\n"
            f"<pre>{html.escape(memory_block)}</pre>"
        )
        await update.effective_message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def _execute_scan_and_reply(self, chat_id: int) -> None:
        if not self.app or not self.on_manual_scan:
            return
        criteria = self.storage.get_criteria_profile()
        await self.app.bot.send_message(
            chat_id=chat_id,
            text=(
                f"🔍 Scanning StreetEasy (via Google Search), Craigslist NYC, RentHop &amp; NYBits "
                f"for <b>{criteria.min_bedrooms:g}+ BR / {criteria.min_bathrooms:g}+ BA</b> homes "
                f"(up to <b>${criteria.max_monthly_rent:,}/mo</b>) in "
                f"{html.escape(', '.join(criteria.neighborhoods))}..."
            ),
            parse_mode=ParseMode.HTML,
        )
        raw_res = await self.on_manual_scan()
        if isinstance(raw_res, int):
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=f"✅ Scan complete. Sent <b>{raw_res}</b> qualifying ranked match(es).",
                parse_mode=ParseMode.HTML,
            )
            return

        alerts_sent = raw_res.get("high_conviction_alerts_sent", 0)
        evaluated_count = raw_res.get("gemini_evaluated_count", 0)
        summary = (
            "✅ <b>Scan Complete — Funnel Breakdown:</b>\n"
            f"• 📥 Candidates Found: <b>{raw_res.get('scraped_count', 0)}</b>\n"
            f"• 🔄 Already Seen (Skipped): <b>{raw_res.get('already_seen_count', 0)}</b>\n"
            f"• 🚫 Filtered by Pre-Filter: <b>{raw_res.get('prefilter_rejected_count', 0)}</b>\n"
            f"• 🧠 Newly Evaluated by Gemini: <b>{evaluated_count}</b>\n"
            f"• ❌ Disqualified by Criteria/Rules: <b>{raw_res.get('below_threshold_count', 0)}</b>\n"
            f"• 🏆 <b>Qualifying Matches Sent (Ranked #1 to #{alerts_sent}): {alerts_sent}</b>"
        )
        reply_markup = None
        if alerts_sent == 0:
            summary += (
                "\n\n💡 <i>Want to see the highest-ranked apartments evaluated so far (including near-misses) "
                "or adjust your filters? Tap below:</i>"
            )
            reply_markup = InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("👀 Show Top 5 Evaluated Apartments", callback_data="show_top")],
                    [InlineKeyboardButton("⚙️ View / Adjust Criteria", callback_data="act|criteria")],
                ]
            )

        await self.app.bot.send_message(
            chat_id=chat_id,
            text=summary,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Trigger an immediate background scan cycle from Telegram with detailed funnel metrics."""
        if not self._is_authorized(update) or not update.effective_chat:
            return
        await self._execute_scan_and_reply(update.effective_chat.id)

    async def _send_top_evaluated(self, chat_id: int) -> None:
        """Send the top 5 highest-scoring evaluated listings in ranked order (#1 to #5)."""
        top_items = self.storage.get_top_evaluated_listings(limit=5)
        if not top_items and self.app:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text="No evaluated listings in database yet. Run /scan first!",
            )
            return

        total = len(top_items)
        for idx, item in enumerate(top_items, start=1):
            await self.send_high_conviction_alert(
                item,
                target_chat_id=chat_id,
                rank_index=idx,
                total_qualified=total,
            )

    async def cmd_top(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /top command to view the top 5 evaluated apartments in rank order."""
        if not self._is_authorized(update) or not update.effective_chat:
            return
        await self._send_top_evaluated(update.effective_chat.id)

    async def handle_show_top_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the [ 👀 Show Top 5 Evaluated Apartments ] button."""
        if not self._is_authorized(update) or not update.callback_query or not update.effective_chat:
            return
        await update.callback_query.answer("Loading top ranked apartments...")
        await self._send_top_evaluated(update.effective_chat.id)

    async def cmd_test_alert(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a sample Carnegie Hill 3BR alert so the user can test feedback buttons."""
        if not self._is_authorized(update) or not update.effective_chat:
            return
        mock_scraper = MockUESScraper()
        sample_listing = (await mock_scraper.fetch_listings())[0]
        evaluation = self.evaluator.evaluate_listing(sample_listing)
        evaluated = EvaluatedListing(listing=sample_listing, evaluation=evaluation, alerted=True)
        self.storage.save_evaluated_listing(evaluated)
        await self.send_high_conviction_alert(
            evaluated, target_chat_id=update.effective_chat.id, rank_index=1, total_qualified=1
        )

    async def handle_vote_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle clicks on [ YES 👍 ] and [ NO 👎 ] inline buttons."""
        if not self._is_authorized(update):
            return
        query = update.callback_query
        if not query or not query.data or not update.effective_chat:
            return

        parts = query.data.split("|", 2)
        if len(parts) != 3:
            await query.answer("Invalid callback payload.")
            return

        _, vote, listing_id = parts
        if vote not in ("YES", "NO"):
            await query.answer("Invalid vote option.")
            return

        chat_id = update.effective_chat.id
        evaluated = self.storage.get_evaluated_listing(listing_id)
        address_label = (
            evaluated.listing.display_title if evaluated else listing_id
        )

        try:
            await query.edit_message_reply_markup(
                reply_markup=build_feedback_keyboard(listing_id, selected_vote=vote)
            )
        except Exception as exc:
            logger.debug("Could not update inline keyboard markup: %s", exc)

        if vote == "YES":
            self.storage.record_feedback(
                listing_id=listing_id,
                vote="YES",
                reason="Approved match via Telegram inline button.",
                feedback_source="inline_button",
            )
            self.storage.clear_pending_feedback_prompt(chat_id)
            await query.answer("👍 Saved to positive training memory!")
            if query.message:
                await query.message.reply_text(
                    f"✅ <b>Saved YES 👍 for {html.escape(address_label)}</b>.\n"
                    "Gemini will boost the ranking score of similar layouts and exposures. "
                    "<i>(Tip: Reply to this message if you want to note what you liked most!)</i>\n"
                    f"<code>Ref: {html.escape(listing_id)}</code>",
                    parse_mode=ParseMode.HTML,
                )
        else:
            self.storage.record_feedback(
                listing_id=listing_id,
                vote="NO",
                reason="",
                feedback_source="inline_button",
            )
            await query.answer("👎 Marked as NO. Tap a reason or send a text/voice note!")
            prompt_msg_id: int | None = None
            if query.message:
                prompt_msg = await query.message.reply_text(
                    f"👎 <b>Marked NO for {html.escape(address_label)}</b>.\n\n"
                    "<b>Why wasn't it a fit?</b>\n"
                    "• Tap a quick reason below, <b>OR</b>\n"
                    "• Send a quick <b>text message</b> or <b>voice note</b> with any detail "
                    "(e.g., <i>\"Pass-through isn't open enough\"</i>, <i>\"Rooms too small\"</i>, "
                    "or <i>\"Too far east on 86th\"</i>).\n"
                    f"<code>Ref: {html.escape(listing_id)}</code>",
                    parse_mode=ParseMode.HTML,
                    reply_markup=build_quick_reasons_keyboard(listing_id),
                )
                prompt_msg_id = prompt_msg.message_id

            self.storage.set_pending_feedback_prompt(
                chat_id=chat_id,
                listing_id=listing_id,
                prompt_message_id=prompt_msg_id,
            )

    async def handle_quick_reason_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle 1-tap quick rejection reason buttons (e.g., 'Rooms too small', 'Location not great')."""
        if not self._is_authorized(update):
            return
        query = update.callback_query
        if not query or not query.data or not update.effective_chat:
            return

        parts = query.data.split("|", 2)
        if len(parts) != 3:
            await query.answer("Invalid reason payload.")
            return

        _, reason_key, listing_id = parts
        reason_text = QUICK_NO_REASONS.get(reason_key, reason_key)

        entry = self.storage.record_feedback(
            listing_id=listing_id,
            vote="NO",
            reason=reason_text,
            feedback_source="inline_button",
        )
        self.storage.clear_pending_feedback_prompt(update.effective_chat.id)
        addr = entry.listing_snapshot.get("address", listing_id)

        await query.answer("🧠 Reason saved to Gemini training memory!")
        if query.message:
            await query.message.reply_text(
                f"🧠 <b>Training Memory Updated!</b>\n"
                f"• <b>Property:</b> {html.escape(str(addr))}\n"
                f"• <b>Learned Disqualifier:</b> <i>\"{html.escape(reason_text)}\"</i>\n\n"
                "You can still send an extra text or voice note if you want to add more detail, "
                "or view all rules with /criteria.",
                parse_mode=ParseMode.HTML,
            )

    def _resolve_feedback_target_listing_id(self, update: Update) -> str | None:
        msg = update.effective_message
        chat = update.effective_chat
        if msg and msg.reply_to_message and msg.reply_to_message.text:
            reply_text = msg.reply_to_message.text
            for prefix in ("Ref: ", "ID: "):
                if prefix in reply_text:
                    after = reply_text.split(prefix, 1)[1].strip()
                    token = after.split()[0].strip()
                    if token:
                        return token

        if chat:
            pending = self.storage.get_pending_feedback_prompt(chat.id)
            if pending:
                return pending.listing_id

        return None

    async def handle_text_feedback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Save user's text rejection reason (or general standing rule) into training memory."""
        if not self._is_authorized(update) or not update.effective_message or not update.effective_chat:
            return

        reason_text = (update.effective_message.text or "").strip()
        if not reason_text:
            return

        listing_id = self._resolve_feedback_target_listing_id(update)
        if not listing_id:
            general_id = f"rule:{int(time.time())}"
            self.storage.record_feedback(
                listing_id=general_id,
                vote="NO",
                reason=f"General User Criterion: {reason_text}",
                feedback_source="text_reply",
            )
            await update.effective_message.reply_text(
                f"🧠 <b>New Standing Rule Added to Memory!</b>\n"
                f"• <b>Rule:</b> <i>\"{html.escape(reason_text)}\"</i>\n\n"
                "Gemini will enforce this across all future listing evaluations. "
                "Send /criteria to view or edit all rules.",
                parse_mode=ParseMode.HTML,
            )
            return

        entry = self.storage.record_feedback(
            listing_id=listing_id,
            vote="NO",
            reason=reason_text,
            feedback_source="text_reply",
        )
        self.storage.clear_pending_feedback_prompt(update.effective_chat.id)
        addr = entry.listing_snapshot.get("address", listing_id)

        await update.effective_message.reply_text(
            f"🧠 <b>Training Memory Updated!</b>\n"
            f"• <b>Property:</b> {html.escape(str(addr))}\n"
            f"• <b>Learned Disqualifier:</b> <i>\"{html.escape(entry.reason)}\"</i>\n\n"
            "This rule will be dynamically injected into all future Gemini evaluations.",
            parse_mode=ParseMode.HTML,
        )

    async def handle_voice_feedback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Transcribe a Telegram voice note via Gemini and save the reason into training memory."""
        if not self._is_authorized(update) or not update.effective_message or not update.effective_chat:
            return

        voice = update.effective_message.voice
        if not voice:
            return

        tg_file = await voice.get_file()
        audio_bytearray = await tg_file.download_as_bytearray()
        mime_type = voice.mime_type or "audio/ogg"

        transcribed_reason = self.evaluator.transcribe_voice_feedback(
            bytes(audio_bytearray), mime_type=mime_type
        )

        listing_id = self._resolve_feedback_target_listing_id(update)
        if not listing_id:
            listing_id = f"rule:{int(time.time())}"

        entry = self.storage.record_feedback(
            listing_id=listing_id,
            vote="NO",
            reason=transcribed_reason,
            feedback_source="voice_note",
        )
        self.storage.clear_pending_feedback_prompt(update.effective_chat.id)
        addr = entry.listing_snapshot.get("address", listing_id)

        await update.effective_message.reply_text(
            f"🎙️ <b>Voice Feedback Transcribed &amp; Saved!</b>\n"
            f"• <b>Property / Rule:</b> {html.escape(str(addr))}\n"
            f"• <b>Transcribed Rule:</b> <i>\"{html.escape(entry.reason)}\"</i>\n\n"
            "Injected into future Gemini evaluation prompts.",
            parse_mode=ParseMode.HTML,
        )
