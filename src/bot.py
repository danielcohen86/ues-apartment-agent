"""Telegram Bot notification UI and Active Training Feedback Loop ([ YES 👍 ] / [ NO 👎 ])."""

from __future__ import annotations

import html
import logging
from typing import Awaitable, Callable

from telegram import (
    ForceReply,
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
from src.models import EvaluatedListing
from src.scrapers.mock_listings import MockUESScraper
from src.storage import StorageBackend

logger = logging.getLogger(__name__)


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


def format_telegram_alert_html(evaluated: EvaluatedListing) -> str:
    """
    Format a high-conviction (>= 85/100) UES listing into a clean, safe HTML Telegram message.
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

    pros_lines = "\n".join(f"  • {html.escape(p)}" for p in ev.pros[:4])

    lines = [
        f"🏛️ <b>HIGH-CONVICTION UES MATCH ({ev.overall_score}/100)</b>",
        f"📍 <b>{title_safe}</b> — <i>{neigh_safe}</i>",
        f"💰 <b>${listing.price:,}/mo</b> • 🛏️ <b>{beds_str} BR</b> / 🛁 <b>{baths_str} BA</b>{sqft_part}",
        f"☀️ <b>Floor &amp; Exposure:</b> {floor_safe}",
        "",
        f"📊 <b>Scores:</b> Overall <b>{ev.overall_score}</b> | Light <b>{ev.light_score}</b> | Kitchen <b>{ev.kitchen_layout_score}</b>",
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
    """Manages Telegram notifications, inline button callbacks, and voice/text training loop."""

    def __init__(
        self,
        config: AgentConfig,
        storage: StorageBackend,
        evaluator: GeminiListingEvaluator,
        on_manual_scan: Callable[[], Awaitable[int]] | None = None,
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
        app.add_handler(CommandHandler(["start", "status"], self.cmd_status))
        app.add_handler(CommandHandler("memory", self.cmd_memory))
        app.add_handler(CommandHandler("scan", self.cmd_scan))
        app.add_handler(CommandHandler("test_alert", self.cmd_test_alert))
        app.add_handler(CallbackQueryHandler(self.handle_vote_callback, pattern=r"^vote\|"))
        app.add_handler(
            MessageHandler(filters.VOICE & ~filters.COMMAND, self.handle_voice_feedback)
        )
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_feedback)
        )

    async def send_high_conviction_alert(
        self, evaluated: EvaluatedListing, target_chat_id: int | None = None
    ) -> int | None:
        """Send a high-conviction listing alert with [ YES 👍 ] and [ NO 👎 ] inline buttons."""
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

        text_html = format_telegram_alert_html(evaluated)
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

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start and /status commands."""
        if not self._is_authorized(update) or not update.effective_message:
            return
        stats = self.storage.get_stats()
        hoods = ", ".join(self.config.target_neighborhoods)
        msg = (
            "🏛️ <b>UES 3+ BR Family Rental Agent — Active</b>\n\n"
            f"• <b>Target Areas:</b> {html.escape(hoods)}\n"
            f"• <b>Size Criteria:</b> {int(self.config.min_bedrooms)}+ Bedrooms\n"
            f"• <b>Conviction Threshold:</b> {self.config.min_conviction_score}/100 (Strict Low-Noise Policy)\n"
            f"• <b>Mandatory Layout:</b> Open kitchen w/ real island or breakfast bar + high natural light\n"
            f"• <b>Scan Interval:</b> Every {self.config.scan_interval_minutes} mins\n\n"
            "📈 <b>Memory &amp; Scan Stats:</b>\n"
            f"• Listings Evaluated: <b>{stats['total_evaluated']}</b>\n"
            f"• High-Conviction Alerts Sent: <b>{stats['total_alerted']}</b>\n"
            f"• Training Votes: <b>{stats['yes_votes']} YES 👍</b> | <b>{stats['no_votes']} NO 👎</b> "
            f"(<b>{stats['feedback_with_reasons']}</b> detailed reasons learned)\n\n"
            "<b>Commands:</b>\n"
            "• /scan — Run an immediate listing scan cycle\n"
            "• /memory — View active training feedback injected into Gemini\n"
            "• /test_alert — Send a sample high-conviction UES card to test [ YES 👍 ] / [ NO 👎 ]"
        )
        await update.effective_message.reply_text(msg, parse_mode=ParseMode.HTML)

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

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Trigger an immediate background scan cycle from Telegram."""
        if not self._is_authorized(update) or not update.effective_message:
            return
        if not self.on_manual_scan:
            await update.effective_message.reply_text("Scan callback is not bound.")
            return
        await update.effective_message.reply_text(
            "🔍 Running on-demand scan across UES / Carnegie Hill / Yorkville / Lenox Hill..."
        )
        alerts_sent = await self.on_manual_scan()
        await update.effective_message.reply_text(
            f"✅ Scan complete. Sent <b>{alerts_sent}</b> new high-conviction alert(s) (&gt;= {self.config.min_conviction_score}/100).",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_test_alert(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a sample high-conviction Carnegie Hill 3BR alert so the user can test feedback buttons."""
        if not self._is_authorized(update) or not update.effective_chat:
            return
        mock_scraper = MockUESScraper()
        sample_listing = (await mock_scraper.fetch_listings())[0]
        evaluation = self.evaluator.evaluate_listing(sample_listing)
        evaluated = EvaluatedListing(listing=sample_listing, evaluation=evaluation, alerted=True)
        self.storage.save_evaluated_listing(evaluated)
        await self.send_high_conviction_alert(evaluated, target_chat_id=update.effective_chat.id)

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

        # Update inline keyboard visual state
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
                reason="Approved high-conviction match via Telegram inline button.",
                feedback_source="inline_button",
            )
            self.storage.clear_pending_feedback_prompt(chat_id)
            await query.answer("👍 Saved to positive training memory!")
            if query.message:
                await query.message.reply_text(
                    f"✅ <b>Saved YES 👍 for {html.escape(address_label)}</b>.\n"
                    "Gemini will calibrate future high scores toward this layout and light profile.",
                    parse_mode=ParseMode.HTML,
                )
        else:
            # Vote == "NO": Record initial NO vote and prompt user for text/voice reason
            self.storage.record_feedback(
                listing_id=listing_id,
                vote="NO",
                reason="",
                feedback_source="inline_button",
            )
            await query.answer("👎 Marked as NO. Send a quick text or voice reason!")
            prompt_msg_id: int | None = None
            if query.message:
                prompt_msg = await query.message.reply_text(
                    f"👎 <b>Marked NO for {html.escape(address_label)}</b>.\n\n"
                    "What disqualified it? Send a quick <b>text message</b> or <b>voice note</b> "
                    "(e.g., <i>\"Pass-through isn't open enough\"</i> or <i>\"Bedrooms look too small\"</i>).\n"
                    f"<code>Ref: {html.escape(listing_id)}</code>",
                    parse_mode=ParseMode.HTML,
                    reply_markup=ForceReply(
                        selective=True,
                        input_field_placeholder="Why was this a NO? (text or voice note)",
                    ),
                )
                prompt_msg_id = prompt_msg.message_id

            self.storage.set_pending_feedback_prompt(
                chat_id=chat_id,
                listing_id=listing_id,
                prompt_message_id=prompt_msg_id,
            )

    def _resolve_feedback_target_listing_id(self, update: Update) -> str | None:
        """
        Determine which listing a user's text or voice note refers to:
        1. Check if the user replied directly to a message containing `ID: <listing_id>` or `Ref: <listing_id>`.
        2. Otherwise check `storage.get_pending_feedback_prompt(chat_id)`.
        """
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
        """Save user's text rejection reason into the active training memory database."""
        if not self._is_authorized(update) or not update.effective_message or not update.effective_chat:
            return

        reason_text = (update.effective_message.text or "").strip()
        if not reason_text:
            return

        listing_id = self._resolve_feedback_target_listing_id(update)
        if not listing_id:
            await update.effective_message.reply_text(
                "💡 Click <b>[ NO 👎 ]</b> on any listing alert first (or reply directly to an alert) "
                "so I know which apartment your feedback applies to.",
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

        listing_id = self._resolve_feedback_target_listing_id(update)
        if not listing_id:
            await update.effective_message.reply_text(
                "💡 Click <b>[ NO 👎 ]</b> on a listing alert first so I can link your voice note to it.",
                parse_mode=ParseMode.HTML,
            )
            return

        tg_file = await voice.get_file()
        audio_bytearray = await tg_file.download_as_bytearray()
        mime_type = voice.mime_type or "audio/ogg"

        transcribed_reason = self.evaluator.transcribe_voice_feedback(
            bytes(audio_bytearray), mime_type=mime_type
        )

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
            f"• <b>Property:</b> {html.escape(str(addr))}\n"
            f"• <b>Transcribed Rule:</b> <i>\"{html.escape(entry.reason)}\"</i>\n\n"
            "Injected into future Gemini evaluation prompts.",
            parse_mode=ParseMode.HTML,
        )
