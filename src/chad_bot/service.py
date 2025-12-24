from dataclasses import dataclass
from typing import Optional

from .config import Settings
from .database import Database
from .grok_client import GrokClient
from .rate_limits import RateLimitResult, RateLimitRule, check_rate_limit
from .spam import ValidationResult, validate_prompt
from .yaml_config import YAMLConfig


@dataclass
class ProcessResult:
    reply: str
    log_id: Optional[int]
    status: str = "auto_responded"
    error: Optional[str] = None


class RequestProcessor:
    def __init__(self, db: Database, grok: GrokClient, settings: Settings, yaml_config: Optional[YAMLConfig] = None):
        self.db = db
        self.grok = grok
        self.settings = settings
        self.yaml_config = yaml_config or YAMLConfig()
        # Model pricing is in USD per 1,000,000 tokens.
        # Prompt (input) tokens are charged at a different rate than completion (output) tokens.
        # From the attached pricing, grok-3-mini appears to be $0.30 (input) and $0.50 (output) per M tokens.
        self.prompt_price_per_m_token = 0.30
        self.completion_price_per_m_token = 0.50

    async def _check_duplicate(self, guild_id: str, user_id: str, content: str, window_seconds: int) -> bool:
        async with self.db.conn.execute(
            """
            SELECT 1 FROM message_log
            WHERE guild_id=? AND user_id=? AND user_content=? AND created_at >= datetime('now', ?)
            LIMIT 1;
            """,
            (guild_id, user_id, content, f"-{window_seconds} seconds"),
        ) as cur:
            return bool(await cur.fetchone())

    async def _check_budgets_chat(self, guild_id: str, user_id: str, config) -> Optional[str]:
        usage = await self.db.get_usage(guild_id, user_id)
        if usage["user"]["chat_tokens_used"] >= config.user_daily_chat_token_limit:
            return self.yaml_config.get_message("chat_budget_user")
        if usage["guild"]["chat_tokens_used"] >= config.global_daily_chat_token_limit:
            return self.yaml_config.get_message("chat_budget_guild")
        return None

    async def process_chat(
        self,
        *,
        guild_id: str,
        channel_id: str,
        user_id: str,
        discord_message_id: Optional[str],
        content: str,
        is_admin: bool,
    ) -> ProcessResult:
        config = await self.db.get_guild_config(guild_id)
        validation: ValidationResult = validate_prompt(content, max_chars=config.max_prompt_chars, yaml_config=self.yaml_config)
        if not validation.ok:
            log_id = await self.db.record_message(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                discord_message_id=discord_message_id,
                command_type="ask",
                user_content=content,
                status="auto_responded",
                error_code=validation.reason,
                error_detail=validation.reply,
            )
            reply = self.yaml_config.format_reply(validation.reply or self.yaml_config.get_message("invalid_input"))
            return ProcessResult(reply=reply, log_id=log_id, status="auto_responded")

        duplicate = await self._check_duplicate(guild_id, user_id, content, config.duplicate_window_seconds)
        if duplicate:
            reply = self.yaml_config.format_reply(self.yaml_config.get_message("duplicate"))
            log_id = await self.db.record_message(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                discord_message_id=discord_message_id,
                command_type="ask",
                user_content=content,
                status="auto_responded",
                error_code="duplicate",
                error_detail=reply,
            )
            return ProcessResult(reply=reply, log_id=log_id, status="auto_responded")

        rate_limit_rule = RateLimitRule(config.ask_window_seconds, config.ask_max_per_window)
        rate_result: RateLimitResult = await check_rate_limit(
            self.db,
            guild_id=guild_id,
            user_id=user_id,
            command_type="ask",
            rule=rate_limit_rule,
            yaml_config=self.yaml_config,
        )
        if not rate_result.allowed:
            reply = self.yaml_config.format_reply(rate_result.reply or self.yaml_config.get_message("rate_limit_chat"))
            log_id = await self.db.record_message(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                discord_message_id=discord_message_id,
                command_type="ask",
                user_content=content,
                status="auto_responded",
                error_code="rate_limited",
                error_detail=reply,
            )
            return ProcessResult(reply=reply, log_id=log_id)

        budget_error = await self._check_budgets_chat(guild_id, user_id, config)
        if budget_error:
            reply = self.yaml_config.format_reply(budget_error)
            log_id = await self.db.record_message(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                discord_message_id=discord_message_id,
                command_type="ask",
                user_content=content,
                status="auto_responded",
                error_code="chat_budget",
                error_detail=reply,
            )
            return ProcessResult(reply=reply, log_id=log_id)

        if config.auto_approve_enabled and not (config.admin_bypass_auto_approve and is_admin):
            log_id = await self.db.record_message(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                discord_message_id=discord_message_id,
                command_type="ask",
                user_content=content,
                status="pending_approval",
                needs_approval=True,
            )
            reply = self.yaml_config.format_reply(self.yaml_config.get_message("pending_approval_chat"))
            return ProcessResult(reply=reply, log_id=log_id, status="pending_approval")

        # Check for user-specific overrides
        user_override = await self.db.get_enabled_user_override(guild_id, user_id)
        
        # Handle custom_response override - return static response without AI
        if user_override and user_override.override_type == "custom_response" and user_override.custom_response:
            log_id = await self.db.record_message(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                discord_message_id=discord_message_id,
                command_type="ask",
                user_content=content,
                status="auto_responded",
                grok_response_content=user_override.custom_response,
                decision="user_override",
            )
            formatted_reply = self.yaml_config.format_reply(user_override.custom_response)
            reply_with_question = f"**Question:** {content}\n\n{formatted_reply}"
            return ProcessResult(reply=reply_with_question, log_id=log_id, status="auto_responded")

        # Determine system prompt: user override > guild config > YAML config
        if user_override and user_override.override_type == "custom_prompt" and user_override.custom_system_prompt:
            system_prompt = user_override.custom_system_prompt
        elif config.system_prompt:
            system_prompt = config.system_prompt
        else:
            system_prompt = self.yaml_config.get_system_prompt()

        try:
            grok_result = await self.grok.chat(
                system_prompt=system_prompt,
                user_content=content,
                temperature=config.temperature,
                max_tokens=config.max_completion_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            reply = self.yaml_config.format_reply(self.yaml_config.get_message("ai_error_chat"))
            log_id = await self.db.record_message(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                discord_message_id=discord_message_id,
                command_type="ask",
                user_content=content,
                status="error",
                error_code="grok_error",
                error_detail=str(exc),
            )
            return ProcessResult(reply=reply, log_id=log_id, status="error", error=str(exc))

        usage = grok_result.usage or {}
        total_tokens = usage.get("total_tokens", 0) or 0
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        cost = (
            (prompt_tokens / 1_000_000.0) * self.prompt_price_per_m_token
            + (completion_tokens / 1_000_000.0) * self.completion_price_per_m_token
        ) if (prompt_tokens or completion_tokens) else None
        log_id = await self.db.record_message(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            discord_message_id=discord_message_id,
            command_type="ask",
            user_content=content,
            grok_request_payload={"model": self.grok.chat_model, "max_tokens": config.max_completion_tokens},
            grok_response_content=grok_result.content,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            status="auto_responded",
        )
        if total_tokens:
            await self.db.increment_daily_chat_usage(guild_id, user_id, total_tokens)
        
        # Format reply with question prefix and prefix/suffix
        formatted_reply = self.yaml_config.format_reply(grok_result.content)
        return ProcessResult(reply=formatted_reply, log_id=log_id, status="auto_responded")
