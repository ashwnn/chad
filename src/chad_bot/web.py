import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from .config import Settings, setup_logging
from .database import GuildConfig, Database, UserOverride
from .discord_api import DiscordApiClient
from .grok_client import GrokClient
from .service import RequestProcessor
from .yaml_config import YAMLConfig

logger = logging.getLogger(__name__)


class ConfigUpdate(BaseModel):
    auto_approve_enabled: Optional[bool] = None
    admin_bypass_auto_approve: Optional[bool] = None
    ask_window_seconds: Optional[int] = Field(None, ge=1)
    ask_max_per_window: Optional[int] = Field(None, ge=1)
    duplicate_window_seconds: Optional[int] = Field(None, ge=1)
    user_daily_chat_token_limit: Optional[int] = Field(None, ge=0)
    global_daily_chat_token_limit: Optional[int] = Field(None, ge=0)
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    max_prompt_chars: Optional[int] = None
    admin_user_ids: Optional[str] = None


class ApprovalDecision(BaseModel):
    decision: str
    manual_reply_content: Optional[str] = None
    reason: Optional[str] = None


class SendMessageRequest(BaseModel):
    channel_id: str
    content: str
    mention_user_id: Optional[str] = None


class UserOverrideRequest(BaseModel):
    discord_user_id: str
    override_type: str = "custom_response"  # "custom_response" or "custom_prompt"
    custom_response: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    enabled: bool = True


class UserOverrideUpdate(BaseModel):
    override_type: Optional[str] = None
    custom_response: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    enabled: Optional[bool] = None


class SimpleRateLimiter:
    """In-memory sliding window limiter keyed by client host."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._buckets: Dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> tuple[bool, int]:
        now = asyncio.get_event_loop().time()
        window_start = now - 60
        async with self._lock:
            timestamps = self._buckets.get(key, [])
            # Drop expired timestamps
            timestamps = [t for t in timestamps if t >= window_start]
            if len(timestamps) >= self.max_per_minute:
                retry_after = max(1, int(timestamps[0] - window_start))
                self._buckets[key] = timestamps
                return False, retry_after
            timestamps.append(now)
            self._buckets[key] = timestamps
        return True, 0


def create_app(settings: Settings) -> FastAPI:
    setup_logging(settings)
    settings.validate(require_grok=True)

    db = Database(settings.database_path)
    grok = GrokClient(
        api_key=settings.grok_api_key,
        api_base=settings.grok_api_base,
        chat_model=settings.grok_chat_model,
    )
    discord_api = DiscordApiClient(settings.discord_token)
    yaml_config = YAMLConfig()
    processor = RequestProcessor(db=db, grok=grok, settings=settings, yaml_config=yaml_config)
    api_rate_limiter = SimpleRateLimiter(settings.api_rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await db.connect()
        await db.create_schema()
        logger.info("Web backend connected to %s", settings.database_path)
        yield
        # Shutdown
        await grok.close()
        await discord_api.close()
        await db.close()

    app = FastAPI(title="Chad Bot Admin", lifespan=lifespan)

    allow_origins = ["*"] if "*" in settings.cors_origins else list(settings.cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path.startswith("/api"):
            client_host = request.client.host if request.client else "unknown"
            allowed, retry_after = await api_rate_limiter.allow(client_host)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests", "retry_after": retry_after},
                )
        return await call_next(request)

    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        guild_ids = await db.list_guilds()
        guilds = []
        for guild_id in guild_ids:
            guild_info = await discord_api.get_guild(guild_id)
            # Construct guild icon URL if available
            icon_url = None
            if guild_info and guild_info.get("icon"):
                icon_hash = guild_info.get("icon")
                icon_url = f"https://cdn.discordapp.com/icons/{guild_id}/{icon_hash}.png"
            
            guilds.append({
                "id": guild_id,
                "name": guild_info.get("name", f"Guild {guild_id}") if guild_info else f"Guild {guild_id}",
                "icon_url": icon_url
            })
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "guilds": guilds},
        )

    @app.get("/guilds/{guild_id}", response_class=HTMLResponse)
    async def overview(request: Request, guild_id: str):
        config = await db.get_guild_config(guild_id)
        pending = await db.pending_messages(guild_id)
        recent = await db.recent_messages(guild_id)
        usage = await db.get_usage(guild_id)
        analytics = await db.analytics(guild_id)
        guild_info = await discord_api.get_guild(guild_id)
        guild_name = guild_info.get("name", f"Guild {guild_id}") if guild_info else f"Guild {guild_id}"
        

        
        return templates.TemplateResponse(
            "overview.html",
            {
                "request": request,
                "page": "dashboard",
                "guild_id": guild_id,
                "guild_name": guild_name,
                "config": config,
                "pending": pending,
                "recent": recent,
                "usage": usage,
                "analytics": analytics,
                "model_pricing": {"prompt": processor.prompt_price_per_m_token, "completion": processor.completion_price_per_m_token, "model": settings.grok_chat_model},

            },
        )

    @app.get("/guilds/{guild_id}/config", response_class=HTMLResponse)
    async def config_page(request: Request, guild_id: str):
        config = await db.get_guild_config(guild_id)
        guild_info = await discord_api.get_guild(guild_id)
        guild_name = guild_info.get("name", f"Guild {guild_id}") if guild_info else f"Guild {guild_id}"

        return templates.TemplateResponse(
            "config.html",
            {
                "request": request,
                "page": "config",
                "guild_id": guild_id,
                "guild_name": guild_name,
                "config": config,

            },
        )

    @app.get("/guilds/{guild_id}/queue", response_class=HTMLResponse)
    async def queue_page(request: Request, guild_id: str):
        pending = await db.pending_messages(guild_id)
        guild_info = await discord_api.get_guild(guild_id)
        guild_name = guild_info.get("name", f"Guild {guild_id}") if guild_info else f"Guild {guild_id}"

        return templates.TemplateResponse(
            "queue.html",
            {
                "request": request,
                "page": "queue",
                "guild_id": guild_id,
                "guild_name": guild_name,
                "pending": pending,

            },
        )

    @app.get("/guilds/{guild_id}/history", response_class=HTMLResponse)
    async def history_page(
        request: Request,
        guild_id: str,
        limit: int = 100,
        status: Optional[str] = None,
        command_type: Optional[str] = None,
    ):
        history = await db.history(guild_id, limit=limit, status=status, command_type=command_type)
        guild_info = await discord_api.get_guild(guild_id)
        guild_name = guild_info.get("name", f"Guild {guild_id}") if guild_info else f"Guild {guild_id}"

        return templates.TemplateResponse(
            "history.html",
            {
                "request": request,
                "page": "history",
                "guild_id": guild_id,
                "guild_name": guild_name,
                "history": history,
                "status_filter": status,
                "command_type_filter": command_type,
                "model_pricing": {"prompt": processor.prompt_price_per_m_token, "completion": processor.completion_price_per_m_token, "model": settings.grok_chat_model},

            },
        )

    @app.get("/guilds/{guild_id}/analytics", response_class=HTMLResponse)
    async def analytics_page(request: Request, guild_id: str):
        analytics = await db.analytics(guild_id)
        recent_messages = await db.recent_messages(guild_id, limit=100)
        guild_info = await discord_api.get_guild(guild_id)
        guild_name = guild_info.get("name", f"Guild {guild_id}") if guild_info else f"Guild {guild_id}"

        return templates.TemplateResponse(
            "analytics.html",
            {
                "request": request,
                "page": "analytics",
                "guild_id": guild_id,
                "guild_name": guild_name,
                "analytics": analytics,
                "recent": recent_messages,
                "model_pricing": {"prompt": processor.prompt_price_per_m_token, "completion": processor.completion_price_per_m_token, "model": settings.grok_chat_model},

            },
        )

    @app.get("/messages", response_class=HTMLResponse)
    async def messages_page(request: Request):
        return templates.TemplateResponse(
            "messages.html",
            {
                "request": request,
                "page": "messages",
            },
        )

    @app.get("/guilds/{guild_id}/send-message", response_class=HTMLResponse)
    async def send_message_page(request: Request, guild_id: str):
        guild_info = await discord_api.get_guild(guild_id)
        guild_name = guild_info.get("name", f"Guild {guild_id}") if guild_info else f"Guild {guild_id}"
        channels = await discord_api.get_guild_channels(guild_id) or []

        return templates.TemplateResponse(
            "send_message.html",
            {
                "request": request,
                "page": "send_message",
                "guild_id": guild_id,
                "guild_name": guild_name,
                "channels": channels,
            },
        )

    @app.get("/guilds/{guild_id}/user-overrides", response_class=HTMLResponse)
    async def user_overrides_page(request: Request, guild_id: str):
        guild_info = await discord_api.get_guild(guild_id)
        guild_name = guild_info.get("name", f"Guild {guild_id}") if guild_info else f"Guild {guild_id}"
        overrides = await db.list_user_overrides(guild_id)

        return templates.TemplateResponse(
            "user_overrides.html",
            {
                "request": request,
                "page": "user_overrides",
                "guild_id": guild_id,
                "guild_name": guild_name,
                "overrides": overrides,
            },
        )

    @app.get("/api/guilds/{guild_id}/config")
    async def get_config(guild_id: str):
        config = await db.get_guild_config(guild_id)
        return config.__dict__

    @app.post("/api/guilds/{guild_id}/config")
    async def update_config(guild_id: str, payload: ConfigUpdate):
        # Validate admin_user_ids if present
        if payload.admin_user_ids is not None:
            # Allow empty string
            if payload.admin_user_ids.strip():
                for uid in payload.admin_user_ids.split(","):
                    clean_uid = uid.strip()
                    if clean_uid and not clean_uid.isdigit():
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Invalid Admin ID: '{clean_uid}'. IDs must be numeric."
                        )

        current = await db.get_guild_config(guild_id)
        update_data = current.__dict__
        for key, value in payload.model_dump(exclude_none=True).items():
            update_data[key] = value
        updated = await db.upsert_guild_config(GuildConfig(**update_data))
        return updated.__dict__

    @app.delete("/api/guilds/{guild_id}")
    async def delete_guild(guild_id: str):
        """Delete guild configuration and all associated data from database."""
        await db.delete_guild(guild_id)
        return {"status": "deleted", "guild_id": guild_id}

    @app.get("/api/guilds/{guild_id}/pending")
    async def list_pending(guild_id: str):
        return await db.pending_messages(guild_id)

    @app.get("/api/guilds/{guild_id}/history")
    async def list_history(
        guild_id: str,
        limit: int = 50,
        status: Optional[str] = None,
        command_type: Optional[str] = None,
    ):
        return await db.history(guild_id, limit=limit, status=status, command_type=command_type)

    @app.get("/api/guilds/{guild_id}/analytics")
    async def get_analytics(guild_id: str):
        return await db.analytics(guild_id)

    @app.get("/api/guilds/{guild_id}/channels")
    async def get_channels(guild_id: str):
        """Get text channels for a guild."""
        channels = await discord_api.get_guild_channels(guild_id)
        if channels is None:
            raise HTTPException(status_code=502, detail="Failed to fetch channels from Discord")
        return channels

    @app.post("/api/guilds/{guild_id}/send-message")
    async def send_custom_message(guild_id: str, payload: SendMessageRequest):
        """Send a custom message to a channel in the guild."""
        try:
            result = await discord_api.send_message(
                channel_id=payload.channel_id,
                content=payload.content,
                mention_user_id=payload.mention_user_id,
            )
            if result:
                return {"status": "sent", "message_id": result.get("id")}
            raise HTTPException(status_code=502, detail="Failed to send message to Discord")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error sending custom message: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))

    # User Override endpoints
    @app.get("/api/guilds/{guild_id}/user-overrides")
    async def list_user_overrides(guild_id: str):
        """List all user overrides for a guild."""
        overrides = await db.list_user_overrides(guild_id)
        return [{"id": o.id, "discord_user_id": o.discord_user_id, "override_type": o.override_type,
                 "custom_response": o.custom_response, "custom_system_prompt": o.custom_system_prompt,
                 "enabled": o.enabled, "created_at": o.created_at, "updated_at": o.updated_at}
                for o in overrides]

    @app.post("/api/guilds/{guild_id}/user-overrides")
    async def create_user_override(guild_id: str, payload: UserOverrideRequest):
        """Create a new user override."""
        # Validate user ID is numeric
        if not payload.discord_user_id.strip().isdigit():
            raise HTTPException(status_code=400, detail="Discord User ID must be numeric")
        
        # Validate override_type
        if payload.override_type not in ("custom_response", "custom_prompt"):
            raise HTTPException(status_code=400, detail="override_type must be 'custom_response' or 'custom_prompt'")
        
        override = UserOverride(
            guild_id=guild_id,
            discord_user_id=payload.discord_user_id.strip(),
            override_type=payload.override_type,
            custom_response=payload.custom_response,
            custom_system_prompt=payload.custom_system_prompt,
            enabled=payload.enabled,
        )
        result = await db.add_user_override(override)
        return {"id": result.id, "discord_user_id": result.discord_user_id, "override_type": result.override_type,
                "custom_response": result.custom_response, "custom_system_prompt": result.custom_system_prompt,
                "enabled": result.enabled, "created_at": result.created_at, "updated_at": result.updated_at}

    @app.put("/api/guilds/{guild_id}/user-overrides/{override_id}")
    async def update_user_override_endpoint(guild_id: str, override_id: int, payload: UserOverrideUpdate):
        """Update a user override."""
        if payload.override_type is not None and payload.override_type not in ("custom_response", "custom_prompt"):
            raise HTTPException(status_code=400, detail="override_type must be 'custom_response' or 'custom_prompt'")
        
        update_data = payload.model_dump(exclude_none=True)
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        result = await db.update_user_override(override_id, **update_data)
        if not result:
            raise HTTPException(status_code=404, detail="User override not found")
        
        return {"id": result.id, "discord_user_id": result.discord_user_id, "override_type": result.override_type,
                "custom_response": result.custom_response, "custom_system_prompt": result.custom_system_prompt,
                "enabled": result.enabled, "created_at": result.created_at, "updated_at": result.updated_at}

    @app.delete("/api/guilds/{guild_id}/user-overrides/{override_id}")
    async def delete_user_override_endpoint(guild_id: str, override_id: int):
        """Delete a user override."""
        success = await db.delete_user_override(override_id)
        if not success:
            raise HTTPException(status_code=404, detail="User override not found")
        return {"status": "deleted", "override_id": override_id}

    # YAML Configuration endpoints
    @app.get("/api/yaml-config")
    async def get_yaml_config():
        """Get all YAML configuration values."""
        return yaml_config.get_all()

    class YAMLConfigUpdate(BaseModel):
        updates: Dict[str, Any]

    @app.post("/api/yaml-config")
    async def update_yaml_config(payload: YAMLConfigUpdate):
        """Update YAML configuration values."""
        yaml_config.update(payload.updates)
        return {"status": "updated", "config": yaml_config.get_all()}

    @app.get("/api/yaml-config/messages")
    async def get_yaml_messages():
        """Get all bot messages from YAML config."""
        return yaml_config.get("messages", {})

    @app.get("/api/yaml-config/system-prompt")
    async def get_yaml_system_prompt():
        """Get system prompt from YAML config."""
        return {"system_prompt": yaml_config.get_system_prompt()}

    @app.get("/api/yaml-config/bot-settings")
    async def get_yaml_bot_settings():
        """Get bot settings (prefix/suffix) from YAML config."""
        return yaml_config.get("bot_settings", {})

    async def _send_discord_message(channel_id: str, content: str, mention_id: Optional[str] = None):
        """Send a message to Discord channel.
        
        Returns:
            dict with response data if successful
            None if failed (with exception logged)
        """
        try:
            result = await discord_api.send_message(channel_id=channel_id, content=content, mention_user_id=mention_id)
            logger.info("Successfully sent message to channel %s", channel_id)
            return result
        except Exception as exc:
            logger.exception("Could not send Discord message to channel %s: %s", channel_id, exc)
            return None

    async def _process_approval(message: Dict[str, Any]) -> Dict[str, Any]:
        """Process approval for a pending message (grok or googl)."""
        cfg = await db.get_guild_config(message["guild_id"])
        
        reply_content = None
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost = 0.0
        
        try:
            if message["command_type"] == "ask":
                # Determine system prompt
                user_override = await db.get_enabled_user_override(message["guild_id"], message["user_id"])
                if user_override and user_override.override_type == "custom_prompt" and user_override.custom_system_prompt:
                    system_prompt = user_override.custom_system_prompt
                elif cfg.system_prompt:
                    system_prompt = cfg.system_prompt
                else:
                    system_prompt = yaml_config.get_system_prompt()
                    
                content_resp, usage, cost = await processor.execute_chat(
                    system_prompt=system_prompt,
                    user_content=message["user_content"],
                    config=cfg
                )
                
                # Update stats
                reply_content = yaml_config.format_reply(content_resp)
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
            elif message["command_type"] == "googl":
                content_resp, usage, cost, _ = await processor.execute_search(
                    query=message["user_content"]
                )
                
                # Format reply (no prefix/suffix for search usually, but kept consistent if needed)
                reply_content = f"{content_resp}"
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            
            else:
                 raise HTTPException(status_code=400, detail=f"Unknown command type: {message['command_type']}")

        except Exception as exc:
            await db.update_message_status(
                message["id"],
                status="error",
                decision="approve",
                error_code="ai_error",
                error_detail=str(exc),
            )
            # Re-raise so UI sees error
            raise HTTPException(status_code=502, detail=f"AI processing failed: {exc}")

        # Update Usage
        if total_tokens:
            await db.increment_daily_chat_usage(message["guild_id"], message["user_id"], total_tokens)
        
        # Update DB
        status_slug = f"approved_{message['command_type']}" # e.g. approved_ask, approved_googl (or keep 'approved_grok' for legacy compat?)
        # Keeping legacy status 'approved_grok' for 'ask' if needed, but let's standardize.
        # The schema uses 'approved_grok' in existing code. Let's check status usage.
        # Just use "approved_ai" or similar? Existing code uses "approved_grok". 
        # For minimal friction, let's use "approved_grok" for chat and "approved_search" for search?
        # Or better yet, just 'approved' + command_type logic?
        # Let's check what 'process_chat' does... it uses 'auto_responded'.
        # Approvals explicitly used 'approved_grok'.
        
        final_status = "approved_grok" if message["command_type"] == "ask" else "approved_search"

        await db.update_message_status(
            message["id"],
            status=final_status,
            decision="approve",
            grok_response_content=reply_content, # Storing formatted reply or raw? Service stores raw usually, but here likely processed.
            # Service stores raw content usually in 'grok_response_content'. 
            # But here we are overwriting it. 
            # Let's store raw reply in DB if possible? 
            # Actually service stores raw in `grok_response_content`, and returns formatted in `reply`.
            # We should probably store the raw content if we want consistent logs, but `reply_content` above is formatted.
            # Let's strip formatting for DB storage if we want purity, or just store what we send.
            # Storing what we send is safer for history.
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            approved_by_admin_id="admin", # We don't have admin ID in this unauth context yet
        )
        
        # Send to Discord
        send_result = await _send_discord_message(
            channel_id=message["channel_id"],
            content=reply_content,
            mention_id=message["user_id"],
        )
        
        if send_result is None:
            await db.update_message_status(
                message["id"],
                status=f"{final_status}_send_failed",
                error_code="discord_send_error",
                error_detail="Failed to send approved message to Discord channel",
            )
            raise HTTPException(status_code=502, detail="Failed to send message to Discord channel")
        
        return {"status": final_status, "reply": reply_content}

    @app.post("/api/approvals/{message_id}")
    async def approve(message_id: int, payload: ApprovalDecision):
        message = await db.get_message(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        if message["status"] != "pending_approval":
            raise HTTPException(status_code=400, detail="Message not pending")
        
        if payload.decision == "grok" or payload.decision == "approve": 
            # "grok" is legacy decision string for "Approve with AI". 
            # We accept "approve" as generic too.
            return await _process_approval(message)
            
        if payload.decision == "manual":
            manual_text = payload.manual_reply_content or yaml_config.get_message("manual_reply_default")
            formatted_manual = yaml_config.format_reply(manual_text)
            await db.update_message_status(
                message_id,
                status="approved_manual",
                decision="manual",
                manual_reply_content=manual_text,
            )
            send_result = await _send_discord_message(
                channel_id=message["channel_id"],
                content=formatted_manual,
                mention_id=message["user_id"],
            )
            if send_result is None:
                # Message failed to send to Discord, update status to reflect this
                await db.update_message_status(
                    message_id,
                    status="approved_manual_send_failed",
                    error_code="discord_send_error",
                    error_detail="Failed to send approved message to Discord channel",
                )
                raise HTTPException(status_code=502, detail="Failed to send message to Discord channel")
            return {"status": "approved_manual", "reply": formatted_manual}
        
        if payload.decision == "reject":
            reply_text = payload.reason or yaml_config.get_message("rejection_default")
            formatted_rejection = yaml_config.format_reply(reply_text)
            await db.update_message_status(
                message_id,
                status="rejected",
                decision="reject",
                error_detail=payload.reason,
            )
            send_result = await _send_discord_message(
                channel_id=message["channel_id"],
                content=formatted_rejection,
                mention_id=message["user_id"],
            )
            if send_result is None:
                # Message failed to send to Discord, update status to reflect this
                await db.update_message_status(
                    message_id,
                    status="rejected_send_failed",
                    error_code="discord_send_error",
                    error_detail="Failed to send rejection message to Discord channel",
                )
                raise HTTPException(status_code=502, detail="Failed to send message to Discord channel")
            return {"status": "rejected", "reply": formatted_rejection}
        raise HTTPException(status_code=400, detail="Invalid decision")

    @app.delete("/api/messages/{message_id}")
    async def delete_message_endpoint(message_id: int):
        """Delete a message from Discord (keeps database record for history)."""
        message = await db.get_message(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        if not message["discord_message_id"]:
            raise HTTPException(status_code=400, detail="Message has no Discord message ID")
        
        # Delete from Discord
        success = await discord_api.delete_message(
            channel_id=message["channel_id"],
            message_id=message["discord_message_id"],
        )
        
        if not success:
            raise HTTPException(status_code=502, detail="Failed to delete message from Discord")
        
        # Mark as deleted in database
        await db.mark_message_deleted(message_id)
        
        return {"status": "deleted", "message_id": message_id}

    return app


app = create_app(Settings())


def run() -> None:
    import uvicorn

    settings = Settings()
    setup_logging(settings)
    uvicorn.run("chad_bot.web:app", host=settings.web_host, port=settings.web_port, reload=False)


if __name__ == "__main__":
    run()
