import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

from .config import DEFAULT_SYSTEM_PROMPT


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def today_str() -> str:
    return utc_now().date().isoformat()


@dataclass
class GuildConfig:
    guild_id: str
    auto_approve_enabled: bool = False
    admin_bypass_auto_approve: bool = True
    ask_window_seconds: int = 60
    ask_max_per_window: int = 5
    duplicate_window_seconds: int = 60
    user_daily_chat_token_limit: int = 20000
    global_daily_chat_token_limit: int = 200000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.5
    max_completion_tokens: int = 1024
    max_prompt_chars: int = 4000
    admin_user_ids: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    # Deprecated fields (kept for database compatibility, not used)
    image_window_seconds: int = 300
    image_max_per_window: int = 3
    user_daily_image_limit: int = 5
    global_daily_image_limit: int = 50


class Database:
    """Async SQLite helper covering schema and a few common queries."""

    def __init__(self, path: str):
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        # Guild config cache: {guild_id: (GuildConfig, timestamp)}
        self._guild_config_cache: Dict[str, Tuple[GuildConfig, float]] = {}
        self._cache_ttl_seconds = 300  # 5 minute cache TTL

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys = ON;")
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
        # Clear cache on close
        self._guild_config_cache.clear()

    @property
    def conn(self) -> aiosqlite.Connection:
        if not self._conn:
            raise RuntimeError("Database not connected")
        return self._conn

    async def create_schema(self) -> None:
        await self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS guild_config (
                guild_id TEXT PRIMARY KEY,
                auto_approve_enabled INTEGER DEFAULT 0,
                admin_bypass_auto_approve INTEGER DEFAULT 1,
                ask_window_seconds INTEGER DEFAULT 60,
                ask_max_per_window INTEGER DEFAULT 5,
                -- DEPRECATED: Image generation removed, kept for backward compatibility
                image_window_seconds INTEGER DEFAULT 300,
                image_max_per_window INTEGER DEFAULT 3,
                duplicate_window_seconds INTEGER DEFAULT 60,
                user_daily_chat_token_limit INTEGER DEFAULT 20000,
                global_daily_chat_token_limit INTEGER DEFAULT 200000,
                -- DEPRECATED: Image generation removed, kept for backward compatibility
                user_daily_image_limit INTEGER DEFAULT 5,
                global_daily_image_limit INTEGER DEFAULT 50,
                system_prompt TEXT NOT NULL,
                temperature REAL DEFAULT 0.5,
                max_completion_tokens INTEGER DEFAULT 1024,
                max_prompt_chars INTEGER DEFAULT 4000,
                admin_user_ids TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS admin_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discord_user_id TEXT NOT NULL,
                guild_id TEXT NOT NULL,
                role TEXT DEFAULT 'admin',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(discord_user_id, guild_id)
            );

            CREATE TABLE IF NOT EXISTS user_daily_usage (
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                day TEXT NOT NULL,
                chat_tokens_used INTEGER DEFAULT 0,
                -- DEPRECATED: Image generation removed, kept for backward compatibility
                images_generated INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (guild_id, user_id, day)
            );

            CREATE TABLE IF NOT EXISTS guild_daily_usage (
                guild_id TEXT NOT NULL,
                day TEXT NOT NULL,
                chat_tokens_used INTEGER DEFAULT 0,
                -- DEPRECATED: Image generation removed, kept for backward compatibility
                images_generated INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (guild_id, day)
            );

            CREATE TABLE IF NOT EXISTS message_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                discord_message_id TEXT,
                command_type TEXT NOT NULL,
                user_content TEXT NOT NULL,
                grok_request_payload TEXT,
                grok_response_content TEXT,
                -- DEPRECATED: Image generation removed, kept for backward compatibility
                grok_image_urls TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                estimated_cost_usd REAL,
                needs_approval INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                decision TEXT,
                approved_by_admin_id TEXT,
                approved_at TEXT,
                manual_reply_content TEXT,
                error_code TEXT,
                error_detail TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                responded_at TEXT,
                deleted_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_message_log_guild_id ON message_log(guild_id);
            CREATE INDEX IF NOT EXISTS idx_message_log_guild_user_created ON message_log(guild_id, user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_message_log_status_created ON message_log(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_message_log_guild_user_cmd_created ON message_log(guild_id, user_id, command_type, created_at);
            """
        )
        await self._ensure_column("guild_config", "duplicate_window_seconds INTEGER DEFAULT 60")
        await self._ensure_column("guild_config", "admin_user_ids TEXT DEFAULT ''")
        await self._ensure_column("message_log", "deleted_at TEXT")
        await self.conn.commit()

    async def _ensure_column(self, table: str, column_def: str) -> None:
        col_name = column_def.split()[0]
        async with self.conn.execute(f"PRAGMA table_info({table});") as cur:
            cols = [row["name"] for row in await cur.fetchall()]
        if col_name not in cols:
            await self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def};")

    async def upsert_guild_config(self, config: GuildConfig) -> GuildConfig:
        fields = asdict(config)
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO guild_config (
                    guild_id, auto_approve_enabled, admin_bypass_auto_approve,
                    ask_window_seconds, ask_max_per_window,
                    image_window_seconds, image_max_per_window, duplicate_window_seconds,
                    user_daily_chat_token_limit, global_daily_chat_token_limit,
                    user_daily_image_limit, global_daily_image_limit,
                    system_prompt, temperature, max_completion_tokens, max_prompt_chars,
                    admin_user_ids, created_at, updated_at
                ) VALUES (
                    :guild_id, :auto_approve_enabled, :admin_bypass_auto_approve,
                    :ask_window_seconds, :ask_max_per_window,
                    :image_window_seconds, :image_max_per_window, :duplicate_window_seconds,
                    :user_daily_chat_token_limit, :global_daily_chat_token_limit,
                    :user_daily_image_limit, :global_daily_image_limit,
                    :system_prompt, :temperature, :max_completion_tokens, :max_prompt_chars,
                    :admin_user_ids, COALESCE(:created_at, datetime('now')), datetime('now')
                )
                ON CONFLICT(guild_id) DO UPDATE SET
                    auto_approve_enabled=excluded.auto_approve_enabled,
                    admin_bypass_auto_approve=excluded.admin_bypass_auto_approve,
                    ask_window_seconds=excluded.ask_window_seconds,
                    ask_max_per_window=excluded.ask_max_per_window,
                    image_window_seconds=excluded.image_window_seconds,
                    image_max_per_window=excluded.image_max_per_window,
                    duplicate_window_seconds=excluded.duplicate_window_seconds,
                    user_daily_chat_token_limit=excluded.user_daily_chat_token_limit,
                    global_daily_chat_token_limit=excluded.global_daily_chat_token_limit,
                    user_daily_image_limit=excluded.user_daily_image_limit,
                    global_daily_image_limit=excluded.global_daily_image_limit,
                    system_prompt=excluded.system_prompt,
                    temperature=excluded.temperature,
                    max_completion_tokens=excluded.max_completion_tokens,
                    max_prompt_chars=excluded.max_prompt_chars,
                    admin_user_ids=excluded.admin_user_ids,
                    updated_at=datetime('now');
                """,
                fields,
            )
            await self.conn.commit()
        
        # Invalidate cache after update
        self._guild_config_cache.pop(config.guild_id, None)
        
        return await self.get_guild_config(config.guild_id)

    async def get_guild_config(self, guild_id: str) -> GuildConfig:
        # Check cache first
        if guild_id in self._guild_config_cache:
            cached_config, timestamp = self._guild_config_cache[guild_id]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return cached_config
        
        # Cache miss or expired, fetch from database
        async with self.conn.execute(
            "SELECT * FROM guild_config WHERE guild_id = ?", (guild_id,)
        ) as cursor:
            row = await cursor.fetchone()
        
        if not row:
            config = await self.upsert_guild_config(GuildConfig(guild_id=guild_id))
        else:
            config = GuildConfig(**dict(row))
        
        # Update cache
        self._guild_config_cache[guild_id] = (config, time.time())
        return config

    async def add_admin(self, discord_user_id: str, guild_id: str, role: str = "admin") -> None:
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO admin_users (discord_user_id, guild_id, role)
                VALUES (?, ?, ?)
                ON CONFLICT(discord_user_id, guild_id) DO NOTHING;
                """,
                (discord_user_id, guild_id, role),
            )
            await self.conn.commit()

    async def is_admin(self, discord_user_id: str, guild_id: str) -> bool:
        async with self.conn.execute(
            "SELECT 1 FROM admin_users WHERE discord_user_id=? AND guild_id=? LIMIT 1",
            (discord_user_id, guild_id),
        ) as cur:
            row = await cur.fetchone()
            return bool(row)

    async def record_message(
        self,
        *,
        guild_id: str,
        channel_id: str,
        user_id: str,
        command_type: str,
        user_content: str,
        status: str,
        discord_message_id: Optional[str] = None,
        needs_approval: bool = False,
        grok_request_payload: Optional[Dict[str, Any]] = None,
        grok_response_content: Optional[str] = None,
        grok_image_urls: Optional[List[str]] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
        decision: Optional[str] = None,
        approved_by_admin_id: Optional[str] = None,
        approved_at: Optional[str] = None,
        manual_reply_content: Optional[str] = None,
        error_code: Optional[str] = None,
        error_detail: Optional[str] = None,
    ) -> int:
        payload_json = json.dumps(grok_request_payload) if grok_request_payload else None
        image_json = json.dumps(grok_image_urls) if grok_image_urls else None
        async with self._lock:
            cursor = await self.conn.execute(
                """
                INSERT INTO message_log (
                    guild_id, channel_id, user_id, discord_message_id, command_type,
                    user_content, grok_request_payload, grok_response_content, grok_image_urls,
                    prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd,
                    needs_approval, status, decision, approved_by_admin_id, approved_at,
                    manual_reply_content, error_code, error_detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guild_id,
                    channel_id,
                    user_id,
                    discord_message_id,
                    command_type,
                    user_content,
                    payload_json,
                    grok_response_content,
                    image_json,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    int(needs_approval),
                    status,
                    decision,
                    approved_by_admin_id,
                    approved_at,
                    manual_reply_content,
                    error_code,
                    error_detail,
                ),
            )
            await self.conn.commit()
            return cursor.lastrowid

    async def update_message_status(
        self,
        message_id: int,
        *,
        status: str,
        decision: Optional[str] = None,
        grok_response_content: Optional[str] = None,
        grok_image_urls: Optional[List[str]] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
        approved_by_admin_id: Optional[str] = None,
        manual_reply_content: Optional[str] = None,
        error_code: Optional[str] = None,
        error_detail: Optional[str] = None,
    ) -> None:
        image_json = json.dumps(grok_image_urls) if grok_image_urls else None
        async with self._lock:
            await self.conn.execute(
                """
                UPDATE message_log
                SET status=?,
                    decision=?,
                    grok_response_content=?,
                    grok_image_urls=COALESCE(?, grok_image_urls),
                    prompt_tokens=?,
                    completion_tokens=?,
                    total_tokens=?,
                    estimated_cost_usd=?,
                    approved_by_admin_id=?,
                    approved_at=CASE WHEN ? IS NOT NULL THEN datetime('now') ELSE approved_at END,
                    manual_reply_content=?,
                    error_code=?,
                    error_detail=?,
                    responded_at=CASE
                        WHEN status IN ('auto_responded','approved_grok','approved_manual','rejected','error')
                        THEN datetime('now') ELSE responded_at END
                WHERE id=?;
                """,
                (
                    status,
                    decision,
                    grok_response_content,
                    image_json,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    approved_by_admin_id,
                    approved_by_admin_id,
                    manual_reply_content,
                    error_code,
                    error_detail,
                    message_id,
                ),
            )
            await self.conn.commit()

    async def update_discord_message_id(self, message_id: int, discord_message_id: str) -> None:
        """Update the actual Discord message ID for a logged message.
        
        This is used to store the ID of the bot's response message after it's sent.
        Args:
            message_id: The database record ID
            discord_message_id: The actual Discord message ID (not interaction ID)
        """
        async with self._lock:
            await self.conn.execute(
                """
                UPDATE message_log
                SET discord_message_id = ?
                WHERE id = ?;
                """,
                (discord_message_id, message_id),
            )
            await self.conn.commit()

    async def get_message(self, message_id: int) -> Optional[Dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM message_log WHERE id=?", (message_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def pending_messages(self, guild_id: str) -> List[Dict[str, Any]]:
        async with self.conn.execute(
            """
            SELECT * FROM message_log
            WHERE guild_id=? AND status='pending_approval'
            ORDER BY created_at DESC;
            """,
            (guild_id,),
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def list_guilds(self) -> List[str]:
        async with self.conn.execute("SELECT guild_id FROM guild_config ORDER BY guild_id") as cur:
            rows = await cur.fetchall()
            return [r["guild_id"] for r in rows]

    async def history(
        self,
        guild_id: str,
        *,
        limit: int = 50,
        status: Optional[str] = None,
        command_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT id, created_at, user_id, command_type, status, user_content, grok_response_content, total_tokens, estimated_cost_usd, decision
            FROM message_log
            WHERE guild_id=?
        """
        params: List[Any] = [guild_id]
        if status:
            query += " AND status=?"
            params.append(status)
        if command_type:
            query += " AND command_type=?"
            params.append(command_type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        async with self.conn.execute(query, params) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def analytics(self, guild_id: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {"status_counts": {}, "token_total": 0, "image_requests": 0, "total_cost_usd": 0.0}
        async with self.conn.execute(
            """
            SELECT status, COUNT(*) as cnt FROM message_log
            WHERE guild_id=? GROUP BY status;
            """,
            (guild_id,),
        ) as cur:
            for row in await cur.fetchall():
                data["status_counts"][row["status"]] = row["cnt"]
        async with self.conn.execute(
            """
            SELECT SUM(total_tokens) as token_total,
                   SUM(CASE WHEN command_type='image' THEN 1 ELSE 0 END) as image_requests,
                   SUM(COALESCE(estimated_cost_usd, 0)) as total_cost_usd
            FROM message_log
            WHERE guild_id=?;
            """,
            (guild_id,),
        ) as cur:
            row = await cur.fetchone()
            if row:
                data["token_total"] = row["token_total"] or 0
                data["image_requests"] = row["image_requests"] or 0
                data["total_cost_usd"] = row["total_cost_usd"] or 0.0
        return data

    async def recent_messages(self, guild_id: str, limit: int = 25) -> List[Dict[str, Any]]:
        async with self.conn.execute(
            """
            SELECT * FROM message_log
            WHERE guild_id=?
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (guild_id, limit),
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def increment_daily_chat_usage(
        self, guild_id: str, user_id: str, tokens: int
    ) -> None:
        day = today_str()
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO user_daily_usage (guild_id, user_id, day, chat_tokens_used)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(guild_id, user_id, day) DO UPDATE SET
                    chat_tokens_used = chat_tokens_used + excluded.chat_tokens_used,
                    last_updated = datetime('now');
                """,
                (guild_id, user_id, day, tokens),
            )
            await self.conn.execute(
                """
                INSERT INTO guild_daily_usage (guild_id, day, chat_tokens_used)
                VALUES (?, ?, ?)
                ON CONFLICT(guild_id, day) DO UPDATE SET
                    chat_tokens_used = chat_tokens_used + excluded.chat_tokens_used,
                    last_updated = datetime('now');
                """,
                (guild_id, day, tokens),
            )
            await self.conn.commit()

    async def increment_daily_image_usage(
        self, guild_id: str, user_id: str, count: int = 1
    ) -> None:
        day = today_str()
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO user_daily_usage (guild_id, user_id, day, images_generated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(guild_id, user_id, day) DO UPDATE SET
                    images_generated = images_generated + excluded.images_generated,
                    last_updated = datetime('now');
                """,
                (guild_id, user_id, day, count),
            )
            await self.conn.execute(
                """
                INSERT INTO guild_daily_usage (guild_id, day, images_generated)
                VALUES (?, ?, ?)
                ON CONFLICT(guild_id, day) DO UPDATE SET
                    images_generated = images_generated + excluded.images_generated,
                    last_updated = datetime('now');
                """,
                (guild_id, day, count),
            )
            await self.conn.commit()

    async def get_usage(self, guild_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        day = today_str()
        usage: Dict[str, Any] = {}
        if user_id:
            async with self.conn.execute(
                """
                SELECT chat_tokens_used, images_generated
                FROM user_daily_usage
                WHERE guild_id=? AND user_id=? AND day=?;
                """,
                (guild_id, user_id, day),
            ) as cur:
                usage["user"] = dict(await cur.fetchone() or {"chat_tokens_used": 0, "images_generated": 0})
        async with self.conn.execute(
            """
            SELECT chat_tokens_used, images_generated
            FROM guild_daily_usage
            WHERE guild_id=? AND day=?;
            """,
            (guild_id, day),
        ) as cur:
            usage["guild"] = dict(await cur.fetchone() or {"chat_tokens_used": 0, "images_generated": 0})
        return usage

    async def count_recent(
        self, *, guild_id: str, user_id: str, command_type: str, window_seconds: int
    ) -> int:
        async with self.conn.execute(
            """
            SELECT COUNT(*) as count
            FROM message_log
            WHERE guild_id=?
              AND user_id=?
              AND command_type=?
              AND created_at >= datetime('now', ?)
              AND status != 'error';
            """,
            (guild_id, user_id, command_type, f"-{window_seconds} seconds"),
        ) as cur:
            row = await cur.fetchone()
            return int(row["count"] if row else 0)

    async def mark_message_deleted(self, message_id: int) -> None:
        """Mark a message as deleted from Discord (keeps database record)."""
        async with self._lock:
            await self.conn.execute(
                """
                UPDATE message_log
                SET deleted_at = datetime('now')
                WHERE id = ?;
                """,
                (message_id,),
            )
            await self.conn.commit()

    async def delete_guild(self, guild_id: str) -> None:
        """Delete all guild configuration and history from database."""
        async with self._lock:
            await self.conn.execute("DELETE FROM guild_config WHERE guild_id = ?", (guild_id,))
            await self.conn.execute("DELETE FROM admin_users WHERE guild_id = ?", (guild_id,))
            await self.conn.execute("DELETE FROM user_daily_usage WHERE guild_id = ?", (guild_id,))
            await self.conn.execute("DELETE FROM guild_daily_usage WHERE guild_id = ?", (guild_id,))
            await self.conn.execute("DELETE FROM message_log WHERE guild_id = ?", (guild_id,))
            await self.conn.commit()
        
        # Invalidate cache for deleted guild
        self._guild_config_cache.pop(guild_id, None)

