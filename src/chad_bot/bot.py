import asyncio
import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from .config import Settings
from .database import Database
from .gemini_client import GeminiClient
from .grok_client import GrokClient
from .service import RequestProcessor
from .yaml_config import YAMLConfig

logger = logging.getLogger(__name__)

# Discord message character limit
DISCORD_MAX_LENGTH = 2000


def split_message(text: str, max_length: int = DISCORD_MAX_LENGTH) -> list[str]:
    """Split a message into chunks that fit within Discord's character limit.
    
    Attempts to split at newlines first, then at word boundaries, preserving
    readability as much as possible.
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    remaining = text
    
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        
        # Find a good split point
        split_point = max_length
        
        # Try to split at a double newline (paragraph break) first
        para_break = remaining.rfind("\n\n", 0, max_length)
        if para_break > max_length // 2:  # Only use if reasonably far into the chunk
            split_point = para_break + 2  # Include the newlines in the first chunk
        else:
            # Try to split at a single newline
            newline = remaining.rfind("\n", 0, max_length)
            if newline > max_length // 2:
                split_point = newline + 1
            else:
                # Try to split at a space
                space = remaining.rfind(" ", 0, max_length)
                if space > max_length // 2:
                    split_point = space + 1
                # Otherwise just hard split at max_length
        
        chunks.append(remaining[:split_point].rstrip())
        remaining = remaining[split_point:].lstrip()
    
    return chunks


class ChadBot(commands.Bot):
    def __init__(self, settings: Settings, db: Database, processor: RequestProcessor):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.settings = settings
        self.db = db
        self.processor = processor

    async def setup_hook(self) -> None:
        await self.db.connect()
        await self.db.create_schema()
        logger.info("Database ready at %s", self.db.path)
        # Sync slash commands with Discord
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash command(s)")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

    async def close(self) -> None:
        """Close the bot and clean up resources."""
        # Close HTTP clients
        await self.processor.grok.close()
        if self.processor.gemini:
            self.processor.gemini.close()
        # Close database
        await self.db.close()
        # Call parent close
        await super().close()

    async def on_ready(self):
        logger.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "unknown")
        await self.change_presence(activity=discord.Game(name="/ask for questions"))

    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User | discord.Member) -> None:
        """Handle reaction additions. Delete bot messages when admins react with ❌.
        
        This handler works on ANY message sent by the bot, including old messages
        from previous sessions. It doesn't rely on message tracking in memory.
        """
        # Ignore reactions from bots
        if user.bot:
            return
        
        # Only handle ❌ emoji
        if reaction.emoji != "❌":
            return
        
        # Ensure message and guild exist
        if not reaction.message.guild:
            return
        
        # Check if the reacting user is an admin
        guild_id = str(reaction.message.guild.id)
        user_id = str(user.id)
        
        # Check if user is a Discord admin or saved admin
        is_admin = False
        if isinstance(user, discord.Member):
            is_admin = user.guild_permissions and (
                user.guild_permissions.administrator or user.guild_permissions.manage_guild
            )
        
        if not is_admin:
            is_admin = await self.db.is_admin(user_id, guild_id)
        
        # Check admin_user_ids config field
        if not is_admin:
            config = await self.db.get_guild_config(guild_id)
            if config.admin_user_ids:
                admin_ids = [uid.strip() for uid in config.admin_user_ids.split(",") if uid.strip()]
                if user_id in admin_ids:
                    is_admin = True
        
        if not is_admin:
            return
        
        # Check if the message was sent by this bot
        if reaction.message.author != self.user:
            return
        
        try:
            message_id = reaction.message.id
            channel_id = reaction.message.channel.id
            
            # Delete the message
            await reaction.message.delete()
            logger.info(
                "Message %s in channel %s deleted by admin %s (%s) via ❌ reaction",
                message_id,
                channel_id,
                user,
                user_id 
            )
            
            # Try to mark the message as deleted in the database (optional, best effort)
            async with self.db.conn.execute(
                "SELECT id FROM message_log WHERE discord_message_id = ? LIMIT 1",
                (str(message_id),)
            ) as cur:
                row = await cur.fetchone()
                if row:
                    log_id = row["id"]
                    await self.db.mark_message_deleted(log_id)
                    logger.debug("Marked log entry %s as deleted", log_id)
        except discord.Forbidden:
            logger.warning(
                "Missing permissions to delete message %s in channel %s",
                reaction.message.id,
                reaction.message.channel.id
            )
        except discord.NotFound:
            logger.debug("Message %s was already deleted", reaction.message.id)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Error deleting message %s: %s",
                reaction.message.id,
                str(e)
            )


def create_bot(settings: Settings) -> ChadBot:
    db = Database(settings.database_path)
    grok = GrokClient(
        api_key=settings.grok_api_key,
        api_base=settings.grok_api_base,
        chat_model=settings.grok_chat_model,
    )
    from .gemini_client import HAS_GEMINI_SDK
    gemini = None
    if settings.has_gemini:
        if HAS_GEMINI_SDK:
            gemini = GeminiClient(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
            )
        else:
            logger.warning("GEMINI_API_KEY is set but 'google-genai' library is missing. /googl will be disabled.")
    
    yaml_config = YAMLConfig()
    processor = RequestProcessor(db=db, grok=grok, settings=settings, yaml_config=yaml_config, gemini=gemini)
    bot = ChadBot(settings=settings, db=db, processor=processor)

    @bot.tree.command(name="ask", description="Ask a question to the AI")
    @app_commands.describe(question="Your question for the AI")
    async def ask_slash(interaction: discord.Interaction, question: str):
        """Slash command for asking questions."""
        guild_id = str(interaction.guild.id) if interaction.guild else None
        if not guild_id:
            await interaction.response.send_message(yaml_config.get_message("dm_not_allowed"), ephemeral=True)
            return
        
        # Check if user is admin
        is_admin = False
        user_id = str(interaction.user.id)
        
        # Check Discord permissions first
        if interaction.user.guild_permissions and (interaction.user.guild_permissions.administrator or interaction.user.guild_permissions.manage_guild):
            is_admin = True
        # Check admin_users table
        elif await db.is_admin(user_id, guild_id):
            is_admin = True
        # Check admin_user_ids config field
        else:
            config = await db.get_guild_config(guild_id)
            if config.admin_user_ids:
                admin_ids = [uid.strip() for uid in config.admin_user_ids.split(",") if uid.strip()]
                if user_id in admin_ids:
                    is_admin = True
        
        # Defer response as processing might take time
        await interaction.response.defer()
        
        result = await processor.process_chat(
            guild_id=guild_id,
            channel_id=str(interaction.channel.id) if interaction.channel else "",
            user_id=str(interaction.user.id),
            discord_message_id=str(interaction.id),
            content=question,
            is_admin=is_admin,
        )
        
        # Split long responses to fit Discord's character limit
        chunks = split_message(result.reply)
        
        # Send the first chunk and capture the message ID
        response_message = await interaction.followup.send(chunks[0])
        
        # Send remaining chunks as additional messages
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
        
        # Update the database with the actual Discord message ID for better tracking
        if result.log_id and response_message:
            await db.update_discord_message_id(result.log_id, str(response_message.id))
            logger.info(
                "Stored bot response message ID %s for log entry %s",
                response_message.id,
                result.log_id
            )
        
        logger.info("Handled /ask from %s (admin: %s, chunks: %d)", interaction.user.id, is_admin, len(chunks))

    @bot.tree.command(name="googl", description="Search for accurate information using Google")
    @app_commands.describe(query="What do you want to search for?")
    async def googl_slash(interaction: discord.Interaction, query: str):
        """Slash command for searching with Gemini + Google Search grounding."""
        guild_id = str(interaction.guild.id) if interaction.guild else None
        if not guild_id:
            await interaction.response.send_message(yaml_config.get_message("dm_not_allowed"), ephemeral=True)
            return
        
        # Defer response as search might take time
        await interaction.response.defer()
        
        result = await processor.process_search(
            guild_id=guild_id,
            channel_id=str(interaction.channel.id) if interaction.channel else "",
            user_id=str(interaction.user.id),
            discord_message_id=str(interaction.id),
            query=query,
        )
        
        # Split long responses to fit Discord's character limit
        chunks = split_message(result.reply)
        
        # Send the first chunk and capture the message ID
        response_message = await interaction.followup.send(chunks[0])
        
        # Send remaining chunks as additional messages
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
        
        # Update the database with the actual Discord message ID for better tracking
        if result.log_id and response_message:
            await db.update_discord_message_id(result.log_id, str(response_message.id))
            logger.info(
                "Stored bot response message ID %s for log entry %s",
                response_message.id,
                result.log_id
            )
        
        logger.info("Handled /googl from %s (chunks: %d)", interaction.user.id, len(chunks))

    @bot.tree.command(name="sync", description="Force sync slash commands (Restricted)")
    @app_commands.describe(scope="Whether to sync 'global' or 'guild' commands (default: global)")
    async def sync_slash(interaction: discord.Interaction, scope: str = "global"):
        """Restricted command to sync app commands."""
        user_id = str(interaction.user.id)
        
        # Get allowed IDs from YAML config
        allowed_ids = processor.yaml_config.get("bot_settings.sync_allowed_user_ids", [])

        if user_id not in [str(uid) for uid in allowed_ids]:
            await interaction.response.send_message("❌ You are not authorized to use this command.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        
        try:
            if scope.lower() == "guild":
                if not interaction.guild:
                    await interaction.followup.send("❌ This command must be run in a guild to sync guild commands.", ephemeral=True)
                    return
                # Copy global commands to guild
                bot.tree.copy_global_to(guild=interaction.guild)
                synced = await bot.tree.sync(guild=interaction.guild)
                message = f"✅ Synced {len(synced)} command(s) to this guild."
            else:
                synced = await bot.tree.sync()
                message = f"✅ Synced {len(synced)} command(s) globally. (Note: Global changes can take up to an hour to propagate)"
            
            await interaction.followup.send(message, ephemeral=True)
            logger.info("Bot commands synced by %s (scope: %s)", user_id, scope)
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to sync: {e}", ephemeral=True)
            logger.error("Failed to sync commands: %s", e)

    return bot


async def run_bot():
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    bot = create_bot(settings)
    if not settings.discord_token:
        logger.error("DISCORD_BOT_TOKEN is required to start the bot.")
        return
    async with bot:
        await bot.start(settings.discord_token)


if __name__ == "__main__":
    asyncio.run(run_bot())
