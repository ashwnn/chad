import asyncio
import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from .config import Settings
from .database import Database
from .grok_client import GrokClient
from .service import RequestProcessor
from .yaml_config import YAMLConfig

logger = logging.getLogger(__name__)


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
        # Close database
        await self.db.close()
        # Call parent close
        await super().close()

    async def on_ready(self):
        logger.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "unknown")
        await self.change_presence(activity=discord.Game(name="/ask for questions"))

    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User | discord.Member) -> None:
        """Handle reaction additions. Delete bot messages when admins react with ❌."""
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
        
        if not is_admin:
            return
        
        # Check if the message was sent by this bot
        if reaction.message.author != self.user:
            return
        
        try:
            await reaction.message.delete()
            logger.info(
                "Message %s deleted by admin %s (%s) via ❌ reaction",
                reaction.message.id,
                user,
                user_id 
            )
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


def _guild_id(ctx: commands.Context) -> Optional[str]:
    return str(ctx.guild.id) if ctx.guild else None


def _channel_id(ctx: commands.Context) -> str:
    return str(ctx.channel.id)


def _user_id(ctx: commands.Context) -> str:
    return str(ctx.author.id)


def _is_admin_user(ctx: commands.Context) -> bool:
    perms = ctx.author.guild_permissions if hasattr(ctx.author, "guild_permissions") else None
    return bool(perms and (perms.administrator or perms.manage_guild))


def _admin_label(ctx: commands.Context) -> str:
    return " (admin)" if _is_admin_user(ctx) else ""


async def _determine_admin(db: Database, ctx: commands.Context) -> bool:
    guild_id = _guild_id(ctx)
    if not guild_id:
        return False
    db_admin = await db.is_admin(_user_id(ctx), guild_id)
    return db_admin or _is_admin_user(ctx)


def create_bot(settings: Settings) -> ChadBot:
    db = Database(settings.database_path)
    grok = GrokClient(
        api_key=settings.grok_api_key,
        api_base=settings.grok_api_base,
        chat_model=settings.grok_chat_model,
        image_model=settings.grok_image_model,
    )
    yaml_config = YAMLConfig()
    processor = RequestProcessor(db=db, grok=grok, settings=settings, yaml_config=yaml_config)
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
        if interaction.user.guild_permissions and (interaction.user.guild_permissions.administrator or interaction.user.guild_permissions.manage_guild):
            is_admin = True
        else:
            is_admin = await db.is_admin(str(interaction.user.id), guild_id)
        
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
        
        # Send the response
        await interaction.followup.send(result.reply)
        logger.info("Handled /ask from %s (admin: %s)", interaction.user.id, is_admin)

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
