"""
Chad Discord bot and admin web UI.

Run the bot with:
    python -m chad_bot.bot

Run the web UI with:
    uvicorn chad_bot.web:app
"""

__all__ = [
    "bot",
    "web",
    "service",
    "database",
    "config",
    "rate_limits",
    "spam",
    "grok_client",
    "gemini_client",
    "yaml_config",
    "discord_api",
]
