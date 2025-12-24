"""
YAML configuration manager for bot messages and settings.
Handles loading, updating, and saving YAML configuration files.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class YAMLConfig:
    """Manages YAML configuration file for bot messages and settings."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            self._config = self._get_defaults()
            self.save()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            self._config = self._get_defaults()
    
    def save(self) -> None:
        """Save current configuration to YAML file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation key (e.g., 'messages.empty_input')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by dot-notation key."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values and save."""
        for key, value in updates.items():
            self.set(key, value)
        self.save()
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self._config.copy()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Return default configuration values."""
        return {
            "bot_settings": {
                "reply_prefix": "",
                "reply_suffix": ""
            },
            "system_prompt": (
                "You are Chad, a Discord AI assistant. Always answer the user's question directly and concisely. "
                "Lead with the helpful answer, then optionally add one short sarcastic or blunt comment. "
                "Tone can be mildly rude but never hateful. Avoid slurs, protected class insults, explicit "
                "sexual content, or graphic violence. If the user prompt is unclear, spammy, or misuses "
                "commands, call it out and tell them briefly what to do instead."
            ),
            "messages": {
                "empty_input": "Try sending an actual question instead of blank air.",
                "too_short": "That barely qualifies as a question. Add some words.",
                "trivial_input": "Wow, groundbreaking. Try a real question.",
                "gibberish": "That looks like keyboard smash. Try again.",
                "too_long": "Message is too long. Trim it under {max_chars} characters.",
                "duplicate": "You literally just asked that. Wait a bit.",
                "rate_limit_chat": "Cool it. You hit the spam limit. Try again later.",
                "rate_limit_image": "Too many image requests. Take a breather.",
                "chat_budget_user": "Your daily chat budget is toast. Ask again tomorrow.",
                "chat_budget_guild": "This guild used up the chat budget for today. Cool your jets.",
                "image_budget_user": "You hit the image quota for today.",
                "image_budget_guild": "Your server burned through the image budget today.",
                "pending_approval_chat": "Your request is waiting for an admin to approve.",
                "pending_approval_image": "Image request queued for admin approval.",
                "image_generated": "Here's your image.",
                "image_approved": "Here's your approved image.",
                "grok_error_chat": "Grok had a meltdown. Try again later.",
                "grok_error_image": "Image service failed. Try later.",
                "no_image_url": "Image generated, but no URL returned.",
                "dm_not_allowed": "This only works in servers, not DMs.",
                "manual_reply_default": "Admin reply.",
                "rejection_default": "Request rejected by an admin.",
                "invalid_input": "Invalid input.",
                "unknown_error": "Something went wrong. Try again."
            }
        }
    
    # Convenience methods for common operations
    def get_message(self, key: str, **kwargs) -> str:
        """Get a message and optionally format it with kwargs."""
        msg = self.get(f"messages.{key}", "Error: Message not found")
        if kwargs:
            try:
                return msg.format(**kwargs)
            except KeyError:
                return msg
        return msg
    
    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.get("system_prompt", "")
    
    def get_reply_prefix(self) -> str:
        """Get the reply prefix."""
        return self.get("bot_settings.reply_prefix", "")
    
    def get_reply_suffix(self) -> str:
        """Get the reply suffix."""
        return self.get("bot_settings.reply_suffix", "")
    
    def format_reply(self, content: str) -> str:
        """Format a reply with prefix and suffix if configured."""
        prefix = self.get_reply_prefix()
        suffix = self.get_reply_suffix()
        
        parts = []
        if prefix:
            parts.append(prefix)
        parts.append(content)
        if suffix:
            parts.append(suffix)
        
        return " ".join(parts)
