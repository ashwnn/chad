import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

class DiscordApiClient:
    def __init__(self, token: Optional[str]):
        self.token = token
        self.base_url = "https://discord.com/api/v10"
        # Persistent HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the persistent async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=15.0,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_guild(self, guild_id: str) -> Optional[Dict[str, Any]]:
        """Fetch guild information from Discord API."""
        if not self.token:
            logger.warning("Discord token missing, skipping guild fetch")
            return None
        
        client = await self._get_client()
        try:
            resp = await client.get(
                f"/guilds/{guild_id}",
                headers={"Authorization": f"Bot {self.token}"},
            )
            if resp.status_code >= 400:
                logger.error("Failed fetching guild %s: %s - %s", guild_id, resp.status_code, resp.text)
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error fetching guild %s: %s", guild_id, exc)
            return None

    async def send_message(
        self,
        *,
        channel_id: str,
        content: str,
        mention_user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.token:
            logger.warning("Discord token missing, skipping send")
            return None
        
        payload: Dict[str, Any] = {}
        
        # Build the message content with optional mention
        if mention_user_id:
            payload["content"] = f"<@{mention_user_id}> {content}"
        else:
            payload["content"] = content
        
        client = await self._get_client()
        resp = await client.post(
            f"/channels/{channel_id}/messages",
            headers={"Authorization": f"Bot {self.token}"},
            json=payload,
        )
        if resp.status_code >= 400:
            logger.error("Failed sending message to Discord: %s - %s", resp.status_code, resp.text)
        resp.raise_for_status()
        return resp.json()

    async def delete_message(self, channel_id: str, message_id: str) -> bool:
        """Delete a message from Discord.
        
        Args:
            channel_id: Discord channel ID
            message_id: Discord message ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.token:
            logger.warning("Discord token missing, skipping delete")
            return False
        
        client = await self._get_client()
        try:
            resp = await client.delete(
                f"/channels/{channel_id}/messages/{message_id}",
                headers={"Authorization": f"Bot {self.token}"},
            )
            if resp.status_code >= 400:
                logger.error("Failed deleting message from Discord: %s - %s", resp.status_code, resp.text)
                return False
            resp.raise_for_status()
            logger.info("Successfully deleted message %s from channel %s", message_id, channel_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Error deleting message %s: %s", message_id, exc)
            return False

    async def get_guild_roles(self, guild_id: str) -> Optional[list]:
        """Fetch guild roles."""
        if not self.token:
            return None
        
        client = await self._get_client()
        try:
            resp = await client.get(
                f"/guilds/{guild_id}/roles",
                headers={"Authorization": f"Bot {self.token}"},
            )
            if resp.status_code >= 400:
                logger.error("Failed fetching roles for guild %s: %s", guild_id, resp.status_code)
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Error fetching roles for guild %s: %s", guild_id, exc)
            return None

    async def get_guild_channels(self, guild_id: str) -> Optional[list]:
        """Fetch guild channels, returning only text channels."""
        if not self.token:
            return None
        
        client = await self._get_client()
        try:
            resp = await client.get(
                f"/guilds/{guild_id}/channels",
                headers={"Authorization": f"Bot {self.token}"},
            )
            if resp.status_code >= 400:
                logger.error("Failed fetching channels for guild %s: %s", guild_id, resp.status_code)
                return None
            resp.raise_for_status()
            channels = resp.json()
            # Filter to text channels only (type 0 = text channel)
            # Also include type 5 (announcement channels) as they support text messages
            text_channels = [
                ch for ch in channels 
                if ch.get("type") in (0, 5)
            ]
            # Sort by position
            text_channels.sort(key=lambda x: x.get("position", 0))
            return text_channels
        except Exception as exc:
            logger.error("Error fetching channels for guild %s: %s", guild_id, exc)
            return None

    async def get_bot_member(self, guild_id: str) -> Optional[Dict[str, Any]]:
        """Fetch bot member in guild."""
        if not self.token:
            return None
            
        client = await self._get_client()
        try:
            resp = await client.get(
                f"/guilds/{guild_id}/members/@me",
                headers={"Authorization": f"Bot {self.token}"},
            )
            if resp.status_code >= 400:
                logger.error("Failed fetching bot member for guild %s: %s", guild_id, resp.status_code)
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Error fetching bot member for guild %s: %s", guild_id, exc)
            return None

    async def get_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Fetch channel information."""
        if not self.token:
            return None
            
        client = await self._get_client()
        try:
            resp = await client.get(
                f"/channels/{channel_id}",
                headers={"Authorization": f"Bot {self.token}"},
            )
            if resp.status_code >= 400:
                logger.error("Failed fetching channel %s: %s", channel_id, resp.status_code)
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Error fetching channel %s: %s", channel_id, exc)
            return None

    def _compute_base_permissions(self, member: Dict[str, Any], roles: list) -> int:
        """Compute base permissions for a member in a guild."""
        # Permissions constants
        ADMINISTRATOR = 1 << 3
        
        # Create role map for easy lookup
        role_map = {r["id"]: int(r["permissions"]) for r in roles}
        guild_id = member.get("guild_id") # Note: member object from API might not have guild_id if not enriched, but we usually pass guild_id to context
        
        permissions = 0
        
        # Get guild_id from roles if possible (usually @everyone role has same ID as guild)
        # But here we need to be careful. The caller should ensure we have the right context.
        # In get_guild_roles, we get all roles. The one with id == guild_id is @everyone.
        
        # Find @everyone role (id == guild_id)
        # We don't have guild_id explicitly here, but we can infer or pass it.
        # Actually, simpler: The caller passes guild_id.
        pass
        
    async def calculate_permissions(self, guild_id: str, channel_id: Optional[str] = None) -> int:
        """Calculate permissions for the bot in a guild (and optionally a channel)."""
        roles = await self.get_guild_roles(guild_id)
        member = await self.get_bot_member(guild_id)
        
        if not roles or not member:
            return 0
            
        # Permissions constants
        ADMINISTRATOR = 1 << 3
        
        # Create role map
        role_map = {r["id"]: int(r["permissions"]) for r in roles}
        
        # 1. Base permissions
        permissions = 0
        
        # Add @everyone permissions
        if guild_id in role_map:
            permissions |= role_map[guild_id]
            
        # Add member role permissions
        for role_id in member.get("roles", []):
            if role_id in role_map:
                permissions |= role_map[role_id]
        
        # If Administrator, return all permissions (roughly)
        # Discord says Administrator overrides everything EXCEPT channel-specific overrides? 
        # No, Administrator overrides ALL channel overrides.
        if permissions & ADMINISTRATOR:
            return -1 # All permissions (conceptually)
            
        # If no channel provided, return guild permissions
        if not channel_id:
            return permissions
            
        # 2. Channel Overwrites
        channel = await self.get_channel(channel_id)
        if not channel:
            return permissions
            
        overwrites = channel.get("permission_overwrites", [])
        
        # Apply @everyone overwrite
        everyone_allow = 0
        everyone_deny = 0
        for ow in overwrites:
            if ow["id"] == guild_id: # @everyone
                everyone_allow = int(ow["allow"])
                everyone_deny = int(ow["deny"])
                break
        
        permissions &= ~everyone_deny
        permissions |= everyone_allow
        
        # Apply role overwrites
        role_allow = 0
        role_deny = 0
        member_roles = set(member.get("roles", []))
        
        for ow in overwrites:
            if ow["type"] == 0 and ow["id"] in member_roles: # Role overwrite
                role_allow |= int(ow["allow"])
                role_deny |= int(ow["deny"])
                
        permissions &= ~role_deny
        permissions |= role_allow
        
        # Apply member overwrite
        member_allow = 0
        member_deny = 0
        user_id = member["user"]["id"]
        
        for ow in overwrites:
            if ow["type"] == 1 and ow["id"] == user_id: # Member overwrite
                member_allow = int(ow["allow"])
                member_deny = int(ow["deny"])
                break
                
        permissions &= ~member_deny
        permissions |= member_allow
        
        return permissions
