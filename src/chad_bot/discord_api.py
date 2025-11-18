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
        embed_url: Optional[str] = None,
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
        
        # Add embed if image URL provided
        if embed_url:
            payload["embeds"] = [
                {
                    "image": {
                        "url": embed_url
                    },
                    "color": 2563755  # Discord blue hex #2563eb
                }
            ]
            # Don't duplicate content in embed if already in message
            if not mention_user_id:
                payload["embeds"][0]["description"] = content
        
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
