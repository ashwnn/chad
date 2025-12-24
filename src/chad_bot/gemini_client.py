from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_SDK = True
except (ImportError, ModuleNotFoundError):
    HAS_GEMINI_SDK = False

@dataclass
class SearchResult:
    """Result from a Gemini grounded search query."""
    content: str
    grounding_metadata: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None


class GeminiClient:
    """Client for Google Gemini API with grounded search capability."""

    def __init__(self, *, api_key: Optional[str], model: str = "gemini-2.5-flash-lite"):
        self.api_key = api_key
        self.model = model
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create the Gemini client."""
        if not HAS_GEMINI_SDK:
            raise ImportError("The 'google-genai' library is not installed. Please install it with 'pip install google-genai'.")
        
        if self._client is None:
            if not self.api_key:
                raise ValueError("Gemini API key is required")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def search(self, query: str) -> SearchResult:
        """
        Perform a grounded search using Google Gemini with Google Search tool.
        
        Args:
            query: The search query from the user.
            
        Returns:
            SearchResult with the AI-generated response grounded in search results.
        """
        if not self.api_key:
            return SearchResult(
                content="[Search unavailable - GEMINI_API_KEY is not configured]",
                grounding_metadata=None,
                usage=None,
            )

        client = self._get_client()
        
        # Configure the request with Google Search grounding tool
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        # Make the API call (the SDK handles async internally)
        response = client.models.generate_content(
            model=self.model,
            contents=query,
            config=config,
        )

        # Extract the text response
        content = ""
        if response.text:
            content = response.text
        elif response.candidates and response.candidates[0].content:
            # Fallback to extracting from parts
            parts = response.candidates[0].content.parts
            if parts:
                content = "".join(part.text for part in parts if hasattr(part, 'text') and part.text)

        # Extract grounding metadata if available
        grounding_metadata = None
        if response.candidates and response.candidates[0].grounding_metadata:
            gm = response.candidates[0].grounding_metadata
            grounding_metadata = {
                "search_entry_point": getattr(gm, 'search_entry_point', None),
                "grounding_chunks": getattr(gm, 'grounding_chunks', None),
                "grounding_supports": getattr(gm, 'grounding_supports', None),
            }

        # Extract usage metadata if available
        usage = None
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return SearchResult(
            content=content,
            grounding_metadata=grounding_metadata,
            usage=usage,
        )

    def close(self) -> None:
        """Clean up resources. The genai client doesn't require explicit cleanup."""
        self._client = None
