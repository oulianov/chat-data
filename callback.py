"""Callback handlers used in the app."""
import asyncio

from typing import Any, Dict, List
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.docstore.document import Document
from schemas import ChatResponse


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream")
        await self.websocket.send_json(resp.dict())


class SendSourceCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def send_sources(self, sources: List[Document], **kwargs: Any) -> None:
        # Just send the source name (for now)
        # Deduplicate
        sources_names = list(set([s.metadata["source"] for s in sources]))
        resp = ChatResponse(
            sender="bot", message="", sources=sources_names, type="stream"
        )
        await self.websocket.send_json(resp.dict())


class QuestionGenCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        resp = ChatResponse(
            sender="bot", message="Synthesizing question...", type="info"
        )
        await self.websocket.send_json(resp.dict())


class InstaRepCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_rep(self, message: str, **kwargs: Any) -> None:
        # Make a nice scrolling effect by sending it word by word
        all_words = message.split(" ")
        for word in all_words:
            resp = ChatResponse(sender="bot", message=word + " ", type="stream")
            await self.websocket.send_json(resp.dict())
            await asyncio.sleep(0.05)
