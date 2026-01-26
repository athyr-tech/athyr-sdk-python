"""Protocol definitions for Athyr agents."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from athyr.kv import Bucket
from athyr.models import (
    CompletionRequest,
    CompletionResponse,
    Model,
    Session,
    SessionProfile,
    StreamChunk,
)
from athyr.subscription import Subscription


@runtime_checkable
class Agent(Protocol):
    """Protocol defining the Agent interface.

    This protocol defines the contract that all Athyr agent implementations
    must fulfill. It can be used for type checking and dependency injection.

    Example:
        def process_with_agent(agent: Agent) -> None:
            # Works with any Agent implementation
            response = await agent.complete(request)
    """

    # Connection

    async def connect(self) -> None:
        """Connect to Athyr server."""
        ...

    async def close(self) -> None:
        """Close the connection."""
        ...

    def connected(self) -> bool:
        """Check if connected."""
        ...

    def agent_id(self) -> str:
        """Get the agent ID."""
        ...

    # Messaging

    async def publish(self, subject: str, data: bytes) -> None:
        """Publish a message to a subject."""
        ...

    async def subscribe(self, subject: str) -> Subscription:
        """Subscribe to messages on a subject."""
        ...

    async def queue_subscribe(self, subject: str, queue: str) -> Subscription:
        """Subscribe with queue group for load balancing."""
        ...

    async def request(self, subject: str, data: bytes, timeout: float = 5.0) -> bytes:
        """Send request and wait for response (RPC pattern)."""
        ...

    # LLM Operations

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Perform a blocking LLM completion."""
        ...

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Perform a streaming LLM completion."""
        ...

    async def models(self) -> list[Model]:
        """Get available LLM models."""
        ...

    # Memory Operations

    async def create_session(
        self,
        profile: SessionProfile | None = None,
        system_prompt: str = "",
    ) -> Session:
        """Create a new memory session."""
        ...

    async def get_session(self, session_id: str) -> Session:
        """Retrieve an existing session."""
        ...

    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        ...

    async def add_hint(self, session_id: str, hint: str) -> None:
        """Add a hint to a session."""
        ...

    # KV Storage

    def kv(self, bucket: str) -> Bucket:
        """Get a KV bucket for key-value operations."""
        ...
