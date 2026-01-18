"""Athyr Agent client."""

from __future__ import annotations

from collections.abc import AsyncIterator

from athyr._internal.connection import ConnectionManager
from athyr._internal.llm import LLMOperations
from athyr._internal.messaging import MessagingOperations
from athyr._internal.sessions import SessionOperations
from athyr.client_state import ConnectionState, StateChangeCallback
from athyr.kv import Bucket
from athyr.models import (
    AgentCard,
    CompletionRequest,
    CompletionResponse,
    Model,
    Session,
    SessionProfile,
    StreamChunk,
)
from athyr.protocol import Agent
from athyr.subscription import Subscription

# Re-export for backwards compatibility
__all__ = ["Agent", "AthyrAgent"]


class AthyrAgent:
    """Athyr Agent client implementation.

    This is the main entry point for interacting with Athyr. It provides
    a unified interface for LLM completions, messaging, sessions, and
    key-value storage.

    The client uses composition internally, delegating to specialized
    components for each capability:
    - ConnectionManager: Connection lifecycle and reconnection
    - LLMOperations: LLM completions and model listing
    - MessagingOperations: Pub/sub and request/reply
    - SessionOperations: Memory session management

    Example:
        async with AthyrAgent("localhost:9090", agent_card=card) as agent:
            response = await agent.complete(request)
    """

    def __init__(
        self,
        address: str,
        *,
        agent_card: AgentCard | None = None,
        tls: bool = False,
        connect_timeout: float = 10.0,
        heartbeat_interval: float = 30.0,
        on_state_change: StateChangeCallback | None = None,
        auto_reconnect: bool = False,
        max_retries: int = 0,
        base_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        """Initialize the agent.

        Args:
            address: Athyr server address (host:port).
            agent_card: Agent metadata for registration.
            tls: Use TLS for connection.
            connect_timeout: Connection timeout in seconds.
            heartbeat_interval: Heartbeat interval in seconds (0 to disable).
            on_state_change: Optional callback for connection state changes.
            auto_reconnect: Enable automatic reconnection on connection loss.
            max_retries: Maximum reconnection attempts (0 = infinite).
            base_backoff: Initial backoff duration in seconds (default: 1.0).
            max_backoff: Maximum backoff duration in seconds (default: 30.0).
        """
        # Initialize connection manager
        self._connection = ConnectionManager(
            address,
            agent_card=agent_card or AgentCard(name="python-agent"),
            tls=tls,
            connect_timeout=connect_timeout,
            heartbeat_interval=heartbeat_interval,
            on_state_change=on_state_change,
            auto_reconnect=auto_reconnect,
            max_retries=max_retries,
            base_backoff=base_backoff,
            max_backoff=max_backoff,
        )

        # Initialize operation components (they use connection for stub access)
        self._llm = LLMOperations(self._connection)
        self._messaging = MessagingOperations(self._connection)
        self._sessions = SessionOperations(self._connection)

    # Connection lifecycle

    async def connect(self) -> None:
        """Connect to Athyr server and register the agent."""
        await self._connection.connect()

    async def close(self) -> None:
        """Close the connection and unregister the agent."""
        await self._connection.close()

    def connected(self) -> bool:
        """Check if connected to Athyr."""
        return self._connection.is_connected

    def agent_id(self) -> str:
        """Get the assigned agent ID."""
        return self._connection.agent_id

    @property
    def state(self) -> ConnectionState:
        """Get the current connection state."""
        return self._connection.state

    @property
    def subscriptions(self) -> set[Subscription]:
        """Get active subscriptions."""
        return self._connection.subscriptions

    # Context manager support

    async def __aenter__(self) -> AthyrAgent:
        """Enter async context."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context."""
        await self.close()

    # LLM Operations (delegated to LLMOperations)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Perform a blocking LLM completion.

        Args:
            request: The completion request with model, messages, and options.

        Returns:
            CompletionResponse with the generated content and metadata.

        Raises:
            ConnectionError: If not connected to Athyr.
            InvalidRequestError: If request parameters are invalid.
            UnavailableError: If the LLM service is unavailable.
        """
        return await self._llm.complete(request)

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Perform a streaming LLM completion.

        Args:
            request: The completion request with model, messages, and options.

        Yields:
            StreamChunk objects with incremental content. The final chunk
            has done=True and includes usage statistics.

        Raises:
            ConnectionError: If not connected to Athyr.
            InvalidRequestError: If request parameters are invalid.
            UnavailableError: If the LLM service is unavailable.
            StreamError: If the stream encounters an error.
        """
        async for chunk in self._llm.complete_stream(request):
            yield chunk

    async def models(self) -> list[Model]:
        """Get available LLM models.

        Returns:
            List of available Model objects.

        Raises:
            ConnectionError: If not connected to Athyr.
            UnavailableError: If the service is unavailable.
        """
        return await self._llm.models()

    # Messaging Operations (delegated to MessagingOperations)

    async def publish(self, subject: str, data: bytes) -> None:
        """Publish a message to a subject.

        Args:
            subject: The subject to publish to (e.g., "events.user.created").
            data: The message payload as bytes.

        Raises:
            ConnectionError: If not connected to Athyr.
            AthyrError: If the publish operation fails.

        Example:
            await agent.publish("events.order", b'{"order_id": 123}')
        """
        await self._messaging.publish(subject, data)

    async def subscribe(self, subject: str) -> Subscription:
        """Subscribe to messages on a subject.

        Creates a subscription that receives messages published to the given
        subject. Use the async iterator interface to receive messages.

        Args:
            subject: The subject to subscribe to. Supports wildcards:
                - "events.>" matches "events.user", "events.order.created", etc.
                - "events.*" matches "events.user" but not "events.user.created"

        Returns:
            Subscription object for receiving messages.

        Raises:
            ConnectionError: If not connected to Athyr.
            AthyrError: If the subscribe operation fails.

        Example:
            sub = await agent.subscribe("events.user.>")
            async for msg in sub.messages():
                print(f"Received on {msg.subject}: {msg.data}")
            await sub.unsubscribe()
        """
        return await self._messaging.subscribe(subject)

    async def queue_subscribe(self, subject: str, queue: str) -> Subscription:
        """Subscribe to messages with queue group for load balancing.

        Creates a queue subscription where messages are distributed across
        all subscribers in the same queue group. Only one subscriber in the
        group receives each message.

        Args:
            subject: The subject to subscribe to. Supports wildcards.
            queue: The queue group name. Subscribers with the same queue
                name share message load.

        Returns:
            Subscription object for receiving messages.

        Raises:
            ConnectionError: If not connected to Athyr.
            AthyrError: If the subscribe operation fails.

        Example:
            sub = await agent.queue_subscribe("tasks.process", "workers")
            async for msg in sub.messages():
                process_task(msg.data)
        """
        return await self._messaging.queue_subscribe(subject, queue)

    async def request(
        self,
        subject: str,
        data: bytes,
        timeout: float = 5.0,
    ) -> bytes:
        """Send a request and wait for a response (RPC pattern).

        Sends a message to a subject and waits for a single response.
        This implements the request/reply messaging pattern for
        synchronous-style communication.

        Args:
            subject: The subject to send the request to.
            data: The request payload as bytes.
            timeout: Response timeout in seconds (default: 5.0).

        Returns:
            The response payload as bytes.

        Raises:
            ConnectionError: If not connected to Athyr.
            TimeoutError: If no response received within timeout.
            AthyrError: If the request fails.

        Example:
            response = await agent.request(
                "services.calculator.add",
                b'{"a": 1, "b": 2}',
                timeout=10.0,
            )
            result = json.loads(response)
        """
        return await self._messaging.request(subject, data, timeout)

    # Session Operations (delegated to SessionOperations)

    async def create_session(
        self,
        profile: SessionProfile | None = None,
        system_prompt: str = "",
    ) -> Session:
        """Create a new memory session.

        Creates a conversation session for maintaining context across
        multiple completions. Sessions can be configured with different
        memory profiles (rolling window, summarization, etc.).

        Args:
            profile: Session memory profile configuration. Defaults to
                rolling window with 4000 max tokens.
            system_prompt: Agent personality/instructions for this session.

        Returns:
            Session object with the assigned session ID.

        Raises:
            ConnectionError: If not connected to Athyr.
            AthyrError: If session creation fails.

        Example:
            session = await agent.create_session(
                profile=SessionProfile(max_tokens=8000),
                system_prompt="You are a helpful assistant.",
            )
        """
        return await self._sessions.create(profile, system_prompt)

    async def get_session(self, session_id: str) -> Session:
        """Retrieve an existing session.

        Gets the full session data including messages, summary,
        hints, and profile configuration.

        Args:
            session_id: The session ID returned from create_session.

        Returns:
            Session object with current state.

        Raises:
            ConnectionError: If not connected to Athyr.
            NotFoundError: If session doesn't exist.
            AthyrError: If retrieval fails.

        Example:
            session = await agent.get_session("session-123")
            print(f"Session has {len(session.messages)} messages")
        """
        return await self._sessions.get(session_id)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session.

        Permanently removes the session and all its messages.
        This operation cannot be undone.

        Args:
            session_id: The session ID to delete.

        Raises:
            ConnectionError: If not connected to Athyr.
            NotFoundError: If session doesn't exist.
            AthyrError: If deletion fails.

        Example:
            await agent.delete_session("session-123")
        """
        await self._sessions.delete(session_id)

    async def add_hint(self, session_id: str, hint: str) -> None:
        """Add a hint to a session.

        Hints are persistent notes that influence LLM responses for
        this session. Use hints to provide context that should be
        remembered across conversations.

        Args:
            session_id: The session ID to add the hint to.
            hint: The hint text to add.

        Raises:
            ConnectionError: If not connected to Athyr.
            NotFoundError: If session doesn't exist.
            AthyrError: If the operation fails.

        Example:
            await agent.add_hint(session.id, "User prefers concise responses")
        """
        await self._sessions.add_hint(session_id, hint)

    # KV Storage

    def kv(self, bucket: str) -> Bucket:
        """Get a KV bucket for key-value operations.

        Returns a bucket-scoped wrapper that provides get, put, delete,
        and keys operations without needing to specify the bucket name
        on each call.

        Args:
            bucket: The bucket name.

        Returns:
            Bucket wrapper for KV operations.

        Raises:
            ConnectionError: If not connected to Athyr.

        Example:
            bucket = agent.kv("user-settings")
            await bucket.put("theme", b"dark")
            entry = await bucket.get("theme")
            print(entry.value)  # b"dark"
        """
        stub = self._connection.get_stub()
        return Bucket(
            bucket=bucket,
            agent_id=self._connection.agent_id,
            stub=stub,
            connected_check=lambda: self._connection.get_stub(),
        )
