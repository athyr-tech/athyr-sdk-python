"""Athyr Agent client."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

import grpc

from athyr._grpc import athyr_pb2, athyr_pb2_grpc
from athyr.exceptions import ConnectionError, from_grpc_error
from athyr.models import (
    AgentCard,
    CompletionRequest,
    CompletionResponse,
    Model,
    Session,
    SessionProfile,
    StreamChunk,
)
from athyr.subscription import Subscription

logger = logging.getLogger(__name__)


@runtime_checkable
class Agent(Protocol):
    """Protocol defining the Agent interface."""

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


class AthyrAgent:
    """Athyr Agent client implementation.

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
    ) -> None:
        """Initialize the agent.

        Args:
            address: Athyr server address (host:port).
            agent_card: Agent metadata for registration.
            tls: Use TLS for connection.
            connect_timeout: Connection timeout in seconds.
            heartbeat_interval: Heartbeat interval in seconds (0 to disable).
        """
        self._address = address
        self._agent_card = agent_card or AgentCard(name="python-agent")
        self._tls = tls
        self._connect_timeout = connect_timeout
        self._heartbeat_interval = heartbeat_interval

        self._channel: grpc.aio.Channel | None = None
        self._stub: athyr_pb2_grpc.AthyrStub | None = None
        self._agent_id: str = ""
        self._connected: bool = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._subscriptions: set[Subscription] = set()

    async def connect(self) -> None:
        """Connect to Athyr server and register the agent."""
        if self._connected:
            return

        try:
            # Create gRPC channel
            if self._tls:
                self._channel = grpc.aio.secure_channel(
                    self._address,
                    grpc.ssl_channel_credentials(),
                )
            else:
                self._channel = grpc.aio.insecure_channel(self._address)

            self._stub = athyr_pb2_grpc.AthyrStub(self._channel)

            # Connect and register
            request = athyr_pb2.ConnectRequest(
                agent_card=self._agent_card.to_proto(),
            )

            response = await asyncio.wait_for(
                self._stub.Connect(request),
                timeout=self._connect_timeout,
            )

            self._agent_id = response.agent_id
            self._connected = True

            logger.info(f"Connected to Athyr as {self._agent_id}")

            # Start heartbeat
            if self._heartbeat_interval > 0:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        except asyncio.TimeoutError as e:
            await self._cleanup()
            raise ConnectionError(
                f"Connection to {self._address} timed out",
                code="TIMEOUT",
            ) from e
        except grpc.RpcError as e:
            await self._cleanup()
            raise from_grpc_error(e) from e
        except Exception as e:
            await self._cleanup()
            raise ConnectionError(
                f"Failed to connect to {self._address}: {e}",
            ) from e

    async def close(self) -> None:
        """Close the connection and unregister the agent."""
        if not self._connected:
            return

        try:
            # Cancel all active subscriptions
            for sub in list(self._subscriptions):
                await sub.unsubscribe()
            self._subscriptions.clear()

            # Stop heartbeat
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None

            # Disconnect
            if self._stub and self._agent_id:
                try:
                    await self._stub.Disconnect(
                        athyr_pb2.DisconnectRequest(agent_id=self._agent_id)
                    )
                except Exception as e:
                    logger.warning(f"Disconnect failed: {e}")

        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._connected = False
        self._agent_id = ""
        self._stub = None

        if self._channel:
            await self._channel.close()
            self._channel = None

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self._connected:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                if self._stub and self._agent_id:
                    await self._stub.Heartbeat(
                        athyr_pb2.HeartbeatRequest(agent_id=self._agent_id)
                    )
                    logger.debug("Heartbeat sent")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

    def connected(self) -> bool:
        """Check if connected to Athyr."""
        return self._connected

    def agent_id(self) -> str:
        """Get the assigned agent ID."""
        return self._agent_id

    @property
    def subscriptions(self) -> set[Subscription]:
        """Get active subscriptions."""
        return self._subscriptions

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

    def _ensure_connected(self) -> athyr_pb2_grpc.AthyrStub:
        """Ensure the agent is connected and return the stub.

        Returns:
            The gRPC stub for making API calls.

        Raises:
            ConnectionError: If not connected or stub unavailable.
        """
        if not self._connected:
            raise ConnectionError("Not connected to Athyr")
        if not self._stub:
            raise ConnectionError("No stub available")
        return self._stub

    # LLM Operations

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
        stub = self._ensure_connected()

        try:
            proto_request = request.to_proto(agent_id=self._agent_id)
            proto_response = await stub.Complete(proto_request)
            return CompletionResponse.from_proto(proto_response)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    async def models(self) -> list[Model]:
        """Get available LLM models.

        Returns:
            List of available Model objects.

        Raises:
            ConnectionError: If not connected to Athyr.
            UnavailableError: If the service is unavailable.
        """
        stub = self._ensure_connected()

        try:
            response = await stub.ListModels(athyr_pb2.ListModelsRequest())
            return [Model.from_proto(m) for m in response.models]
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

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
        stub = self._ensure_connected()

        try:
            proto_request = request.to_proto(agent_id=self._agent_id)
            stream = stub.CompleteStream(proto_request)
            async for chunk in stream:
                yield StreamChunk.from_proto(chunk)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    # Messaging Operations

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
        stub = self._ensure_connected()

        try:
            await stub.Publish(
                athyr_pb2.PublishRequest(
                    agent_id=self._agent_id,
                    subject=subject,
                    data=data,
                )
            )
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

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
        return await self._subscribe(subject, queue="")

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
            # Multiple workers sharing load
            sub = await agent.queue_subscribe("tasks.process", "workers")
            async for msg in sub.messages():
                process_task(msg.data)
        """
        return await self._subscribe(subject, queue=queue)

    async def _subscribe(self, subject: str, queue: str) -> Subscription:
        """Internal subscribe implementation.

        Args:
            subject: The subject to subscribe to.
            queue: Queue group name (empty string for regular subscription).

        Returns:
            Subscription object.
        """
        stub = self._ensure_connected()

        try:
            stream = stub.Subscribe(
                athyr_pb2.SubscribeRequest(
                    agent_id=self._agent_id,
                    subject=subject,
                    queue_group=queue,
                )
            )

            sub = Subscription(
                subject=subject,
                stream=stream,
                cancel_fn=stream.cancel,
            )

            self._subscriptions.add(sub)
            return sub
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

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
        stub = self._ensure_connected()

        try:
            response = await stub.Request(
                athyr_pb2.RequestMessage(
                    agent_id=self._agent_id,
                    subject=subject,
                    data=data,
                    timeout_ms=int(timeout * 1000),
                )
            )
            return response.data
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    # Memory Session Operations

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
            # Use session.id in completion requests
        """
        stub = self._ensure_connected()

        try:
            request = athyr_pb2.CreateSessionRequest(
                agent_id=self._agent_id,
                profile=profile.to_proto() if profile else None,
                system_prompt=system_prompt,
            )
            response = await stub.CreateSession(request)
            return Session.from_proto(response)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

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
        stub = self._ensure_connected()

        try:
            request = athyr_pb2.GetSessionRequest(
                agent_id=self._agent_id,
                session_id=session_id,
            )
            response = await stub.GetSession(request)
            return Session.from_proto(response)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

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
        stub = self._ensure_connected()

        try:
            request = athyr_pb2.DeleteSessionRequest(
                agent_id=self._agent_id,
                session_id=session_id,
            )
            await stub.DeleteSession(request)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e
