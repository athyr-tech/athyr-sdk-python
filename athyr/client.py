"""Athyr Agent client."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

import grpc

from athyr._grpc import athyr_pb2, athyr_pb2_grpc
from athyr.exceptions import AthyrError, ConnectionError, from_grpc_error
from athyr.models import AgentCard, CompletionRequest, CompletionResponse, Model


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

    def _ensure_connected(self) -> None:
        """Ensure the agent is connected."""
        if not self._connected:
            raise ConnectionError("Not connected to Athyr")
        if not self._stub:
            raise ConnectionError("No stub available")

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
        self._ensure_connected()

        try:
            proto_request = request.to_proto(agent_id=self._agent_id)
            proto_response = await self._stub.Complete(proto_request)
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
        self._ensure_connected()

        try:
            response = await self._stub.ListModels(athyr_pb2.ListModelsRequest())
            return [Model.from_proto(m) for m in response.models]
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e