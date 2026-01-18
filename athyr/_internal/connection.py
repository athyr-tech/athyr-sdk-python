"""Connection management for Athyr agents."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

import grpc

from athyr._grpc import athyr_pb2, athyr_pb2_grpc
from athyr.client_state import ConnectionState, ConnectionStateMachine, StateChangeCallback
from athyr.exceptions import ConnectionError
from athyr.models import AgentCard

if TYPE_CHECKING:
    from athyr.subscription import Subscription

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages gRPC connection lifecycle for Athyr agents.

    Handles:
    - Connection establishment (TLS/insecure)
    - Agent registration
    - Heartbeat loop
    - Auto-reconnection with exponential backoff
    - Subscription tracking for reconnection

    Example:
        connection = ConnectionManager("localhost:9090", agent_card=card)
        await connection.connect()
        stub = connection.get_stub()
        await connection.close()
    """

    def __init__(
        self,
        address: str,
        *,
        agent_card: AgentCard,
        tls: bool = False,
        connect_timeout: float = 10.0,
        heartbeat_interval: float = 30.0,
        on_state_change: StateChangeCallback | None = None,
        auto_reconnect: bool = False,
        max_retries: int = 0,
        base_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        """Initialize the connection manager.

        Args:
            address: Athyr server address (host:port).
            agent_card: Agent metadata for registration.
            tls: Use TLS for connection.
            connect_timeout: Connection timeout in seconds.
            heartbeat_interval: Heartbeat interval in seconds (0 to disable).
            on_state_change: Optional callback for connection state changes.
            auto_reconnect: Enable automatic reconnection on connection loss.
            max_retries: Maximum reconnection attempts (0 = infinite).
            base_backoff: Initial backoff duration in seconds.
            max_backoff: Maximum backoff duration in seconds.
        """
        self._address = address
        self._agent_card = agent_card
        self._tls = tls
        self._connect_timeout = connect_timeout
        self._heartbeat_interval = heartbeat_interval

        # Auto-reconnect configuration
        self._auto_reconnect = auto_reconnect
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff

        # Connection state
        self._channel: grpc.aio.Channel | None = None
        self._stub: athyr_pb2_grpc.AthyrStub | None = None
        self._agent_id: str = ""
        self._state_machine = ConnectionStateMachine(on_state_change=on_state_change)

        # Background tasks
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None

        # Subscription tracking for reconnection
        self._subscriptions: set[Subscription] = set()

    # Properties

    @property
    def agent_id(self) -> str:
        """Get the assigned agent ID."""
        return self._agent_id

    @property
    def is_connected(self) -> bool:
        """Check if connected to Athyr."""
        return self._state_machine.is_connected

    @property
    def state(self) -> ConnectionState:
        """Get the current connection state."""
        return self._state_machine.state

    @property
    def subscriptions(self) -> set[Subscription]:
        """Get tracked subscriptions."""
        return self._subscriptions

    def get_stub(self) -> athyr_pb2_grpc.AthyrStub:
        """Get the gRPC stub, ensuring connection is active.

        Returns:
            The gRPC stub for making API calls.

        Raises:
            ConnectionError: If not connected or stub unavailable.
        """
        if not self._state_machine.is_connected:
            raise ConnectionError("Not connected to Athyr")
        if not self._stub:
            raise ConnectionError("No stub available")
        return self._stub

    # Connection lifecycle

    async def connect(self) -> None:
        """Connect to Athyr server and register the agent."""
        if self._state_machine.is_connected:
            return

        await self._state_machine.set_state(ConnectionState.CONNECTING)

        try:
            self._channel = self._create_channel()
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
            await self._state_machine.set_state(ConnectionState.CONNECTED)

            logger.info(f"Connected to Athyr as {self._agent_id}")

            # Start heartbeat
            if self._heartbeat_interval > 0:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        except asyncio.TimeoutError as e:
            await self._cleanup(error=e)
            raise ConnectionError(
                f"Connection to {self._address} timed out",
                code="TIMEOUT",
            ) from e
        except grpc.RpcError as e:
            await self._cleanup(error=e)
            from athyr.exceptions import from_grpc_error
            raise from_grpc_error(e) from e
        except Exception as e:
            await self._cleanup(error=e)
            raise ConnectionError(
                f"Failed to connect to {self._address}: {e}",
            ) from e

    async def close(self) -> None:
        """Close the connection and unregister the agent."""
        # Cancel any in-progress reconnection
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        # Check if we need to disconnect
        if not self._state_machine.is_connected and \
           self._state_machine.state != ConnectionState.RECONNECTING:
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

    def _create_channel(self) -> grpc.aio.Channel:
        """Create a gRPC channel based on TLS configuration."""
        if self._tls:
            return grpc.aio.secure_channel(
                self._address,
                grpc.ssl_channel_credentials(),
            )
        else:
            return grpc.aio.insecure_channel(self._address)

    async def _cleanup(self, error: Exception | None = None) -> None:
        """Clean up connection resources."""
        await self._state_machine.set_state(ConnectionState.DISCONNECTED, error)
        self._agent_id = ""
        self._stub = None

        if self._channel:
            await self._channel.close()
            self._channel = None

    # Heartbeat

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self._state_machine.is_connected:
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
                if self._auto_reconnect:
                    self._trigger_reconnect(e)
                    return

    # Auto-reconnect

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        backoff = min(self._base_backoff * (2**attempt), self._max_backoff)
        jitter = 0.5 + random.random()
        return backoff * jitter

    def _trigger_reconnect(self, error: Exception) -> None:
        """Start the reconnection process if not already reconnecting."""
        if self._state_machine.state == ConnectionState.RECONNECTING:
            return

        # Mark all subscriptions as reconnecting
        for sub in self._subscriptions:
            sub._set_reconnecting()

        self._state_machine.set_state_sync(ConnectionState.RECONNECTING, error)
        self._reconnect_task = asyncio.create_task(self._reconnect_loop(error))

    async def _reconnect_loop(self, initial_error: Exception) -> None:
        """Attempt to reconnect with exponential backoff."""
        attempt = 0
        last_error = initial_error

        while True:
            if self._max_retries > 0 and attempt >= self._max_retries:
                logger.error(
                    f"Max reconnection attempts reached ({attempt}), "
                    f"last error: {last_error}"
                )
                await self._state_machine.set_state(
                    ConnectionState.DISCONNECTED, last_error
                )
                return

            backoff = self._calculate_backoff(attempt)
            logger.info(
                f"Reconnecting in {backoff:.2f}s (attempt {attempt + 1}"
                + (f"/{self._max_retries})" if self._max_retries > 0 else ")")
            )

            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                logger.info("Reconnection cancelled")
                return

            try:
                await self._do_reconnect()
                logger.info(f"Reconnected successfully after {attempt + 1} attempt(s)")
                return
            except asyncio.CancelledError:
                logger.info("Reconnection cancelled")
                return
            except Exception as e:
                last_error = e
                attempt += 1
                logger.warning(f"Reconnection attempt {attempt} failed: {e}")

    async def _do_reconnect(self) -> None:
        """Perform the actual reconnection."""
        # Close existing channel
        if self._channel:
            try:
                await self._channel.close()
            except Exception:
                pass
            self._channel = None
        self._stub = None

        # Create new channel
        self._channel = self._create_channel()
        self._stub = athyr_pb2_grpc.AthyrStub(self._channel)

        # Re-register agent
        request = athyr_pb2.ConnectRequest(
            agent_card=self._agent_card.to_proto(),
        )

        response = await asyncio.wait_for(
            self._stub.Connect(request),
            timeout=self._connect_timeout,
        )

        self._agent_id = response.agent_id

        # Re-subscribe all tracked subscriptions
        await self._resubscribe_all()

        await self._state_machine.set_state(ConnectionState.CONNECTED)
        self._reconnect_task = None

        # Start new heartbeat loop
        if self._heartbeat_interval > 0:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _resubscribe_all(self) -> None:
        """Re-subscribe to all tracked subscriptions after reconnection."""
        if not self._subscriptions:
            return

        failed_subs: list[Subscription] = []

        for sub in list(self._subscriptions):
            if not sub.active:
                self._subscriptions.discard(sub)
                continue

            try:
                stream = self._stub.Subscribe(
                    athyr_pb2.SubscribeRequest(
                        agent_id=self._agent_id,
                        subject=sub.subject,
                        queue_group=sub.queue,
                    )
                )
                sub._set_stream(stream, stream.cancel)
                logger.debug(f"Restored subscription: {sub.subject}")

            except Exception as e:
                logger.warning(f"Failed to restore subscription {sub.subject}: {e}")
                failed_subs.append(sub)

        for sub in failed_subs:
            sub._active = False
            sub._reconnecting = False
            sub._reconnect_event.set()
            self._subscriptions.discard(sub)

        if failed_subs:
            logger.warning(f"Failed to restore {len(failed_subs)} subscription(s)")

    # Subscription tracking

    def track_subscription(self, subscription: Subscription) -> None:
        """Add a subscription to tracking for reconnection."""
        self._subscriptions.add(subscription)

    def untrack_subscription(self, subscription: Subscription) -> None:
        """Remove a subscription from tracking."""
        self._subscriptions.discard(subscription)
