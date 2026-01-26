"""Messaging operations for Athyr agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc

from athyr._grpc import athyr_pb2
from athyr.exceptions import from_grpc_error
from athyr.subscription import Subscription

if TYPE_CHECKING:
    from athyr._internal.connection import ConnectionManager


class MessagingOperations:
    """Handles pub/sub and request/reply messaging operations.

    Provides methods for:
    - Publishing messages
    - Subscribing to subjects
    - Queue subscriptions for load balancing
    - Request/reply RPC pattern

    Example:
        messaging = MessagingOperations(connection)
        await messaging.publish("events.user", b'{"id": 1}')
        sub = await messaging.subscribe("events.>")
    """

    def __init__(self, connection: ConnectionManager) -> None:
        """Initialize messaging operations.

        Args:
            connection: The connection manager for stub access.
        """
        self._connection = connection

    async def publish(self, subject: str, data: bytes) -> None:
        """Publish a message to a subject.

        Args:
            subject: The subject to publish to (e.g., "events.user.created").
            data: The message payload as bytes.

        Raises:
            ConnectionError: If not connected to Athyr.
            AthyrError: If the publish operation fails.

        Example:
            await messaging.publish("events.order", b'{"order_id": 123}')
        """
        stub = self._connection.get_stub()

        try:
            await stub.Publish(
                athyr_pb2.PublishRequest(
                    agent_id=self._connection.agent_id,
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
            sub = await messaging.subscribe("events.user.>")
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
            sub = await messaging.queue_subscribe("tasks.process", "workers")
            async for msg in sub.messages():
                process_task(msg.data)
        """
        return await self._subscribe(subject, queue=queue)

    async def _subscribe(self, subject: str, queue: str) -> Subscription:
        """Internal subscribe implementation."""
        stub = self._connection.get_stub()

        try:
            stream = stub.Subscribe(
                athyr_pb2.SubscribeRequest(
                    agent_id=self._connection.agent_id,
                    subject=subject,
                    queue_group=queue,
                )
            )

            sub = Subscription(
                subject=subject,
                queue=queue,
                stream=stream,
                cancel_fn=stream.cancel,
                on_unsubscribe=lambda s: self._connection.untrack_subscription(s),
            )

            self._connection.track_subscription(sub)
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
            response = await messaging.request(
                "services.calculator.add",
                b'{"a": 1, "b": 2}',
                timeout=10.0,
            )
            result = json.loads(response)
        """
        stub = self._connection.get_stub()

        try:
            response = await stub.Request(
                athyr_pb2.RequestMessage(
                    agent_id=self._connection.agent_id,
                    subject=subject,
                    data=data,
                    timeout_ms=int(timeout * 1000),
                )
            )
            return response.data
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e
