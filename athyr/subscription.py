"""Subscription management for Athyr messaging."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Callable

from athyr.models import SubscribeMessage

if TYPE_CHECKING:
    from athyr._grpc import athyr_pb2

logger = logging.getLogger(__name__)


class Subscription:
    """Active subscription to a messaging subject.

    Subscriptions are created via AthyrAgent.subscribe() and provide
    an async iterator interface for receiving messages. Subscriptions
    automatically recover after connection loss when auto_reconnect is enabled.

    Example:
        sub = await agent.subscribe("events.user")
        async for msg in sub.messages():
            print(f"Received: {msg.data}")
        await sub.unsubscribe()
    """

    def __init__(
        self,
        subject: str,
        queue: str,
        stream: AsyncIterator[athyr_pb2.SubscribeMessage],
        cancel_fn: Callable[[], None],
        on_unsubscribe: Callable[[Subscription], None] | None = None,
    ) -> None:
        """Initialize subscription.

        Args:
            subject: The subject being subscribed to.
            queue: Queue group name (empty string for regular subscription).
            stream: The gRPC async stream of messages.
            cancel_fn: Function to call to cancel the subscription.
            on_unsubscribe: Optional callback invoked when unsubscribe() is called.
                Used by the agent to remove this subscription from tracking.
        """
        self._subject = subject
        self._queue = queue
        self._stream = stream
        self._cancel_fn = cancel_fn
        self._on_unsubscribe = on_unsubscribe
        self._active = True
        self._reconnecting = False
        self._reconnect_event = asyncio.Event()

    @property
    def subject(self) -> str:
        """Get the subscribed subject."""
        return self._subject

    @property
    def queue(self) -> str:
        """Get the queue group name (empty for regular subscriptions)."""
        return self._queue

    @property
    def active(self) -> bool:
        """Check if the subscription is still active."""
        return self._active

    @property
    def reconnecting(self) -> bool:
        """Check if the subscription is waiting for reconnection."""
        return self._reconnecting

    def _set_reconnecting(self) -> None:
        """Mark subscription as waiting for reconnection.

        Called by the agent when connection is lost.
        """
        if self._active:
            self._reconnecting = True
            self._reconnect_event.clear()

    def _set_stream(
        self,
        stream: AsyncIterator[athyr_pb2.SubscribeMessage],
        cancel_fn: Callable[[], None],
    ) -> None:
        """Set a new stream after reconnection.

        Called by the agent after successfully re-subscribing.

        Args:
            stream: The new gRPC async stream.
            cancel_fn: New cancel function for the stream.
        """
        self._stream = stream
        self._cancel_fn = cancel_fn
        self._reconnecting = False
        self._reconnect_event.set()

    async def messages(self) -> AsyncIterator[SubscribeMessage]:
        """Iterate over incoming messages.

        Yields:
            SubscribeMessage objects as they arrive.

        Note:
            The iterator automatically recovers after reconnection.
            It completes when unsubscribe() is called.
        """
        if not self._active:
            return

        while self._active:
            try:
                async for proto_msg in self._stream:
                    if not self._active:
                        return
                    yield SubscribeMessage.from_proto(proto_msg)

                # Stream ended normally - check if reconnecting
                if self._reconnecting:
                    logger.debug(f"Subscription {self._subject}: waiting for reconnect")
                    await self._reconnect_event.wait()
                    if not self._active:
                        return
                    logger.debug(f"Subscription {self._subject}: reconnected")
                    continue  # Loop back to iterate new stream
                else:
                    # Stream ended and not reconnecting - we're done
                    break

            except asyncio.CancelledError:
                # Check if this is a reconnection cancellation
                if self._reconnecting and self._active:
                    logger.debug(f"Subscription {self._subject}: stream cancelled, waiting for reconnect")
                    await self._reconnect_event.wait()
                    if not self._active:
                        return
                    continue
                # Real cancellation
                break
            except Exception as e:
                # Stream error - check if reconnecting
                if self._reconnecting and self._active:
                    logger.debug(f"Subscription {self._subject}: error during reconnect: {e}")
                    await self._reconnect_event.wait()
                    if not self._active:
                        return
                    continue
                # Real error and not reconnecting
                logger.warning(f"Subscription {self._subject}: stream error: {e}")
                break

        self._active = False

    async def unsubscribe(self) -> None:
        """Cancel the subscription.

        After calling this, the messages() iterator will complete
        and no more messages will be received.
        """
        if not self._active:
            return

        self._active = False
        self._reconnecting = False
        self._reconnect_event.set()  # Wake up any waiting iterator
        try:
            self._cancel_fn()
        except Exception:
            pass  # Ignore cancel errors (stream may already be closed)

        # Notify agent to remove this subscription from tracking
        if self._on_unsubscribe is not None:
            try:
                self._on_unsubscribe(self)
            except Exception:
                pass  # Don't let cleanup errors propagate
