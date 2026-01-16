"""Subscription management for Athyr messaging."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable

from athyr.models import SubscribeMessage

if TYPE_CHECKING:
    from athyr._grpc import athyr_pb2


class Subscription:
    """Active subscription to a messaging subject.

    Subscriptions are created via AthyrAgent.subscribe() and provide
    an async iterator interface for receiving messages.

    Example:
        sub = await agent.subscribe("events.user")
        async for msg in sub.messages():
            print(f"Received: {msg.data}")
        await sub.unsubscribe()
    """

    def __init__(
        self,
        subject: str,
        stream: AsyncIterator[athyr_pb2.SubscribeMessage],
        cancel_fn: Callable[[], None],
    ) -> None:
        """Initialize subscription.

        Args:
            subject: The subject being subscribed to.
            stream: The gRPC async stream of messages.
            cancel_fn: Function to call to cancel the subscription.
        """
        self._subject = subject
        self._stream = stream
        self._cancel_fn = cancel_fn
        self._active = True

    @property
    def subject(self) -> str:
        """Get the subscribed subject."""
        return self._subject

    @property
    def active(self) -> bool:
        """Check if the subscription is still active."""
        return self._active

    async def messages(self) -> AsyncIterator[SubscribeMessage]:
        """Iterate over incoming messages.

        Yields:
            SubscribeMessage objects as they arrive.

        Note:
            The iterator completes when unsubscribe() is called
            or the server closes the stream.
        """
        if not self._active:
            return

        try:
            async for proto_msg in self._stream:
                if not self._active:
                    break
                yield SubscribeMessage.from_proto(proto_msg)
        except asyncio.CancelledError:
            pass
        finally:
            self._active = False

    async def unsubscribe(self) -> None:
        """Cancel the subscription.

        After calling this, the messages() iterator will complete
        and no more messages will be received.
        """
        if not self._active:
            return

        self._active = False
        self._cancel_fn()
