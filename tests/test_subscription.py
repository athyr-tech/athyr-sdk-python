"""Tests for subscription management."""

import asyncio
from unittest.mock import MagicMock

import pytest

from athyr._grpc import athyr_pb2
from athyr.client import AthyrAgent
from athyr.subscription import Subscription
from athyr.models import SubscribeMessage


async def async_iter(items):
    """Convert a list to an async iterator."""
    for item in items:
        yield item


class TestSubscription:
    """Tests for Subscription class."""

    def test_subscription_properties(self) -> None:
        """Test subscription has correct initial properties."""
        cancel_fn = MagicMock()
        stream = async_iter([])

        sub = Subscription("test.subject", "", stream, cancel_fn)

        assert sub.subject == "test.subject"
        assert sub.active is True

    @pytest.mark.asyncio
    async def test_subscription_messages_yields_messages(self) -> None:
        """Test messages() yields SubscribeMessage objects."""
        messages = [
            athyr_pb2.SubscribeMessage(subject="test", data=b"hello", reply=""),
            athyr_pb2.SubscribeMessage(subject="test", data=b"world", reply="reply.to"),
        ]
        cancel_fn = MagicMock()
        stream = async_iter(messages)

        sub = Subscription("test", "", stream, cancel_fn)

        received = []
        async for msg in sub.messages():
            received.append(msg)

        assert len(received) == 2
        assert isinstance(received[0], SubscribeMessage)
        assert received[0].data == b"hello"
        assert received[0].reply == ""
        assert received[1].data == b"world"
        assert received[1].reply == "reply.to"

    @pytest.mark.asyncio
    async def test_subscription_becomes_inactive_after_iteration(self) -> None:
        """Test subscription becomes inactive after messages complete."""
        messages = [
            athyr_pb2.SubscribeMessage(subject="test", data=b"one", reply=""),
        ]
        cancel_fn = MagicMock()
        stream = async_iter(messages)

        sub = Subscription("test", "", stream, cancel_fn)
        assert sub.active is True

        async for _ in sub.messages():
            pass

        assert sub.active is False

    @pytest.mark.asyncio
    async def test_unsubscribe_calls_cancel_function(self) -> None:
        """Test unsubscribe() calls the cancel function."""
        cancel_fn = MagicMock()
        stream = async_iter([])

        sub = Subscription("test", "", stream, cancel_fn)
        await sub.unsubscribe()

        cancel_fn.assert_called_once()
        assert sub.active is False

    @pytest.mark.asyncio
    async def test_unsubscribe_is_idempotent(self) -> None:
        """Test calling unsubscribe() multiple times is safe."""
        cancel_fn = MagicMock()
        stream = async_iter([])

        sub = Subscription("test", "", stream, cancel_fn)
        await sub.unsubscribe()
        await sub.unsubscribe()
        await sub.unsubscribe()

        # Should only be called once
        cancel_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_messages_returns_empty_when_inactive(self) -> None:
        """Test messages() returns immediately when subscription is inactive."""
        cancel_fn = MagicMock()
        stream = async_iter([
            athyr_pb2.SubscribeMessage(subject="test", data=b"should not see", reply=""),
        ])

        sub = Subscription("test", "", stream, cancel_fn)
        await sub.unsubscribe()

        received = []
        async for msg in sub.messages():
            received.append(msg)

        assert len(received) == 0


class TestSubscribeMessage:
    """Tests for SubscribeMessage model."""

    def test_create_minimal(self) -> None:
        """Test creating a minimal SubscribeMessage."""
        msg = SubscribeMessage(subject="events.user", data=b"payload")

        assert msg.subject == "events.user"
        assert msg.data == b"payload"
        assert msg.reply == ""

    def test_create_with_reply(self) -> None:
        """Test creating a SubscribeMessage with reply subject."""
        msg = SubscribeMessage(
            subject="requests.process",
            data=b'{"action": "start"}',
            reply="_INBOX.abc123",
        )

        assert msg.subject == "requests.process"
        assert msg.reply == "_INBOX.abc123"

    def test_from_proto(self) -> None:
        """Test creating from protobuf message."""
        proto = athyr_pb2.SubscribeMessage(
            subject="test.subject",
            data=b"test data",
            reply="reply.subject",
        )

        msg = SubscribeMessage.from_proto(proto)

        assert msg.subject == "test.subject"
        assert msg.data == b"test data"
        assert msg.reply == "reply.subject"


class TestAgentSubscriptionTracking:
    """Tests for agent subscription tracking."""

    def test_agent_has_empty_subscriptions_initially(self) -> None:
        """Test agent starts with no subscriptions."""
        agent = AthyrAgent("localhost:9090")
        assert len(agent.subscriptions) == 0

    def test_subscriptions_property_returns_set(self) -> None:
        """Test subscriptions property returns a set."""
        agent = AthyrAgent("localhost:9090")
        assert isinstance(agent.subscriptions, set)


class TestSubscriptionRecovery:
    """Tests for subscription recovery after reconnection."""

    def test_subscription_has_queue_property(self) -> None:
        """Test subscription exposes queue property."""
        cancel_fn = MagicMock()
        stream = async_iter([])

        sub = Subscription("test.subject", "my-queue", stream, cancel_fn)

        assert sub.subject == "test.subject"
        assert sub.queue == "my-queue"

    def test_subscription_reconnecting_property(self) -> None:
        """Test subscription has reconnecting property."""
        cancel_fn = MagicMock()
        stream = async_iter([])

        sub = Subscription("test", "", stream, cancel_fn)

        assert sub.reconnecting is False

    def test_set_reconnecting_marks_subscription(self) -> None:
        """Test _set_reconnecting marks subscription as reconnecting."""
        cancel_fn = MagicMock()
        stream = async_iter([])

        sub = Subscription("test", "", stream, cancel_fn)
        sub._set_reconnecting()

        assert sub.reconnecting is True
        assert sub.active is True  # Still active

    def test_set_reconnecting_ignored_when_inactive(self) -> None:
        """Test _set_reconnecting is ignored for inactive subscriptions."""
        cancel_fn = MagicMock()
        stream = async_iter([])

        sub = Subscription("test", "", stream, cancel_fn)
        sub._active = False
        sub._set_reconnecting()

        assert sub.reconnecting is False

    @pytest.mark.asyncio
    async def test_set_stream_updates_subscription(self) -> None:
        """Test _set_stream updates stream and wakes up waiting iterator."""
        cancel_fn = MagicMock()
        new_cancel_fn = MagicMock()
        old_stream = async_iter([])
        new_stream = async_iter([
            athyr_pb2.SubscribeMessage(subject="test", data=b"new", reply=""),
        ])

        sub = Subscription("test", "", old_stream, cancel_fn)
        sub._set_reconnecting()

        assert sub.reconnecting is True

        sub._set_stream(new_stream, new_cancel_fn)

        assert sub.reconnecting is False
        assert sub._cancel_fn is new_cancel_fn

    @pytest.mark.asyncio
    async def test_messages_waits_for_reconnect(self) -> None:
        """Test messages() waits for reconnect when stream ends during reconnection."""
        cancel_fn = MagicMock()
        new_cancel_fn = MagicMock()

        # First stream ends after one message
        old_stream = async_iter([
            athyr_pb2.SubscribeMessage(subject="test", data=b"first", reply=""),
        ])

        # New stream with another message
        new_stream = async_iter([
            athyr_pb2.SubscribeMessage(subject="test", data=b"second", reply=""),
        ])

        sub = Subscription("test", "", old_stream, cancel_fn)

        # Mark as reconnecting BEFORE starting iteration
        # This simulates: messages arrive -> connection drops -> reconnecting
        sub._set_reconnecting()

        received = []

        async def consume():
            async for msg in sub.messages():
                received.append(msg.data)

        async def provide_new_stream():
            # Wait a bit for consumer to start waiting for reconnect
            await asyncio.sleep(0.05)
            sub._set_stream(new_stream, new_cancel_fn)

        # Run both concurrently
        await asyncio.gather(consume(), provide_new_stream())

        assert received == [b"first", b"second"]

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_reconnect_waiting(self) -> None:
        """Test unsubscribe() stops messages() waiting for reconnect."""
        cancel_fn = MagicMock()
        stream = async_iter([])

        sub = Subscription("test", "", stream, cancel_fn)
        sub._set_reconnecting()

        async def consume():
            async for _ in sub.messages():
                pass

        # Start consuming (will wait for reconnect)
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)

        # Unsubscribe should stop the waiting
        await sub.unsubscribe()
        await asyncio.sleep(0.01)

        assert task.done()
        assert sub.active is False
        assert sub.reconnecting is False