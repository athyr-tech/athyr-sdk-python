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

        sub = Subscription("test.subject", stream, cancel_fn)

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

        sub = Subscription("test", stream, cancel_fn)

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

        sub = Subscription("test", stream, cancel_fn)
        assert sub.active is True

        async for _ in sub.messages():
            pass

        assert sub.active is False

    @pytest.mark.asyncio
    async def test_unsubscribe_calls_cancel_function(self) -> None:
        """Test unsubscribe() calls the cancel function."""
        cancel_fn = MagicMock()
        stream = async_iter([])

        sub = Subscription("test", stream, cancel_fn)
        await sub.unsubscribe()

        cancel_fn.assert_called_once()
        assert sub.active is False

    @pytest.mark.asyncio
    async def test_unsubscribe_is_idempotent(self) -> None:
        """Test calling unsubscribe() multiple times is safe."""
        cancel_fn = MagicMock()
        stream = async_iter([])

        sub = Subscription("test", stream, cancel_fn)
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

        sub = Subscription("test", stream, cancel_fn)
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