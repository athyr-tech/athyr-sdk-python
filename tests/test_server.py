"""Tests for Server module."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from athyr.server import Server
from athyr.service import Middleware, RawHandler, Service


# Test fixtures


@dataclass
class EchoRequest:
    message: str


@dataclass
class EchoResponse:
    reply: str


class TestServerCreation:
    """Tests for Server creation and configuration."""

    def test_create_server_with_defaults(self) -> None:
        """Test creating server with default values."""
        server = Server(address="localhost:9090")

        assert server.address == "localhost:9090"
        assert server.agent_name == "service-agent"
        assert server.version == "1.0.0"
        assert server.services == []
        assert server.middleware == []

    def test_create_server_with_options(self) -> None:
        """Test creating server with custom options."""
        server = Server(
            address="athyr.example.com:443",
            agent_name="my-service",
            agent_description="A test service",
            version="2.0.0",
            tls=True,
        )

        assert server.address == "athyr.example.com:443"
        assert server.agent_name == "my-service"
        assert server.agent_description == "A test service"
        assert server.version == "2.0.0"
        assert server.tls is True


class TestServerServiceRegistration:
    """Tests for Server service registration."""

    def test_add_service(self) -> None:
        """Test adding a pre-built service."""
        server = Server(address="localhost:9090")

        async def handler(ctx: Any, data: bytes) -> bytes:
            return data

        service = Service.raw(subject="echo", handler=handler)
        result = server.add(service)

        assert result is server  # Returns self for chaining
        assert len(server.services) == 1
        assert server.services[0] is service

    def test_add_multiple_services(self) -> None:
        """Test adding multiple services."""
        server = Server(address="localhost:9090")

        async def handler(ctx: Any, data: bytes) -> bytes:
            return data

        server.add(Service.raw(subject="echo1", handler=handler))
        server.add(Service.raw(subject="echo2", handler=handler))
        server.add(Service.raw(subject="echo3", handler=handler))

        assert len(server.services) == 3

    def test_handle_typed_service(self) -> None:
        """Test adding a typed handler."""
        server = Server(address="localhost:9090")

        async def handler(ctx: Any, req: EchoRequest) -> EchoResponse:
            return EchoResponse(reply=f"Echo: {req.message}")

        result = server.handle(
            subject="echo.invoke",
            handler=handler,
            request_type=EchoRequest,
            response_type=EchoResponse,
            name="Echo Service",
            queue_group="echo-workers",
        )

        assert result is server
        assert len(server.services) == 1
        assert server.services[0].subject == "echo.invoke"
        assert server.services[0].name == "Echo Service"
        assert server.services[0].queue_group == "echo-workers"

    def test_handle_raw_service(self) -> None:
        """Test adding a raw handler."""
        server = Server(address="localhost:9090")

        async def handler(ctx: Any, data: bytes) -> bytes:
            return data

        result = server.handle_raw(
            subject="echo",
            handler=handler,
            name="Raw Echo",
            queue_group="raw-workers",
        )

        assert result is server
        assert len(server.services) == 1
        assert server.services[0].subject == "echo"
        assert server.services[0].name == "Raw Echo"


class TestServerMiddleware:
    """Tests for Server middleware configuration."""

    def test_use_middleware(self) -> None:
        """Test adding global middleware."""
        server = Server(address="localhost:9090")

        def middleware1(handler: RawHandler) -> RawHandler:
            return handler

        def middleware2(handler: RawHandler) -> RawHandler:
            return handler

        result = server.use(middleware1, middleware2)

        assert result is server
        assert len(server.middleware) == 2
        assert server.middleware[0] is middleware1
        assert server.middleware[1] is middleware2

    def test_chained_configuration(self) -> None:
        """Test fluent API with chained calls."""
        server = Server(address="localhost:9090")

        async def handler(ctx: Any, data: bytes) -> bytes:
            return data

        def mw(h: RawHandler) -> RawHandler:
            return h

        # Chain multiple configuration calls
        (
            server
            .use(mw)
            .add(Service.raw(subject="svc1", handler=handler))
            .handle_raw(subject="svc2", handler=handler)
        )

        assert len(server.middleware) == 1
        assert len(server.services) == 2


class TestServerRunValidation:
    """Tests for Server.run() validation."""

    @pytest.mark.asyncio
    async def test_run_raises_without_services(self) -> None:
        """Test run() raises error when no services registered."""
        server = Server(address="localhost:9090")

        with pytest.raises(ValueError, match="No services registered"):
            await server.run()


class TestServerLifecycle:
    """Tests for Server lifecycle management."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_event(self) -> None:
        """Test shutdown() sets the shutdown event."""
        server = Server(address="localhost:9090")

        assert not server._shutdown_event.is_set()
        await server.shutdown()
        assert server._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_run_connects_and_subscribes(self) -> None:
        """Test run() connects agent and subscribes services."""
        server = Server(address="localhost:9090")

        async def handler(ctx: Any, data: bytes) -> bytes:
            return data

        server.add(Service.raw(subject="test", handler=handler))

        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent.agent_id = MagicMock(return_value="test-agent-id")
        mock_agent.connect = AsyncMock()
        mock_agent.subscribe = AsyncMock()
        mock_agent.close = AsyncMock()

        # Create a mock subscription that yields no messages
        mock_sub = MagicMock()
        mock_sub.messages = MagicMock(return_value=async_iter([]))
        mock_sub.unsubscribe = AsyncMock()
        mock_agent.subscribe.return_value = mock_sub

        with patch("athyr.server.AthyrAgent", return_value=mock_agent):
            # Trigger shutdown after a short delay
            async def trigger_shutdown() -> None:
                await server.shutdown()

            import asyncio
            asyncio.create_task(trigger_shutdown())

            await server.run()

        # Verify agent was connected
        mock_agent.connect.assert_called_once()

        # Verify subscription was created
        mock_agent.subscribe.assert_called_once_with("test")

        # Verify cleanup happened
        mock_sub.unsubscribe.assert_called_once()
        mock_agent.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_uses_queue_subscribe_when_configured(self) -> None:
        """Test run() uses queue_subscribe for services with queue groups."""
        server = Server(address="localhost:9090")

        async def handler(ctx: Any, data: bytes) -> bytes:
            return data

        server.add(Service.raw(
            subject="test",
            handler=handler,
            queue_group="workers",
        ))

        mock_agent = AsyncMock()
        mock_agent.agent_id = MagicMock(return_value="test-agent-id")
        mock_agent.connect = AsyncMock()
        mock_agent.queue_subscribe = AsyncMock()
        mock_agent.close = AsyncMock()

        mock_sub = MagicMock()
        mock_sub.messages = MagicMock(return_value=async_iter([]))
        mock_sub.unsubscribe = AsyncMock()
        mock_agent.queue_subscribe.return_value = mock_sub

        with patch("athyr.server.AthyrAgent", return_value=mock_agent):
            async def trigger_shutdown() -> None:
                await server.shutdown()

            import asyncio
            asyncio.create_task(trigger_shutdown())

            await server.run()

        # Verify queue_subscribe was used
        mock_agent.queue_subscribe.assert_called_once_with("test", "workers")


class TestServerMessageHandling:
    """Tests for Server message handling."""

    @pytest.mark.asyncio
    async def test_handler_receives_messages(self) -> None:
        """Test that handlers receive and process messages."""
        server = Server(address="localhost:9090")
        received_data: list[bytes] = []

        async def handler(ctx: Any, data: bytes) -> bytes:
            received_data.append(data)
            return b"response"

        server.add(Service.raw(subject="test", handler=handler))

        mock_agent = AsyncMock()
        mock_agent.agent_id = MagicMock(return_value="test-agent-id")
        mock_agent.connect = AsyncMock()
        mock_agent.subscribe = AsyncMock()
        mock_agent.publish = AsyncMock()
        mock_agent.close = AsyncMock()

        # Create message to be received
        mock_msg = MagicMock()
        mock_msg.subject = "test"
        mock_msg.data = b"test data"
        mock_msg.reply = "_INBOX.123"

        mock_sub = MagicMock()
        mock_sub.messages = MagicMock(return_value=async_iter([mock_msg]))
        mock_sub.unsubscribe = AsyncMock()
        mock_agent.subscribe.return_value = mock_sub

        with patch("athyr.server.AthyrAgent", return_value=mock_agent):
            async def trigger_shutdown() -> None:
                # Give time for message to be processed
                import asyncio
                await asyncio.sleep(0.01)
                await server.shutdown()

            import asyncio
            asyncio.create_task(trigger_shutdown())

            await server.run()

        # Verify handler received the message
        assert received_data == [b"test data"]

        # Verify response was sent
        mock_agent.publish.assert_called_with("_INBOX.123", b"response")


# Helper for async iteration in tests


async def async_iter(items: list[Any]) -> Any:
    """Create an async iterator from a list."""
    for item in items:
        yield item
