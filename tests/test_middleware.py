"""Tests for middleware system."""

import pytest
from unittest.mock import MagicMock

from athyr.middleware import (
    Handler,
    Middleware,
    MiddlewareContext,
    chain,
    middleware,
    passthrough_handler,
)


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for testing."""
    return MagicMock()


@pytest.fixture
def ctx(mock_agent: MagicMock) -> MiddlewareContext:
    """Create a test middleware context."""
    return MiddlewareContext(
        agent=mock_agent,
        subject="test.subject",
        reply_subject="_INBOX.123",
    )


class TestMiddlewareContext:
    """Tests for MiddlewareContext dataclass."""

    def test_create_minimal(self, mock_agent: MagicMock) -> None:
        """Test creating context with minimal fields."""
        ctx = MiddlewareContext(
            agent=mock_agent,
            subject="test.subject",
        )

        assert ctx.agent is mock_agent
        assert ctx.subject == "test.subject"
        assert ctx.reply_subject == ""
        assert ctx.metadata is None

    def test_create_full(self, mock_agent: MagicMock) -> None:
        """Test creating context with all fields."""
        ctx = MiddlewareContext(
            agent=mock_agent,
            subject="test.subject",
            reply_subject="_INBOX.456",
            metadata={"key": "value"},
        )

        assert ctx.reply_subject == "_INBOX.456"
        assert ctx.metadata == {"key": "value"}

    def test_with_metadata_creates_new_context(self, ctx: MiddlewareContext) -> None:
        """Test with_metadata creates a new context."""
        new_ctx = ctx.with_metadata("request_id", "abc123")

        assert new_ctx is not ctx
        assert new_ctx.get_metadata("request_id") == "abc123"
        assert ctx.get_metadata("request_id") is None

    def test_with_metadata_preserves_existing(self, mock_agent: MagicMock) -> None:
        """Test with_metadata preserves existing metadata."""
        ctx = MiddlewareContext(
            agent=mock_agent,
            subject="test",
            metadata={"existing": "value"},
        )

        new_ctx = ctx.with_metadata("new_key", "new_value")

        assert new_ctx.get_metadata("existing") == "value"
        assert new_ctx.get_metadata("new_key") == "new_value"

    def test_get_metadata_returns_default(self, ctx: MiddlewareContext) -> None:
        """Test get_metadata returns default for missing keys."""
        assert ctx.get_metadata("missing") is None
        assert ctx.get_metadata("missing", "default") == "default"

    def test_get_metadata_with_metadata(self, mock_agent: MagicMock) -> None:
        """Test get_metadata with existing metadata."""
        ctx = MiddlewareContext(
            agent=mock_agent,
            subject="test",
            metadata={"key": "value"},
        )

        assert ctx.get_metadata("key") == "value"
        assert ctx.get_metadata("missing", "default") == "default"


class TestMiddlewareDecorator:
    """Tests for @middleware decorator."""

    @pytest.mark.asyncio
    async def test_middleware_wraps_handler(self, ctx: MiddlewareContext) -> None:
        """Test @middleware decorator wraps a handler."""
        calls: list[str] = []

        @middleware
        async def tracking_middleware(ctx, data, call_next):
            calls.append("before")
            result = await call_next(ctx, data)
            calls.append("after")
            return result

        async def handler(ctx, data):
            calls.append("handler")
            return b"response"

        wrapped = tracking_middleware(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"response"
        assert calls == ["before", "handler", "after"]

    @pytest.mark.asyncio
    async def test_middleware_can_modify_data(self, ctx: MiddlewareContext) -> None:
        """Test middleware can modify input data."""

        @middleware
        async def prefix_middleware(ctx, data, call_next):
            modified_data = b"PREFIX:" + data
            return await call_next(ctx, modified_data)

        async def echo_handler(ctx, data):
            return data

        wrapped = prefix_middleware(echo_handler)
        result = await wrapped(ctx, b"input")

        assert result == b"PREFIX:input"

    @pytest.mark.asyncio
    async def test_middleware_can_modify_response(self, ctx: MiddlewareContext) -> None:
        """Test middleware can modify response."""

        @middleware
        async def suffix_middleware(ctx, data, call_next):
            result = await call_next(ctx, data)
            return result + b":SUFFIX"

        async def handler(ctx, data):
            return b"response"

        wrapped = suffix_middleware(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"response:SUFFIX"

    @pytest.mark.asyncio
    async def test_middleware_can_short_circuit(self, ctx: MiddlewareContext) -> None:
        """Test middleware can return early without calling next."""
        handler_called = False

        @middleware
        async def blocking_middleware(ctx, data, call_next):
            return b"blocked"

        async def handler(ctx, data):
            nonlocal handler_called
            handler_called = True
            return b"should not see"

        wrapped = blocking_middleware(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"blocked"
        assert not handler_called

    @pytest.mark.asyncio
    async def test_middleware_can_catch_errors(self, ctx: MiddlewareContext) -> None:
        """Test middleware can catch and handle errors."""

        @middleware
        async def error_handling_middleware(ctx, data, call_next):
            try:
                return await call_next(ctx, data)
            except ValueError as e:
                return f"Error: {e}".encode()

        async def failing_handler(ctx, data):
            raise ValueError("something went wrong")

        wrapped = error_handling_middleware(failing_handler)
        result = await wrapped(ctx, b"input")

        assert result == b"Error: something went wrong"


class TestChain:
    """Tests for chain() function."""

    @pytest.mark.asyncio
    async def test_chain_empty(self, ctx: MiddlewareContext) -> None:
        """Test chain with no middlewares returns identity."""
        combined = chain()

        async def handler(ctx, data):
            return b"response"

        wrapped = combined(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"response"

    @pytest.mark.asyncio
    async def test_chain_single(self, ctx: MiddlewareContext) -> None:
        """Test chain with single middleware."""

        @middleware
        async def upper_middleware(ctx, data, call_next):
            result = await call_next(ctx, data)
            return result.upper()

        combined = chain(upper_middleware)

        async def handler(ctx, data):
            return b"hello"

        wrapped = combined(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"HELLO"

    @pytest.mark.asyncio
    async def test_chain_multiple_order(self, ctx: MiddlewareContext) -> None:
        """Test chain applies middlewares in correct order."""
        order: list[str] = []

        @middleware
        async def first_middleware(ctx, data, call_next):
            order.append("first-before")
            result = await call_next(ctx, data)
            order.append("first-after")
            return result

        @middleware
        async def second_middleware(ctx, data, call_next):
            order.append("second-before")
            result = await call_next(ctx, data)
            order.append("second-after")
            return result

        @middleware
        async def third_middleware(ctx, data, call_next):
            order.append("third-before")
            result = await call_next(ctx, data)
            order.append("third-after")
            return result

        combined = chain(first_middleware, second_middleware, third_middleware)

        async def handler(ctx, data):
            order.append("handler")
            return b"done"

        wrapped = combined(handler)
        await wrapped(ctx, b"input")

        # First middleware is outermost
        assert order == [
            "first-before",
            "second-before",
            "third-before",
            "handler",
            "third-after",
            "second-after",
            "first-after",
        ]

    @pytest.mark.asyncio
    async def test_chain_data_transformation(self, ctx: MiddlewareContext) -> None:
        """Test chain with data transformations."""

        @middleware
        async def add_a(ctx, data, call_next):
            return await call_next(ctx, data + b"A")

        @middleware
        async def add_b(ctx, data, call_next):
            return await call_next(ctx, data + b"B")

        @middleware
        async def add_c(ctx, data, call_next):
            return await call_next(ctx, data + b"C")

        combined = chain(add_a, add_b, add_c)

        async def echo(ctx, data):
            return data

        wrapped = combined(echo)
        result = await wrapped(ctx, b"START:")

        # Data flows through: START: -> A -> B -> C
        assert result == b"START:ABC"

    @pytest.mark.asyncio
    async def test_chain_response_transformation(self, ctx: MiddlewareContext) -> None:
        """Test chain with response transformations."""

        @middleware
        async def wrap_brackets(ctx, data, call_next):
            result = await call_next(ctx, data)
            return b"[" + result + b"]"

        @middleware
        async def wrap_parens(ctx, data, call_next):
            result = await call_next(ctx, data)
            return b"(" + result + b")"

        combined = chain(wrap_brackets, wrap_parens)

        async def handler(ctx, data):
            return b"core"

        wrapped = combined(handler)
        result = await wrapped(ctx, b"input")

        # Response flows back: core -> (core) -> [(core)]
        assert result == b"[(core)]"


class TestPassthroughHandler:
    """Tests for passthrough_handler utility."""

    @pytest.mark.asyncio
    async def test_passthrough_returns_input(self, ctx: MiddlewareContext) -> None:
        """Test passthrough handler returns input unchanged."""
        result = await passthrough_handler(ctx, b"test data")
        assert result == b"test data"

    @pytest.mark.asyncio
    async def test_passthrough_with_middleware(self, ctx: MiddlewareContext) -> None:
        """Test passthrough handler works with middleware."""

        @middleware
        async def double_middleware(ctx, data, call_next):
            result = await call_next(ctx, data)
            return result + result

        wrapped = double_middleware(passthrough_handler)
        result = await wrapped(ctx, b"AB")

        assert result == b"ABAB"


class TestMiddlewareIntegration:
    """Integration tests for middleware system."""

    @pytest.mark.asyncio
    async def test_logging_style_middleware(self, ctx: MiddlewareContext) -> None:
        """Test a logging-style middleware pattern."""
        logs: list[str] = []

        @middleware
        async def log_middleware(ctx, data, call_next):
            logs.append(f"-> {ctx.subject}: {len(data)} bytes")
            result = await call_next(ctx, data)
            logs.append(f"<- {ctx.subject}: {len(result)} bytes")
            return result

        async def handler(ctx, data):
            return b"response"

        wrapped = log_middleware(handler)
        await wrapped(ctx, b"input")

        assert logs == [
            "-> test.subject: 5 bytes",
            "<- test.subject: 8 bytes",
        ]

    @pytest.mark.asyncio
    async def test_retry_style_middleware(self, ctx: MiddlewareContext) -> None:
        """Test a retry-style middleware pattern."""
        attempts = 0

        @middleware
        async def retry_middleware(ctx, data, call_next):
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    return await call_next(ctx, data)
                except ValueError:
                    if attempt == max_attempts - 1:
                        raise
            return b""  # Should not reach

        async def flaky_handler(ctx, data):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("temporary failure")
            return b"success"

        wrapped = retry_middleware(flaky_handler)
        result = await wrapped(ctx, b"input")

        assert result == b"success"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_timeout_style_middleware(self, ctx: MiddlewareContext) -> None:
        """Test a timeout-style middleware pattern (simplified)."""
        import asyncio

        @middleware
        async def timeout_middleware(ctx, data, call_next):
            try:
                return await asyncio.wait_for(call_next(ctx, data), timeout=0.1)
            except asyncio.TimeoutError:
                return b"timeout"

        async def slow_handler(ctx, data):
            await asyncio.sleep(1.0)
            return b"slow response"

        wrapped = timeout_middleware(slow_handler)
        result = await wrapped(ctx, b"input")

        assert result == b"timeout"
