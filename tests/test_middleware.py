"""Tests for middleware system."""

import asyncio
import json
import logging
import pytest
from unittest.mock import MagicMock

from athyr.middleware import (
    Handler,
    Middleware,
    MiddlewareContext,
    chain,
    log_requests,
    middleware,
    passthrough_handler,
    recover,
    retry,
    timeout,
)
from athyr.exceptions import (
    InternalError,
    TimeoutError as AthyrTimeoutError,
    UnavailableError,
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


class TestLogRequests:
    """Tests for log_requests built-in middleware."""

    @pytest.fixture
    def log_capture(self) -> logging.Logger:
        """Create a logger that captures log records."""
        logger = logging.getLogger("test.log_requests")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        return logger

    @pytest.fixture
    def log_records(self, log_capture: logging.Logger) -> list[logging.LogRecord]:
        """Capture log records from the test logger."""
        records: list[logging.LogRecord] = []

        class RecordHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        log_capture.addHandler(RecordHandler())
        return records

    @pytest.mark.asyncio
    async def test_logs_successful_request(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test log_requests logs INFO for successful requests."""

        async def handler(ctx, data):
            return b"response data"

        mw = log_requests(log_capture)
        wrapped = mw(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"response data"
        assert len(log_records) == 1
        assert log_records[0].levelno == logging.INFO

        log_data = json.loads(log_records[0].getMessage())
        assert log_data["subject"] == "test.subject"
        assert log_data["response_bytes"] == 13
        assert "duration_ms" in log_data
        assert log_data["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_logs_failed_request(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test log_requests logs ERROR for failed requests."""

        async def failing_handler(ctx, data):
            raise ValueError("something went wrong")

        mw = log_requests(log_capture)
        wrapped = mw(failing_handler)

        with pytest.raises(ValueError, match="something went wrong"):
            await wrapped(ctx, b"input")

        assert len(log_records) == 1
        assert log_records[0].levelno == logging.ERROR

        log_data = json.loads(log_records[0].getMessage())
        assert log_data["subject"] == "test.subject"
        assert log_data["error"] == "ValueError: something went wrong"
        assert "duration_ms" in log_data

    @pytest.mark.asyncio
    async def test_uses_default_logger(self, ctx: MiddlewareContext) -> None:
        """Test log_requests uses default logger when none provided."""
        # Just verify it doesn't raise
        mw = log_requests()

        async def handler(ctx, data):
            return b"ok"

        wrapped = mw(handler)
        result = await wrapped(ctx, b"input")
        assert result == b"ok"

    @pytest.mark.asyncio
    async def test_chains_with_other_middleware(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test log_requests works in middleware chain."""

        @middleware
        async def prefix_middleware(ctx, data, call_next):
            result = await call_next(ctx, data)
            return b"PREFIX:" + result

        async def handler(ctx, data):
            return b"data"

        combined = chain(log_requests(log_capture), prefix_middleware)
        wrapped = combined(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"PREFIX:data"
        assert len(log_records) == 1

        log_data = json.loads(log_records[0].getMessage())
        # Logged response includes the prefix since log_requests is outermost
        assert log_data["response_bytes"] == 11  # len(b"PREFIX:data")

    @pytest.mark.asyncio
    async def test_measures_accurate_duration(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test log_requests measures duration accurately."""
        import asyncio

        async def slow_handler(ctx, data):
            await asyncio.sleep(0.05)  # 50ms
            return b"done"

        mw = log_requests(log_capture)
        wrapped = mw(slow_handler)
        await wrapped(ctx, b"input")

        log_data = json.loads(log_records[0].getMessage())
        # Should be at least 50ms
        assert log_data["duration_ms"] >= 50
        # But not too long (allow some overhead)
        assert log_data["duration_ms"] < 200

    @pytest.mark.asyncio
    async def test_preserves_exception_type(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test log_requests re-raises the original exception."""

        class CustomError(Exception):
            pass

        async def handler(ctx, data):
            raise CustomError("custom error")

        mw = log_requests(log_capture)
        wrapped = mw(handler)

        with pytest.raises(CustomError, match="custom error"):
            await wrapped(ctx, b"input")

        log_data = json.loads(log_records[0].getMessage())
        assert "CustomError" in log_data["error"]


class TestTimeout:
    """Tests for timeout built-in middleware."""

    @pytest.mark.asyncio
    async def test_fast_handler_completes(self, ctx: MiddlewareContext) -> None:
        """Test fast handler completes successfully within timeout."""

        async def fast_handler(ctx, data):
            return b"fast response"

        mw = timeout(1.0)
        wrapped = mw(fast_handler)
        result = await wrapped(ctx, b"input")

        assert result == b"fast response"

    @pytest.mark.asyncio
    async def test_slow_handler_times_out(self, ctx: MiddlewareContext) -> None:
        """Test slow handler raises TimeoutError."""
        import asyncio

        async def slow_handler(ctx, data):
            await asyncio.sleep(1.0)
            return b"slow response"

        mw = timeout(0.05)  # 50ms timeout
        wrapped = mw(slow_handler)

        with pytest.raises(AthyrTimeoutError) as exc_info:
            await wrapped(ctx, b"input")

        assert "timed out" in str(exc_info.value)
        assert exc_info.value.code == "timeout"
        assert "test.subject" in exc_info.value.details

    @pytest.mark.asyncio
    async def test_timeout_cancels_handler(self, ctx: MiddlewareContext) -> None:
        """Test timeout cancels the underlying handler task."""
        import asyncio

        cancelled = False

        async def cancellable_handler(ctx, data):
            nonlocal cancelled
            try:
                await asyncio.sleep(10.0)
                return b"should not reach"
            except asyncio.CancelledError:
                cancelled = True
                raise

        mw = timeout(0.05)
        wrapped = mw(cancellable_handler)

        with pytest.raises(AthyrTimeoutError):
            await wrapped(ctx, b"input")

        # Give a moment for cancellation to propagate
        await asyncio.sleep(0.01)
        assert cancelled is True

    @pytest.mark.asyncio
    async def test_timeout_with_exact_duration(self, ctx: MiddlewareContext) -> None:
        """Test handler completing just under timeout succeeds."""
        import asyncio

        async def just_in_time_handler(ctx, data):
            await asyncio.sleep(0.02)  # 20ms
            return b"made it"

        mw = timeout(0.1)  # 100ms timeout
        wrapped = mw(just_in_time_handler)
        result = await wrapped(ctx, b"input")

        assert result == b"made it"

    @pytest.mark.asyncio
    async def test_timeout_chains_with_other_middleware(
        self, ctx: MiddlewareContext
    ) -> None:
        """Test timeout works in middleware chain."""

        @middleware
        async def prefix_middleware(ctx, data, call_next):
            result = await call_next(ctx, data)
            return b"PREFIX:" + result

        async def handler(ctx, data):
            return b"data"

        combined = chain(timeout(1.0), prefix_middleware)
        wrapped = combined(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"PREFIX:data"

    @pytest.mark.asyncio
    async def test_timeout_preserves_handler_exceptions(
        self, ctx: MiddlewareContext
    ) -> None:
        """Test timeout doesn't swallow handler exceptions."""

        async def failing_handler(ctx, data):
            raise ValueError("handler error")

        mw = timeout(1.0)
        wrapped = mw(failing_handler)

        with pytest.raises(ValueError, match="handler error"):
            await wrapped(ctx, b"input")


class TestRetry:
    """Tests for retry built-in middleware."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self, ctx: MiddlewareContext) -> None:
        """Test successful handler doesn't retry."""
        attempts = 0

        async def handler(ctx, data):
            nonlocal attempts
            attempts += 1
            return b"success"

        mw = retry(max_attempts=3)
        wrapped = mw(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"success"
        assert attempts == 1

    @pytest.mark.asyncio
    async def test_retries_on_unavailable_error(self, ctx: MiddlewareContext) -> None:
        """Test retries on UnavailableError."""
        attempts = 0

        async def flaky_handler(ctx, data):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise UnavailableError("temporarily unavailable")
            return b"success"

        mw = retry(max_attempts=3, base_backoff=0.01)
        wrapped = mw(flaky_handler)
        result = await wrapped(ctx, b"input")

        assert result == b"success"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self, ctx: MiddlewareContext) -> None:
        """Test raises UnavailableError after max attempts exceeded."""
        attempts = 0

        async def always_fails(ctx, data):
            nonlocal attempts
            attempts += 1
            raise UnavailableError("always fails")

        mw = retry(max_attempts=3, base_backoff=0.01)
        wrapped = mw(always_fails)

        with pytest.raises(UnavailableError) as exc_info:
            await wrapped(ctx, b"input")

        assert attempts == 3
        assert "Max retries (3) exceeded" in str(exc_info.value)
        assert "test.subject" in exc_info.value.details

    @pytest.mark.asyncio
    async def test_does_not_retry_other_exceptions(
        self, ctx: MiddlewareContext
    ) -> None:
        """Test non-UnavailableError exceptions are not retried."""
        attempts = 0

        async def handler(ctx, data):
            nonlocal attempts
            attempts += 1
            raise ValueError("not retryable")

        mw = retry(max_attempts=3, base_backoff=0.01)
        wrapped = mw(handler)

        with pytest.raises(ValueError, match="not retryable"):
            await wrapped(ctx, b"input")

        assert attempts == 1  # No retry

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, ctx: MiddlewareContext) -> None:
        """Test backoff increases exponentially."""
        import time

        timestamps: list[float] = []

        async def failing_handler(ctx, data):
            timestamps.append(time.perf_counter())
            raise UnavailableError("fail")

        mw = retry(max_attempts=4, base_backoff=0.05, max_backoff=10.0)
        wrapped = mw(failing_handler)

        with pytest.raises(UnavailableError):
            await wrapped(ctx, b"input")

        # Check gaps between attempts increase
        # With jitter, we can't be exact, but gaps should generally increase
        assert len(timestamps) == 4
        gap1 = timestamps[1] - timestamps[0]
        gap2 = timestamps[2] - timestamps[1]
        gap3 = timestamps[3] - timestamps[2]

        # First gap: ~0.05s * jitter, second: ~0.1s * jitter, third: ~0.2s * jitter
        # With jitter range 0.5-1.5, gaps should be in reasonable ranges
        assert gap1 >= 0.02  # base * 0.5 minimum
        assert gap2 >= 0.04  # 2x base * 0.5 minimum
        assert gap3 >= 0.08  # 4x base * 0.5 minimum

    @pytest.mark.asyncio
    async def test_max_backoff_caps_delay(self, ctx: MiddlewareContext) -> None:
        """Test backoff is capped at max_backoff."""
        import time

        timestamps: list[float] = []

        async def failing_handler(ctx, data):
            timestamps.append(time.perf_counter())
            raise UnavailableError("fail")

        # Very small max_backoff to cap quickly
        mw = retry(max_attempts=3, base_backoff=1.0, max_backoff=0.05)
        wrapped = mw(failing_handler)

        with pytest.raises(UnavailableError):
            await wrapped(ctx, b"input")

        # All gaps should be around max_backoff (0.05s) with jitter
        gap1 = timestamps[1] - timestamps[0]
        gap2 = timestamps[2] - timestamps[1]

        # Both should be capped at ~0.05 * jitter
        assert gap1 < 0.15  # max_backoff * 1.5 (max jitter) + overhead
        assert gap2 < 0.15

    @pytest.mark.asyncio
    async def test_default_max_attempts(self, ctx: MiddlewareContext) -> None:
        """Test retry uses default max_attempts=3."""
        attempts = 0

        async def handler(ctx, data):
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise UnavailableError("fail once")
            return b"ok"

        # Use small backoff but default max_attempts
        mw = retry(base_backoff=0.01)
        wrapped = mw(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"ok"
        assert attempts == 2

    @pytest.mark.asyncio
    async def test_chains_with_other_middleware(
        self, ctx: MiddlewareContext
    ) -> None:
        """Test retry works in middleware chain."""
        attempts = 0

        @middleware
        async def prefix_middleware(ctx, data, call_next):
            result = await call_next(ctx, data)
            return b"PREFIX:" + result

        async def flaky_handler(ctx, data):
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise UnavailableError("fail once")
            return b"data"

        combined = chain(retry(max_attempts=3, base_backoff=0.01), prefix_middleware)
        wrapped = combined(flaky_handler)
        result = await wrapped(ctx, b"input")

        assert result == b"PREFIX:data"
        assert attempts == 2


class TestRecover:
    """Tests for recover built-in middleware."""

    @pytest.fixture
    def log_capture(self) -> logging.Logger:
        """Create a logger that captures log records."""
        logger = logging.getLogger("test.recover")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        return logger

    @pytest.fixture
    def log_records(self, log_capture: logging.Logger) -> list[logging.LogRecord]:
        """Capture log records from the test logger."""
        records: list[logging.LogRecord] = []

        class RecordHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        log_capture.addHandler(RecordHandler())
        return records

    @pytest.mark.asyncio
    async def test_success_passes_through(self, ctx: MiddlewareContext) -> None:
        """Test successful handler result passes through unchanged."""

        async def handler(ctx, data):
            return b"success"

        mw = recover()
        wrapped = mw(handler)
        result = await wrapped(ctx, b"input")

        assert result == b"success"

    @pytest.mark.asyncio
    async def test_converts_exception_to_internal_error(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test generic exceptions are converted to InternalError."""

        async def failing_handler(ctx, data):
            raise ValueError("something broke")

        mw = recover(log_capture)
        wrapped = mw(failing_handler)

        with pytest.raises(InternalError) as exc_info:
            await wrapped(ctx, b"input")

        assert exc_info.value.code == "internal"
        assert "test.subject" in exc_info.value.details

    @pytest.mark.asyncio
    async def test_logs_exception_with_traceback(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test exceptions are logged with full traceback."""

        async def failing_handler(ctx, data):
            raise RuntimeError("detailed error message")

        mw = recover(log_capture)
        wrapped = mw(failing_handler)

        with pytest.raises(InternalError):
            await wrapped(ctx, b"input")

        assert len(log_records) == 1
        assert log_records[0].levelno == logging.ERROR

        log_data = json.loads(log_records[0].getMessage())
        assert log_data["subject"] == "test.subject"
        assert "RuntimeError: detailed error message" in log_data["error"]
        assert "traceback" in log_data
        assert "RuntimeError" in log_data["traceback"]

    @pytest.mark.asyncio
    async def test_passes_through_internal_error(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test InternalError passes through without conversion."""

        async def handler(ctx, data):
            raise InternalError("already internal", code="custom_code")

        mw = recover(log_capture)
        wrapped = mw(handler)

        with pytest.raises(InternalError) as exc_info:
            await wrapped(ctx, b"input")

        # Should be the original error, not wrapped
        assert exc_info.value.code == "custom_code"
        assert "already internal" in str(exc_info.value)
        # Should not log since it's already a proper error
        assert len(log_records) == 0

    @pytest.mark.asyncio
    async def test_passes_through_timeout_error(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test TimeoutError passes through without conversion."""

        async def handler(ctx, data):
            raise AthyrTimeoutError("timed out", code="timeout")

        mw = recover(log_capture)
        wrapped = mw(handler)

        with pytest.raises(AthyrTimeoutError) as exc_info:
            await wrapped(ctx, b"input")

        assert exc_info.value.code == "timeout"
        assert len(log_records) == 0

    @pytest.mark.asyncio
    async def test_passes_through_unavailable_error(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test UnavailableError passes through without conversion."""

        async def handler(ctx, data):
            raise UnavailableError("service down", code="unavailable")

        mw = recover(log_capture)
        wrapped = mw(handler)

        with pytest.raises(UnavailableError) as exc_info:
            await wrapped(ctx, b"input")

        assert exc_info.value.code == "unavailable"
        assert len(log_records) == 0

    @pytest.mark.asyncio
    async def test_uses_default_logger(self, ctx: MiddlewareContext) -> None:
        """Test recover uses default logger when none provided."""
        # Just verify it doesn't raise
        mw = recover()

        async def handler(ctx, data):
            return b"ok"

        wrapped = mw(handler)
        result = await wrapped(ctx, b"input")
        assert result == b"ok"

    @pytest.mark.asyncio
    async def test_chains_as_outermost_middleware(
        self, ctx: MiddlewareContext, log_capture: logging.Logger, log_records: list
    ) -> None:
        """Test recover works as outermost middleware in chain."""

        @middleware
        async def inner_middleware(ctx, data, call_next):
            # This middleware itself throws
            raise KeyError("middleware error")

        async def handler(ctx, data):
            return b"never reached"

        combined = chain(recover(log_capture), inner_middleware)
        wrapped = combined(handler)

        with pytest.raises(InternalError):
            await wrapped(ctx, b"input")

        # Should have logged the KeyError
        log_data = json.loads(log_records[0].getMessage())
        assert "KeyError" in log_data["error"]
