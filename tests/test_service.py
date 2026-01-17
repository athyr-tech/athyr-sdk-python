"""Tests for Service abstraction."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from athyr.service import (
    BadRequestError,
    DefaultServiceContext,
    InternalError,
    NotFoundError,
    Service,
    ServiceError,
    UnavailableError,
    bad_request,
    internal,
    not_found,
    unavailable,
)


# Test fixtures


@dataclass
class GreetRequest:
    name: str


@dataclass
class GreetResponse:
    message: str


class TestServiceContext:
    """Tests for ServiceContext."""

    def test_default_context_properties(self) -> None:
        """Test DefaultServiceContext exposes all properties."""
        mock_agent = MagicMock()
        ctx = DefaultServiceContext(
            _agent=mock_agent,
            _subject="test.subject",
            _reply_subject="_INBOX.123",
        )

        assert ctx.agent is mock_agent
        assert ctx.subject == "test.subject"
        assert ctx.reply_subject == "_INBOX.123"


class TestService:
    """Tests for Service class."""

    def test_create_typed_service(self) -> None:
        """Test creating a service with typed handler."""

        async def handler(ctx: Any, req: GreetRequest) -> GreetResponse:
            return GreetResponse(message=f"Hello, {req.name}!")

        service = Service(
            subject="greet.invoke",
            handler=handler,
            request_type=GreetRequest,
            response_type=GreetResponse,
        )

        assert service.subject == "greet.invoke"
        assert service.name == "greet.invoke"  # Defaults to subject
        assert service.handler is handler
        assert service.queue_group == ""

    def test_create_typed_service_with_name(self) -> None:
        """Test creating a service with custom name."""

        async def handler(ctx: Any, req: GreetRequest) -> GreetResponse:
            return GreetResponse(message=f"Hello, {req.name}!")

        service = Service(
            subject="greet.invoke",
            handler=handler,
            request_type=GreetRequest,
            response_type=GreetResponse,
            name="Greeting Service",
        )

        assert service.name == "Greeting Service"

    def test_create_typed_service_with_queue_group(self) -> None:
        """Test creating a service with queue group."""

        async def handler(ctx: Any, req: GreetRequest) -> GreetResponse:
            return GreetResponse(message=f"Hello, {req.name}!")

        service = Service(
            subject="greet.invoke",
            handler=handler,
            request_type=GreetRequest,
            response_type=GreetResponse,
            queue_group="greet-workers",
        )

        assert service.queue_group == "greet-workers"

    def test_create_raw_service(self) -> None:
        """Test creating a service with raw handler."""

        async def raw_handler(ctx: Any, data: bytes) -> bytes:
            return data  # Echo

        service = Service.raw(
            subject="echo",
            handler=raw_handler,
        )

        assert service.subject == "echo"
        assert service.raw_handler is raw_handler
        assert service.handler is None

    def test_create_raw_service_with_options(self) -> None:
        """Test creating raw service with all options."""

        async def raw_handler(ctx: Any, data: bytes) -> bytes:
            return data

        service = Service.raw(
            subject="echo",
            handler=raw_handler,
            name="Echo Service",
            queue_group="echo-workers",
        )

        assert service.name == "Echo Service"
        assert service.queue_group == "echo-workers"

    def test_service_requires_handler(self) -> None:
        """Test service raises error without handler."""
        with pytest.raises(ValueError, match="Must specify either handler or raw_handler"):
            Service(subject="test")

    def test_service_rejects_both_handlers(self) -> None:
        """Test service rejects both handler and raw_handler."""

        async def typed_handler(ctx: Any, req: GreetRequest) -> GreetResponse:
            return GreetResponse(message="Hi")

        async def raw_handler(ctx: Any, data: bytes) -> bytes:
            return data

        with pytest.raises(ValueError, match="Cannot specify both"):
            Service(
                subject="test",
                handler=typed_handler,
                raw_handler=raw_handler,
                request_type=GreetRequest,
                response_type=GreetResponse,
            )

    def test_typed_service_requires_request_type(self) -> None:
        """Test typed service requires request_type."""

        async def handler(ctx: Any, req: GreetRequest) -> GreetResponse:
            return GreetResponse(message="Hi")

        with pytest.raises(ValueError, match="request_type required"):
            Service(
                subject="test",
                handler=handler,
                response_type=GreetResponse,
            )

    def test_typed_service_requires_response_type(self) -> None:
        """Test typed service requires response_type."""

        async def handler(ctx: Any, req: GreetRequest) -> GreetResponse:
            return GreetResponse(message="Hi")

        with pytest.raises(ValueError, match="response_type required"):
            Service(
                subject="test",
                handler=handler,
                request_type=GreetRequest,
            )


class TestServiceBuildHandler:
    """Tests for Service.build_handler()."""

    @pytest.mark.asyncio
    async def test_build_raw_handler(self) -> None:
        """Test building handler for raw service."""
        calls: list[bytes] = []

        async def raw_handler(ctx: Any, data: bytes) -> bytes:
            calls.append(data)
            return b"response"

        service = Service.raw(subject="test", handler=raw_handler)
        handler = service.build_handler()

        mock_ctx = MagicMock()
        result = await handler(mock_ctx, b"input")

        assert result == b"response"
        assert calls == [b"input"]

    @pytest.mark.asyncio
    async def test_build_typed_handler_deserializes_request(self) -> None:
        """Test typed handler deserializes JSON request."""
        received_req: GreetRequest | None = None

        async def handler(ctx: Any, req: GreetRequest) -> GreetResponse:
            nonlocal received_req
            received_req = req
            return GreetResponse(message=f"Hello, {req.name}!")

        service = Service(
            subject="greet",
            handler=handler,
            request_type=GreetRequest,
            response_type=GreetResponse,
        )
        built_handler = service.build_handler()

        mock_ctx = MagicMock()
        result = await built_handler(mock_ctx, b'{"name": "World"}')

        assert received_req is not None
        assert received_req.name == "World"
        assert b"Hello, World!" in result

    @pytest.mark.asyncio
    async def test_build_typed_handler_serializes_response(self) -> None:
        """Test typed handler serializes response to JSON."""

        async def handler(ctx: Any, req: GreetRequest) -> GreetResponse:
            return GreetResponse(message="Hi there")

        service = Service(
            subject="greet",
            handler=handler,
            request_type=GreetRequest,
            response_type=GreetResponse,
        )
        built_handler = service.build_handler()

        mock_ctx = MagicMock()
        result = await built_handler(mock_ctx, b'{"name": "Test"}')

        import json
        resp = json.loads(result)
        assert resp["message"] == "Hi there"

    @pytest.mark.asyncio
    async def test_build_typed_handler_invalid_json(self) -> None:
        """Test typed handler raises BadRequestError for invalid JSON."""

        async def handler(ctx: Any, req: GreetRequest) -> GreetResponse:
            return GreetResponse(message="Hi")

        service = Service(
            subject="greet",
            handler=handler,
            request_type=GreetRequest,
            response_type=GreetResponse,
        )
        built_handler = service.build_handler()

        mock_ctx = MagicMock()
        with pytest.raises(BadRequestError, match="Invalid request"):
            await built_handler(mock_ctx, b"not valid json")

    @pytest.mark.asyncio
    async def test_build_handler_with_middleware(self) -> None:
        """Test middleware is applied in correct order."""
        order: list[str] = []

        async def base_handler(ctx: Any, data: bytes) -> bytes:
            order.append("handler")
            return data

        def middleware_a(next_handler: Any) -> Any:
            async def wrapper(ctx: Any, data: bytes) -> bytes:
                order.append("A-before")
                result = await next_handler(ctx, data)
                order.append("A-after")
                return result
            return wrapper

        def middleware_b(next_handler: Any) -> Any:
            async def wrapper(ctx: Any, data: bytes) -> bytes:
                order.append("B-before")
                result = await next_handler(ctx, data)
                order.append("B-after")
                return result
            return wrapper

        service = Service.raw(
            subject="test",
            handler=base_handler,
            middleware=[middleware_a, middleware_b],
        )
        handler = service.build_handler()

        mock_ctx = MagicMock()
        await handler(mock_ctx, b"test")

        # Service middleware applied inner-to-outer (A wraps B wraps handler)
        assert order == ["A-before", "B-before", "handler", "B-after", "A-after"]

    @pytest.mark.asyncio
    async def test_build_handler_with_global_middleware(self) -> None:
        """Test global middleware is applied as outer layer."""
        order: list[str] = []

        async def base_handler(ctx: Any, data: bytes) -> bytes:
            order.append("handler")
            return data

        def service_mw(next_handler: Any) -> Any:
            async def wrapper(ctx: Any, data: bytes) -> bytes:
                order.append("service-before")
                result = await next_handler(ctx, data)
                order.append("service-after")
                return result
            return wrapper

        def global_mw(next_handler: Any) -> Any:
            async def wrapper(ctx: Any, data: bytes) -> bytes:
                order.append("global-before")
                result = await next_handler(ctx, data)
                order.append("global-after")
                return result
            return wrapper

        service = Service.raw(
            subject="test",
            handler=base_handler,
            middleware=[service_mw],
        )
        handler = service.build_handler(global_middleware=[global_mw])

        mock_ctx = MagicMock()
        await handler(mock_ctx, b"test")

        # Global is outer, service is inner
        assert order == ["global-before", "service-before", "handler", "service-after", "global-after"]


class TestServiceErrors:
    """Tests for ServiceError classes."""

    def test_service_error_str(self) -> None:
        """Test ServiceError string representation."""
        err = ServiceError(code="test", message="Test error")
        assert str(err) == "Test error"

    def test_service_error_to_dict(self) -> None:
        """Test ServiceError to_dict()."""
        err = ServiceError(code="test", message="Test error", details={"key": "value"})
        d = err.to_dict()

        assert d["error"] == "Test error"
        assert d["code"] == "test"
        assert d["details"] == {"key": "value"}

    def test_service_error_to_dict_without_details(self) -> None:
        """Test ServiceError to_dict() without details."""
        err = ServiceError(code="test", message="Test error")
        d = err.to_dict()

        assert "details" not in d

    def test_service_error_to_json(self) -> None:
        """Test ServiceError to_json()."""
        err = ServiceError(code="test", message="Test error")
        json_bytes = err.to_json()

        import json
        data = json.loads(json_bytes)
        assert data["error"] == "Test error"
        assert data["code"] == "test"

    def test_bad_request_error(self) -> None:
        """Test BadRequestError has correct code."""
        err = BadRequestError("Invalid input")
        assert err.code == "bad_request"
        assert err.message == "Invalid input"

    def test_internal_error(self) -> None:
        """Test InternalError has correct code."""
        err = InternalError("Server failed")
        assert err.code == "internal"

    def test_unavailable_error(self) -> None:
        """Test UnavailableError has correct code."""
        err = UnavailableError("Try again later")
        assert err.code == "unavailable"

    def test_not_found_error(self) -> None:
        """Test NotFoundError has correct code."""
        err = NotFoundError("Resource missing")
        assert err.code == "not_found"


class TestErrorHelperFunctions:
    """Tests for error helper functions."""

    def test_bad_request_function(self) -> None:
        """Test bad_request() creates BadRequestError."""
        err = bad_request("Invalid field: %s", "name")
        assert isinstance(err, BadRequestError)
        assert err.message == "Invalid field: name"

    def test_internal_function(self) -> None:
        """Test internal() creates InternalError."""
        err = internal("Database error")
        assert isinstance(err, InternalError)
        assert err.message == "Database error"

    def test_unavailable_function(self) -> None:
        """Test unavailable() creates UnavailableError."""
        err = unavailable("Service %s is down", "auth")
        assert isinstance(err, UnavailableError)
        assert err.message == "Service auth is down"

    def test_not_found_function(self) -> None:
        """Test not_found() creates NotFoundError."""
        err = not_found("User %d not found", 123)
        assert isinstance(err, NotFoundError)
        assert err.message == "User 123 not found"
