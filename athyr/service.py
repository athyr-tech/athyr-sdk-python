"""Service abstraction for Athyr request handling.

This module provides the foundation for building services that handle
requests with typed handlers and middleware support.

Example:
    from athyr.service import Service, ServiceContext

    async def handle_greet(ctx: ServiceContext, req: GreetRequest) -> GreetResponse:
        return GreetResponse(message=f"Hello, {req.name}!")

    service = Service(
        subject="greet.invoke",
        handler=handle_greet,
        request_type=GreetRequest,
        response_type=GreetResponse,
    )
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from athyr.client import Agent

logger = logging.getLogger(__name__)

# Type variables for generic handlers
Req = TypeVar("Req")
Resp = TypeVar("Resp")


class ServiceContext(Protocol):
    """Context passed to service handlers.

    Provides access to the agent and request metadata.
    """

    @property
    def agent(self) -> Agent:
        """Get the underlying agent connection."""
        ...

    @property
    def subject(self) -> str:
        """Get the subject this request arrived on."""
        ...

    @property
    def reply_subject(self) -> str:
        """Get where to send the response (used internally)."""
        ...


@dataclass
class DefaultServiceContext:
    """Default implementation of ServiceContext."""

    _agent: Agent
    _subject: str
    _reply_subject: str

    @property
    def agent(self) -> Agent:
        """Get the underlying agent connection."""
        return self._agent

    @property
    def subject(self) -> str:
        """Get the subject this request arrived on."""
        return self._subject

    @property
    def reply_subject(self) -> str:
        """Get where to send the response."""
        return self._reply_subject


# Handler type aliases
RawHandler = Callable[[ServiceContext, bytes], Awaitable[bytes]]
"""Handler that works with raw bytes for full control over serialization."""

Middleware = Callable[[RawHandler], RawHandler]
"""Middleware wraps a RawHandler to add cross-cutting concerns."""


@dataclass
class Service(Generic[Req, Resp]):
    """Represents a single endpoint that an agent exposes.

    Services handle requests on a subject with optional queue group
    for load balancing. Handlers can be typed (with automatic JSON
    marshaling) or raw (for full control).

    Example:
        # Typed handler with automatic JSON marshaling
        service = Service(
            subject="math.add",
            handler=add_numbers,
            request_type=AddRequest,
            response_type=AddResponse,
        )

        # Raw handler for full control
        raw_service = Service.raw(
            subject="echo",
            handler=echo_handler,
        )
    """

    subject: str
    """The subject to subscribe to for requests."""

    handler: Callable[[ServiceContext, Req], Awaitable[Resp]] | None = None
    """Typed handler function (mutually exclusive with raw_handler)."""

    raw_handler: RawHandler | None = None
    """Raw handler function (mutually exclusive with handler)."""

    request_type: type[Req] | None = None
    """Type for deserializing requests (required for typed handlers)."""

    response_type: type[Resp] | None = None
    """Type for serializing responses (required for typed handlers)."""

    name: str = ""
    """Display name for the service (defaults to subject)."""

    queue_group: str = ""
    """Queue group for load balancing (empty for broadcast)."""

    middleware: list[Middleware] = field(default_factory=list)
    """Service-specific middleware (applied before global middleware)."""

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        if not self.name:
            self.name = self.subject

        if self.handler is not None and self.raw_handler is not None:
            raise ValueError("Cannot specify both handler and raw_handler")

        if self.handler is None and self.raw_handler is None:
            raise ValueError("Must specify either handler or raw_handler")

        if self.handler is not None:
            if self.request_type is None:
                raise ValueError("request_type required for typed handlers")
            if self.response_type is None:
                raise ValueError("response_type required for typed handlers")

    @classmethod
    def raw(
        cls,
        subject: str,
        handler: RawHandler,
        *,
        name: str = "",
        queue_group: str = "",
        middleware: list[Middleware] | None = None,
    ) -> Service[Any, Any]:
        """Create a service with a raw byte handler.

        Use this when you need full control over serialization.

        Args:
            subject: The subject to subscribe to.
            handler: Raw handler function.
            name: Display name (defaults to subject).
            queue_group: Queue group for load balancing.
            middleware: Service-specific middleware.

        Returns:
            A Service configured with the raw handler.
        """
        return cls(
            subject=subject,
            raw_handler=handler,
            name=name or subject,
            queue_group=queue_group,
            middleware=middleware or [],
        )

    def build_handler(self, global_middleware: list[Middleware] | None = None) -> RawHandler:
        """Build the final handler with all middleware applied.

        Args:
            global_middleware: Global middleware to apply (outer layer).

        Returns:
            The composed handler with all middleware.
        """
        # Start with the base handler
        if self.raw_handler is not None:
            handler = self.raw_handler
        else:
            handler = self._wrap_typed_handler()

        # Apply service-specific middleware (inner layer)
        for mw in reversed(self.middleware):
            handler = mw(handler)

        # Apply global middleware (outer layer)
        if global_middleware:
            for mw in reversed(global_middleware):
                handler = mw(handler)

        return handler

    def _wrap_typed_handler(self) -> RawHandler:
        """Wrap the typed handler with JSON marshaling."""
        assert self.handler is not None
        assert self.request_type is not None
        assert self.response_type is not None

        typed_handler = self.handler
        req_type = self.request_type
        # resp_type = self.response_type  # Not needed for serialization

        async def wrapper(ctx: ServiceContext, data: bytes) -> bytes:
            # Deserialize request
            try:
                if data:
                    req_dict = json.loads(data)
                    # Handle Pydantic models and dataclasses
                    if hasattr(req_type, "model_validate"):
                        req = req_type.model_validate(req_dict)
                    elif hasattr(req_type, "__dataclass_fields__"):
                        req = req_type(**req_dict)
                    else:
                        req = req_type(req_dict)  # type: ignore
                else:
                    # Empty request - try to create empty instance
                    if hasattr(req_type, "model_validate"):
                        req = req_type.model_validate({})
                    elif hasattr(req_type, "__dataclass_fields__"):
                        req = req_type()  # type: ignore
                    else:
                        req = req_type()  # type: ignore
            except Exception as e:
                raise BadRequestError(f"Invalid request: {e}") from e

            # Call the typed handler
            resp = await typed_handler(ctx, req)

            # Serialize response
            try:
                if hasattr(resp, "model_dump"):
                    resp_dict = resp.model_dump()
                elif hasattr(resp, "__dataclass_fields__"):
                    import dataclasses
                    resp_dict = dataclasses.asdict(resp)
                elif hasattr(resp, "__dict__"):
                    resp_dict = resp.__dict__
                else:
                    resp_dict = resp

                return json.dumps(resp_dict).encode()
            except Exception as e:
                raise InternalError(f"Failed to marshal response: {e}") from e

        return wrapper


# Service Errors


@dataclass
class ServiceError(Exception):
    """Structured error with code and details.

    Service errors are automatically serialized to JSON responses
    with the error code and optional details.
    """

    code: str
    """Error code (e.g., 'bad_request', 'internal', 'not_found')."""

    message: str
    """Human-readable error message."""

    details: Any = None
    """Optional additional details."""

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "error": self.message,
            "code": self.code,
        }
        if self.details is not None:
            result["details"] = self.details
        return result

    def to_json(self) -> bytes:
        """Serialize to JSON bytes."""
        return json.dumps(self.to_dict()).encode()


class BadRequestError(ServiceError):
    """Client error - invalid input."""

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(code="bad_request", message=message, details=details)


class InternalError(ServiceError):
    """Server error - internal failure."""

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(code="internal", message=message, details=details)


class UnavailableError(ServiceError):
    """Temporary failure - retry later."""

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(code="unavailable", message=message, details=details)


class NotFoundError(ServiceError):
    """Resource not found."""

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(code="not_found", message=message, details=details)


# Convenience functions for creating errors (Go SDK style)


def bad_request(message: str, *args: Any) -> BadRequestError:
    """Create a client error (invalid input)."""
    if args:
        message = message % args
    return BadRequestError(message)


def internal(message: str, *args: Any) -> InternalError:
    """Create a server error."""
    if args:
        message = message % args
    return InternalError(message)


def unavailable(message: str, *args: Any) -> UnavailableError:
    """Create a temporary failure error (retry later)."""
    if args:
        message = message % args
    return UnavailableError(message)


def not_found(message: str, *args: Any) -> NotFoundError:
    """Create a not found error."""
    if args:
        message = message % args
    return NotFoundError(message)
