"""Middleware system for request/response interception.

This module provides a Pythonic middleware pattern similar to FastAPI/Starlette.
Middlewares can intercept requests and responses to add cross-cutting concerns
like logging, timeouts, retries, and error recovery.

Example:
    from athyr.middleware import middleware, chain, MiddlewareContext

    @middleware
    async def log_requests(ctx: MiddlewareContext, data: bytes, call_next) -> bytes:
        print(f"Request to {ctx.subject}")
        start = time.time()
        result = await call_next(ctx, data)
        print(f"Response in {time.time() - start:.3f}s")
        return result

    @middleware
    async def add_header(ctx: MiddlewareContext, data: bytes, call_next) -> bytes:
        result = await call_next(ctx, data)
        return b"HEADER:" + result

    # Compose middlewares (first is outermost)
    combined = chain(log_requests, add_header)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload

if TYPE_CHECKING:
    from athyr.client import AthyrAgent


@dataclass
class MiddlewareContext:
    """Context passed to middleware handlers.

    Contains information about the current request and access to the agent.

    Attributes:
        agent: The Athyr agent handling this request.
        subject: The subject/topic this request arrived on.
        reply_subject: Where to send the response (used internally).
        metadata: Optional dictionary for middleware to pass data down the chain.
    """

    agent: AthyrAgent
    subject: str
    reply_subject: str = ""
    metadata: dict[str, Any] | None = None

    def with_metadata(self, key: str, value: Any) -> MiddlewareContext:
        """Create a new context with additional metadata.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            New MiddlewareContext with the metadata added.
        """
        new_metadata = dict(self.metadata) if self.metadata else {}
        new_metadata[key] = value
        return MiddlewareContext(
            agent=self.agent,
            subject=self.subject,
            reply_subject=self.reply_subject,
            metadata=new_metadata,
        )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key.

        Args:
            key: Metadata key to retrieve.
            default: Default value if key not found.

        Returns:
            The metadata value or default.
        """
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)


# Type aliases for handlers and middleware

# Handler: async function that processes a request
# (ctx, data) -> response_data
Handler = Callable[[MiddlewareContext, bytes], Awaitable[bytes]]

# NextHandler: the next handler in the chain (same signature as Handler)
NextHandler = Handler

# MiddlewareFunc: the user-defined middleware function signature
# (ctx, data, call_next) -> response_data
MiddlewareFunc = Callable[
    [MiddlewareContext, bytes, NextHandler],
    Awaitable[bytes],
]

# Middleware: a wrapped middleware that can be composed
# Takes a handler and returns a wrapped handler
Middleware = Callable[[Handler], Handler]


class MiddlewareProtocol(Protocol):
    """Protocol for middleware objects.

    Middleware can be either:
    - A function decorated with @middleware
    - A class implementing __call__(handler: Handler) -> Handler
    """

    def __call__(self, handler: Handler) -> Handler:
        """Wrap a handler with this middleware.

        Args:
            handler: The next handler in the chain.

        Returns:
            A wrapped handler that includes this middleware's logic.
        """
        ...


def middleware(func: MiddlewareFunc) -> Middleware:
    """Decorator to create a middleware from a simple async function.

    The decorated function receives (ctx, data, call_next) and should:
    1. Optionally modify the context or data
    2. Call `await call_next(ctx, data)` to invoke the next handler
    3. Optionally modify the response
    4. Return the response

    Args:
        func: Async function with signature (ctx, data, call_next) -> bytes

    Returns:
        A Middleware that can be used with chain() or applied directly.

    Example:
        @middleware
        async def log_requests(ctx, data, call_next):
            print(f"-> {ctx.subject}")
            result = await call_next(ctx, data)
            print(f"<- {len(result)} bytes")
            return result

        # Use directly
        handler = log_requests(my_handler)

        # Or compose with chain()
        combined = chain(log_requests, other_middleware)
    """

    @wraps(func)
    def wrapper(next_handler: Handler) -> Handler:
        @wraps(func)
        async def wrapped_handler(ctx: MiddlewareContext, data: bytes) -> bytes:
            return await func(ctx, data, next_handler)

        return wrapped_handler

    return wrapper


def chain(*middlewares: Middleware) -> Middleware:
    """Compose multiple middlewares into a single middleware.

    Middlewares are applied in order: the first middleware in the list
    is the outermost (executes first on request, last on response).

    Args:
        *middlewares: Middleware functions to compose.

    Returns:
        A single Middleware that applies all middlewares in order.

    Example:
        # Order matters: log -> timeout -> retry -> handler
        combined = chain(log_requests, timeout(5.0), retry(3))

        # Apply to a handler
        final_handler = combined(my_handler)

        # Execution order for a request:
        # 1. log_requests (before)
        # 2. timeout (before)
        # 3. retry (before)
        # 4. my_handler
        # 5. retry (after)
        # 6. timeout (after)
        # 7. log_requests (after)
    """
    if not middlewares:
        # Identity middleware - just returns the handler unchanged
        return lambda handler: handler

    def combined(handler: Handler) -> Handler:
        # Apply middlewares in reverse order so first is outermost
        result = handler
        for mw in reversed(middlewares):
            result = mw(result)
        return result

    return combined


# Utility function to create a simple passthrough handler for testing
async def passthrough_handler(ctx: MiddlewareContext, data: bytes) -> bytes:
    """A simple handler that returns the input data unchanged.

    Useful for testing middlewares.

    Args:
        ctx: The middleware context.
        data: The input data.

    Returns:
        The input data unchanged.
    """
    return data
