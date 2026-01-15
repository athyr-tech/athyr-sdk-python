"""Exception hierarchy for Athyr SDK."""

from __future__ import annotations

from grpc import StatusCode


class AthyrError(Exception):
    """Base exception for all Athyr SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details

    def __str__(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"[{self.code}]")
        if self.details:
            parts.append(f"({self.details})")
        return " ".join(parts)


class ConnectionError(AthyrError):
    """Connection to Athyr server failed."""

    pass


class AuthenticationError(AthyrError):
    """Authentication failed."""

    pass


class NotFoundError(AthyrError):
    """Requested resource not found."""

    pass


class UnavailableError(AthyrError):
    """Service temporarily unavailable."""

    pass


class TimeoutError(AthyrError):
    """Operation timed out."""

    pass


class InvalidRequestError(AthyrError):
    """Invalid request parameters."""

    pass


class InternalError(AthyrError):
    """Internal server error."""

    pass


class CancelledError(AthyrError):
    """Operation was cancelled."""

    pass


class ResourceExhaustedError(AthyrError):
    """Resource exhausted (rate limit, quota)."""

    pass


class StreamError(AthyrError):
    """Error during streaming operation."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: str | None = None,
        accumulated_content: str = "",
        partial_response: bool = False,
    ) -> None:
        super().__init__(message, code=code, details=details)
        self.accumulated_content = accumulated_content
        self.partial_response = partial_response


# gRPC status code to exception mapping
_GRPC_STATUS_MAP: dict[StatusCode, type[AthyrError]] = {
    StatusCode.CANCELLED: CancelledError,
    StatusCode.UNKNOWN: InternalError,
    StatusCode.INVALID_ARGUMENT: InvalidRequestError,
    StatusCode.DEADLINE_EXCEEDED: TimeoutError,
    StatusCode.NOT_FOUND: NotFoundError,
    StatusCode.ALREADY_EXISTS: InvalidRequestError,
    StatusCode.PERMISSION_DENIED: AuthenticationError,
    StatusCode.RESOURCE_EXHAUSTED: ResourceExhaustedError,
    StatusCode.FAILED_PRECONDITION: InvalidRequestError,
    StatusCode.ABORTED: InternalError,
    StatusCode.OUT_OF_RANGE: InvalidRequestError,
    StatusCode.UNIMPLEMENTED: InvalidRequestError,
    StatusCode.INTERNAL: InternalError,
    StatusCode.UNAVAILABLE: UnavailableError,
    StatusCode.DATA_LOSS: InternalError,
    StatusCode.UNAUTHENTICATED: AuthenticationError,
}


def from_grpc_error(error: Exception) -> AthyrError:
    """Convert a gRPC error to an Athyr exception.

    Args:
        error: A gRPC RpcError or similar exception.

    Returns:
        Appropriate AthyrError subclass.
    """
    # Handle grpc.RpcError
    code = getattr(error, "code", lambda: None)()
    details = getattr(error, "details", lambda: str(error))()

    if code is None:
        return AthyrError(str(error))

    exc_class = _GRPC_STATUS_MAP.get(code, AthyrError)
    return exc_class(details, code=code.name if code else None)


# Helper functions for error checking


def is_not_found(error: Exception) -> bool:
    """Check if error is a NotFoundError."""
    return isinstance(error, NotFoundError)


def is_unavailable(error: Exception) -> bool:
    """Check if error is an UnavailableError."""
    return isinstance(error, UnavailableError)


def is_timeout(error: Exception) -> bool:
    """Check if error is a TimeoutError."""
    return isinstance(error, TimeoutError)


def is_cancelled(error: Exception) -> bool:
    """Check if error is a CancelledError."""
    return isinstance(error, CancelledError)


def is_auth_error(error: Exception) -> bool:
    """Check if error is an authentication error."""
    return isinstance(error, AuthenticationError)


def is_retriable(error: Exception) -> bool:
    """Check if error is retriable (transient)."""
    return isinstance(error, (UnavailableError, TimeoutError, ResourceExhaustedError))