"""Tests for exception hierarchy."""

import pytest
from grpc import StatusCode

from athyr.exceptions import (
    AthyrError,
    AuthenticationError,
    CancelledError,
    ConnectionError,
    InternalError,
    InvalidRequestError,
    NotFoundError,
    ResourceExhaustedError,
    StreamError,
    TimeoutError,
    UnavailableError,
    from_grpc_error,
    is_auth_error,
    is_cancelled,
    is_not_found,
    is_retriable,
    is_timeout,
    is_unavailable,
)


class TestAthyrError:
    """Tests for base AthyrError."""

    def test_create_simple(self) -> None:
        """Test creating error with message only."""
        err = AthyrError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.code is None
        assert err.details is None

    def test_create_with_code(self) -> None:
        """Test creating error with code."""
        err = AthyrError("Error", code="ERR_CODE")
        assert str(err) == "Error [ERR_CODE]"
        assert err.code == "ERR_CODE"

    def test_create_with_details(self) -> None:
        """Test creating error with details."""
        err = AthyrError("Error", details="extra info")
        assert str(err) == "Error (extra info)"

    def test_create_full(self) -> None:
        """Test creating error with all fields."""
        err = AthyrError("Error", code="CODE", details="details")
        assert str(err) == "Error [CODE] (details)"


class TestSpecificExceptions:
    """Tests for specific exception types."""

    def test_connection_error(self) -> None:
        """Test ConnectionError."""
        err = ConnectionError("Connection failed")
        assert isinstance(err, AthyrError)
        assert "Connection failed" in str(err)

    def test_not_found_error(self) -> None:
        """Test NotFoundError."""
        err = NotFoundError("Resource not found", code="NOT_FOUND")
        assert isinstance(err, AthyrError)
        assert is_not_found(err)

    def test_unavailable_error(self) -> None:
        """Test UnavailableError."""
        err = UnavailableError("Service down")
        assert isinstance(err, AthyrError)
        assert is_unavailable(err)

    def test_timeout_error(self) -> None:
        """Test TimeoutError."""
        err = TimeoutError("Operation timed out")
        assert isinstance(err, AthyrError)
        assert is_timeout(err)

    def test_cancelled_error(self) -> None:
        """Test CancelledError."""
        err = CancelledError("Operation cancelled")
        assert isinstance(err, AthyrError)
        assert is_cancelled(err)

    def test_auth_error(self) -> None:
        """Test AuthenticationError."""
        err = AuthenticationError("Invalid credentials")
        assert isinstance(err, AthyrError)
        assert is_auth_error(err)


class TestStreamError:
    """Tests for StreamError."""

    def test_create_simple(self) -> None:
        """Test creating StreamError."""
        err = StreamError("Stream failed")
        assert err.accumulated_content == ""
        assert err.partial_response is False

    def test_create_with_partial(self) -> None:
        """Test creating StreamError with partial response."""
        err = StreamError(
            "Stream failed",
            accumulated_content="partial content",
            partial_response=True,
        )
        assert err.accumulated_content == "partial content"
        assert err.partial_response is True


class TestHelperFunctions:
    """Tests for error checking helpers."""

    def test_is_not_found(self) -> None:
        """Test is_not_found helper."""
        assert is_not_found(NotFoundError("test"))
        assert not is_not_found(UnavailableError("test"))
        assert not is_not_found(Exception("test"))

    def test_is_unavailable(self) -> None:
        """Test is_unavailable helper."""
        assert is_unavailable(UnavailableError("test"))
        assert not is_unavailable(NotFoundError("test"))

    def test_is_timeout(self) -> None:
        """Test is_timeout helper."""
        assert is_timeout(TimeoutError("test"))
        assert not is_timeout(NotFoundError("test"))

    def test_is_retriable(self) -> None:
        """Test is_retriable helper."""
        assert is_retriable(UnavailableError("test"))
        assert is_retriable(TimeoutError("test"))
        assert is_retriable(ResourceExhaustedError("test"))
        assert not is_retriable(NotFoundError("test"))
        assert not is_retriable(InvalidRequestError("test"))


class TestGrpcErrorMapping:
    """Tests for gRPC error mapping."""

    def test_unknown_error(self) -> None:
        """Test mapping unknown error."""
        err = from_grpc_error(Exception("unknown"))
        assert isinstance(err, AthyrError)

    # Note: Full gRPC error mapping tests would require mocking
    # grpc.RpcError which has a specific structure
