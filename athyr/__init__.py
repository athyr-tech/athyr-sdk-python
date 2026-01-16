"""Athyr Python SDK - AI agent runtime client."""

__version__ = "0.1.0"

from athyr.client import Agent, AthyrAgent
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
)
from athyr.kv import Bucket, KVEntry
from athyr.models import AgentCard, Session, SessionMessage, SessionProfile, Tool
from athyr.tools import get_tool, is_tool, tool

__all__ = [
    # Version
    "__version__",
    # Client
    "Agent",
    "AthyrAgent",
    # Models
    "AgentCard",
    "Session",
    "SessionMessage",
    "SessionProfile",
    "Tool",
    # KV Storage
    "Bucket",
    "KVEntry",
    # Tools
    "tool",
    "get_tool",
    "is_tool",
    # Exceptions
    "AthyrError",
    "AuthenticationError",
    "CancelledError",
    "ConnectionError",
    "InternalError",
    "InvalidRequestError",
    "NotFoundError",
    "ResourceExhaustedError",
    "StreamError",
    "TimeoutError",
    "UnavailableError",
]
