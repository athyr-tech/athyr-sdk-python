"""Athyr Python SDK - AI agent runtime client."""

__version__ = "0.1.0"

from athyr.client import Agent, AthyrAgent
from athyr.client_state import ConnectionState, StateChangeCallback
from athyr.exceptions import (
    AthyrError,
    AuthenticationError,
    CancelledError,
    ConnectionError,
    InternalError,
    InvalidRequestError,
    MaxIterationsError,
    NotFoundError,
    ResourceExhaustedError,
    StreamError,
    TimeoutError,
    UnavailableError,
)
from athyr.kv import Bucket, KVEntry
from athyr.models import AgentCard, Message, Session, SessionMessage, SessionProfile, Tool
from athyr.tools import ToolRegistry, get_tool, is_tool, run_tool_loop, tool

__all__ = [
    # Version
    "__version__",
    # Client
    "Agent",
    "AthyrAgent",
    # Connection State
    "ConnectionState",
    "StateChangeCallback",
    # Models
    "AgentCard",
    "Message",
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
    "ToolRegistry",
    "run_tool_loop",
    # Exceptions
    "AthyrError",
    "AuthenticationError",
    "CancelledError",
    "ConnectionError",
    "InternalError",
    "InvalidRequestError",
    "MaxIterationsError",
    "NotFoundError",
    "ResourceExhaustedError",
    "StreamError",
    "TimeoutError",
    "UnavailableError",
]
