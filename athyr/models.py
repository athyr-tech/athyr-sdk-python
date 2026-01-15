"""Pydantic models for Athyr SDK types."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from athyr._grpc import athyr_pb2


class AgentCard(BaseModel):
    """Agent metadata for registration."""

    name: str
    description: str = ""
    version: str = ""
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    def to_proto(self) -> athyr_pb2.AgentCard:
        """Convert to protobuf message."""
        return athyr_pb2.AgentCard(
            name=self.name,
            description=self.description,
            version=self.version,
            capabilities=self.capabilities,
            metadata=self.metadata,
        )

    @classmethod
    def from_proto(cls, proto: athyr_pb2.AgentCard) -> AgentCard:
        """Create from protobuf message."""
        return cls(
            name=proto.name,
            description=proto.description,
            version=proto.version,
            capabilities=list(proto.capabilities),
            metadata=dict(proto.metadata),
        )


class ToolCall(BaseModel):
    """Tool call requested by LLM."""

    id: str
    name: str
    arguments: str  # JSON string

    def to_proto(self) -> athyr_pb2.ToolCall:
        """Convert to protobuf message."""
        return athyr_pb2.ToolCall(
            id=self.id,
            name=self.name,
            arguments=self.arguments,
        )

    @classmethod
    def from_proto(cls, proto: athyr_pb2.ToolCall) -> ToolCall:
        """Create from protobuf message."""
        return cls(
            id=proto.id,
            name=proto.name,
            arguments=proto.arguments,
        )


class Tool(BaseModel):
    """Tool definition for LLM function calling."""

    name: str
    description: str = ""
    parameters: str = "{}"  # JSON Schema string

    def to_proto(self) -> athyr_pb2.Tool:
        """Convert to protobuf message."""
        return athyr_pb2.Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    @classmethod
    def from_proto(cls, proto: athyr_pb2.Tool) -> Tool:
        """Create from protobuf message."""
        return cls(
            name=proto.name,
            description=proto.description,
            parameters=proto.parameters,
        )


class Message(BaseModel):
    """Chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str = ""

    def to_proto(self) -> athyr_pb2.Message:
        """Convert to protobuf message."""
        return athyr_pb2.Message(
            role=self.role,
            content=self.content,
            tool_calls=[tc.to_proto() for tc in self.tool_calls],
            tool_call_id=self.tool_call_id,
        )

    @classmethod
    def from_proto(cls, proto: athyr_pb2.Message) -> Message:
        """Create from protobuf message."""
        return cls(
            role=proto.role,
            content=proto.content,
            tool_calls=[ToolCall.from_proto(tc) for tc in proto.tool_calls],
            tool_call_id=proto.tool_call_id,
        )


class CompletionConfig(BaseModel):
    """LLM completion configuration."""

    temperature: float = 0.7
    max_tokens: int = 0  # 0 = model default
    top_p: float = 1.0
    stop: list[str] = Field(default_factory=list)

    def to_proto(self) -> athyr_pb2.CompletionConfig:
        """Convert to protobuf message."""
        return athyr_pb2.CompletionConfig(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=self.stop,
        )

    @classmethod
    def from_proto(cls, proto: athyr_pb2.CompletionConfig) -> CompletionConfig:
        """Create from protobuf message."""
        return cls(
            temperature=proto.temperature,
            max_tokens=proto.max_tokens,
            top_p=proto.top_p,
            stop=list(proto.stop),
        )


class TokenUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_proto(self) -> athyr_pb2.TokenUsage:
        """Convert to protobuf message."""
        return athyr_pb2.TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
        )

    @classmethod
    def from_proto(cls, proto: athyr_pb2.TokenUsage) -> TokenUsage:
        """Create from protobuf message."""
        return cls(
            prompt_tokens=proto.prompt_tokens,
            completion_tokens=proto.completion_tokens,
            total_tokens=proto.total_tokens,
        )


class CompletionRequest(BaseModel):
    """LLM completion request."""

    model: str
    messages: list[Message]
    config: CompletionConfig | None = None
    session_id: str = ""
    include_memory: bool = False
    durable: bool = False
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: str = ""  # "auto", "none", "required", or tool name

    def to_proto(self, agent_id: str) -> athyr_pb2.CompletionRequest:
        """Convert to protobuf message."""
        return athyr_pb2.CompletionRequest(
            agent_id=agent_id,
            model=self.model,
            messages=[m.to_proto() for m in self.messages],
            config=self.config.to_proto() if self.config else None,
            session_id=self.session_id,
            include_memory=self.include_memory,
            durable=self.durable,
            tools=[t.to_proto() for t in self.tools],
            tool_choice=self.tool_choice,
        )


class CompletionResponse(BaseModel):
    """LLM completion response."""

    content: str
    model: str
    backend: str
    usage: TokenUsage
    finish_reason: str  # "stop", "length", "tool_calls"
    latency_ms: int
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @classmethod
    def from_proto(cls, proto: athyr_pb2.CompletionResponse) -> CompletionResponse:
        """Create from protobuf message."""
        return cls(
            content=proto.content,
            model=proto.model,
            backend=proto.backend,
            usage=TokenUsage.from_proto(proto.usage) if proto.usage else TokenUsage(),
            finish_reason=proto.finish_reason,
            latency_ms=proto.latency_ms,
            tool_calls=[ToolCall.from_proto(tc) for tc in proto.tool_calls],
        )


class Model(BaseModel):
    """Available LLM model."""

    id: str
    name: str
    backend: str
    available: bool

    @classmethod
    def from_proto(cls, proto: athyr_pb2.Model) -> Model:
        """Create from protobuf message."""
        return cls(
            id=proto.id,
            name=proto.name,
            backend=proto.backend,
            available=proto.available,
        )


# Session-related models (for Epic 3, but defining here for completeness)


class SessionProfile(BaseModel):
    """Session memory profile configuration."""

    type: str = "rolling_window"
    max_tokens: int = 4000
    summarization_threshold: int = 3000

    def to_proto(self) -> athyr_pb2.SessionProfile:
        """Convert to protobuf message."""
        return athyr_pb2.SessionProfile(
            type=self.type,
            max_tokens=self.max_tokens,
            summarization_threshold=self.summarization_threshold,
        )

    @classmethod
    def from_proto(cls, proto: athyr_pb2.SessionProfile) -> SessionProfile:
        """Create from protobuf message."""
        return cls(
            type=proto.type,
            max_tokens=proto.max_tokens,
            summarization_threshold=proto.summarization_threshold,
        )


class SessionMessage(BaseModel):
    """Message in a session."""

    role: str
    content: str
    timestamp: datetime | None = None
    tokens: int = 0

    @classmethod
    def from_proto(cls, proto: athyr_pb2.SessionMessage) -> SessionMessage:
        """Create from protobuf message."""
        ts = None
        if proto.timestamp.seconds > 0:
            ts = datetime.fromtimestamp(proto.timestamp.seconds)
        return cls(
            role=proto.role,
            content=proto.content,
            timestamp=ts,
            tokens=proto.tokens,
        )


class Session(BaseModel):
    """Memory session."""

    id: str
    agent_id: str
    messages: list[SessionMessage] = Field(default_factory=list)
    summary: str = ""
    hints: list[str] = Field(default_factory=list)
    profile: SessionProfile | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    system_prompt: str = ""

    @classmethod
    def from_proto(cls, proto: athyr_pb2.Session) -> Session:
        """Create from protobuf message."""
        created_at = None
        if proto.created_at.seconds > 0:
            created_at = datetime.fromtimestamp(proto.created_at.seconds)
        updated_at = None
        if proto.updated_at.seconds > 0:
            updated_at = datetime.fromtimestamp(proto.updated_at.seconds)

        return cls(
            id=proto.id,
            agent_id=proto.agent_id,
            messages=[SessionMessage.from_proto(m) for m in proto.messages],
            summary=proto.summary,
            hints=list(proto.hints),
            profile=SessionProfile.from_proto(proto.profile) if proto.profile else None,
            created_at=created_at,
            updated_at=updated_at,
            system_prompt=proto.system_prompt,
        )
