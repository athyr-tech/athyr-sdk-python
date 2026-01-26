"""Tests for Pydantic models."""

import pytest

from athyr.models import (
    AgentCard,
    CompletionConfig,
    CompletionRequest,
    Message,
    Model,
    SessionProfile,
    TokenUsage,
    Tool,
    ToolCall,
)


class TestAgentCard:
    """Tests for AgentCard model."""

    def test_create_minimal(self) -> None:
        """Test creating AgentCard with minimal fields."""
        card = AgentCard(name="test-agent")
        assert card.name == "test-agent"
        assert card.description == ""
        assert card.version == ""
        assert card.capabilities == []
        assert card.metadata == {}

    def test_create_full(self) -> None:
        """Test creating AgentCard with all fields."""
        card = AgentCard(
            name="test-agent",
            description="A test agent",
            version="1.0.0",
            capabilities=["llm", "tools"],
            metadata={"env": "test"},
        )
        assert card.name == "test-agent"
        assert card.description == "A test agent"
        assert card.version == "1.0.0"
        assert card.capabilities == ["llm", "tools"]
        assert card.metadata == {"env": "test"}

    def test_proto_roundtrip(self) -> None:
        """Test conversion to/from protobuf."""
        card = AgentCard(
            name="test",
            description="desc",
            version="1.0",
            capabilities=["a", "b"],
            metadata={"k": "v"},
        )
        proto = card.to_proto()
        card2 = AgentCard.from_proto(proto)
        assert card == card2


class TestMessage:
    """Tests for Message model."""

    def test_create_simple(self) -> None:
        """Test creating a simple message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls == []
        assert msg.tool_call_id == ""

    def test_create_with_tool_calls(self) -> None:
        """Test creating message with tool calls."""
        tc = ToolCall(id="tc1", name="get_weather", arguments='{"city": "NYC"}')
        msg = Message(role="assistant", content="", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "get_weather"

    def test_proto_roundtrip(self) -> None:
        """Test conversion to/from protobuf."""
        msg = Message(role="user", content="Hi")
        proto = msg.to_proto()
        msg2 = Message.from_proto(proto)
        assert msg == msg2


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create(self) -> None:
        """Test creating ToolCall."""
        tc = ToolCall(
            id="call_123",
            name="get_weather",
            arguments='{"city": "NYC"}',
        )
        assert tc.id == "call_123"
        assert tc.name == "get_weather"
        assert tc.arguments == '{"city": "NYC"}'

    def test_proto_roundtrip(self) -> None:
        """Test conversion to/from protobuf."""
        tc = ToolCall(id="1", name="test", arguments="{}")
        proto = tc.to_proto()
        tc2 = ToolCall.from_proto(proto)
        assert tc == tc2


class TestTool:
    """Tests for Tool model."""

    def test_create(self) -> None:
        """Test creating Tool."""
        tool = Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters='{"type": "object"}',
        )
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a city"

    def test_proto_roundtrip(self) -> None:
        """Test conversion to/from protobuf."""
        tool = Tool(name="test", description="desc", parameters="{}")
        proto = tool.to_proto()
        tool2 = Tool.from_proto(proto)
        assert tool == tool2


class TestCompletionConfig:
    """Tests for CompletionConfig model."""

    def test_defaults(self) -> None:
        """Test default values."""
        cfg = CompletionConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 0
        assert cfg.top_p == 1.0
        assert cfg.stop == []

    def test_custom_values(self) -> None:
        """Test custom values."""
        cfg = CompletionConfig(
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 100
        assert cfg.stop == ["END"]

    def test_proto_roundtrip(self) -> None:
        """Test conversion to/from protobuf."""
        cfg = CompletionConfig(temperature=0.8, max_tokens=50)
        proto = cfg.to_proto()
        cfg2 = CompletionConfig.from_proto(proto)
        assert cfg == cfg2


class TestTokenUsage:
    """Tests for TokenUsage model."""

    def test_create(self) -> None:
        """Test creating TokenUsage."""
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_proto_roundtrip(self) -> None:
        """Test conversion to/from protobuf."""
        usage = TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        proto = usage.to_proto()
        usage2 = TokenUsage.from_proto(proto)
        assert usage == usage2


class TestCompletionRequest:
    """Tests for CompletionRequest model."""

    def test_create_minimal(self) -> None:
        """Test creating minimal CompletionRequest."""
        req = CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="Hi")],
        )
        assert req.model == "llama3"
        assert len(req.messages) == 1
        assert req.config is None
        assert req.session_id == ""

    def test_to_proto(self) -> None:
        """Test converting to protobuf."""
        req = CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="Hi")],
            config=CompletionConfig(temperature=0.5),
        )
        proto = req.to_proto(agent_id="agent-123")
        assert proto.agent_id == "agent-123"
        assert proto.model == "llama3"
        assert len(proto.messages) == 1
        assert proto.config.temperature == 0.5


class TestSessionProfile:
    """Tests for SessionProfile model."""

    def test_defaults(self) -> None:
        """Test default values."""
        profile = SessionProfile()
        assert profile.type == "rolling_window"
        assert profile.max_tokens == 4000
        assert profile.summarization_threshold == 3000

    def test_proto_roundtrip(self) -> None:
        """Test conversion to/from protobuf."""
        profile = SessionProfile(type="test", max_tokens=1000)
        proto = profile.to_proto()
        profile2 = SessionProfile.from_proto(proto)
        assert profile == profile2