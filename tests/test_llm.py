"""Tests for LLM completion functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from athyr._grpc import athyr_pb2
from athyr.client import AthyrAgent
from athyr.exceptions import ConnectionError, InvalidRequestError, UnavailableError
from athyr.models import (
    CompletionConfig,
    CompletionRequest,
    CompletionResponse,
    Message,
    Tool,
    ToolCall,
    TokenUsage,
)


class TestCompleteNotConnected:
    """Tests for complete() when not connected."""

    @pytest.mark.asyncio
    async def test_complete_raises_when_not_connected(self) -> None:
        """Test complete() raises ConnectionError when not connected."""
        agent = AthyrAgent("localhost:9090")
        request = CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="Hello")],
        )
        with pytest.raises(ConnectionError):
            await agent.complete(request)


class TestCompleteWithMock:
    """Tests for complete() with mocked gRPC stub."""

    @pytest.fixture
    def mock_agent(self) -> AthyrAgent:
        """Create an agent with mocked connection state."""
        agent = AthyrAgent("localhost:9090")
        # Simulate connected state
        agent._connected = True
        agent._agent_id = "test-agent-123"
        agent._stub = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_complete_simple_response(self, mock_agent: AthyrAgent) -> None:
        """Test complete() returns proper response."""
        # Setup mock response
        mock_response = athyr_pb2.CompletionResponse(
            content="Hello! How can I help you?",
            model="llama3",
            backend="ollama",
            finish_reason="stop",
            latency_ms=150,
            usage=athyr_pb2.TokenUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )
        mock_agent._stub.Complete = AsyncMock(return_value=mock_response)

        # Execute
        request = CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="Hello")],
        )
        response = await mock_agent.complete(request)

        # Verify response
        assert isinstance(response, CompletionResponse)
        assert response.content == "Hello! How can I help you?"
        assert response.model == "llama3"
        assert response.backend == "ollama"
        assert response.finish_reason == "stop"
        assert response.latency_ms == 150
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_complete_with_config(self, mock_agent: AthyrAgent) -> None:
        """Test complete() passes config to request."""
        mock_response = athyr_pb2.CompletionResponse(
            content="Response",
            model="llama3",
            backend="ollama",
            finish_reason="stop",
            latency_ms=100,
            usage=athyr_pb2.TokenUsage(),
        )
        mock_agent._stub.Complete = AsyncMock(return_value=mock_response)

        request = CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="Hello")],
            config=CompletionConfig(
                temperature=0.5,
                max_tokens=100,
                top_p=0.9,
                stop=["END"],
            ),
        )
        await mock_agent.complete(request)

        # Verify the proto request was built correctly
        call_args = mock_agent._stub.Complete.call_args
        proto_request = call_args[0][0]
        assert proto_request.config.temperature == 0.5
        assert proto_request.config.max_tokens == 100
        assert proto_request.config.top_p == 0.9
        assert list(proto_request.config.stop) == ["END"]

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self, mock_agent: AthyrAgent) -> None:
        """Test complete() handles tool calls in response."""
        mock_response = athyr_pb2.CompletionResponse(
            content="",
            model="llama3",
            backend="ollama",
            finish_reason="tool_calls",
            latency_ms=100,
            usage=athyr_pb2.TokenUsage(
                prompt_tokens=50,
                completion_tokens=30,
                total_tokens=80,
            ),
            tool_calls=[
                athyr_pb2.ToolCall(
                    id="call_123",
                    name="get_weather",
                    arguments='{"city": "NYC"}',
                ),
                athyr_pb2.ToolCall(
                    id="call_456",
                    name="get_time",
                    arguments='{"timezone": "EST"}',
                ),
            ],
        )
        mock_agent._stub.Complete = AsyncMock(return_value=mock_response)

        request = CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="What's the weather in NYC?")],
            tools=[
                Tool(
                    name="get_weather",
                    description="Get weather for a city",
                    parameters='{"type": "object", "properties": {"city": {"type": "string"}}}',
                ),
            ],
        )
        response = await mock_agent.complete(request)

        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].id == "call_123"
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == '{"city": "NYC"}'
        assert response.tool_calls[1].id == "call_456"
        assert response.tool_calls[1].name == "get_time"

    @pytest.mark.asyncio
    async def test_complete_passes_agent_id(self, mock_agent: AthyrAgent) -> None:
        """Test complete() includes agent_id in request."""
        mock_response = athyr_pb2.CompletionResponse(
            content="Response",
            model="llama3",
            backend="ollama",
            finish_reason="stop",
            latency_ms=100,
            usage=athyr_pb2.TokenUsage(),
        )
        mock_agent._stub.Complete = AsyncMock(return_value=mock_response)

        request = CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="Hello")],
        )
        await mock_agent.complete(request)

        call_args = mock_agent._stub.Complete.call_args
        proto_request = call_args[0][0]
        assert proto_request.agent_id == "test-agent-123"

    @pytest.mark.asyncio
    async def test_complete_with_session(self, mock_agent: AthyrAgent) -> None:
        """Test complete() passes session_id and include_memory."""
        mock_response = athyr_pb2.CompletionResponse(
            content="Response",
            model="llama3",
            backend="ollama",
            finish_reason="stop",
            latency_ms=100,
            usage=athyr_pb2.TokenUsage(),
        )
        mock_agent._stub.Complete = AsyncMock(return_value=mock_response)

        request = CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="Hello")],
            session_id="session-abc",
            include_memory=True,
        )
        await mock_agent.complete(request)

        call_args = mock_agent._stub.Complete.call_args
        proto_request = call_args[0][0]
        assert proto_request.session_id == "session-abc"
        assert proto_request.include_memory is True

    @pytest.mark.asyncio
    async def test_complete_with_tool_choice(self, mock_agent: AthyrAgent) -> None:
        """Test complete() passes tool_choice."""
        mock_response = athyr_pb2.CompletionResponse(
            content="Response",
            model="llama3",
            backend="ollama",
            finish_reason="stop",
            latency_ms=100,
            usage=athyr_pb2.TokenUsage(),
        )
        mock_agent._stub.Complete = AsyncMock(return_value=mock_response)

        request = CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="Hello")],
            tool_choice="required",
        )
        await mock_agent.complete(request)

        call_args = mock_agent._stub.Complete.call_args
        proto_request = call_args[0][0]
        assert proto_request.tool_choice == "required"


import grpc


class MockGrpcError(grpc.RpcError):
    """Mock gRPC error for testing."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__()
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class TestCompleteGrpcErrors:
    """Tests for gRPC error handling in complete()."""

    @pytest.fixture
    def mock_agent(self) -> AthyrAgent:
        """Create an agent with mocked connection state."""
        agent = AthyrAgent("localhost:9090")
        agent._connected = True
        agent._agent_id = "test-agent-123"
        agent._stub = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_complete_wraps_grpc_unavailable(self, mock_agent: AthyrAgent) -> None:
        """Test complete() wraps gRPC UNAVAILABLE error."""
        mock_error = MockGrpcError(grpc.StatusCode.UNAVAILABLE, "Service unavailable")
        mock_agent._stub.Complete = AsyncMock(side_effect=mock_error)

        request = CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="Hello")],
        )

        with pytest.raises(UnavailableError):
            await mock_agent.complete(request)

    @pytest.mark.asyncio
    async def test_complete_wraps_grpc_invalid_argument(
        self, mock_agent: AthyrAgent
    ) -> None:
        """Test complete() wraps gRPC INVALID_ARGUMENT error."""
        mock_error = MockGrpcError(grpc.StatusCode.INVALID_ARGUMENT, "Invalid model")
        mock_agent._stub.Complete = AsyncMock(side_effect=mock_error)

        request = CompletionRequest(
            model="invalid-model",
            messages=[Message(role="user", content="Hello")],
        )

        with pytest.raises(InvalidRequestError):
            await mock_agent.complete(request)
