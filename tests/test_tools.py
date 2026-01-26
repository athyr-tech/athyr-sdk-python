"""Tests for tool decorator and ToolRegistry."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from athyr.exceptions import MaxIterationsError
from athyr.models import CompletionResponse, Message, TokenUsage, ToolCall
from athyr.tools import ToolRegistry, get_tool, is_tool, run_tool_loop, tool


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator_basic(self) -> None:
        """Test basic @tool decoration."""

        @tool
        def my_func(x: int) -> str:
            """Do something."""
            return str(x)

        assert is_tool(my_func)
        tool_def = get_tool(my_func)
        assert tool_def is not None
        assert tool_def.name == "my_func"
        assert tool_def.description == "Do something."

    def test_tool_decorator_with_args(self) -> None:
        """Test @tool with explicit name and description."""

        @tool(name="custom_name", description="Custom description")
        def my_func(x: int) -> str:
            """Original docstring."""
            return str(x)

        tool_def = get_tool(my_func)
        assert tool_def is not None
        assert tool_def.name == "custom_name"
        assert tool_def.description == "Custom description"

    def test_is_tool_returns_false_for_undecorated(self) -> None:
        """Test is_tool returns False for undecorated function."""

        def plain_func(x: int) -> str:
            return str(x)

        assert not is_tool(plain_func)
        assert get_tool(plain_func) is None


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        registry = ToolRegistry()

        @tool
        def my_func(x: int) -> str:
            """Do something."""
            return str(x)

        registry.register(my_func)
        assert "my_func" in registry
        assert len(registry) == 1

    def test_register_multiple_tools(self) -> None:
        """Test registering multiple tools."""
        registry = ToolRegistry()

        @tool
        def func_a(x: int) -> str:
            """Function A."""
            return str(x)

        @tool
        def func_b(y: str) -> int:
            """Function B."""
            return len(y)

        registry.register(func_a)
        registry.register(func_b)

        assert len(registry) == 2
        assert "func_a" in registry
        assert "func_b" in registry

    def test_register_as_decorator(self) -> None:
        """Test using register as a decorator."""
        registry = ToolRegistry()

        @registry.register
        @tool
        def my_func(x: int) -> str:
            """Do something."""
            return str(x)

        assert "my_func" in registry
        assert my_func(42) == "42"

    def test_register_raises_for_undecorated(self) -> None:
        """Test register raises for non-@tool function."""
        registry = ToolRegistry()

        def plain_func(x: int) -> str:
            return str(x)

        with pytest.raises(ValueError, match="not decorated with @tool"):
            registry.register(plain_func)

    def test_register_raises_for_duplicate(self) -> None:
        """Test register raises for duplicate tool name."""
        registry = ToolRegistry()

        @tool
        def my_func(x: int) -> str:
            """First function."""
            return str(x)

        @tool(name="my_func")
        def another_func(y: str) -> int:
            """Second function with same name."""
            return len(y)

        registry.register(my_func)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(another_func)

    def test_get_tool(self) -> None:
        """Test getting a tool by name."""
        registry = ToolRegistry()

        @tool
        def my_func(x: int) -> str:
            """Do something."""
            return str(x)

        registry.register(my_func)

        func = registry.get("my_func")
        assert func is my_func

        assert registry.get("nonexistent") is None

    def test_tools_returns_list(self) -> None:
        """Test tools() returns list of Tool objects."""
        registry = ToolRegistry()

        @tool
        def func_a(x: int) -> str:
            """Function A."""
            return str(x)

        @tool
        def func_b(y: str) -> int:
            """Function B."""
            return len(y)

        registry.register(func_a)
        registry.register(func_b)

        tool_list = registry.tools()
        assert len(tool_list) == 2

        names = {t.name for t in tool_list}
        assert names == {"func_a", "func_b"}

    def test_execute_tool(self) -> None:
        """Test executing a tool by name."""
        registry = ToolRegistry()

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        registry.register(add)

        result = registry.execute("add", '{"a": 2, "b": 3}')
        assert json.loads(result) == 5

    def test_execute_tool_with_dict_result(self) -> None:
        """Test executing a tool that returns a dict."""
        registry = ToolRegistry()

        @tool
        def get_info(name: str) -> dict:
            """Get info for a name."""
            return {"name": name, "length": len(name)}

        registry.register(get_info)

        result = registry.execute("get_info", '{"name": "Alice"}')
        parsed = json.loads(result)
        assert parsed == {"name": "Alice", "length": 5}

    def test_execute_raises_for_unknown_tool(self) -> None:
        """Test execute raises for unknown tool."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.execute("unknown", '{"x": 1}')

    def test_execute_raises_for_invalid_json(self) -> None:
        """Test execute raises for invalid JSON arguments."""
        registry = ToolRegistry()

        @tool
        def my_func(x: int) -> str:
            """Do something."""
            return str(x)

        registry.register(my_func)

        with pytest.raises(json.JSONDecodeError):
            registry.execute("my_func", "not valid json")

    def test_iter_and_names(self) -> None:
        """Test iteration and names() method."""
        registry = ToolRegistry()

        @tool
        def func_a(x: int) -> str:
            """Function A."""
            return str(x)

        @tool
        def func_b(y: str) -> int:
            """Function B."""
            return len(y)

        registry.register(func_a)
        registry.register(func_b)

        # Test iteration
        names_from_iter = set(registry)
        assert names_from_iter == {"func_a", "func_b"}

        # Test names()
        assert set(registry.names()) == {"func_a", "func_b"}

    def test_empty_registry(self) -> None:
        """Test empty registry behavior."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.tools() == []
        assert registry.names() == []
        assert "anything" not in registry


class TestRunToolLoop:
    """Tests for run_tool_loop function."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Create a registry with test tools."""
        reg = ToolRegistry()

        @tool
        def get_weather(city: str) -> dict:
            """Get weather for a city."""
            return {"city": city, "temp": "22°C", "conditions": "sunny"}

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        reg.register(get_weather)
        reg.register(add)
        return reg

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock()
        agent.complete = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_returns_immediately_when_no_tool_calls(
        self, mock_agent: MagicMock, registry: ToolRegistry
    ) -> None:
        """Test loop returns immediately when LLM doesn't call tools."""
        mock_agent.complete.return_value = CompletionResponse(
            content="Hello! How can I help you?",
            finish_reason="stop",
            model="test-model",
            backend="ollama",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        response = await run_tool_loop(
            agent=mock_agent,
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            registry=registry,
        )

        assert response.content == "Hello! How can I help you?"
        assert mock_agent.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_executes_single_tool_call(
        self, mock_agent: MagicMock, registry: ToolRegistry
    ) -> None:
        """Test loop executes a single tool call."""
        # First response: LLM calls a tool
        first_response = CompletionResponse(
            content="",
            finish_reason="tool_calls",
            model="test-model",
            backend="ollama",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="get_weather",
                    arguments='{"city": "Paris"}',
                )
            ],
        )

        # Second response: LLM responds with text
        second_response = CompletionResponse(
            content="The weather in Paris is 22°C and sunny.",
            finish_reason="stop",
            model="test-model",
            backend="ollama",
            latency_ms=150,
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )

        mock_agent.complete.side_effect = [first_response, second_response]

        response = await run_tool_loop(
            agent=mock_agent,
            model="test-model",
            messages=[Message(role="user", content="What's the weather in Paris?")],
            registry=registry,
        )

        assert response.content == "The weather in Paris is 22°C and sunny."
        assert mock_agent.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_executes_multiple_tool_calls(
        self, mock_agent: MagicMock, registry: ToolRegistry
    ) -> None:
        """Test loop executes multiple tool calls in one response."""
        # First response: LLM calls two tools
        first_response = CompletionResponse(
            content="",
            finish_reason="tool_calls",
            model="test-model",
            backend="ollama",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments='{"city": "Paris"}'),
                ToolCall(id="call_2", name="add", arguments='{"a": 2, "b": 3}'),
            ],
        )

        # Second response: LLM responds with text
        second_response = CompletionResponse(
            content="Paris is 22°C and 2+3=5.",
            finish_reason="stop",
            model="test-model",
            backend="ollama",
            latency_ms=200,
            usage=TokenUsage(prompt_tokens=30, completion_tokens=10, total_tokens=40),
        )

        mock_agent.complete.side_effect = [first_response, second_response]

        response = await run_tool_loop(
            agent=mock_agent,
            model="test-model",
            messages=[Message(role="user", content="Weather in Paris and 2+3?")],
            registry=registry,
        )

        assert response.content == "Paris is 22°C and 2+3=5."
        assert mock_agent.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_max_iterations_error(
        self, mock_agent: MagicMock, registry: ToolRegistry
    ) -> None:
        """Test loop raises MaxIterationsError when limit exceeded."""
        # Always return tool calls (infinite loop scenario)
        mock_agent.complete.return_value = CompletionResponse(
            content="",
            finish_reason="tool_calls",
            model="test-model",
            backend="ollama",
            latency_ms=50,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            tool_calls=[
                ToolCall(id="call_1", name="add", arguments='{"a": 1, "b": 1}'),
            ],
        )

        with pytest.raises(MaxIterationsError) as exc_info:
            await run_tool_loop(
                agent=mock_agent,
                model="test-model",
                messages=[Message(role="user", content="Loop forever")],
                registry=registry,
                max_iterations=3,
            )

        assert exc_info.value.iterations == 3
        assert mock_agent.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_includes_system_prompt(
        self, mock_agent: MagicMock, registry: ToolRegistry
    ) -> None:
        """Test system_prompt is prepended to messages."""
        mock_agent.complete.return_value = CompletionResponse(
            content="I'm a helpful assistant!",
            finish_reason="stop",
            model="test-model",
            backend="ollama",
            latency_ms=80,
            usage=TokenUsage(prompt_tokens=15, completion_tokens=5, total_tokens=20),
        )

        await run_tool_loop(
            agent=mock_agent,
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            registry=registry,
            system_prompt="You are a helpful assistant.",
        )

        # Check the messages passed to complete
        call_args = mock_agent.complete.call_args
        request = call_args[0][0]
        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.messages[0].content == "You are a helpful assistant."
        assert request.messages[1].role == "user"

    @pytest.mark.asyncio
    async def test_on_tool_call_callback(
        self, mock_agent: MagicMock, registry: ToolRegistry
    ) -> None:
        """Test on_tool_call callback is invoked."""
        first_response = CompletionResponse(
            content="",
            finish_reason="tool_calls",
            model="test-model",
            backend="ollama",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            tool_calls=[
                ToolCall(id="call_1", name="add", arguments='{"a": 5, "b": 3}'),
            ],
        )
        second_response = CompletionResponse(
            content="5+3=8",
            finish_reason="stop",
            model="test-model",
            backend="ollama",
            latency_ms=120,
            usage=TokenUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        )
        mock_agent.complete.side_effect = [first_response, second_response]

        callback_calls: list[tuple[str, str, str]] = []

        def on_tool_call(name: str, args: str, result: str) -> None:
            callback_calls.append((name, args, result))

        await run_tool_loop(
            agent=mock_agent,
            model="test-model",
            messages=[Message(role="user", content="5+3?")],
            registry=registry,
            on_tool_call=on_tool_call,
        )

        assert len(callback_calls) == 1
        name, args, result = callback_calls[0]
        assert name == "add"
        assert args == '{"a": 5, "b": 3}'
        assert json.loads(result) == 8

    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_tool(
        self, mock_agent: MagicMock, registry: ToolRegistry
    ) -> None:
        """Test loop raises KeyError when LLM calls unregistered tool."""
        mock_agent.complete.return_value = CompletionResponse(
            content="",
            finish_reason="tool_calls",
            model="test-model",
            backend="ollama",
            latency_ms=100,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            tool_calls=[
                ToolCall(id="call_1", name="unknown_tool", arguments='{}'),
            ],
        )

        with pytest.raises(KeyError, match="unknown_tool"):
            await run_tool_loop(
                agent=mock_agent,
                model="test-model",
                messages=[Message(role="user", content="Call unknown")],
                registry=registry,
            )
