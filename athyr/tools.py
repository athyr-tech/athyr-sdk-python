"""Tool decorator for marking functions as LLM-callable tools."""

from __future__ import annotations

import functools
import inspect
import json
from typing import TYPE_CHECKING, Any, Callable, TypeVar, get_type_hints

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from athyr.exceptions import MaxIterationsError
from athyr.models import CompletionRequest, CompletionResponse, Message, Tool

if TYPE_CHECKING:
    from athyr.client import Agent

F = TypeVar("F", bound=Callable[..., Any])

# Marker attribute for decorated functions
TOOL_ATTR = "__athyr_tool__"


def tool(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> F | Callable[[F], F]:
    """Decorator to mark a function as an LLM-callable tool.

    Extracts metadata from the function to create a Tool definition:
    - Name: from function name or explicit `name` parameter
    - Description: from docstring or explicit `description` parameter
    - Parameters: JSON Schema generated from type hints

    Can be used with or without parentheses:
        @tool
        def my_func(x: int) -> str: ...

        @tool(name="custom_name", description="Custom desc")
        def my_func(x: int) -> str: ...

    Args:
        func: The function to decorate (when used without parentheses).
        name: Override the tool name (defaults to function name).
        description: Override the description (defaults to docstring).

    Returns:
        The decorated function with a `__athyr_tool__` attribute.

    Example:
        @tool
        def get_weather(city: str, unit: str = "celsius") -> str:
            '''Get the current weather for a city.

            Args:
                city: The city name to get weather for.
                unit: Temperature unit (celsius or fahrenheit).

            Returns:
                Weather description string.
            '''
            return f"Weather in {city}: sunny, 22 {unit}"

        # Access the tool definition
        tool_def = get_weather.__athyr_tool__
        print(tool_def.name)  # "get_weather"
        print(tool_def.parameters)  # JSON Schema string
    """

    def decorator(fn: F) -> F:
        tool_name = name if name is not None else fn.__name__
        tool_description = description if description is not None else _extract_description(fn)
        parameters_schema = _generate_schema(fn)

        tool_def = Tool(
            name=tool_name,
            description=tool_description,
            parameters=json.dumps(parameters_schema),
        )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        # Attach tool definition to the wrapper
        setattr(wrapper, TOOL_ATTR, tool_def)

        return wrapper  # type: ignore[return-value]

    # Support @tool and @tool() syntax
    if func is not None:
        return decorator(func)
    return decorator


def get_tool(func: Callable[..., Any]) -> Tool | None:
    """Get the Tool definition from a decorated function.

    Args:
        func: A function that may be decorated with @tool.

    Returns:
        The Tool definition if the function is decorated, None otherwise.

    Example:
        @tool
        def my_func(x: int) -> str:
            '''Do something.'''
            return str(x)

        tool_def = get_tool(my_func)
        if tool_def:
            print(tool_def.name)  # "my_func"
    """
    return getattr(func, TOOL_ATTR, None)


def is_tool(func: Callable[..., Any]) -> bool:
    """Check if a function is decorated with @tool.

    Args:
        func: The function to check.

    Returns:
        True if the function has a Tool definition, False otherwise.
    """
    return hasattr(func, TOOL_ATTR)


def _extract_description(func: Callable[..., Any]) -> str:
    """Extract description from function docstring.

    Takes the first paragraph of the docstring (up to first blank line)
    as the description.
    """
    doc = func.__doc__
    if not doc:
        return ""

    # Get first paragraph (up to first blank line)
    lines = doc.strip().split("\n")
    description_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            break
        description_lines.append(stripped)

    return " ".join(description_lines)


def _generate_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON Schema from function type hints.

    Creates a Pydantic model from the function's parameters and uses
    Pydantic's schema generation for a proper JSON Schema.
    """
    sig = inspect.signature(func)

    # Try to get type hints, fall back to empty dict on failure
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    # Build fields for Pydantic model
    fields: dict[str, tuple[type, FieldInfo | Any]] = {}

    for param_name, param in sig.parameters.items():
        # Skip self, cls, *args, **kwargs
        if param_name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Get type annotation (default to Any)
        param_type = hints.get(param_name, Any)

        # Get default value
        if param.default is inspect.Parameter.empty:
            # Required parameter
            default = ...
        else:
            default = param.default

        # Extract description from docstring Args section
        param_description = _extract_param_description(func, param_name)

        if param_description:
            fields[param_name] = (param_type, Field(default=default, description=param_description))
        else:
            fields[param_name] = (param_type, default)

    # Create dynamic Pydantic model
    if not fields:
        return {"type": "object", "properties": {}}

    model = create_model(f"{func.__name__}_params", **fields)  # type: ignore[call-overload]

    # Generate JSON Schema
    schema = model.model_json_schema()

    # Clean up schema - remove title and $defs if not needed
    schema.pop("title", None)
    if "$defs" in schema and not schema.get("properties"):
        schema.pop("$defs", None)

    return schema


def _extract_param_description(func: Callable[..., Any], param_name: str) -> str:
    """Extract parameter description from docstring Args section.

    Parses Google-style docstrings to find parameter descriptions.
    """
    doc = func.__doc__
    if not doc:
        return ""

    lines = doc.split("\n")
    in_args_section = False
    current_param: str | None = None
    descriptions: dict[str, list[str]] = {}

    for line in lines:
        stripped = line.strip()

        # Detect Args section
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue

        # Detect end of Args section (another section or end of doc)
        if in_args_section and stripped.endswith(":") and not stripped.startswith(" "):
            if stripped.lower() in ("returns:", "return:", "raises:", "yields:", "example:", "examples:", "note:", "notes:"):
                in_args_section = False
                continue

        if in_args_section:
            # Check for new parameter line: "param_name: description" or "param_name (type): description"
            if ":" in stripped and not stripped.startswith(" "):
                # Parse "param_name: desc" or "param_name (type): desc"
                parts = stripped.split(":", 1)
                param_part = parts[0].strip()

                # Handle "param_name (type)" format
                if "(" in param_part:
                    param_part = param_part.split("(")[0].strip()

                current_param = param_part
                if len(parts) > 1:
                    descriptions[current_param] = [parts[1].strip()]
                else:
                    descriptions[current_param] = []
            elif current_param and stripped:
                # Continuation of previous parameter description
                descriptions[current_param].append(stripped)

    param_desc_parts = descriptions.get(param_name, [])
    return " ".join(param_desc_parts)


class ToolRegistry:
    """Registry for collecting and managing decorated tools.

    Provides a central place to register @tool decorated functions,
    retrieve Tool definitions for completion requests, and execute
    tools by name.

    Example:
        from athyr import tool, ToolRegistry

        registry = ToolRegistry()

        @tool
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Sunny in {city}"

        @tool
        def calculate(expr: str) -> str:
            '''Evaluate math expression.'''
            return str(eval(expr))

        # Register tools
        registry.register(get_weather)
        registry.register(calculate)

        # Use in completion request
        request = CompletionRequest(
            model=model,
            messages=messages,
            tools=registry.tools(),
        )

        # Execute tool calls from LLM response
        for tc in response.tool_calls:
            result = registry.execute(tc.name, tc.arguments)
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Register a @tool decorated function.

        Args:
            func: A function decorated with @tool.

        Returns:
            The same function (allows use as decorator).

        Raises:
            ValueError: If the function is not decorated with @tool.
            ValueError: If a tool with the same name is already registered.

        Example:
            @tool
            def my_func(x: int) -> str:
                '''Do something.'''
                return str(x)

            registry.register(my_func)

            # Or use as a decorator
            @registry.register
            @tool
            def another_func(y: str) -> int:
                '''Do something else.'''
                return len(y)
        """
        tool_def = get_tool(func)
        if tool_def is None:
            raise ValueError(
                f"Function '{func.__name__}' is not decorated with @tool"
            )

        if tool_def.name in self._tools:
            raise ValueError(
                f"Tool '{tool_def.name}' is already registered"
            )

        self._tools[tool_def.name] = func
        return func

    def get(self, name: str) -> Callable[..., Any] | None:
        """Get a tool function by name.

        Args:
            name: The tool name.

        Returns:
            The registered function, or None if not found.
        """
        return self._tools.get(name)

    def tools(self) -> list[Tool]:
        """Get all Tool definitions for completion requests.

        Returns:
            List of Tool objects ready to pass to CompletionRequest.

        Example:
            request = CompletionRequest(
                model=model,
                messages=messages,
                tools=registry.tools(),
            )
        """
        result: list[Tool] = []
        for func in self._tools.values():
            tool_def = get_tool(func)
            if tool_def:
                result.append(tool_def)
        return result

    def execute(self, name: str, arguments: str) -> str:
        """Execute a tool by name with JSON arguments.

        Parses the JSON arguments, calls the tool function, and
        returns the result as a JSON string.

        Args:
            name: The tool name to execute.
            arguments: JSON string of arguments.

        Returns:
            The tool result as a JSON string.

        Raises:
            KeyError: If the tool is not registered.
            json.JSONDecodeError: If arguments is not valid JSON.
            Exception: Any exception from the tool function.

        Example:
            result = registry.execute("get_weather", '{"city": "Paris"}')
            # result: '{"temperature": "22°C", "conditions": "sunny"}'
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        func = self._tools[name]
        args = json.loads(arguments)
        result = func(**args)
        return json.dumps(result)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered.

        Example:
            if "get_weather" in registry:
                ...
        """
        return name in self._tools

    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)

    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._tools)

    def names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())


async def run_tool_loop(
    agent: Agent,
    model: str,
    messages: list[Message],
    registry: ToolRegistry,
    *,
    max_iterations: int = 10,
    system_prompt: str | None = None,
    on_tool_call: Callable[[str, str, str], None] | None = None,
) -> CompletionResponse:
    """Run an agentic tool-calling loop until completion.

    Executes a loop that:
    1. Sends messages + tools to the LLM
    2. If LLM requests tools → executes them, adds results to messages
    3. Repeats until LLM responds with text (no tool calls)

    Args:
        agent: Connected Athyr agent.
        model: Model identifier to use.
        messages: Initial conversation messages.
        registry: ToolRegistry containing available tools.
        max_iterations: Maximum loop iterations (safety limit).
        system_prompt: Optional system prompt to prepend.
        on_tool_call: Optional callback called for each tool execution.
            Receives (tool_name, arguments, result).

    Returns:
        Final CompletionResponse with LLM's text response.

    Raises:
        MaxIterationsError: If loop exceeds max_iterations.
        KeyError: If LLM calls an unregistered tool.
        AthyrError: If LLM completion fails.

    Example:
        from athyr import AthyrAgent, ToolRegistry, run_tool_loop, tool, Message

        registry = ToolRegistry()

        @tool
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Sunny in {city}"

        registry.register(get_weather)

        async with AthyrAgent("localhost:9090") as agent:
            response = await run_tool_loop(
                agent=agent,
                model="gpt-4",
                messages=[Message(role="user", content="What's the weather in Paris?")],
                registry=registry,
            )
            print(response.content)
    """
    # Build message list with optional system prompt
    msgs: list[Message] = []
    if system_prompt:
        msgs.append(Message(role="system", content=system_prompt))
    msgs.extend(messages)

    for iteration in range(max_iterations):
        # Make completion request with tools
        response = await agent.complete(
            CompletionRequest(
                model=model,
                messages=msgs,
                tools=registry.tools(),
                tool_choice="auto",
            )
        )

        # No tool calls = LLM is done
        if not response.tool_calls:
            return response

        # Add assistant message with tool calls to history
        msgs.append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )

        # Execute each tool and add results
        for tc in response.tool_calls:
            result = registry.execute(tc.name, tc.arguments)

            # Optional callback for observability
            if on_tool_call:
                on_tool_call(tc.name, tc.arguments, result)

            # Add tool result message
            msgs.append(
                Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                )
            )

    # Exceeded max iterations
    raise MaxIterationsError(
        f"Tool loop exceeded {max_iterations} iterations",
        iterations=max_iterations,
    )