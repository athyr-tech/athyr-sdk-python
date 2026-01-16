"""Tool decorator for marking functions as LLM-callable tools."""

from __future__ import annotations

import functools
import inspect
import json
from typing import Any, Callable, TypeVar, get_type_hints

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from athyr.models import Tool

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