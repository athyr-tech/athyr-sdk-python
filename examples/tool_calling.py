#!/usr/bin/env python
"""Tool calling example for Athyr SDK.

Demonstrates using the @tool decorator with LLM function calling:
- Define tools with automatic JSON Schema generation
- Send tools to LLM with completion requests
- Handle tool call responses
- Execute tools and return results to LLM

NOTE: Tool calling requires LLM models that support function calling.
Some models (like OpenAI's GPT-4, Claude, etc.) have full support,
while smaller local models may not properly return tool_calls.
If the model doesn't support tools, it will respond with text instead.

Usage:
    # Start Athyr server first
    athyr serve

    # Run the example (ATHYR_MODEL is required)
    ATHYR_MODEL=gpt-4 python examples/tool_calling.py

    # Run with mock mode to demonstrate the tool calling flow
    python examples/tool_calling.py --mock
"""

import argparse
import asyncio
import json
import os
import random

from athyr import AthyrAgent, AgentCard, Tool, get_tool, tool
from athyr.models import CompletionRequest, Message, ToolCall

# Configuration via environment variables
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")
ATHYR_MODEL = os.environ.get("ATHYR_MODEL")  # Required: model to use


# Define tools using the @tool decorator
# The decorator automatically extracts:
# - Name from function name
# - Description from docstring
# - JSON Schema from type hints


@tool
def get_weather(location: str, unit: str = "celsius") -> dict[str, str]:
    """Get the current weather for a location.

    Args:
        location: City name or location to get weather for.
        unit: Temperature unit - either 'celsius' or 'fahrenheit'.

    Returns:
        Weather information including temperature and conditions.
    """
    # Simulated weather data
    temps = {"celsius": random.randint(15, 30), "fahrenheit": random.randint(59, 86)}
    conditions = ["sunny", "cloudy", "partly cloudy", "rainy"]

    return {
        "location": location,
        "temperature": f"{temps.get(unit, temps['celsius'])}°{'C' if unit == 'celsius' else 'F'}",
        "conditions": random.choice(conditions),
        "humidity": f"{random.randint(40, 80)}%",
    }


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 2', '10 * 5').

    Returns:
        The result of the calculation as a string.
    """
    # Simple and safe expression evaluation for basic math
    # In production, use a proper math parser
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression"

    try:
        result = eval(expression)  # noqa: S307 - Safe for numeric expressions
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(name="search_database", description="Search for records in the database")
def search_db(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search the database for matching records.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.

    Returns:
        List of matching records.
    """
    # Simulated database search
    results = [
        {"id": f"rec_{i}", "title": f"Result {i} for '{query}'", "score": f"{0.9 - i * 0.1:.1f}"}
        for i in range(min(limit, 3))
    ]
    return results


# Registry of available tools
TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_database": search_db,
}


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool by name with the given arguments.

    Args:
        name: The tool name to execute.
        arguments: JSON string of arguments.

    Returns:
        The tool result as a JSON string.
    """
    if name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {name}"})

    func = TOOLS[name]
    try:
        args = json.loads(arguments)
        result = func(**args)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def show_tool_definitions() -> None:
    """Display registered tool definitions."""
    print("\nRegistered tools:")
    for name, func in TOOLS.items():
        tool_def = get_tool(func)
        if tool_def:
            print(f"\n  {tool_def.name}:")
            print(f"    Description: {tool_def.description}")
            schema = json.loads(tool_def.parameters)
            props = schema.get("properties", {})
            required = schema.get("required", [])
            for param, info in props.items():
                req = "(required)" if param in required else "(optional)"
                print(f"    - {param}: {info.get('type', 'any')} {req}")


async def run_mock_demo() -> None:
    """Run a mock demonstration of tool calling flow.

    This shows exactly how tool calling works without requiring
    a model that supports function calling.
    """
    print("Athyr SDK - Tool Calling Demo (Mock Mode)")
    print("=" * 50)
    print("\nThis demo shows the tool calling flow without a real LLM.")
    print("In production, the LLM would decide when to call tools.\n")

    show_tool_definitions()

    # Simulate user message
    user_message = "What's the weather like in Paris? Also, what is 15 * 7?"
    print("\n" + "-" * 50)
    print(f"User: {user_message}")
    print("-" * 50)

    # Simulate LLM deciding to call tools
    print("\n[Mock] LLM decides to call 2 tools:")

    # Create mock tool calls (as if from LLM)
    mock_tool_calls = [
        ToolCall(
            id="call_001",
            name="get_weather",
            arguments='{"location": "Paris", "unit": "celsius"}',
        ),
        ToolCall(
            id="call_002",
            name="calculate",
            arguments='{"expression": "15 * 7"}',
        ),
    ]

    # Execute each tool
    for tc in mock_tool_calls:
        print(f"\n  Calling: {tc.name}")
        print(f"  Arguments: {tc.arguments}")
        result = execute_tool(tc.name, tc.arguments)
        print(f"  Result: {result}")

    # Show the message flow
    print("\n" + "-" * 50)
    print("Message flow for tool calling:")
    print("-" * 50)

    print("\n1. Initial messages sent to LLM:")
    print('   [{"role": "system", "content": "You are a helpful assistant..."}]')
    print(f'   [{{"role": "user", "content": "{user_message}"}}]')
    print("   + tools: [get_weather, calculate, search_database]")

    print("\n2. LLM responds with tool_calls (finish_reason='tool_calls'):")
    print('   {"content": "", "tool_calls": [')
    for tc in mock_tool_calls:
        print(f'     {{"id": "{tc.id}", "name": "{tc.name}", "arguments": {tc.arguments}}}')
    print("   ]}")

    print("\n3. Client executes tools and adds results as 'tool' messages:")
    for tc in mock_tool_calls:
        result = execute_tool(tc.name, tc.arguments)
        print(f'   {{"role": "tool", "content": {result}, "tool_call_id": "{tc.id}"}}')

    print("\n4. Send updated messages back to LLM for final response:")
    print("   LLM receives: original messages + assistant tool_calls + tool results")

    print("\n5. LLM generates final response using tool results:")
    final_response = (
        "The weather in Paris is currently 22°C and sunny with 65% humidity. "
        "And 15 multiplied by 7 equals 105."
    )
    print(f"   Assistant: {final_response}")

    print("\n" + "=" * 50)
    print("Mock demo complete!")
    print("\nTo test with a real LLM, run without --mock flag")
    print("(requires a model that supports function calling)")


async def run_live_demo() -> None:
    """Run the tool calling example with a real Athyr server."""
    print("Athyr SDK - Tool Calling Example")
    print("=" * 50)

    show_tool_definitions()

    # Create agent
    card = AgentCard(
        name="tool-calling-agent",
        description="Agent demonstrating tool calling capabilities",
        version="1.0.0",
        capabilities=["tools", "function_calling"],
    )

    async with AthyrAgent(ATHYR_SERVER, agent_card=card) as agent:
        print(f"\nConnected as: {agent.agent_id()}")

        # Check model is specified
        if not ATHYR_MODEL:
            print("\nError: ATHYR_MODEL environment variable is required")
            print("Example: ATHYR_MODEL=gpt-4 python examples/tool_calling.py")
            return

        model_id = ATHYR_MODEL
        print(f"Using model: {model_id}")

        # Convert decorated functions to Tool objects for the API
        tool_defs: list[Tool] = []
        for func in TOOLS.values():
            tool_def = get_tool(func)
            if tool_def:
                tool_defs.append(tool_def)

        # Create a conversation that will require tool use
        messages = [
            Message(
                role="system",
                content="You are a helpful assistant with access to tools. "
                "Use the available tools to help answer questions. "
                "Always use tools when they can provide accurate information.",
            ),
            Message(
                role="user",
                content="What's the weather like in Paris? Also, what is 15 * 7?",
            ),
        ]

        print("\n" + "-" * 50)
        print(f"User: {messages[-1].content}")
        print("-" * 50)

        # Send request with tools
        request = CompletionRequest(
            model=model_id,
            messages=messages,
            tools=tool_defs,
            tool_choice="auto",  # Let the model decide when to use tools
        )

        response = await agent.complete(request)

        # Check if the model wants to call tools
        if response.tool_calls:
            print(f"\nModel requested {len(response.tool_calls)} tool call(s):")

            # Execute each tool call
            tool_results: list[Message] = []

            # First, add the assistant's message with tool calls
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            for tc in response.tool_calls:
                print(f"\n  Calling: {tc.name}")
                print(f"  Arguments: {tc.arguments}")

                # Execute the tool
                result = execute_tool(tc.name, tc.arguments)
                print(f"  Result: {result}")

                # Add tool result message
                tool_results.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                    )
                )

            # Add all tool results to messages
            messages.extend(tool_results)

            # Send follow-up request with tool results
            print("\n" + "-" * 50)
            print("Sending tool results back to model...")

            follow_up = CompletionRequest(
                model=model_id,
                messages=messages,
                tools=tool_defs,
            )

            final_response = await agent.complete(follow_up)
            print(f"\nAssistant: {final_response.content}")
            print(f"\nFinish reason: {final_response.finish_reason}")
            print(f"Total tokens: {final_response.usage.total_tokens}")
        else:
            # Model responded directly without tools
            print(f"\nAssistant: {response.content}")
            print(f"\nFinish reason: {response.finish_reason}")
            print("\nNote: The model did not use tools. This may be because:")
            print("  - The model doesn't support function calling")
            print("  - The model decided tools weren't needed")
            print("  - Try using a model with tool calling support (GPT-4, Claude, etc.)")

        print("\n" + "=" * 50)
        print("Tool calling example complete!")


async def main() -> None:
    """Entry point for the tool calling example."""
    parser = argparse.ArgumentParser(description="Athyr SDK Tool Calling Example")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode to demonstrate tool calling flow",
    )
    args = parser.parse_args()

    if args.mock:
        await run_mock_demo()
    else:
        await run_live_demo()


if __name__ == "__main__":
    asyncio.run(main())
