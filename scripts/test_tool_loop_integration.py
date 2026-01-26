#!/usr/bin/env python
"""Integration test for run_tool_loop with live Athyr + Ollama.

Usage:
    # Start Athyr server first
    athyr serve

    # Run this test
    uv run python scripts/test_tool_loop_integration.py
"""

import asyncio

from athyr import (
    AgentCard,
    AthyrAgent,
    Message,
    ToolRegistry,
    run_tool_loop,
    tool,
)


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The city name to get weather for.

    Returns:
        Weather information.
    """
    # Simulated weather data
    weather_data = {
        "paris": {"temp": "18°C", "conditions": "partly cloudy"},
        "london": {"temp": "14°C", "conditions": "rainy"},
        "tokyo": {"temp": "26°C", "conditions": "sunny"},
    }
    data = weather_data.get(city.lower(), {"temp": "20°C", "conditions": "unknown"})
    return {"city": city, **data}


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of the two numbers.
    """
    return a + b


async def main() -> None:
    """Run the integration test."""
    print("=" * 60)
    print("Integration Test: run_tool_loop with Athyr + Ollama")
    print("=" * 60)

    # Set up tool registry
    registry = ToolRegistry()
    registry.register(get_weather)
    registry.register(add_numbers)

    print(f"\nRegistered tools: {registry.names()}")

    # Connect to Athyr
    card = AgentCard(
        name="tool-loop-test",
        description="Integration test for run_tool_loop",
        version="1.0.0",
        capabilities=["tools"],
    )

    try:
        async with AthyrAgent("localhost:9090", agent_card=card) as agent:
            print(f"Connected as: {agent.agent_id()}")

            # Get available models
            models = await agent.models()
            available = [m for m in models if m.available]

            if not available:
                print("No available models!")
                return

            # Prefer qwen3:4b for tool calling, fallback to first available
            model = "qwen3:4b"
            model_ids = [m.id for m in available]
            if model not in model_ids:
                model = available[0].id
                print(f"Note: qwen3:4b not available, using {model}")
            print(f"Using model: {model}")

            # Debug: Show tool definitions
            print("\nTool definitions being sent:")
            for t in registry.tools():
                print(f"  - {t.name}: {t.description[:50]}...")
                print(f"    Schema: {t.parameters[:80]}...")

            # Tool call tracking
            tool_calls_made: list[str] = []

            def on_tool_call(name: str, args: str, result: str) -> None:
                tool_calls_made.append(name)
                print(f"  → Tool: {name}")
                print(f"    Args: {args}")
                print(f"    Result: {result}")

            # Test 1: Simple question that should use tools
            print("\n" + "-" * 60)
            print("Test 1: Weather query (should call get_weather)")
            print("-" * 60)

            response = await run_tool_loop(
                agent=agent,
                model=model,
                messages=[
                    Message(role="user", content="What's the weather in Paris?")
                ],
                registry=registry,
                system_prompt="You are a helpful assistant. Use tools when needed.",
                on_tool_call=on_tool_call,
                max_iterations=5,
            )

            print(f"\nFinal response: {response.content}")
            print(f"Tools called: {tool_calls_made}")

            # Test 2: Math question
            tool_calls_made.clear()
            print("\n" + "-" * 60)
            print("Test 2: Math query (should call add_numbers)")
            print("-" * 60)

            response = await run_tool_loop(
                agent=agent,
                model=model,
                messages=[
                    Message(role="user", content="What is 42 + 17?")
                ],
                registry=registry,
                system_prompt="You are a helpful assistant. Use the add_numbers tool for math.",
                on_tool_call=on_tool_call,
                max_iterations=5,
            )

            print(f"\nFinal response: {response.content}")
            print(f"Tools called: {tool_calls_made}")

            print("\n" + "=" * 60)
            print("Integration test completed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Athyr is running: athyr serve")
        raise


if __name__ == "__main__":
    asyncio.run(main())
