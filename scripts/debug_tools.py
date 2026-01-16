#!/usr/bin/env python
"""Debug tool calling to understand what's happening.

Usage:
    uv run python scripts/debug_tools.py
"""

import asyncio
import json

from athyr import AgentCard, AthyrAgent, Message, Tool
from athyr.models import CompletionRequest


async def main() -> None:
    """Debug tool calling."""
    print("=" * 60)
    print("Debug: Tool Calling")
    print("=" * 60)

    # Create a simple tool definition manually
    tool = Tool(
        name="add",
        description="Add two numbers together",
        parameters=json.dumps({
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        }),
    )

    print("\nTool definition:")
    print(f"  Name: {tool.name}")
    print(f"  Description: {tool.description}")
    print(f"  Parameters: {tool.parameters}")

    # Create agent
    card = AgentCard(
        name="debug-tools",
        description="Debug tool calling",
        version="1.0.0",
        capabilities=["tools"],
    )

    async with AthyrAgent("localhost:9090", agent_card=card) as agent:
        print(f"\nConnected as: {agent.agent_id()}")

        # Get available models
        models = await agent.models()
        available = [m for m in models if m.available]

        if not available:
            print("No available models!")
            return

        # Prefer qwen3:4b for tool calling
        model = "qwen3:4b"
        model_ids = [m.id for m in available]
        if model not in model_ids:
            model = available[0].id
            print(f"Note: qwen3:4b not available, using {model}")
        print(f"Using model: {model}")

        # Create request with tools
        messages = [
            Message(role="user", content="What is 42 + 17? Use the add tool."),
        ]

        request = CompletionRequest(
            model=model,
            messages=messages,
            tools=[tool],
            tool_choice="auto",
        )

        # Show what we're sending
        proto = request.to_proto(agent_id=agent.agent_id())
        print("\n" + "-" * 60)
        print("Proto request details:")
        print(f"  Model: {proto.model}")
        print(f"  Tool choice: {proto.tool_choice}")
        print(f"  Number of tools: {len(proto.tools)}")
        for t in proto.tools:
            print(f"    - {t.name}: {t.description}")
            print(f"      Parameters: {t.parameters[:100]}...")
        print("-" * 60)

        # Send request
        print("\nSending completion request...")
        response = await agent.complete(request)

        print("\n" + "-" * 60)
        print("Response:")
        print(f"  Content: {response.content[:200] if response.content else '(empty)'}")
        print(f"  Finish reason: {response.finish_reason}")
        print(f"  Tool calls: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            print(f"    - {tc.name}({tc.arguments})")
        print(f"  Backend: {response.backend}")
        print(f"  Latency: {response.latency_ms}ms")
        print("-" * 60)

        if not response.tool_calls:
            print("\n⚠️  No tool calls returned!")
            print("The model may not be using tools despite them being available.")


if __name__ == "__main__":
    asyncio.run(main())
