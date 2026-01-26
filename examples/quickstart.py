#!/usr/bin/env python
"""Quickstart example for Athyr SDK.

Demonstrates basic SDK usage:
- Agent connection
- Making LLM completion requests

Usage:
    # Start Athyr server first
    athyr serve

    # Run the example (ATHYR_MODEL is required)
    ATHYR_MODEL=llama2 python examples/quickstart.py
    ATHYR_MODEL=liquid/lfm-2.5-1.2b-instruct:free python examples/quickstart.py
"""

import asyncio
import os

from athyr import AthyrAgent, AgentCard
from athyr.models import CompletionRequest, Message

# Configuration via environment variables
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")
ATHYR_MODEL = os.environ.get("ATHYR_MODEL")  # Required: model to use


async def main() -> None:
    """Run the quickstart example."""
    print("Athyr SDK - Quickstart Example")
    print("=" * 40)

    # Create agent with metadata
    card = AgentCard(
        name="quickstart-agent",
        description="A simple demo agent",
        version="1.0.0",
    )

    # Use context manager for automatic connection/disconnection
    async with AthyrAgent(ATHYR_SERVER, agent_card=card) as agent:
        print(f"Connected! Agent ID: {agent.agent_id()}")

        # Check model is specified
        if not ATHYR_MODEL:
            print("\nError: ATHYR_MODEL environment variable is required")
            print("Example: ATHYR_MODEL=llama2 python examples/quickstart.py")
            return

        model_id = ATHYR_MODEL
        print(f"\nUsing model: {model_id}")

        # Make a completion request
        request = CompletionRequest(
            model=model_id,
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="What is 2 + 2? Reply in one word."),
            ],
        )

        print("\nSending completion request...")
        response = await agent.complete(request)

        print(f"\nResponse: {response.content}")
        print(f"Model: {response.model} ({response.backend})")
        print(f"Tokens: {response.usage.total_tokens} total")
        print(f"Latency: {response.latency_ms}ms")


if __name__ == "__main__":
    asyncio.run(main())