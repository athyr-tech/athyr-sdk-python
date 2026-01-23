#!/usr/bin/env python
"""Quickstart example for Athyr SDK.

Demonstrates basic SDK usage:
- Agent connection
- Listing available models
- Making LLM completion requests

Usage:
    # Start Athyr server first
    athyr serve

    # Run the example
    python examples/quickstart.py
"""

import asyncio
import os

from athyr import AthyrAgent, AgentCard
from athyr.models import CompletionRequest, Message

# Server address (configurable via ATHYR_SERVER env var for Docker)
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")


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

        # List available models
        print("\nAvailable models:")
        models = await agent.models()
        for model in models:
            status = "available" if model.available else "unavailable"
            print(f"  - {model.name} ({model.backend}) [{status}]")

        if not models:
            print("  No models available")
            return

        # Find first available model
        available_models = [m for m in models if m.available]
        if not available_models:
            print("\nNo available models to use")
            return

        model_id = available_models[0].id
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