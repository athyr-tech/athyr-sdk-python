#!/usr/bin/env python
"""Streaming completions example for Athyr SDK.

Demonstrates streaming LLM responses with real-time token output:
- Async iteration over response chunks
- Progressive content display
- Final usage statistics

Usage:
    # Start Athyr server first
    athyr serve

    # Run the example
    python examples/streaming.py
"""

import asyncio
import os
import sys

from athyr import AthyrAgent, AgentCard
from athyr.models import CompletionRequest, Message

# Server address (configurable via ATHYR_SERVER env var for Docker)
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")


async def main() -> None:
    """Run the streaming example."""
    print("Athyr SDK - Streaming Example")
    print("=" * 40)

    card = AgentCard(
        name="streaming-agent",
        description="Demonstrates streaming completions",
        version="1.0.0",
    )

    async with AthyrAgent(ATHYR_SERVER, agent_card=card) as agent:
        print(f"Connected! Agent ID: {agent.agent_id()}")

        # Get available models
        models = await agent.models()
        available = [m for m in models if m.available]
        if not available:
            print("\nNo available models")
            return

        model_id = available[0].id
        print(f"Using model: {model_id}\n")

        # Create a request that will generate a longer response
        request = CompletionRequest(
            model=model_id,
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(
                    role="user",
                    content="Explain what an API is in 2-3 sentences.",
                ),
            ],
        )

        print("Response: ", end="", flush=True)

        # Stream the response
        full_content = ""
        async for chunk in agent.complete_stream(request):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_content += chunk.content

            if chunk.done:
                print("\n")
                print("-" * 40)
                print(f"Model: {chunk.model} ({chunk.backend})")
                if chunk.usage:
                    print(f"Tokens: {chunk.usage.total_tokens} total")
                    print(f"  Prompt: {chunk.usage.prompt_tokens}")
                    print(f"  Completion: {chunk.usage.completion_tokens}")


if __name__ == "__main__":
    asyncio.run(main())