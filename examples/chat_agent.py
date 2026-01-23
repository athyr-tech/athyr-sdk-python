#!/usr/bin/env python
"""Chat agent example - memory sessions with tools.

Combines session memory with tool calling for stateful assistants.
Use for: chatbots, personal assistants, support agents.

Usage:
    python examples/chat_agent.py
"""

import asyncio
import os

from athyr import AthyrAgent, AgentCard, tool, ToolRegistry, run_tool_loop
from athyr.models import Message

# Server address (configurable via ATHYR_SERVER env var for Docker)
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In production, call a real weather API
    return f"Weather in {city}: Sunny, 22°C"


@tool
def set_reminder(text: str, minutes: int = 5) -> str:
    """Set a reminder for the user."""
    return f"Reminder set: '{text}' in {minutes} minutes"


async def main() -> None:
    """Run the chat agent example."""
    registry = ToolRegistry()
    registry.register(get_weather)
    registry.register(set_reminder)

    card = AgentCard(name="chat-agent", description="Stateful chat assistant")

    async with AthyrAgent(ATHYR_SERVER, agent_card=card) as agent:
        print(f"Connected: {agent.agent_id()}")
        print(f"Tools: {registry.names()}")

        # Get model
        models = await agent.models()
        available = [m for m in models if m.available]
        if not available:
            print("No models available")
            return
        model_id = available[0].id

        # Create session with personality
        session = await agent.create_session(
            system_prompt="You are a helpful assistant. Be concise."
        )
        print(f"Session: {session.id}")

        # Add persistent hint
        await agent.add_hint(session.id, "User prefers metric units")

        # Chat loop
        messages: list[Message] = []
        print("\nChat (type 'quit' to exit):\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input or user_input.lower() in ("quit", "exit"):
                break

            messages.append(Message(role="user", content=user_input))

            response = await run_tool_loop(
                agent,
                model=model_id,
                messages=messages,
                registry=registry,
                on_tool_call=lambda n, a, r: print(f"  [Tool: {n}]"),
            )

            messages.append(Message(role="assistant", content=response.content))
            print(f"Assistant: {response.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
