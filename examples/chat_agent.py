#!/usr/bin/env python
"""Chat agent example - memory sessions with tools.

Combines session memory with tool calling for stateful assistants.
Use for: chatbots, personal assistants, support agents.
"""

import asyncio

from athyr import AthyrAgent, AgentCard, tool, ToolRegistry, run_tool_loop
from athyr.models import Message


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"


@tool
def set_reminder(text: str, minutes: int = 5) -> str:
    """Set a reminder for the user."""
    return f"Reminder set: '{text}' in {minutes} minutes"


async def main() -> None:
    registry = ToolRegistry()
    registry.register(get_weather)
    registry.register(set_reminder)

    card = AgentCard(name="chat-agent", description="Stateful chat assistant")

    # Connect and create session (requires running Athyr server)
    # async with AthyrAgent("localhost:9090", agent_card=card) as agent:
    #     # Create session with personality
    #     session = await agent.create_session(
    #         system_prompt="You are a helpful assistant. Be concise."
    #     )
    #
    #     # Add user preference as hint (persists across turns)
    #     await agent.add_hint(session.id, "User prefers metric units")
    #
    #     # First turn - uses tools + remembers context
    #     response = await run_tool_loop(
    #         agent,
    #         model="llama3",
    #         messages=[Message(role="user", content="What's the weather in Paris?")],
    #         registry=registry,
    #     )
    #     print(f"Assistant: {response.content}")
    #
    #     # Second turn - remembers previous conversation
    #     response = await run_tool_loop(
    #         agent,
    #         model="llama3",
    #         messages=[Message(role="user", content="Set a reminder to check it again")],
    #         registry=registry,
    #     )
    #     print(f"Assistant: {response.content}")

    print("Chat agent with memory + tools")
    print(f"Tools: {registry.names()}")
    print("Session features: system_prompt, hints, message history")


if __name__ == "__main__":
    asyncio.run(main())
