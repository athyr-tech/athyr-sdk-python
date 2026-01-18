#!/usr/bin/env python
"""Resilient agent example - auto-reconnect on connection loss.

Automatically reconnects with exponential backoff when server disconnects.
Use for: long-running agents, production deployments, unreliable networks.
"""

import asyncio

from athyr import AthyrAgent, AgentCard
from athyr.client_state import ConnectionState


def on_state_change(state: ConnectionState, error: Exception | None) -> None:
    """Handle connection state changes."""
    if error:
        print(f"State: {state.name} (error: {error})")
    else:
        print(f"State: {state.name}")


async def main() -> None:
    card = AgentCard(name="resilient-agent", description="Auto-reconnecting agent")

    # Configure auto-reconnect (requires running Athyr server)
    # agent = AthyrAgent(
    #     "localhost:9090",
    #     agent_card=card,
    #     auto_reconnect=True,       # Enable reconnection
    #     max_retries=10,            # Max attempts (0 = infinite)
    #     base_backoff=1.0,          # Initial delay: 1s
    #     max_backoff=30.0,          # Max delay: 30s
    #     on_state_change=on_state_change,
    # )
    #
    # async with agent:
    #     print(f"Connected: {agent.agent_id()}")
    #
    #     # Agent will auto-reconnect if server restarts
    #     while True:
    #         if agent.connected():
    #             models = await agent.models()
    #             print(f"Available models: {len(models)}")
    #         await asyncio.sleep(10)

    print("Resilient agent with auto-reconnect")
    print("Features:")
    print("  - auto_reconnect=True: Enable reconnection")
    print("  - max_retries: Limit attempts (0 = infinite)")
    print("  - base_backoff/max_backoff: Exponential backoff")
    print("  - on_state_change: Callback for state monitoring")


if __name__ == "__main__":
    asyncio.run(main())
