#!/usr/bin/env python
"""Resilient agent example - auto-reconnect on connection loss.

Automatically reconnects with exponential backoff when server disconnects.
Use for: long-running agents, production deployments, unreliable networks.

Usage:
    python examples/resilient_agent.py
"""

import asyncio
import os

from athyr import AthyrAgent, AgentCard
from athyr.client_state import ConnectionState

# Server address (configurable via ATHYR_SERVER env var for Docker)
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")


def on_state_change(state: ConnectionState, error: Exception | None) -> None:
    """Handle connection state changes."""
    if error:
        print(f"  State: {state.name} (error: {error})")
    else:
        print(f"  State: {state.name}")


async def main() -> None:
    """Run the resilient agent example."""
    card = AgentCard(name="resilient-agent", description="Auto-reconnecting agent")

    # Configure auto-reconnect
    agent = AthyrAgent(
        ATHYR_SERVER,
        agent_card=card,
        auto_reconnect=True,
        max_retries=10,       # 0 = infinite
        base_backoff=1.0,     # Initial delay (seconds)
        max_backoff=30.0,     # Max delay (seconds)
        on_state_change=on_state_change,
    )

    print("Resilient agent with auto-reconnect")
    print("  max_retries=10, backoff=1s-30s")
    print("\nTry restarting the Athyr server to see reconnection...")
    print("Press Ctrl+C to stop.\n")

    async with agent:
        print(f"Connected: {agent.agent_id()}")

        # Long-running loop
        try:
            while True:
                if agent.connected():
                    models = await agent.models()
                    print(f"[OK] {len(models)} models available")
                else:
                    print("[...] Waiting for reconnection")

                await asyncio.sleep(5)

        except KeyboardInterrupt:
            print("\nStopping...")


if __name__ == "__main__":
    asyncio.run(main())