#!/usr/bin/env python
"""HandoffRouter example - route requests to specialists.

A triage agent classifies requests and routes to appropriate specialists.
Use for: support systems, task routing, expert delegation.

Usage:
    python examples/handoff.py
"""

import asyncio
import os

from athyr import AthyrAgent, AgentCard
from athyr.orchestration import HandoffRouter

# Server address (configurable via ATHYR_SERVER env var for Docker)
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")


async def main() -> None:
    """Run the HandoffRouter example."""
    # Define router with triage and specialists (4 agents to fit within free tier limit)
    # Triage returns JSON: {"route": "billing"} or {"handled": true, "response": "..."}
    router = (
        HandoffRouter("support-router")
        .triage("agent.triage.invoke")
        .route("billing", "agent.billing.invoke")
        .route("technical", "agent.technical.invoke")
        .fallback("agent.general.invoke")
        .max_handoffs(3)
    )

    card = AgentCard(name="handoff-demo", description="HandoffRouter demo")

    async with AthyrAgent(ATHYR_SERVER, agent_card=card) as agent:
        print(f"Connected: {agent.agent_id()}")
        print(f"Routes: {list(router._routes.keys())}")

        # Route a request
        request = b'{"message": "I was charged twice for my subscription"}'
        print(f"\nRequest: {request.decode()}")

        trace = await router.handle_with_trace(agent, request)

        if trace.error:
            print(f"Routing failed: {trace.error}")
            return

        # Show routing path
        print(f"\nRoute path: {' -> '.join(trace.route_names())}")
        for step in trace.path:
            ms = step.duration.total_seconds() * 1000
            print(f"  {step.agent}: {ms:.0f}ms (-> {step.route or 'final'})")

        print(f"\nResponse: {trace.output.decode()[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())