#!/usr/bin/env python
"""HandoffRouter example - route requests to specialists.

A triage agent decides which specialist should handle the request.
Use for: support systems, task routing, expert delegation.
"""

import asyncio

from athyr.orchestration import HandoffRouter


async def main() -> None:
    # Define router with triage and specialist routes
    router = (
        HandoffRouter("support-router")
        .triage("agents.triage")  # Decides where to route
        .route("billing", "agents.billing-specialist")
        .route("technical", "agents.tech-specialist")
        .route("sales", "agents.sales-specialist")
        .fallback("agents.general-support")  # When route unknown
        .max_handoffs(3)  # Prevent infinite loops
    )

    # Triage agent should return JSON: {"route": "billing", "context": "..."}
    # Or handle directly: {"handled": true, "response": "..."}

    # Execute (requires running Athyr server with these agents)
    # async with AthyrAgent("localhost:9090") as agent:
    #     response = await router.handle(agent, b"I have a billing question")

    # With tracing:
    # trace = await router.handle_with_trace(agent, request)
    # print(f"Route path: {trace.route_names}")

    print("HandoffRouter defined with triage and 3 specialists")
    print(f"Routes: {list(router._routes.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
