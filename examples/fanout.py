#!/usr/bin/env python
"""FanOut example - run agents in parallel.

All agents receive the same input and execute concurrently.
Use for: parallel analysis, voting systems, redundancy.
"""

import asyncio

from athyr.orchestration import FanOut, FanOutAgent
from athyr.orchestration.fanout import all_must_succeed, first_success, collect_all


async def main() -> None:
    # Define agents to run in parallel
    fanout = (
        FanOut("sentiment-analysis")
        .add(FanOutAgent(name="analyzer-1", subject="agents.sentiment.v1"))
        .add(FanOutAgent(name="analyzer-2", subject="agents.sentiment.v2"))
        .add(FanOutAgent(name="analyzer-3", subject="agents.sentiment.v3"))
    )

    # Aggregation strategies:
    # - all_must_succeed(): All must complete successfully
    # - first_success(): Return first successful result
    # - collect_all(): Gather all results (including failures)
    # - require_quorum(n): Need at least n successes

    # Execute (requires running Athyr server with these agents)
    # async with AthyrAgent("localhost:9090") as agent:
    #     result = await fanout.execute(agent, b"Analyze this text", all_must_succeed())

    # With tracing:
    # trace = await fanout.execute_with_trace(agent, data, collect_all())
    # for r in trace.results:
    #     print(f"{r.agent}: {'OK' if r.success else r.error}")

    print("FanOut defined with 3 parallel agents")
    print(f"Agents: {[a.name for a in fanout._agents]}")


if __name__ == "__main__":
    asyncio.run(main())
