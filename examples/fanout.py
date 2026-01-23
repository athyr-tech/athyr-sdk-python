#!/usr/bin/env python
"""FanOut example - run agents in parallel.

All agents receive the same input and execute concurrently.
Use for: parallel analysis, voting systems, redundancy, ensemble methods.

Usage:
    python examples/fanout.py
"""

import asyncio
import os

from athyr import AthyrAgent, AgentCard
from athyr.orchestration import FanOut, all_must_succeed

# Server address (configurable via ATHYR_SERVER env var for Docker)
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")


async def main() -> None:
    """Run the FanOut example."""
    # Define parallel agents (uses DEFAULT_TIMEOUT for LLM processing)
    fanout = (
        FanOut("sentiment-analysis")
        .agent("model-a", "agent.sentiment.a.invoke")
        .agent("model-b", "agent.sentiment.b.invoke")
        .agent("model-c", "agent.sentiment.c.invoke")
    )

    card = AgentCard(name="fanout-demo", description="FanOut orchestration demo")

    async with AthyrAgent(ATHYR_SERVER, agent_card=card) as agent:
        print(f"Connected: {agent.agent_id()}")
        print(f"Agents: {', '.join(name for name, _ in fanout.agents)}")

        # Execute in parallel
        input_data = b'{"text": "This product exceeded my expectations!"}'
        print(f"\nInput: {input_data.decode()}")

        # all_must_succeed: fails if any agent fails
        # Other options: first_success(), collect_all(), require_quorum(n)
        trace = await fanout.execute_with_trace(agent, input_data, all_must_succeed())

        if trace.error:
            print(f"Failed: {trace.error}")
            return

        # Show results
        print(f"\nResults ({trace.success_count()}/{len(trace.agents)} succeeded):")
        for name, result in trace.agents.items():
            ms = result.duration.total_seconds() * 1000
            status = "OK" if result.error is None else f"ERR: {result.error}"
            print(f"  {name}: {ms:.0f}ms [{status}]")

        print(f"\nAggregated: {trace.output.decode()[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())