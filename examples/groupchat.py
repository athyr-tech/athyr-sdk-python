#!/usr/bin/env python
"""GroupChat example - multi-agent discussion.

Agents take turns discussing a topic until consensus or max rounds.
Use for: collaborative review, debate, brainstorming, code review.

Usage:
    python examples/groupchat.py
"""

import asyncio
import os

from athyr import AthyrAgent, AgentCard
from athyr.orchestration import GroupChat, round_robin

# Server address (configurable via ATHYR_SERVER env var for Docker)
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")


async def main() -> None:
    """Run the GroupChat example."""
    # Define participants
    chat = (
        GroupChat("code-review")
        .participant("security", "agent.reviewer.security.invoke")
        .participant("performance", "agent.reviewer.performance.invoke")
        .participant("style", "agent.reviewer.style.invoke")
        .manager(round_robin())
        .max_rounds(5)
    )

    # Optional: end early on consensus
    chat.consensus_check(lambda msgs: msgs and "LGTM" in msgs[-1].content.upper())

    card = AgentCard(name="groupchat-demo", description="GroupChat demo")

    async with AthyrAgent(ATHYR_SERVER, agent_card=card) as agent:
        print(f"Connected: {agent.agent_id()}")
        print(f"Participants: {[p.name for p in chat._participants]}")

        # Start discussion
        topic = b"Review the authentication module changes in PR #142"
        print(f"\nTopic: {topic.decode()}")

        trace = await chat.discuss_with_trace(agent, topic)

        if trace.error:
            print(f"Discussion failed: {trace.error}")
            return

        # Show transcript
        print(f"\nTranscript ({trace.rounds_completed} rounds):")
        for msg in trace.messages:
            print(f"  [{msg.speaker}]: {msg.content[:60]}...")

        print(f"\nConsensus: {'Yes' if trace.consensus_reached else 'No'}")
        print(f"Conclusion: {trace.conclusion.decode()[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())