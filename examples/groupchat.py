#!/usr/bin/env python
"""GroupChat example - multi-agent discussion.

Agents take turns discussing a topic until consensus or max rounds.
Use for: collaborative review, debate, brainstorming.
"""

import asyncio

from athyr.orchestration import GroupChat, Participant
from athyr.orchestration.groupchat import round_robin


async def main() -> None:
    # Define participants
    chat = (
        GroupChat("code-review")
        .add(Participant(name="security-reviewer", subject="agents.security"))
        .add(Participant(name="performance-reviewer", subject="agents.performance"))
        .add(Participant(name="style-reviewer", subject="agents.style"))
        .manager(round_robin())  # Or: func_manager(custom_selector)
        .max_rounds(5)
    )

    # Optional: end early when consensus reached
    # def check_consensus(messages):
    #     return "LGTM" in messages[-1].content
    # chat.consensus_check(check_consensus)

    # Execute (requires running Athyr server with these agents)
    # async with AthyrAgent("localhost:9090") as agent:
    #     trace = await chat.discuss(agent, "Review this code: ...")
    #     print(f"Conclusion: {trace.conclusion}")
    #     for msg in trace.messages:
    #         print(f"{msg.sender}: {msg.content[:50]}...")

    print("GroupChat defined with 3 participants")
    print(f"Participants: {[p.name for p in chat._participants]}")


if __name__ == "__main__":
    asyncio.run(main())
