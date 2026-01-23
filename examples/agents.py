#!/usr/bin/env python
"""LLM-powered agents for the orchestration examples.

Run agents for a specific example (respects Athyr free edition 5 agent limit).

Usage:
    # Terminal 1: Start Athyr
    athyr serve

    # Terminal 2: Start agents for the example you want to run
    python examples/agents.py pipeline                  # 3 agents, auto-select model
    python examples/agents.py fanout --model llama2     # 3 agents, use llama2
    python examples/agents.py handoff --model qwen3     # 5 agents, use qwen3
    python examples/agents.py groupchat                 # 3 agents

    # Terminal 3: Run the example
    python examples/pipeline.py
"""

import asyncio
import json
import os
import sys

from athyr import AthyrAgent, AgentCard
from athyr.models import CompletionRequest, Message

# Server address (configurable via ATHYR_SERVER env var for Docker)
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")

# Global model override (set via --model argument)
MODEL_OVERRIDE: str | None = None


async def llm_agent(name: str, subject: str, system_prompt: str) -> None:
    """Run an LLM-powered agent."""
    card = AgentCard(name=name, description=f"LLM agent: {name}")

    async with AthyrAgent(ATHYR_SERVER, agent_card=card) as agent:
        models = await agent.models()
        available = [m for m in models if m.available]
        if not available:
            print(f"  {name}: No models available!", flush=True)
            return

        # Use specified model or first available
        if MODEL_OVERRIDE:
            matches = [m for m in available if MODEL_OVERRIDE in m.id]
            if matches:
                model_id = matches[0].id
            else:
                print(f"  {name}: Model '{MODEL_OVERRIDE}' not found, using {available[0].id}", flush=True)
                model_id = available[0].id
        else:
            model_id = available[0].id
        print(f"  {name}: ready ({model_id})", flush=True)

        sub = await agent.subscribe(subject)

        async for msg in sub.messages():
            if not msg.reply:
                continue

            try:
                user_content = msg.data.decode() if msg.data else "{}"
                print(f"  {name}: received request", flush=True)

                response = await agent.complete(CompletionRequest(
                    model=model_id,
                    messages=[
                        Message(role="system", content=system_prompt),
                        Message(role="user", content=user_content),
                    ],
                    max_tokens=500,
                ))

                print(f"  {name}: sending response", flush=True)
                await agent.publish(msg.reply, response.content.encode())

            except Exception as e:
                print(f"  {name}: error - {e}", flush=True)
                error = json.dumps({"error": str(e)}).encode()
                await agent.publish(msg.reply, error)


# Agent configs grouped by example
PIPELINE_AGENTS = [
    ("extract-agent", "agent.extract.invoke",
     "Extract key information from the input. Return JSON with 'extracted' field."),
    ("summarize-agent", "agent.summarize.invoke",
     "Summarize the input concisely. Return JSON with 'summary' field."),
    ("translate-agent", "agent.translate.invoke",
     "Translate the input to French. Return JSON with 'translated' field."),
]

FANOUT_AGENTS = [
    ("sentiment-a", "agent.sentiment.a.invoke",
     "Analyze sentiment. Return JSON: {\"model\": \"a\", \"sentiment\": \"positive/negative/neutral\", \"confidence\": 0.0-1.0}"),
    ("sentiment-b", "agent.sentiment.b.invoke",
     "Analyze sentiment. Return JSON: {\"model\": \"b\", \"sentiment\": \"positive/negative/neutral\", \"confidence\": 0.0-1.0}"),
    ("sentiment-c", "agent.sentiment.c.invoke",
     "Analyze sentiment. Return JSON: {\"model\": \"c\", \"sentiment\": \"positive/negative/neutral\", \"confidence\": 0.0-1.0}"),
]

HANDOFF_AGENTS = [
    ("triage-agent", "agent.triage.invoke",
     "Classify request into: billing, technical, or general. Return ONLY: {\"route\": \"<category>\"}"),
    ("billing-agent", "agent.billing.invoke",
     "Handle billing inquiries. Return: {\"response\": \"<answer>\", \"handled_by\": \"billing\"}"),
    ("technical-agent", "agent.technical.invoke",
     "Handle technical issues. Return: {\"response\": \"<answer>\", \"handled_by\": \"technical\"}"),
    ("general-agent", "agent.general.invoke",
     "Handle general inquiries. Return: {\"response\": \"<answer>\", \"handled_by\": \"general\"}"),
]

GROUPCHAT_AGENTS = [
    ("security-reviewer", "agent.reviewer.security.invoke",
     "Review from security perspective. Return: {\"content\": \"<your comments>\"}"),
    ("performance-reviewer", "agent.reviewer.performance.invoke",
     "Review from performance perspective. Return: {\"content\": \"<your comments>\"}"),
    ("style-reviewer", "agent.reviewer.style.invoke",
     "Review code style. Return: {\"content\": \"<your comments>\"}"),
]

AGENT_GROUPS = {
    "pipeline": PIPELINE_AGENTS,
    "fanout": FANOUT_AGENTS,
    "handoff": HANDOFF_AGENTS,
    "groupchat": GROUPCHAT_AGENTS,
}


async def main() -> None:
    global MODEL_OVERRIDE

    # Parse arguments
    args = sys.argv[1:]
    example = None
    model = None

    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] in AGENT_GROUPS:
            example = args[i]
            i += 1
        else:
            i += 1

    if not example:
        print("Usage: python examples/agents.py <example> [--model <name>]")
        print(f"Examples: {', '.join(AGENT_GROUPS.keys())}")
        print("Models: llama2, qwen3, granite4, etc. (partial match supported)")
        sys.exit(1)

    MODEL_OVERRIDE = model
    agents = AGENT_GROUPS[example]

    model_info = f" with model '{model}'" if model else ""
    print(f"Starting {len(agents)} agents for {example}{model_info}...\n", flush=True)

    tasks = [llm_agent(name, subject, prompt) for name, subject, prompt in agents]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())