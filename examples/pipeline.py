#!/usr/bin/env python
"""Pipeline example - chain agents sequentially.

Each step receives the output of the previous step as input.
Use for: document processing, multi-stage analysis, ETL workflows.

Usage:
    python examples/pipeline.py
"""

import asyncio
import os

from athyr import AthyrAgent, AgentCard
from athyr.orchestration import Pipeline

# Server address (configurable via ATHYR_SERVER env var for Docker)
ATHYR_SERVER = os.environ.get("ATHYR_SERVER", "localhost:9090")


async def main() -> None:
    """Run the pipeline example."""
    # Define pipeline: extract -> summarize -> translate
    # Uses DEFAULT_TIMEOUT (120s) for LLM processing
    pipeline = (
        Pipeline("document-processor")
        .step("extract", "agent.extract.invoke")
        .step("summarize", "agent.summarize.invoke")
        .step("translate", "agent.translate.invoke")
    )

    card = AgentCard(name="pipeline-demo", description="Pipeline orchestration demo")

    async with AthyrAgent(ATHYR_SERVER, agent_card=card) as agent:
        print(f"Connected: {agent.agent_id()}")
        print(f"Pipeline: {' -> '.join(name for name, _ in pipeline.steps)}")

        # Execute pipeline
        input_data = b'{"document": "Technical paper about distributed systems..."}'
        print(f"\nInput: {input_data.decode()[:50]}...")

        trace = await pipeline.execute_with_trace(agent, input_data)

        if trace.error:
            print(f"Failed at step '{trace.error.step}': {trace.error.cause}")
            return

        # Show results
        print("\nExecution trace:")
        for step in trace.steps:
            ms = step.duration.total_seconds() * 1000
            print(f"  {step.name}: {ms:.0f}ms")

        output = trace.output()
        if output:
            print(f"\nFinal output: {output.decode()[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
