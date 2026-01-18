#!/usr/bin/env python
"""Pipeline example - chain agents sequentially.

Each step receives the output of the previous step as input.
Use for: document processing, multi-stage analysis, workflows.
"""

import asyncio

from athyr.orchestration import Pipeline, PipelineStep


async def main() -> None:
    # Define pipeline steps (each step is an agent subject)
    pipeline = (
        Pipeline("document-processor")
        .add(PipelineStep(name="extract", subject="agents.extractor"))
        .add(PipelineStep(name="summarize", subject="agents.summarizer"))
        .add(PipelineStep(name="translate", subject="agents.translator"))
    )

    # Execute pipeline (requires running Athyr server with these agents)
    # async with AthyrAgent("localhost:9090") as agent:
    #     result = await pipeline.execute(agent, b"Raw document content...")
    #     print(f"Final output: {result}")

    # For tracing/debugging:
    # trace = await pipeline.execute_with_trace(agent, input_data)
    # for step in trace.steps:
    #     print(f"{step.name}: {step.duration}")

    print("Pipeline defined with 3 steps: extract -> summarize -> translate")
    print(f"Steps: {[s.name for s in pipeline._steps]}")


if __name__ == "__main__":
    asyncio.run(main())
