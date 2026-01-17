"""Pipeline pattern for sequential agent chains.

A Pipeline chains agents in sequential order where each agent
processes the output from the previous agent.

Example:
    pipeline = (
        Pipeline("doc-pipeline")
        .step("draft", "agent.draft.invoke")
        .step("review", "agent.review.invoke")
        .step("polish", "agent.polish.invoke")
    )

    result = await pipeline.execute(agent, input_data)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from athyr.client import Agent


@dataclass
class StepTrace:
    """Execution details for a single pipeline step."""

    index: int
    """Step index in the pipeline (0-based)."""

    name: str
    """Step name."""

    subject: str
    """Subject the step was sent to."""

    input: bytes
    """Input data sent to the step."""

    output: bytes = b""
    """Output data received from the step."""

    started_at: datetime = field(default_factory=datetime.now)
    """When the step started executing."""

    duration: timedelta = field(default_factory=lambda: timedelta(0))
    """How long the step took to execute."""

    error: Exception | None = None
    """Error if the step failed."""


@dataclass
class PipelineTrace:
    """Detailed execution information for a pipeline run."""

    pipeline: str
    """Pipeline name."""

    steps: list[StepTrace] = field(default_factory=list)
    """Execution trace for each step."""

    started_at: datetime = field(default_factory=datetime.now)
    """When the pipeline started executing."""

    duration: timedelta = field(default_factory=lambda: timedelta(0))
    """Total pipeline execution time."""

    error: PipelineError | None = None
    """Error if the pipeline failed."""

    def output(self) -> bytes | None:
        """Get the final output of the pipeline.

        Returns:
            The output from the last successful step, or None if
            the pipeline has no steps or failed.
        """
        if not self.steps:
            return None
        last_step = self.steps[-1]
        if last_step.error is not None:
            return None
        return last_step.output


@dataclass
class PipelineError(Exception):
    """Error indicating which pipeline step failed."""

    step: str
    """Name of the step that failed."""

    index: int
    """Index of the step that failed (0-based)."""

    subject: str
    """Subject the step was sent to."""

    cause: Exception
    """The underlying error."""

    def __str__(self) -> str:
        return (
            f"pipeline step {self.step!r} (index {self.index}, "
            f"subject {self.subject}) failed: {self.cause}"
        )

    def __repr__(self) -> str:
        return (
            f"PipelineError(step={self.step!r}, index={self.index}, "
            f"subject={self.subject!r}, cause={self.cause!r})"
        )


@dataclass
class _StepOptions:
    """Internal options for a pipeline step."""

    timeout: float | None = None
    """Timeout for this step in seconds."""

    transform: Callable[[bytes], bytes] | None = None
    """Transform function to apply to step output."""


@dataclass
class _PipelineStep:
    """Internal representation of a pipeline step."""

    name: str
    subject: str
    options: _StepOptions = field(default_factory=_StepOptions)


class Pipeline:
    """Chains agents in sequential order.

    Each step receives the output of the previous step as input.
    The pipeline can be executed with or without detailed tracing.

    Example:
        pipeline = (
            Pipeline("content-pipeline")
            .step("draft", "agent.draft.invoke")
            .step("review", "agent.review.invoke", timeout=30.0)
            .step("format", "agent.format.invoke")
        )

        # Simple execution
        result = await pipeline.execute(agent, b'{"topic": "AI"}')

        # With tracing for debugging
        trace = await pipeline.execute_with_trace(agent, b'{"topic": "AI"}')
        print(f"Pipeline took {trace.duration}")
        for step in trace.steps:
            print(f"  {step.name}: {step.duration}")
    """

    def __init__(self, name: str) -> None:
        """Create a new pipeline.

        Args:
            name: Name for the pipeline (used in tracing).
        """
        self._name = name
        self._steps: list[_PipelineStep] = []

    @property
    def name(self) -> str:
        """Get the pipeline name."""
        return self._name

    @property
    def steps(self) -> list[tuple[str, str]]:
        """Get list of (name, subject) tuples for all steps."""
        return [(s.name, s.subject) for s in self._steps]

    def step(
        self,
        name: str,
        subject: str,
        *,
        timeout: float | None = None,
        transform: Callable[[bytes], bytes] | None = None,
    ) -> Pipeline:
        """Add a step to the pipeline.

        Steps are executed in the order they are added.

        Args:
            name: Display name for the step.
            subject: Subject to send requests to.
            timeout: Optional timeout for this step in seconds.
            transform: Optional function to transform step output
                before passing to the next step.

        Returns:
            Self for method chaining.
        """
        self._steps.append(
            _PipelineStep(
                name=name,
                subject=subject,
                options=_StepOptions(timeout=timeout, transform=transform),
            )
        )
        return self

    async def execute(self, agent: Agent, input: bytes) -> bytes:
        """Execute the pipeline and return the final result.

        Each step receives the output of the previous step as input.

        Args:
            agent: The agent to use for requests.
            input: Initial input data for the first step.

        Returns:
            The output from the final step.

        Raises:
            PipelineError: If any step fails.
        """
        trace = await self.execute_with_trace(agent, input)
        if trace.error:
            raise trace.error
        if not trace.steps:
            return input
        return trace.steps[-1].output

    async def execute_with_trace(
        self, agent: Agent, input: bytes
    ) -> PipelineTrace:
        """Execute the pipeline with detailed tracing.

        Useful for debugging and monitoring pipeline execution.

        Args:
            agent: The agent to use for requests.
            input: Initial input data for the first step.

        Returns:
            PipelineTrace with execution details for each step.
        """
        trace = PipelineTrace(
            pipeline=self._name,
            started_at=datetime.now(),
        )

        current = input

        for i, step in enumerate(self._steps):
            step_trace = StepTrace(
                index=i,
                name=step.name,
                subject=step.subject,
                input=current,
                started_at=datetime.now(),
            )

            try:
                # Execute with optional timeout
                if step.options.timeout is not None:
                    output = await asyncio.wait_for(
                        agent.request(step.subject, current),
                        timeout=step.options.timeout,
                    )
                else:
                    output = await agent.request(step.subject, current)

                step_trace.duration = datetime.now() - step_trace.started_at

                # Apply transform if configured
                if step.options.transform is not None:
                    try:
                        output = step.options.transform(output)
                    except Exception as e:
                        step_trace.error = e
                        trace.steps.append(step_trace)
                        trace.duration = datetime.now() - trace.started_at
                        trace.error = PipelineError(
                            step=step.name,
                            index=i,
                            subject=step.subject,
                            cause=Exception(f"transform failed: {e}"),
                        )
                        return trace

                step_trace.output = output
                trace.steps.append(step_trace)
                current = output

            except asyncio.TimeoutError as e:
                step_trace.duration = datetime.now() - step_trace.started_at
                step_trace.error = e
                trace.steps.append(step_trace)
                trace.duration = datetime.now() - trace.started_at
                trace.error = PipelineError(
                    step=step.name,
                    index=i,
                    subject=step.subject,
                    cause=e,
                )
                return trace

            except Exception as e:
                step_trace.duration = datetime.now() - step_trace.started_at
                step_trace.error = e
                trace.steps.append(step_trace)
                trace.duration = datetime.now() - trace.started_at
                trace.error = PipelineError(
                    step=step.name,
                    index=i,
                    subject=step.subject,
                    cause=e,
                )
                return trace

        trace.duration = datetime.now() - trace.started_at
        return trace
