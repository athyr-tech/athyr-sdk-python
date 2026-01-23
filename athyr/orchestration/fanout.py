"""FanOut pattern for parallel agent execution.

FanOut runs multiple agents in parallel on the same input
and aggregates their results using configurable strategies.

Example:
    fanout = (
        FanOut("stock-analysis")
        .agent("fundamental", "agent.fundamental.invoke")
        .agent("technical", "agent.technical.invoke")
        .agent("sentiment", "agent.sentiment.invoke")
    )

    result = await fanout.execute(agent, input_data, all_must_succeed())
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable

from athyr.orchestration import DEFAULT_TIMEOUT

if TYPE_CHECKING:
    from athyr.client import Agent


@dataclass
class AgentResult:
    """Result from a single agent in a fan-out."""

    output: bytes = b""
    """Output data from the agent."""

    error: Exception | None = None
    """Error if the agent failed."""

    duration: timedelta = field(default_factory=lambda: timedelta(0))
    """How long the agent took to execute."""

    def success(self) -> bool:
        """Check if the agent completed without error."""
        return self.error is None


@dataclass
class AgentTrace:
    """Execution details for a single agent in a fan-out."""

    name: str
    """Agent name."""

    subject: str
    """Subject the agent was called on."""

    output: bytes = b""
    """Output data from the agent."""

    started_at: datetime = field(default_factory=datetime.now)
    """When the agent started executing."""

    duration: timedelta = field(default_factory=lambda: timedelta(0))
    """How long the agent took to execute."""

    error: Exception | None = None
    """Error if the agent failed."""


@dataclass
class FanOutTrace:
    """Detailed execution information for a fan-out run."""

    fanout: str
    """FanOut name."""

    agents: dict[str, AgentTrace] = field(default_factory=dict)
    """Execution trace for each agent."""

    output: bytes = b""
    """Aggregated output."""

    started_at: datetime = field(default_factory=datetime.now)
    """When the fan-out started executing."""

    duration: timedelta = field(default_factory=lambda: timedelta(0))
    """Total execution time."""

    error: Exception | None = None
    """Error if the fan-out failed."""

    def success_count(self) -> int:
        """Get the number of agents that succeeded."""
        return sum(1 for a in self.agents.values() if a.error is None)

    def failure_count(self) -> int:
        """Get the number of agents that failed."""
        return len(self.agents) - self.success_count()


@dataclass
class FanOutError(Exception):
    """Error indicating fan-out execution failure."""

    failed_agent: str = ""
    """Primary agent that caused failure (if applicable)."""

    cause: Exception | None = None
    """The underlying error."""

    results: dict[str, AgentResult] = field(default_factory=dict)
    """All agent results for inspection."""

    def __str__(self) -> str:
        if self.failed_agent:
            return f"fan-out failed: agent {self.failed_agent!r}: {self.cause}"
        return f"fan-out failed: {self.cause}"

    def __repr__(self) -> str:
        return (
            f"FanOutError(failed_agent={self.failed_agent!r}, "
            f"cause={self.cause!r})"
        )


# Aggregator type
Aggregator = Callable[[dict[str, AgentResult]], tuple[bytes, Exception | None]]
"""Function that combines results from multiple agents into a single result."""


def _marshal_results(results: dict[str, bytes]) -> bytes:
    """Convert results map to JSON."""
    # Convert bytes to strings for JSON serialization
    str_results = {k: v.decode("utf-8", errors="replace") for k, v in results.items()}
    return json.dumps(str_results).encode()


# Aggregation strategies


def all_must_succeed() -> Aggregator:
    """Create an aggregator that fails if any agent fails.

    On success, returns all results as a JSON object.

    Example:
        result = await fanout.execute(agent, input, all_must_succeed())
    """

    def aggregator(results: dict[str, AgentResult]) -> tuple[bytes, Exception | None]:
        combined: dict[str, bytes] = {}
        for name, result in results.items():
            if result.error is not None:
                return b"", FanOutError(
                    failed_agent=name,
                    cause=result.error,
                    results=results,
                )
            combined[name] = result.output
        return _marshal_results(combined), None

    return aggregator


def first_success() -> Aggregator:
    """Create an aggregator that returns the first successful result.

    Fails only if all agents fail.

    Example:
        result = await fanout.execute(agent, input, first_success())
    """

    def aggregator(results: dict[str, AgentResult]) -> tuple[bytes, Exception | None]:
        first_output: bytes | None = None
        all_errors: list[str] = []

        for name, result in results.items():
            if result.error is None:
                if first_output is None:
                    first_output = result.output
            else:
                all_errors.append(f"{name}: {result.error}")

        if first_output is not None:
            return first_output, None

        return b"", FanOutError(
            cause=Exception("; ".join(all_errors)),
            results=results,
        )

    return aggregator


def collect_all() -> Aggregator:
    """Create an aggregator that collects all results (including failures).

    Never fails - returns all successful results as JSON.

    Example:
        result = await fanout.execute(agent, input, collect_all())
    """

    def aggregator(results: dict[str, AgentResult]) -> tuple[bytes, Exception | None]:
        combined: dict[str, bytes] = {}
        for name, result in results.items():
            if result.error is None:
                combined[name] = result.output
        return _marshal_results(combined), None

    return aggregator


def require_quorum(n: int) -> Aggregator:
    """Create an aggregator that requires at least n agents to succeed.

    Args:
        n: Minimum number of agents that must succeed.

    Example:
        result = await fanout.execute(agent, input, require_quorum(2))
    """

    def aggregator(results: dict[str, AgentResult]) -> tuple[bytes, Exception | None]:
        combined: dict[str, bytes] = {}
        success_count = 0

        for name, result in results.items():
            if result.error is None:
                combined[name] = result.output
                success_count += 1

        if success_count < n:
            return b"", FanOutError(
                cause=Exception(
                    f"quorum not reached: {success_count}/{len(results)} "
                    f"succeeded, need {n}"
                ),
                results=results,
            )

        return _marshal_results(combined), None

    return aggregator


@dataclass
class _FanOutAgent:
    """Internal representation of a fan-out agent."""

    name: str
    subject: str
    timeout: float | None = None


class FanOut:
    """Runs multiple agents in parallel and aggregates results.

    All agents receive the same input and execute concurrently.
    Results are combined using a configurable aggregation strategy.

    Example:
        fanout = (
            FanOut("analysis")
            .agent("fast", "agent.fast.invoke", timeout=5.0)
            .agent("slow", "agent.slow.invoke", timeout=30.0)
            .agent("reliable", "agent.reliable.invoke")
        )

        # All must succeed
        result = await fanout.execute(agent, b'{"query": "test"}', all_must_succeed())

        # First success wins
        result = await fanout.execute(agent, b'{"query": "test"}', first_success())

        # With tracing
        trace = await fanout.execute_with_trace(agent, b'{"query": "test"}')
        print(f"Successes: {trace.success_count()}, Failures: {trace.failure_count()}")
    """

    def __init__(self, name: str) -> None:
        """Create a new fan-out orchestrator.

        Args:
            name: Name for the fan-out (used in tracing).
        """
        self._name = name
        self._agents: list[_FanOutAgent] = []

    @property
    def name(self) -> str:
        """Get the fan-out name."""
        return self._name

    @property
    def agents(self) -> list[tuple[str, str]]:
        """Get list of (name, subject) tuples for all agents."""
        return [(a.name, a.subject) for a in self._agents]

    def agent(
        self,
        name: str,
        subject: str,
        *,
        timeout: float | None = None,
    ) -> FanOut:
        """Add an agent to the fan-out.

        All agents receive the same input and run in parallel.

        Args:
            name: Display name for the agent.
            subject: Subject to send requests to.
            timeout: Optional timeout for this agent in seconds.

        Returns:
            Self for method chaining.
        """
        self._agents.append(
            _FanOutAgent(name=name, subject=subject, timeout=timeout)
        )
        return self

    async def execute(
        self,
        client: Agent,
        input: bytes,
        aggregator: Aggregator | None = None,
    ) -> bytes:
        """Execute all agents in parallel and aggregate results.

        Args:
            client: The agent to use for requests.
            input: Input data sent to all agents.
            aggregator: Strategy for combining results.
                Defaults to collect_all().

        Returns:
            Aggregated output from all agents.

        Raises:
            FanOutError: If aggregation fails (depends on strategy).
        """
        trace = await self.execute_with_trace(client, input, aggregator)
        if trace.error:
            raise trace.error
        return trace.output

    async def execute_with_trace(
        self,
        client: Agent,
        input: bytes,
        aggregator: Aggregator | None = None,
    ) -> FanOutTrace:
        """Execute all agents with detailed tracing.

        Useful for debugging and monitoring fan-out execution.

        Args:
            client: The agent to use for requests.
            input: Input data sent to all agents.
            aggregator: Strategy for combining results.
                Defaults to collect_all().

        Returns:
            FanOutTrace with execution details for each agent.
        """
        if aggregator is None:
            aggregator = collect_all()

        trace = FanOutTrace(
            fanout=self._name,
            started_at=datetime.now(),
        )

        # Run all agents in parallel
        results: dict[str, AgentResult] = {}

        async def run_agent(agent: _FanOutAgent) -> tuple[str, AgentTrace, AgentResult]:
            agent_trace = AgentTrace(
                name=agent.name,
                subject=agent.subject,
                started_at=datetime.now(),
            )

            result = AgentResult()

            try:
                # Pass timeout to request for gRPC-level timeout
                timeout = agent.timeout if agent.timeout is not None else DEFAULT_TIMEOUT
                output = await client.request(agent.subject, input, timeout=timeout)

                agent_trace.output = output
                result.output = output

            except Exception as e:
                agent_trace.error = e
                result.error = e

            agent_trace.duration = datetime.now() - agent_trace.started_at
            result.duration = agent_trace.duration

            return agent.name, agent_trace, result

        # Execute all agents concurrently
        tasks = [run_agent(agent) for agent in self._agents]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                # This shouldn't happen since we catch exceptions inside run_agent
                continue
            name, agent_trace, result = item
            trace.agents[name] = agent_trace
            results[name] = result

        trace.duration = datetime.now() - trace.started_at

        # Aggregate results
        output, error = aggregator(results)
        if error is not None:
            trace.error = error
        else:
            trace.output = output

        return trace
