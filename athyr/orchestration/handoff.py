"""HandoffRouter pattern for dynamic task routing.

HandoffRouter provides dynamic task routing where a triage agent
classifies requests and delegates to appropriate specialist agents.

Example:
    router = (
        HandoffRouter("support-triage")
        .triage("agent.triage.invoke")
        .route("billing", "agent.billing.invoke")
        .route("technical", "agent.technical.invoke")
        .route("account", "agent.account.invoke")
        .max_handoffs(5)
        .fallback("agent.escalation.invoke")
    )

    result = await router.handle(agent, customer_request)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from athyr.orchestration import DEFAULT_TIMEOUT

if TYPE_CHECKING:
    from athyr.client import Agent


@dataclass
class HandoffDecision:
    """Expected response format from triage agents.

    Triage agents should return this as JSON to indicate routing decisions.

    Attributes:
        route: Name of the route to hand off to (e.g., "billing", "technical")
        context: Additional context to pass to the next agent (optional)
        handled: True if triage handled the request directly without routing
        response: The final response if handled is True
    """

    route: str = ""
    """Name of the route to hand off to."""

    context: bytes = b""
    """Additional context to pass to the next agent."""

    handled: bool = False
    """True if triage handled the request directly."""

    response: bytes = b""
    """Final response if handled is True."""

    @classmethod
    def from_json(cls, data: bytes) -> HandoffDecision:
        """Parse a HandoffDecision from JSON bytes.

        Args:
            data: JSON bytes to parse.

        Returns:
            HandoffDecision instance.

        Raises:
            json.JSONDecodeError: If data is not valid JSON.
        """
        obj = json.loads(data)
        return cls(
            route=obj.get("route", ""),
            context=json.dumps(obj.get("context", {})).encode() if obj.get("context") else b"",
            handled=obj.get("handled", False),
            response=json.dumps(obj.get("response", {})).encode() if obj.get("response") else b"",
        )

    def to_json(self) -> bytes:
        """Serialize to JSON bytes."""
        obj: dict[str, Any] = {}
        if self.route:
            obj["route"] = self.route
        if self.context:
            obj["context"] = json.loads(self.context)
        if self.handled:
            obj["handled"] = True
        if self.response:
            obj["response"] = json.loads(self.response)
        return json.dumps(obj).encode()


@dataclass
class HandoffStep:
    """Details about a single handoff step."""

    index: int
    """Step index in the routing path."""

    agent: str
    """Agent name (e.g., "triage", "billing")."""

    subject: str
    """Athyr subject called."""

    route: str = ""
    """Route returned by agent."""

    output: bytes = b""
    """Output from the agent."""

    started_at: datetime = field(default_factory=datetime.now)
    """When the step started."""

    duration: timedelta = field(default_factory=lambda: timedelta(0))
    """How long the step took."""

    error: Exception | None = None
    """Error if the step failed."""


@dataclass
class HandoffTrace:
    """Detailed routing information for a handoff run."""

    router: str
    """Router name."""

    path: list[HandoffStep] = field(default_factory=list)
    """Steps taken during routing."""

    input: bytes = b""
    """Original input."""

    output: bytes = b""
    """Final output."""

    started_at: datetime = field(default_factory=datetime.now)
    """When routing started."""

    duration: timedelta = field(default_factory=lambda: timedelta(0))
    """Total routing time."""

    error: Exception | None = None
    """Error if routing failed."""

    def route_names(self) -> list[str]:
        """Get the sequence of agent names traversed."""
        return [step.agent for step in self.path]


@dataclass
class HandoffError(Exception):
    """Error indicating routing failure."""

    agent: str = ""
    """Agent that caused the failure."""

    route: str = ""
    """Route that was attempted."""

    cause: Exception | None = None
    """The underlying error."""

    path: list[HandoffStep] = field(default_factory=list)
    """Path up to failure."""

    def __str__(self) -> str:
        if self.agent and self.route:
            return (
                f"handoff failed: agent {self.agent!r} tried to route to "
                f"{self.route!r}: {self.cause}"
            )
        if self.agent:
            return f"handoff failed: agent {self.agent!r}: {self.cause}"
        return f"handoff failed: {self.cause}"


class HandoffRouter:
    """Routes requests through triage to specialist agents.

    A HandoffRouter uses a triage agent to classify incoming requests
    and route them to appropriate specialist agents. Supports multi-hop
    routing where specialists can hand off to other specialists.

    Example:
        router = (
            HandoffRouter("customer-support")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
            .route("technical", "agent.tech.invoke")
            .fallback("agent.human.invoke")
        )

        # Simple execution
        result = await router.handle(agent, request)

        # With tracing
        trace = await router.handle_with_trace(agent, request)
        print(f"Route path: {' -> '.join(trace.route_names())}")
    """

    def __init__(self, name: str) -> None:
        """Create a new handoff router.

        Args:
            name: Name for the router (used in tracing).
        """
        self._name = name
        self._triage_subject: str = ""
        self._routes: dict[str, str] = {}
        self._max_handoffs: int = 10
        self._fallback_subject: str = ""

    @property
    def name(self) -> str:
        """Get the router name."""
        return self._name

    def triage(self, subject: str) -> HandoffRouter:
        """Set the triage agent that classifies requests.

        The triage agent must return a HandoffDecision as JSON.

        Args:
            subject: Subject of the triage agent.

        Returns:
            Self for method chaining.
        """
        self._triage_subject = subject
        return self

    def route(self, name: str, subject: str) -> HandoffRouter:
        """Add a route from a decision name to an agent subject.

        Args:
            name: Route name (returned by triage in HandoffDecision.route)
            subject: Subject of the specialist agent.

        Returns:
            Self for method chaining.
        """
        self._routes[name] = subject
        return self

    def max_handoffs(self, n: int) -> HandoffRouter:
        """Set the maximum number of handoffs allowed.

        Prevents infinite loops when agents keep re-routing.
        Default is 10.

        Args:
            n: Maximum handoffs.

        Returns:
            Self for method chaining.
        """
        self._max_handoffs = n
        return self

    def fallback(self, subject: str) -> HandoffRouter:
        """Set the fallback agent for unhandled routes.

        Called when triage returns an unknown route name.

        Args:
            subject: Subject of the fallback agent.

        Returns:
            Self for method chaining.
        """
        self._fallback_subject = subject
        return self

    async def handle(self, client: Agent, input: bytes) -> bytes:
        """Process a request through the handoff router.

        Args:
            client: The agent to use for requests.
            input: Input data to route.

        Returns:
            Output from the final agent.

        Raises:
            HandoffError: If routing fails.
        """
        trace = await self.handle_with_trace(client, input)
        if trace.error:
            raise trace.error
        return trace.output

    async def handle_with_trace(
        self, client: Agent, input: bytes
    ) -> HandoffTrace:
        """Process a request with detailed routing trace.

        Args:
            client: The agent to use for requests.
            input: Input data to route.

        Returns:
            HandoffTrace with routing details.
        """
        if not self._triage_subject:
            error = HandoffError(cause=Exception("no triage agent configured"))
            return HandoffTrace(
                router=self._name,
                input=input,
                error=error,
            )

        trace = HandoffTrace(
            router=self._name,
            started_at=datetime.now(),
            input=input,
        )

        current = input
        i = 0

        while i < self._max_handoffs:
            step = HandoffStep(
                index=i,
                started_at=datetime.now(),
                agent="",
                subject="",
            )

            # First iteration uses triage, subsequent use the routed agent
            if i == 0:
                step.agent = "triage"
                step.subject = self._triage_subject
            else:
                # Get the last route from previous step
                last_route = trace.path[-1].route
                step.agent = last_route
                step.subject = self._routes.get(last_route, "")

            # Call the agent
            try:
                output = await client.request(step.subject, current, timeout=DEFAULT_TIMEOUT)
            except Exception as e:
                step.duration = datetime.now() - step.started_at
                step.error = e
                trace.path.append(step)
                trace.duration = datetime.now() - trace.started_at
                trace.error = HandoffError(
                    agent=step.agent,
                    cause=e,
                    path=list(trace.path),
                )
                return trace

            step.duration = datetime.now() - step.started_at

            # Try to parse as HandoffDecision
            try:
                decision = HandoffDecision.from_json(output)
            except json.JSONDecodeError:
                # Not a decision - treat as final response
                step.route = "final"
                step.output = output
                trace.path.append(step)
                trace.duration = datetime.now() - trace.started_at
                trace.output = output
                return trace

            step.route = decision.route

            # Check if agent handled directly
            if decision.handled:
                step.output = decision.response
                trace.path.append(step)
                trace.duration = datetime.now() - trace.started_at
                trace.output = decision.response
                return trace

            # If no route specified, treat as final response
            if not decision.route:
                step.route = "final"
                step.output = output
                trace.path.append(step)
                trace.duration = datetime.now() - trace.started_at
                trace.output = output
                return trace

            # Check if route exists
            if decision.route not in self._routes:
                # Unknown route - use fallback if available
                if self._fallback_subject:
                    step.route = "fallback"
                    trace.path.append(step)

                    # Call fallback
                    fallback_step = HandoffStep(
                        index=i + 1,
                        agent="fallback",
                        subject=self._fallback_subject,
                        started_at=datetime.now(),
                    )

                    try:
                        fallback_output = await client.request(
                            self._fallback_subject, current, timeout=DEFAULT_TIMEOUT
                        )
                    except Exception as e:
                        fallback_step.duration = datetime.now() - fallback_step.started_at
                        fallback_step.error = e
                        trace.path.append(fallback_step)
                        trace.duration = datetime.now() - trace.started_at
                        trace.error = HandoffError(
                            agent="fallback",
                            cause=e,
                            path=list(trace.path),
                        )
                        return trace

                    fallback_step.duration = datetime.now() - fallback_step.started_at
                    fallback_step.output = fallback_output
                    fallback_step.route = "final"
                    trace.path.append(fallback_step)
                    trace.duration = datetime.now() - trace.started_at
                    trace.output = fallback_output
                    return trace

                # No fallback - error
                trace.path.append(step)
                trace.duration = datetime.now() - trace.started_at
                trace.error = HandoffError(
                    agent=step.agent,
                    route=decision.route,
                    cause=Exception(
                        f"unknown route {decision.route!r} and no fallback configured"
                    ),
                    path=list(trace.path),
                )
                return trace

            # Prepare context for next agent
            if decision.context:
                current = decision.context
            else:
                current = input

            trace.path.append(step)

            # Route to the specialist agent
            specialist_step = HandoffStep(
                index=i + 1,
                agent=decision.route,
                subject=self._routes[decision.route],
                started_at=datetime.now(),
            )

            try:
                specialist_output = await client.request(
                    self._routes[decision.route], current, timeout=DEFAULT_TIMEOUT
                )
            except Exception as e:
                specialist_step.duration = datetime.now() - specialist_step.started_at
                specialist_step.error = e
                trace.path.append(specialist_step)
                trace.duration = datetime.now() - trace.started_at
                trace.error = HandoffError(
                    agent=decision.route,
                    cause=e,
                    path=list(trace.path),
                )
                return trace

            specialist_step.duration = datetime.now() - specialist_step.started_at

            # Check if specialist returns another handoff decision
            try:
                next_decision = HandoffDecision.from_json(specialist_output)
            except json.JSONDecodeError:
                # Not a decision - treat as final response
                specialist_step.output = specialist_output
                specialist_step.route = "final"
                trace.path.append(specialist_step)
                trace.duration = datetime.now() - trace.started_at
                trace.output = specialist_output
                return trace

            # Specialist returned a decision
            if next_decision.handled:
                specialist_step.output = next_decision.response
                specialist_step.route = "final"
                trace.path.append(specialist_step)
                trace.duration = datetime.now() - trace.started_at
                trace.output = next_decision.response
                return trace

            # If no route specified, treat as final response
            if not next_decision.route:
                specialist_step.output = specialist_output
                specialist_step.route = "final"
                trace.path.append(specialist_step)
                trace.duration = datetime.now() - trace.started_at
                trace.output = specialist_output
                return trace

            # Another handoff - continue loop
            specialist_step.route = next_decision.route
            trace.path.append(specialist_step)

            if next_decision.context:
                current = next_decision.context
            else:
                current = input

            i += 2  # We processed two steps (triage/prev + specialist)

        # Max handoffs exceeded
        trace.duration = datetime.now() - trace.started_at
        trace.error = HandoffError(
            cause=Exception(f"max handoffs ({self._max_handoffs}) exceeded"),
            path=list(trace.path),
        )
        return trace
