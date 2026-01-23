"""GroupChat pattern for multi-agent collaborative discussions.

GroupChat orchestrates conversations where multiple agents take turns
contributing to a shared discussion on a topic.

Example:
    chat = (
        GroupChat("product-review")
        .manager(round_robin())
        .participant("pm", "agent.pm.invoke")
        .participant("engineer", "agent.engineer.invoke")
        .participant("designer", "agent.designer.invoke")
        .max_rounds(10)
        .consensus_check(lambda msgs: detect_agreement(msgs))
    )

    result = await chat.discuss(agent, topic)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

from athyr.orchestration import DEFAULT_TIMEOUT

if TYPE_CHECKING:
    from athyr.client import Agent


@dataclass
class Message:
    """A single contribution to the discussion.

    Attributes:
        speaker: Name of the participant who sent the message.
        content: The message content.
        timestamp: When the message was sent.
    """

    speaker: str
    """Name of the participant who sent the message."""

    content: str
    """The message content."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When the message was sent."""


@dataclass
class ChatRequest:
    """Input format for group chat participants.

    Participants receive this as JSON containing the topic
    and full conversation history.

    Attributes:
        topic: The discussion topic.
        messages: Previous messages in the conversation.
        round: Current round number (0-indexed).
    """

    topic: str
    """The discussion topic."""

    messages: list[Message] = field(default_factory=list)
    """Previous messages in the conversation."""

    round: int = 0
    """Current round number (0-indexed)."""

    def to_json(self) -> bytes:
        """Serialize to JSON bytes."""
        return json.dumps({
            "topic": self.topic,
            "messages": [
                {
                    "speaker": m.speaker,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in self.messages
            ],
            "round": self.round,
        }).encode()


@dataclass
class ChatResponse:
    """Expected output format from participants.

    Participants should return this as JSON.

    Attributes:
        content: The participant's contribution.
    """

    content: str = ""
    """The participant's contribution."""

    @classmethod
    def from_json(cls, data: bytes) -> ChatResponse:
        """Parse a ChatResponse from JSON bytes.

        If parsing fails, treats raw data as content.
        """
        try:
            obj = json.loads(data)
            return cls(content=obj.get("content", ""))
        except json.JSONDecodeError:
            # Treat raw response as content
            return cls(content=data.decode("utf-8", errors="replace"))


@dataclass
class ChatTrace:
    """Detailed discussion information for a group chat run.

    Attributes:
        groupchat: The GroupChat name.
        topic: The discussion topic.
        messages: All messages in the discussion.
        participants: Names of all participants.
        rounds_completed: Number of rounds that completed.
        consensus_reached: Whether consensus was detected.
        conclusion: The final message content.
        started_at: When the discussion started.
        duration: Total discussion time.
        error: Error if the discussion failed.
    """

    groupchat: str
    """The GroupChat name."""

    topic: str = ""
    """The discussion topic."""

    messages: list[Message] = field(default_factory=list)
    """All messages in the discussion."""

    participants: list[str] = field(default_factory=list)
    """Names of all participants."""

    rounds_completed: int = 0
    """Number of rounds that completed."""

    consensus_reached: bool = False
    """Whether consensus was detected."""

    conclusion: bytes = b""
    """The final message content."""

    started_at: datetime = field(default_factory=datetime.now)
    """When the discussion started."""

    duration: timedelta = field(default_factory=lambda: timedelta(0))
    """Total discussion time."""

    error: ChatError | None = None
    """Error if the discussion failed."""

    def transcript(self) -> str:
        """Get a formatted string of the discussion."""
        lines = []
        for msg in self.messages:
            lines.append(f"[{msg.speaker}]: {msg.content}")
        return "\n".join(lines)


@dataclass
class ChatError(Exception):
    """Error indicating group chat failure.

    Attributes:
        participant: Which participant caused the error (if applicable).
        round: Which round failed (if applicable).
        cause: The underlying error.
    """

    participant: str = ""
    """Which participant caused the error."""

    round: int = -1
    """Which round failed (-1 if not applicable)."""

    cause: Exception | None = None
    """The underlying error."""

    def __str__(self) -> str:
        if self.participant:
            return (
                f"group chat failed: participant {self.participant!r} "
                f"in round {self.round}: {self.cause}"
            )
        return f"group chat failed: {self.cause}"


# Manager Protocol and implementations


class Manager(ABC):
    """Selects which participant speaks next.

    Managers control the flow of conversation by deciding
    who speaks in each round. They can maintain state and
    use message history to make decisions.
    """

    @abstractmethod
    def init(self, participants: list[str]) -> Any:
        """Initialize manager state with participant list.

        Args:
            participants: Names of all participants.

        Returns:
            Opaque state object passed to subsequent calls.
        """
        ...

    @abstractmethod
    def next(self, state: Any, messages: list[Message]) -> str:
        """Select the next speaker.

        Args:
            state: State object from init().
            messages: All messages so far.

        Returns:
            Name of the next speaker, or empty string to end discussion.
        """
        ...


@dataclass
class _RoundRobinState:
    """Internal state for round-robin manager."""

    participants: list[str]
    index: int = 0


class _RoundRobinManager(Manager):
    """Selects speakers in rotating order."""

    def init(self, participants: list[str]) -> _RoundRobinState:
        return _RoundRobinState(participants=participants, index=0)

    def next(self, state: Any, messages: list[Message]) -> str:
        s: _RoundRobinState = state
        if not s.participants:
            return ""
        speaker = s.participants[s.index]
        s.index = (s.index + 1) % len(s.participants)
        return speaker


def round_robin() -> Manager:
    """Create a manager that selects speakers in rotating order.

    Each participant speaks once per cycle, repeating until max rounds.

    Example:
        chat = GroupChat("review").manager(round_robin())
    """
    return _RoundRobinManager()


@dataclass
class _FuncState:
    """Internal state for function manager."""

    participants: list[str]


class _FuncManager(Manager):
    """Wraps a custom selection function."""

    def __init__(
        self, fn: Callable[[list[str], list[Message]], str]
    ) -> None:
        self._fn = fn

    def init(self, participants: list[str]) -> _FuncState:
        return _FuncState(participants=participants)

    def next(self, state: Any, messages: list[Message]) -> str:
        s: _FuncState = state
        return self._fn(s.participants, messages)


def func_manager(
    fn: Callable[[list[str], list[Message]], str]
) -> Manager:
    """Create a manager using a custom selection function.

    The function receives the participant list and message history,
    and returns the name of the next speaker (or empty to end).

    Example:
        def pick_least_active(participants, messages):
            counts = {p: 0 for p in participants}
            for m in messages:
                if m.speaker in counts:
                    counts[m.speaker] += 1
            return min(counts, key=counts.get)

        chat = GroupChat("balanced").manager(func_manager(pick_least_active))
    """
    return _FuncManager(fn)


# Consensus function type
ConsensusFunc = Callable[[list[Message]], bool]
"""Function that checks if the group has reached consensus."""


@dataclass
class _Participant:
    """Internal representation of a group chat participant."""

    name: str
    subject: str


class GroupChat:
    """Orchestrates multi-agent collaborative discussions.

    Multiple agents take turns contributing to a shared conversation
    on a topic. A manager controls who speaks next, and an optional
    consensus function can end the discussion early.

    Example:
        chat = (
            GroupChat("product-review")
            .manager(round_robin())
            .participant("pm", "agent.pm.invoke")
            .participant("engineer", "agent.engineer.invoke")
            .participant("designer", "agent.designer.invoke")
            .max_rounds(10)
            .consensus_check(lambda msgs: "agreed" in msgs[-1].content.lower())
        )

        # Simple execution
        result = await chat.discuss(agent, b"Review the new feature")

        # With tracing
        trace = await chat.discuss_with_trace(agent, b"Review the new feature")
        print(trace.transcript())
    """

    def __init__(self, name: str) -> None:
        """Create a new group chat orchestrator.

        Args:
            name: Name for the group chat (used in tracing).
        """
        self._name = name
        self._manager: Manager = round_robin()
        self._participants: list[_Participant] = []
        self._max_rounds: int = 10
        self._consensus: ConsensusFunc | None = None

    @property
    def name(self) -> str:
        """Get the group chat name."""
        return self._name

    def manager(self, m: Manager) -> GroupChat:
        """Set the selection strategy for choosing the next speaker.

        Args:
            m: Manager that controls speaker selection.

        Returns:
            Self for method chaining.
        """
        self._manager = m
        return self

    def participant(self, name: str, subject: str) -> GroupChat:
        """Add a participant to the group chat.

        Args:
            name: Display name for the participant.
            subject: Subject to send requests to.

        Returns:
            Self for method chaining.
        """
        self._participants.append(_Participant(name=name, subject=subject))
        return self

    def max_rounds(self, n: int) -> GroupChat:
        """Set the maximum number of discussion rounds.

        Each round allows one participant to speak.
        Default is 10.

        Args:
            n: Maximum rounds.

        Returns:
            Self for method chaining.
        """
        self._max_rounds = n
        return self

    def consensus_check(self, fn: ConsensusFunc) -> GroupChat:
        """Set a function to detect when consensus is reached.

        If the function returns True, the discussion ends early.

        Args:
            fn: Function that takes message history and returns True
                if consensus is reached.

        Returns:
            Self for method chaining.
        """
        self._consensus = fn
        return self

    async def discuss(self, client: Agent, topic: bytes) -> bytes:
        """Start a group discussion on the given topic.

        Args:
            client: The agent to use for requests.
            topic: The discussion topic.

        Returns:
            The conclusion (last message content).

        Raises:
            ChatError: If the discussion fails.
        """
        trace = await self.discuss_with_trace(client, topic)
        if trace.error:
            raise trace.error
        return trace.conclusion

    async def discuss_with_trace(
        self, client: Agent, topic: bytes
    ) -> ChatTrace:
        """Run the discussion and return detailed trace.

        Args:
            client: The agent to use for requests.
            topic: The discussion topic.

        Returns:
            ChatTrace with discussion details.
        """
        if not self._participants:
            error = ChatError(cause=Exception("no participants configured"))
            return ChatTrace(
                groupchat=self._name,
                topic=topic.decode("utf-8", errors="replace"),
                error=error,
            )

        trace = ChatTrace(
            groupchat=self._name,
            topic=topic.decode("utf-8", errors="replace"),
            started_at=datetime.now(),
            participants=[p.name for p in self._participants],
        )

        # Build participant lookup
        participant_map = {p.name: p for p in self._participants}

        # Initialize manager state
        state = self._manager.init(trace.participants)

        for round_num in range(self._max_rounds):
            # Select next speaker
            speaker = self._manager.next(state, trace.messages)
            if not speaker:
                # Manager signals done
                break

            p = participant_map.get(speaker)
            if not p:
                trace.duration = datetime.now() - trace.started_at
                trace.error = ChatError(
                    participant=speaker,
                    cause=Exception(
                        f"unknown participant {speaker!r} selected by manager"
                    ),
                )
                return trace

            # Prepare request for participant
            req = ChatRequest(
                topic=trace.topic,
                messages=list(trace.messages),
                round=round_num,
            )

            # Call participant
            try:
                resp_bytes = await client.request(p.subject, req.to_json(), timeout=DEFAULT_TIMEOUT)
            except Exception as e:
                trace.duration = datetime.now() - trace.started_at
                trace.error = ChatError(
                    participant=p.name,
                    round=round_num,
                    cause=e,
                )
                return trace

            # Parse response
            resp = ChatResponse.from_json(resp_bytes)

            # Add message to history
            msg = Message(
                speaker=p.name,
                content=resp.content,
                timestamp=datetime.now(),
            )
            trace.messages.append(msg)

            # Check for consensus
            if self._consensus is not None and self._consensus(trace.messages):
                trace.consensus_reached = True
                break

        trace.duration = datetime.now() - trace.started_at
        trace.rounds_completed = len(trace.messages)

        # Set conclusion to last message content
        if trace.messages:
            trace.conclusion = trace.messages[-1].content.encode()

        return trace
