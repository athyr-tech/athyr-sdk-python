"""Tests for GroupChat orchestration pattern."""

import json
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from athyr.orchestration.groupchat import (
    ChatError,
    ChatRequest,
    ChatResponse,
    ChatTrace,
    GroupChat,
    Message,
    func_manager,
    round_robin,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self) -> None:
        """Test creating a message."""
        msg = Message(speaker="alice", content="Hello world")

        assert msg.speaker == "alice"
        assert msg.content == "Hello world"
        assert msg.timestamp is not None


class TestChatRequest:
    """Tests for ChatRequest dataclass."""

    def test_to_json(self) -> None:
        """Test serializing ChatRequest to JSON."""
        msg = Message(speaker="alice", content="Hi")
        req = ChatRequest(topic="test topic", messages=[msg], round=1)

        data = req.to_json()
        parsed = json.loads(data)

        assert parsed["topic"] == "test topic"
        assert parsed["round"] == 1
        assert len(parsed["messages"]) == 1
        assert parsed["messages"][0]["speaker"] == "alice"
        assert parsed["messages"][0]["content"] == "Hi"


class TestChatResponse:
    """Tests for ChatResponse dataclass."""

    def test_from_json_valid(self) -> None:
        """Test parsing valid JSON response."""
        data = json.dumps({"content": "My response"}).encode()
        resp = ChatResponse.from_json(data)

        assert resp.content == "My response"

    def test_from_json_invalid_treats_as_content(self) -> None:
        """Test that invalid JSON is treated as raw content."""
        resp = ChatResponse.from_json(b"Plain text response")

        assert resp.content == "Plain text response"


class TestGroupChatCreation:
    """Tests for GroupChat creation and configuration."""

    def test_create_groupchat(self) -> None:
        """Test creating a group chat with name."""
        chat = GroupChat("product-review")

        assert chat.name == "product-review"

    def test_add_participant(self) -> None:
        """Test adding a participant."""
        chat = GroupChat("test").participant("alice", "agent.alice.invoke")

        assert len(chat._participants) == 1
        assert chat._participants[0].name == "alice"
        assert chat._participants[0].subject == "agent.alice.invoke"

    def test_add_multiple_participants(self) -> None:
        """Test adding multiple participants."""
        chat = (
            GroupChat("test")
            .participant("alice", "agent.alice.invoke")
            .participant("bob", "agent.bob.invoke")
            .participant("charlie", "agent.charlie.invoke")
        )

        assert len(chat._participants) == 3

    def test_set_max_rounds(self) -> None:
        """Test setting max rounds."""
        chat = GroupChat("test").max_rounds(5)

        assert chat._max_rounds == 5

    def test_set_manager(self) -> None:
        """Test setting a custom manager."""
        custom_manager = round_robin()
        chat = GroupChat("test").manager(custom_manager)

        assert chat._manager is custom_manager

    def test_set_consensus_check(self) -> None:
        """Test setting a consensus check function."""
        check_fn = lambda msgs: len(msgs) >= 3
        chat = GroupChat("test").consensus_check(check_fn)

        assert chat._consensus is check_fn

    def test_fluent_api_returns_self(self) -> None:
        """Test that all methods return self for chaining."""
        chat = GroupChat("test")

        assert chat.participant("a", "a") is chat
        assert chat.max_rounds(5) is chat
        assert chat.manager(round_robin()) is chat
        assert chat.consensus_check(lambda m: False) is chat


class TestGroupChatExecution:
    """Tests for GroupChat execution."""

    @pytest.mark.asyncio
    async def test_discuss_no_participants(self) -> None:
        """Test that discuss fails without participants."""
        chat = GroupChat("empty")
        mock_agent = AsyncMock()

        with pytest.raises(ChatError) as exc_info:
            await chat.discuss(mock_agent, b"topic")

        assert "no participants configured" in str(exc_info.value.cause)

    @pytest.mark.asyncio
    async def test_discuss_single_participant(self) -> None:
        """Test discussion with a single participant."""
        chat = (
            GroupChat("single")
            .participant("alice", "agent.alice.invoke")
            .max_rounds(1)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            return_value=json.dumps({"content": "My thoughts"}).encode()
        )

        result = await chat.discuss(mock_agent, b"topic")

        assert result == b"My thoughts"
        mock_agent.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_discuss_round_robin_order(self) -> None:
        """Test that round robin visits participants in order."""
        chat = (
            GroupChat("ordered")
            .participant("alice", "agent.alice.invoke")
            .participant("bob", "agent.bob.invoke")
            .participant("charlie", "agent.charlie.invoke")
            .max_rounds(6)
        )

        speakers: list[str] = []

        async def track_speaker(subject: str, data: bytes) -> bytes:
            req = json.loads(data)
            # Extract participant name from subject
            name = subject.split(".")[1]
            speakers.append(name)
            return json.dumps({"content": f"Response from {name}"}).encode()

        mock_agent = AsyncMock()
        mock_agent.request = track_speaker

        await chat.discuss(mock_agent, b"topic")

        # Should cycle through alice, bob, charlie twice
        assert speakers == ["alice", "bob", "charlie", "alice", "bob", "charlie"]

    @pytest.mark.asyncio
    async def test_discuss_passes_message_history(self) -> None:
        """Test that participants receive message history."""
        chat = (
            GroupChat("history")
            .participant("alice", "agent.alice.invoke")
            .participant("bob", "agent.bob.invoke")
            .max_rounds(2)
        )

        received_requests: list[dict] = []

        async def capture_request(subject: str, data: bytes) -> bytes:
            req = json.loads(data)
            received_requests.append(req)
            name = subject.split(".")[1]
            return json.dumps({"content": f"From {name}"}).encode()

        mock_agent = AsyncMock()
        mock_agent.request = capture_request

        await chat.discuss(mock_agent, b"test topic")

        # First participant receives empty history
        assert received_requests[0]["topic"] == "test topic"
        assert received_requests[0]["round"] == 0
        assert len(received_requests[0]["messages"]) == 0

        # Second participant receives first message
        assert received_requests[1]["round"] == 1
        assert len(received_requests[1]["messages"]) == 1
        assert received_requests[1]["messages"][0]["speaker"] == "alice"

    @pytest.mark.asyncio
    async def test_discuss_consensus_ends_early(self) -> None:
        """Test that consensus check can end discussion early."""
        chat = (
            GroupChat("consensus")
            .participant("alice", "agent.alice.invoke")
            .participant("bob", "agent.bob.invoke")
            .max_rounds(10)
            .consensus_check(lambda msgs: len(msgs) >= 2)
        )

        call_count = 0

        async def count_calls(subject: str, data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            return json.dumps({"content": "response"}).encode()

        mock_agent = AsyncMock()
        mock_agent.request = count_calls

        trace = await chat.discuss_with_trace(mock_agent, b"topic")

        # Should stop after 2 rounds due to consensus
        assert call_count == 2
        assert trace.consensus_reached is True

    @pytest.mark.asyncio
    async def test_discuss_participant_error(self) -> None:
        """Test error handling when participant fails."""
        chat = (
            GroupChat("failing")
            .participant("alice", "agent.alice.invoke")
            .participant("bob", "agent.bob.invoke")
            .max_rounds(5)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[
                json.dumps({"content": "ok"}).encode(),
                Exception("Bob is unavailable"),
            ]
        )

        with pytest.raises(ChatError) as exc_info:
            await chat.discuss(mock_agent, b"topic")

        error = exc_info.value
        assert error.participant == "bob"
        assert error.round == 1
        assert "Bob is unavailable" in str(error.cause)

    @pytest.mark.asyncio
    async def test_discuss_raw_response_as_content(self) -> None:
        """Test that non-JSON responses are treated as content."""
        chat = (
            GroupChat("raw")
            .participant("alice", "agent.alice.invoke")
            .max_rounds(1)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=b"Plain text response")

        result = await chat.discuss(mock_agent, b"topic")

        assert result == b"Plain text response"


class TestManagerStrategies:
    """Tests for manager selection strategies."""

    @pytest.mark.asyncio
    async def test_round_robin_manager(self) -> None:
        """Test round robin manager cycles through participants."""
        mgr = round_robin()
        state = mgr.init(["a", "b", "c"])

        assert mgr.next(state, []) == "a"
        assert mgr.next(state, []) == "b"
        assert mgr.next(state, []) == "c"
        assert mgr.next(state, []) == "a"  # Cycles back

    @pytest.mark.asyncio
    async def test_round_robin_empty_participants(self) -> None:
        """Test round robin with no participants returns empty."""
        mgr = round_robin()
        state = mgr.init([])

        assert mgr.next(state, []) == ""

    @pytest.mark.asyncio
    async def test_func_manager(self) -> None:
        """Test custom function manager."""
        # Manager that always picks the least active participant
        def pick_least_active(participants: list[str], messages: list[Message]) -> str:
            if not participants:
                return ""
            counts = {p: 0 for p in participants}
            for m in messages:
                if m.speaker in counts:
                    counts[m.speaker] += 1
            return min(counts, key=lambda p: counts[p])

        mgr = func_manager(pick_least_active)
        state = mgr.init(["alice", "bob"])

        # Initially both have 0, returns first alphabetically (alice)
        assert mgr.next(state, []) == "alice"

        # After alice speaks, bob should be next
        msgs = [Message(speaker="alice", content="hi")]
        assert mgr.next(state, msgs) == "bob"

    @pytest.mark.asyncio
    async def test_func_manager_can_end_discussion(self) -> None:
        """Test that func_manager can end discussion by returning empty."""
        # Manager that ends after 2 messages
        def end_after_two(participants: list[str], messages: list[Message]) -> str:
            if len(messages) >= 2:
                return ""
            return participants[len(messages) % len(participants)]

        chat = (
            GroupChat("ending")
            .manager(func_manager(end_after_two))
            .participant("alice", "agent.alice.invoke")
            .participant("bob", "agent.bob.invoke")
            .max_rounds(10)
        )

        call_count = 0

        async def count_calls(subject: str, data: bytes) -> bytes:
            nonlocal call_count
            call_count += 1
            return json.dumps({"content": "msg"}).encode()

        mock_agent = AsyncMock()
        mock_agent.request = count_calls

        await chat.discuss(mock_agent, b"topic")

        # Should stop after 2 rounds when manager returns ""
        assert call_count == 2


class TestGroupChatTracing:
    """Tests for GroupChat tracing."""

    @pytest.mark.asyncio
    async def test_trace_records_messages(self) -> None:
        """Test that trace records all messages."""
        chat = (
            GroupChat("traced")
            .participant("alice", "agent.alice.invoke")
            .participant("bob", "agent.bob.invoke")
            .max_rounds(2)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[
                json.dumps({"content": "Alice says hi"}).encode(),
                json.dumps({"content": "Bob responds"}).encode(),
            ]
        )

        trace = await chat.discuss_with_trace(mock_agent, b"test topic")

        assert trace.groupchat == "traced"
        assert trace.topic == "test topic"
        assert len(trace.messages) == 2
        assert trace.messages[0].speaker == "alice"
        assert trace.messages[0].content == "Alice says hi"
        assert trace.messages[1].speaker == "bob"
        assert trace.messages[1].content == "Bob responds"

    @pytest.mark.asyncio
    async def test_trace_records_participants(self) -> None:
        """Test that trace records participant list."""
        chat = (
            GroupChat("test")
            .participant("alice", "agent.alice.invoke")
            .participant("bob", "agent.bob.invoke")
            .max_rounds(1)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            return_value=json.dumps({"content": "msg"}).encode()
        )

        trace = await chat.discuss_with_trace(mock_agent, b"topic")

        assert trace.participants == ["alice", "bob"]

    @pytest.mark.asyncio
    async def test_trace_rounds_completed(self) -> None:
        """Test trace.rounds_completed."""
        chat = (
            GroupChat("test")
            .participant("alice", "agent.alice.invoke")
            .max_rounds(3)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            return_value=json.dumps({"content": "msg"}).encode()
        )

        trace = await chat.discuss_with_trace(mock_agent, b"topic")

        assert trace.rounds_completed == 3

    @pytest.mark.asyncio
    async def test_trace_consensus_reached(self) -> None:
        """Test trace.consensus_reached."""
        chat = (
            GroupChat("test")
            .participant("alice", "agent.alice.invoke")
            .max_rounds(10)
            .consensus_check(lambda msgs: len(msgs) >= 1)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            return_value=json.dumps({"content": "done"}).encode()
        )

        trace = await chat.discuss_with_trace(mock_agent, b"topic")

        assert trace.consensus_reached is True
        assert trace.rounds_completed == 1

    @pytest.mark.asyncio
    async def test_trace_conclusion(self) -> None:
        """Test trace.conclusion is last message content."""
        chat = (
            GroupChat("test")
            .participant("alice", "agent.alice.invoke")
            .participant("bob", "agent.bob.invoke")
            .max_rounds(2)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[
                json.dumps({"content": "first"}).encode(),
                json.dumps({"content": "final answer"}).encode(),
            ]
        )

        trace = await chat.discuss_with_trace(mock_agent, b"topic")

        assert trace.conclusion == b"final answer"

    @pytest.mark.asyncio
    async def test_trace_duration(self) -> None:
        """Test trace records duration."""
        chat = (
            GroupChat("test")
            .participant("alice", "agent.alice.invoke")
            .max_rounds(1)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            return_value=json.dumps({"content": "done"}).encode()
        )

        trace = await chat.discuss_with_trace(mock_agent, b"topic")

        assert trace.duration >= timedelta(0)

    @pytest.mark.asyncio
    async def test_trace_transcript(self) -> None:
        """Test trace.transcript() formatting."""
        chat = (
            GroupChat("test")
            .participant("alice", "agent.alice.invoke")
            .participant("bob", "agent.bob.invoke")
            .max_rounds(2)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[
                json.dumps({"content": "Hello"}).encode(),
                json.dumps({"content": "Hi there"}).encode(),
            ]
        )

        trace = await chat.discuss_with_trace(mock_agent, b"topic")

        transcript = trace.transcript()
        assert "[alice]: Hello" in transcript
        assert "[bob]: Hi there" in transcript

    @pytest.mark.asyncio
    async def test_trace_records_error(self) -> None:
        """Test that trace records errors."""
        chat = (
            GroupChat("test")
            .participant("alice", "agent.alice.invoke")
            .max_rounds(1)
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=Exception("Network error"))

        trace = await chat.discuss_with_trace(mock_agent, b"topic")

        assert trace.error is not None
        assert trace.error.participant == "alice"
        assert "Network error" in str(trace.error.cause)


class TestChatTrace:
    """Tests for ChatTrace dataclass."""

    def test_trace_defaults(self) -> None:
        """Test ChatTrace default values."""
        trace = ChatTrace(groupchat="test")

        assert trace.messages == []
        assert trace.participants == []
        assert trace.rounds_completed == 0
        assert trace.consensus_reached is False
        assert trace.conclusion == b""
        assert trace.error is None

    def test_transcript_empty(self) -> None:
        """Test transcript() on empty trace."""
        trace = ChatTrace(groupchat="test")

        assert trace.transcript() == ""


class TestChatError:
    """Tests for ChatError."""

    def test_error_string_with_participant(self) -> None:
        """Test error string with participant info."""
        error = ChatError(
            participant="bob",
            round=3,
            cause=Exception("Connection failed"),
        )

        error_str = str(error)
        assert "bob" in error_str
        assert "round 3" in error_str
        assert "Connection failed" in error_str

    def test_error_string_without_participant(self) -> None:
        """Test error string without participant."""
        error = ChatError(
            cause=Exception("General failure"),
        )

        error_str = str(error)
        assert "General failure" in error_str
