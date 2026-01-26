"""Tests for FanOut orchestration pattern."""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from athyr.orchestration.fanout import (
    AgentResult,
    FanOut,
    FanOutError,
    all_must_succeed,
    collect_all,
    first_success,
    require_quorum,
)


class TestFanOutCreation:
    """Tests for FanOut creation and configuration."""

    def test_create_fanout(self) -> None:
        """Test creating a fan-out with name."""
        fanout = FanOut("test-fanout")

        assert fanout.name == "test-fanout"
        assert fanout.agents == []

    def test_add_agent(self) -> None:
        """Test adding an agent to the fan-out."""
        fanout = FanOut("test").agent("first", "agent.first.invoke")

        assert len(fanout.agents) == 1
        assert fanout.agents[0] == ("first", "agent.first.invoke")

    def test_add_multiple_agents(self) -> None:
        """Test adding multiple agents."""
        fanout = (
            FanOut("test")
            .agent("a", "agent.a.invoke")
            .agent("b", "agent.b.invoke")
            .agent("c", "agent.c.invoke")
        )

        assert len(fanout.agents) == 3

    def test_fluent_api_returns_self(self) -> None:
        """Test that agent() returns self for chaining."""
        fanout = FanOut("test")
        result = fanout.agent("first", "agent.first.invoke")

        assert result is fanout


class TestFanOutExecution:
    """Tests for FanOut execution."""

    @pytest.mark.asyncio
    async def test_execute_empty_fanout(self) -> None:
        """Test executing an empty fan-out returns empty JSON."""
        fanout = FanOut("empty")
        mock_agent = AsyncMock()

        result = await fanout.execute(mock_agent, b"input")

        assert result == b"{}"
        mock_agent.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_single_agent(self) -> None:
        """Test executing a single-agent fan-out."""
        fanout = FanOut("single").agent("process", "agent.process.invoke")

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=b"result")

        result = await fanout.execute(mock_agent, b"input")

        assert b"process" in result
        mock_agent.request.assert_called_once_with("agent.process.invoke", b"input")

    @pytest.mark.asyncio
    async def test_execute_runs_agents_in_parallel(self) -> None:
        """Test that agents run in parallel."""
        fanout = (
            FanOut("parallel")
            .agent("a", "agent.a.invoke")
            .agent("b", "agent.b.invoke")
            .agent("c", "agent.c.invoke")
        )

        call_times: list[float] = []

        async def track_request(subject: str, data: bytes) -> bytes:
            import time
            call_times.append(time.time())
            await asyncio.sleep(0.05)  # 50ms
            return f"result-{subject}".encode()

        mock_agent = AsyncMock()
        mock_agent.request = track_request

        trace = await fanout.execute_with_trace(mock_agent, b"input")

        # All agents should have started nearly simultaneously
        # (within 20ms of each other)
        assert len(call_times) == 3
        time_spread = max(call_times) - min(call_times)
        assert time_spread < 0.02  # Less than 20ms

        # Total time should be ~50ms (parallel), not ~150ms (sequential)
        assert trace.duration < timedelta(milliseconds=100)


class TestAggregationStrategies:
    """Tests for aggregation strategies."""

    @pytest.mark.asyncio
    async def test_all_must_succeed_success(self) -> None:
        """Test all_must_succeed when all agents succeed."""
        fanout = (
            FanOut("test")
            .agent("a", "agent.a.invoke")
            .agent("b", "agent.b.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=[b"result-a", b"result-b"])

        result = await fanout.execute(mock_agent, b"input", all_must_succeed())

        assert b"a" in result
        assert b"b" in result

    @pytest.mark.asyncio
    async def test_all_must_succeed_failure(self) -> None:
        """Test all_must_succeed when one agent fails."""
        fanout = (
            FanOut("test")
            .agent("good", "agent.good.invoke")
            .agent("bad", "agent.bad.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[b"ok", Exception("Failed")]
        )

        with pytest.raises(FanOutError) as exc_info:
            await fanout.execute(mock_agent, b"input", all_must_succeed())

        error = exc_info.value
        assert error.failed_agent == "bad"
        assert "Failed" in str(error.cause)

    @pytest.mark.asyncio
    async def test_first_success_returns_first(self) -> None:
        """Test first_success returns first successful result."""
        fanout = (
            FanOut("test")
            .agent("fail", "agent.fail.invoke")
            .agent("ok", "agent.ok.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[Exception("Failed"), b"success"]
        )

        result = await fanout.execute(mock_agent, b"input", first_success())

        assert result == b"success"

    @pytest.mark.asyncio
    async def test_first_success_all_fail(self) -> None:
        """Test first_success fails when all agents fail."""
        fanout = (
            FanOut("test")
            .agent("fail1", "agent.fail1.invoke")
            .agent("fail2", "agent.fail2.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[Exception("Error 1"), Exception("Error 2")]
        )

        with pytest.raises(FanOutError) as exc_info:
            await fanout.execute(mock_agent, b"input", first_success())

        error = exc_info.value
        assert "Error 1" in str(error.cause) or "Error 2" in str(error.cause)

    @pytest.mark.asyncio
    async def test_collect_all_never_fails(self) -> None:
        """Test collect_all collects all results and never fails."""
        fanout = (
            FanOut("test")
            .agent("good", "agent.good.invoke")
            .agent("bad", "agent.bad.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[b"result", Exception("Failed")]
        )

        # Should not raise
        result = await fanout.execute(mock_agent, b"input", collect_all())

        # Only successful results included
        assert b"good" in result
        assert b"bad" not in result

    @pytest.mark.asyncio
    async def test_require_quorum_success(self) -> None:
        """Test require_quorum succeeds when quorum is met."""
        fanout = (
            FanOut("test")
            .agent("a", "agent.a.invoke")
            .agent("b", "agent.b.invoke")
            .agent("c", "agent.c.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[b"result-a", Exception("Failed"), b"result-c"]
        )

        # Require 2 successes - we have 2
        result = await fanout.execute(mock_agent, b"input", require_quorum(2))

        assert b"a" in result
        assert b"c" in result

    @pytest.mark.asyncio
    async def test_require_quorum_failure(self) -> None:
        """Test require_quorum fails when quorum is not met."""
        fanout = (
            FanOut("test")
            .agent("a", "agent.a.invoke")
            .agent("b", "agent.b.invoke")
            .agent("c", "agent.c.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[b"result-a", Exception("Failed"), Exception("Failed")]
        )

        # Require 2 successes - we only have 1
        with pytest.raises(FanOutError) as exc_info:
            await fanout.execute(mock_agent, b"input", require_quorum(2))

        error = exc_info.value
        assert "quorum not reached" in str(error.cause)


class TestFanOutTracing:
    """Tests for FanOut tracing."""

    @pytest.mark.asyncio
    async def test_trace_records_all_agents(self) -> None:
        """Test that tracing records all agent executions."""
        fanout = (
            FanOut("traced")
            .agent("a", "agent.a.invoke")
            .agent("b", "agent.b.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=[b"out-a", b"out-b"])

        trace = await fanout.execute_with_trace(mock_agent, b"input")

        assert trace.fanout == "traced"
        assert len(trace.agents) == 2
        assert "a" in trace.agents
        assert "b" in trace.agents
        assert trace.agents["a"].output == b"out-a"
        assert trace.agents["b"].output == b"out-b"

    @pytest.mark.asyncio
    async def test_trace_success_count(self) -> None:
        """Test trace.success_count()."""
        fanout = (
            FanOut("test")
            .agent("good1", "agent.good1.invoke")
            .agent("bad", "agent.bad.invoke")
            .agent("good2", "agent.good2.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[b"ok", Exception("Failed"), b"ok"]
        )

        trace = await fanout.execute_with_trace(mock_agent, b"input")

        assert trace.success_count() == 2
        assert trace.failure_count() == 1

    @pytest.mark.asyncio
    async def test_trace_records_error(self) -> None:
        """Test that tracing records errors."""
        fanout = FanOut("test").agent("fail", "agent.fail.invoke")

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=Exception("Boom"))

        trace = await fanout.execute_with_trace(mock_agent, b"input")

        assert trace.agents["fail"].error is not None
        assert "Boom" in str(trace.agents["fail"].error)


class TestFanOutTimeout:
    """Tests for FanOut agent timeout."""

    @pytest.mark.asyncio
    async def test_agent_timeout(self) -> None:
        """Test that agent timeout works."""
        fanout = FanOut("timeout").agent(
            "slow", "agent.slow.invoke", timeout=0.05
        )

        async def slow_request(subject: str, data: bytes) -> bytes:
            await asyncio.sleep(1.0)  # Too slow
            return b"done"

        mock_agent = AsyncMock()
        mock_agent.request = slow_request

        trace = await fanout.execute_with_trace(mock_agent, b"input")

        assert trace.agents["slow"].error is not None
        assert isinstance(trace.agents["slow"].error, asyncio.TimeoutError)


class TestAgentResult:
    """Tests for AgentResult."""

    def test_success_true_when_no_error(self) -> None:
        """Test success() returns True when no error."""
        result = AgentResult(output=b"data")
        assert result.success() is True

    def test_success_false_when_error(self) -> None:
        """Test success() returns False when error."""
        result = AgentResult(error=Exception("Failed"))
        assert result.success() is False


class TestFanOutError:
    """Tests for FanOutError."""

    def test_error_string_with_agent(self) -> None:
        """Test FanOutError string with failed agent."""
        error = FanOutError(
            failed_agent="bad",
            cause=Exception("Network error"),
        )

        error_str = str(error)
        assert "bad" in error_str
        assert "Network error" in error_str

    def test_error_string_without_agent(self) -> None:
        """Test FanOutError string without specific agent."""
        error = FanOutError(
            cause=Exception("General failure"),
        )

        error_str = str(error)
        assert "General failure" in error_str
