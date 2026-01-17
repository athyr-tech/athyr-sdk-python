"""Tests for auto-reconnect functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from athyr.client import AthyrAgent
from athyr.client_state import ConnectionState


class TestAutoReconnectConfig:
    """Tests for auto-reconnect configuration options."""

    def test_default_auto_reconnect_disabled(self) -> None:
        """Test auto-reconnect is disabled by default."""
        agent = AthyrAgent("localhost:9090")
        assert agent._auto_reconnect is False

    def test_auto_reconnect_enabled(self) -> None:
        """Test enabling auto-reconnect."""
        agent = AthyrAgent("localhost:9090", auto_reconnect=True)
        assert agent._auto_reconnect is True

    def test_default_max_retries(self) -> None:
        """Test default max_retries is 0 (infinite)."""
        agent = AthyrAgent("localhost:9090")
        assert agent._max_retries == 0

    def test_custom_max_retries(self) -> None:
        """Test custom max_retries value."""
        agent = AthyrAgent("localhost:9090", max_retries=5)
        assert agent._max_retries == 5

    def test_default_backoff_values(self) -> None:
        """Test default backoff configuration."""
        agent = AthyrAgent("localhost:9090")
        assert agent._base_backoff == 1.0
        assert agent._max_backoff == 30.0

    def test_custom_backoff_values(self) -> None:
        """Test custom backoff configuration."""
        agent = AthyrAgent(
            "localhost:9090",
            base_backoff=0.5,
            max_backoff=60.0,
        )
        assert agent._base_backoff == 0.5
        assert agent._max_backoff == 60.0

    def test_reconnect_task_initially_none(self) -> None:
        """Test reconnect task is None initially."""
        agent = AthyrAgent("localhost:9090")
        assert agent._reconnect_task is None


class TestBackoffCalculation:
    """Tests for exponential backoff with jitter."""

    def test_backoff_increases_exponentially(self) -> None:
        """Test backoff increases with each attempt."""
        agent = AthyrAgent("localhost:9090", base_backoff=1.0, max_backoff=100.0)

        # Get multiple backoff values to average out jitter
        samples_per_attempt = 50
        avg_backoffs = []

        for attempt in range(5):
            total = sum(
                agent._calculate_backoff(attempt)
                for _ in range(samples_per_attempt)
            )
            avg_backoffs.append(total / samples_per_attempt)

        # Each subsequent backoff should be roughly double
        # (allowing for jitter variation)
        for i in range(1, len(avg_backoffs)):
            assert avg_backoffs[i] > avg_backoffs[i - 1]

    def test_backoff_capped_at_max(self) -> None:
        """Test backoff doesn't exceed max_backoff."""
        agent = AthyrAgent("localhost:9090", base_backoff=1.0, max_backoff=10.0)

        # With high attempt number, backoff should be capped
        for _ in range(100):
            backoff = agent._calculate_backoff(100)
            # Max with jitter could be 1.5x max_backoff
            assert backoff <= 10.0 * 1.5

    def test_backoff_with_jitter_varies(self) -> None:
        """Test backoff includes jitter (not constant)."""
        agent = AthyrAgent("localhost:9090", base_backoff=1.0)

        backoffs = [agent._calculate_backoff(0) for _ in range(10)]

        # Backoffs should vary due to jitter
        assert len(set(backoffs)) > 1

    def test_backoff_jitter_within_range(self) -> None:
        """Test jitter is between 0.5x and 1.5x."""
        agent = AthyrAgent("localhost:9090", base_backoff=2.0, max_backoff=100.0)

        for attempt in range(5):
            expected_base = min(2.0 * (2**attempt), 100.0)
            for _ in range(50):
                backoff = agent._calculate_backoff(attempt)
                # Should be between 0.5x and 1.5x of base
                assert backoff >= expected_base * 0.5
                assert backoff <= expected_base * 1.5


class TestTriggerReconnect:
    """Tests for _trigger_reconnect method."""

    @pytest.mark.asyncio
    async def test_trigger_reconnect_sets_state(self) -> None:
        """Test trigger_reconnect sets state to RECONNECTING."""
        agent = AthyrAgent("localhost:9090", auto_reconnect=True)
        error = ConnectionError("test error")

        with patch.object(agent, "_reconnect_loop", new_callable=AsyncMock):
            agent._trigger_reconnect(error)
            await asyncio.sleep(0)  # Let task start

        assert agent._state_machine.state == ConnectionState.RECONNECTING

        # Cleanup
        if agent._reconnect_task:
            agent._reconnect_task.cancel()
            try:
                await agent._reconnect_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_trigger_reconnect_creates_task(self) -> None:
        """Test trigger_reconnect creates a reconnect task."""
        agent = AthyrAgent("localhost:9090", auto_reconnect=True)
        error = ConnectionError("test error")

        with patch.object(agent, "_reconnect_loop", new_callable=AsyncMock):
            agent._trigger_reconnect(error)

        assert agent._reconnect_task is not None

        # Cleanup
        agent._reconnect_task.cancel()
        try:
            await agent._reconnect_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_trigger_reconnect_idempotent(self) -> None:
        """Test calling trigger_reconnect twice doesn't create duplicate tasks."""
        agent = AthyrAgent("localhost:9090", auto_reconnect=True)
        error = ConnectionError("test error")

        with patch.object(agent, "_reconnect_loop", new_callable=AsyncMock):
            agent._trigger_reconnect(error)
            first_task = agent._reconnect_task

            # Trigger again - should be no-op since already reconnecting
            agent._trigger_reconnect(error)
            second_task = agent._reconnect_task

        assert first_task is second_task

        # Cleanup
        if first_task:
            first_task.cancel()
            try:
                await first_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_trigger_reconnect_passes_error_to_callback(self) -> None:
        """Test state change callback receives the error."""
        errors_received: list[Exception | None] = []

        def callback(state: ConnectionState, error: Exception | None) -> None:
            errors_received.append(error)

        agent = AthyrAgent(
            "localhost:9090",
            auto_reconnect=True,
            on_state_change=callback,
        )
        test_error = ValueError("connection lost")

        with patch.object(agent, "_reconnect_loop", new_callable=AsyncMock):
            agent._trigger_reconnect(test_error)

        assert test_error in errors_received

        # Cleanup
        if agent._reconnect_task:
            agent._reconnect_task.cancel()
            try:
                await agent._reconnect_task
            except asyncio.CancelledError:
                pass


class TestReconnectLoop:
    """Tests for _reconnect_loop method."""

    @pytest.mark.asyncio
    async def test_reconnect_loop_succeeds_on_first_attempt(self) -> None:
        """Test reconnect loop succeeds immediately if do_reconnect works."""
        agent = AthyrAgent("localhost:9090", auto_reconnect=True, base_backoff=0.01)
        agent._state_machine.set_state_sync(ConnectionState.RECONNECTING)

        with patch.object(agent, "_do_reconnect", new_callable=AsyncMock) as mock_reconnect:
            await agent._reconnect_loop(ConnectionError("initial"))

        mock_reconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_loop_retries_on_failure(self) -> None:
        """Test reconnect loop retries after failure."""
        agent = AthyrAgent(
            "localhost:9090",
            auto_reconnect=True,
            max_retries=3,
            base_backoff=0.01,
        )
        agent._state_machine.set_state_sync(ConnectionState.RECONNECTING)

        call_count = 0

        async def failing_reconnect() -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError(f"attempt {call_count}")

        with patch.object(agent, "_do_reconnect", side_effect=failing_reconnect):
            await agent._reconnect_loop(ConnectionError("initial"))

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_reconnect_loop_gives_up_after_max_retries(self) -> None:
        """Test reconnect loop stops after max_retries."""
        agent = AthyrAgent(
            "localhost:9090",
            auto_reconnect=True,
            max_retries=2,
            base_backoff=0.01,
        )
        agent._state_machine.set_state_sync(ConnectionState.RECONNECTING)

        with patch.object(
            agent, "_do_reconnect", side_effect=ConnectionError("fail")
        ):
            await agent._reconnect_loop(ConnectionError("initial"))

        assert agent._state_machine.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_reconnect_loop_cancellation(self) -> None:
        """Test reconnect loop can be cancelled during backoff."""
        agent = AthyrAgent(
            "localhost:9090",
            auto_reconnect=True,
            base_backoff=10.0,  # Long backoff so we can cancel during sleep
        )
        agent._state_machine.set_state_sync(ConnectionState.RECONNECTING)

        task = asyncio.create_task(
            agent._reconnect_loop(ConnectionError("initial"))
        )
        await asyncio.sleep(0.01)  # Let it start the backoff sleep
        task.cancel()

        # The task should complete (it catches CancelledError internally)
        # and should not raise
        await task


class TestDoReconnect:
    """Tests for _do_reconnect method."""

    @pytest.mark.asyncio
    async def test_do_reconnect_closes_old_channel(self) -> None:
        """Test do_reconnect closes existing channel."""
        agent = AthyrAgent("localhost:9090")
        mock_channel = AsyncMock()
        agent._channel = mock_channel

        with patch("athyr.client.grpc.aio.insecure_channel") as mock_new_channel:
            mock_new_channel.return_value = AsyncMock()
            mock_stub = MagicMock()
            mock_stub.Connect = AsyncMock(
                return_value=MagicMock(agent_id="new-agent-123")
            )
            with patch("athyr.client.athyr_pb2_grpc.AthyrStub", return_value=mock_stub):
                await agent._do_reconnect()

        mock_channel.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_do_reconnect_creates_new_channel(self) -> None:
        """Test do_reconnect creates a new channel."""
        agent = AthyrAgent("localhost:9090")

        with patch("athyr.client.grpc.aio.insecure_channel") as mock_new_channel:
            mock_channel = AsyncMock()
            mock_new_channel.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Connect = AsyncMock(
                return_value=MagicMock(agent_id="new-agent-123")
            )
            with patch("athyr.client.athyr_pb2_grpc.AthyrStub", return_value=mock_stub):
                await agent._do_reconnect()

        mock_new_channel.assert_called_once_with("localhost:9090")

    @pytest.mark.asyncio
    async def test_do_reconnect_registers_agent(self) -> None:
        """Test do_reconnect re-registers the agent."""
        agent = AthyrAgent("localhost:9090")

        with patch("athyr.client.grpc.aio.insecure_channel") as mock_new_channel:
            mock_channel = AsyncMock()
            mock_new_channel.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Connect = AsyncMock(
                return_value=MagicMock(agent_id="reconnected-agent")
            )
            with patch("athyr.client.athyr_pb2_grpc.AthyrStub", return_value=mock_stub):
                await agent._do_reconnect()

        assert agent._agent_id == "reconnected-agent"
        mock_stub.Connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_do_reconnect_sets_connected_state(self) -> None:
        """Test do_reconnect sets state to CONNECTED."""
        agent = AthyrAgent("localhost:9090")
        agent._state_machine.set_state_sync(ConnectionState.RECONNECTING)

        with patch("athyr.client.grpc.aio.insecure_channel") as mock_new_channel:
            mock_channel = AsyncMock()
            mock_new_channel.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Connect = AsyncMock(
                return_value=MagicMock(agent_id="test-agent")
            )
            with patch("athyr.client.athyr_pb2_grpc.AthyrStub", return_value=mock_stub):
                await agent._do_reconnect()

        assert agent._state_machine.state == ConnectionState.CONNECTED


class TestCloseWithReconnect:
    """Tests for close() behavior during reconnection."""

    @pytest.mark.asyncio
    async def test_close_cancels_reconnect_task(self) -> None:
        """Test close() cancels in-progress reconnection."""
        agent = AthyrAgent(
            "localhost:9090",
            auto_reconnect=True,
            base_backoff=10.0,  # Long backoff
        )

        # Manually set up a "reconnecting" state
        agent._state_machine.set_state_sync(ConnectionState.RECONNECTING)

        # Create a long-running reconnect task
        async def long_reconnect(error: Exception) -> None:
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                pass

        agent._reconnect_task = asyncio.create_task(
            long_reconnect(ConnectionError("test"))
        )

        # Close should cancel it
        await agent.close()

        assert agent._reconnect_task is None

    @pytest.mark.asyncio
    async def test_close_while_not_connected_or_reconnecting(self) -> None:
        """Test close() is no-op when disconnected and not reconnecting."""
        agent = AthyrAgent("localhost:9090")

        # Should not raise
        await agent.close()

        assert agent._state_machine.state == ConnectionState.DISCONNECTED


class TestHeartbeatTriggersReconnect:
    """Tests for heartbeat failure triggering reconnection."""

    @pytest.mark.asyncio
    async def test_heartbeat_failure_triggers_reconnect_when_enabled(self) -> None:
        """Test heartbeat failure triggers reconnection when auto_reconnect=True."""
        agent = AthyrAgent(
            "localhost:9090",
            auto_reconnect=True,
            heartbeat_interval=0.01,
        )
        agent._state_machine.set_state_sync(ConnectionState.CONNECTED)
        agent._stub = MagicMock()
        agent._stub.Heartbeat = AsyncMock(side_effect=ConnectionError("heartbeat failed"))
        agent._agent_id = "test-agent"

        # Mock _trigger_reconnect to verify it's called
        triggered = False

        def mock_trigger(error: Exception) -> None:
            nonlocal triggered
            triggered = True

        with patch.object(agent, "_trigger_reconnect", side_effect=mock_trigger):
            # Run one iteration of heartbeat loop
            task = asyncio.create_task(agent._heartbeat_loop())
            await asyncio.sleep(0.05)  # Let heartbeat fail
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert triggered

    @pytest.mark.asyncio
    async def test_heartbeat_failure_no_reconnect_when_disabled(self) -> None:
        """Test heartbeat failure doesn't trigger reconnection when disabled."""
        agent = AthyrAgent(
            "localhost:9090",
            auto_reconnect=False,
            heartbeat_interval=0.01,
        )
        agent._state_machine.set_state_sync(ConnectionState.CONNECTED)
        agent._stub = MagicMock()
        agent._stub.Heartbeat = AsyncMock(side_effect=ConnectionError("heartbeat failed"))
        agent._agent_id = "test-agent"

        # Mock _trigger_reconnect to verify it's NOT called
        triggered = False

        def mock_trigger(error: Exception) -> None:
            nonlocal triggered
            triggered = True

        with patch.object(agent, "_trigger_reconnect", side_effect=mock_trigger):
            # Run one iteration of heartbeat loop
            task = asyncio.create_task(agent._heartbeat_loop())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert not triggered
