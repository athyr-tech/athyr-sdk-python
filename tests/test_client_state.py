"""Tests for connection state management."""

import pytest

from athyr.client_state import ConnectionState, ConnectionStateMachine


class TestConnectionState:
    """Tests for ConnectionState enum."""

    def test_state_values(self) -> None:
        """Test state enum has expected values."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"

    def test_state_str(self) -> None:
        """Test state string representation."""
        assert str(ConnectionState.DISCONNECTED) == "disconnected"
        assert str(ConnectionState.CONNECTED) == "connected"


class TestConnectionStateMachine:
    """Tests for ConnectionStateMachine."""

    def test_initial_state_is_disconnected(self) -> None:
        """Test state machine starts disconnected."""
        sm = ConnectionStateMachine()
        assert sm.state == ConnectionState.DISCONNECTED
        assert not sm.is_connected

    def test_is_connected_only_when_connected(self) -> None:
        """Test is_connected property."""
        sm = ConnectionStateMachine()

        sm.set_state_sync(ConnectionState.CONNECTING)
        assert not sm.is_connected

        sm.set_state_sync(ConnectionState.CONNECTED)
        assert sm.is_connected

        sm.set_state_sync(ConnectionState.RECONNECTING)
        assert not sm.is_connected

        sm.set_state_sync(ConnectionState.DISCONNECTED)
        assert not sm.is_connected

    def test_set_state_sync(self) -> None:
        """Test synchronous state transitions."""
        sm = ConnectionStateMachine()

        sm.set_state_sync(ConnectionState.CONNECTING)
        assert sm.state == ConnectionState.CONNECTING

        sm.set_state_sync(ConnectionState.CONNECTED)
        assert sm.state == ConnectionState.CONNECTED

    def test_set_state_sync_same_state_is_noop(self) -> None:
        """Test setting same state doesn't trigger callback."""
        callback_count = 0

        def callback(state: ConnectionState, error: Exception | None) -> None:
            nonlocal callback_count
            callback_count += 1

        sm = ConnectionStateMachine(on_state_change=callback)
        sm.set_state_sync(ConnectionState.CONNECTING)
        assert callback_count == 1

        # Same state again
        sm.set_state_sync(ConnectionState.CONNECTING)
        assert callback_count == 1  # No additional callback

    def test_callback_invoked_on_state_change(self) -> None:
        """Test callback is invoked on state transitions."""
        states_received: list[ConnectionState] = []
        errors_received: list[Exception | None] = []

        def callback(state: ConnectionState, error: Exception | None) -> None:
            states_received.append(state)
            errors_received.append(error)

        sm = ConnectionStateMachine(on_state_change=callback)

        sm.set_state_sync(ConnectionState.CONNECTING)
        sm.set_state_sync(ConnectionState.CONNECTED)

        assert states_received == [ConnectionState.CONNECTING, ConnectionState.CONNECTED]
        assert errors_received == [None, None]

    def test_callback_receives_error(self) -> None:
        """Test callback receives error when provided."""
        received_error: Exception | None = None

        def callback(state: ConnectionState, error: Exception | None) -> None:
            nonlocal received_error
            received_error = error

        sm = ConnectionStateMachine(on_state_change=callback)
        test_error = ValueError("test error")

        # First connect, then disconnect with error
        sm.set_state_sync(ConnectionState.CONNECTED)
        sm.set_state_sync(ConnectionState.DISCONNECTED, error=test_error)
        assert received_error is test_error

    def test_callback_error_does_not_break_state_transition(self) -> None:
        """Test callback errors don't prevent state changes."""

        def bad_callback(state: ConnectionState, error: Exception | None) -> None:
            raise RuntimeError("Callback failed")

        sm = ConnectionStateMachine(on_state_change=bad_callback)

        # Should not raise, state should still change
        sm.set_state_sync(ConnectionState.CONNECTING)
        assert sm.state == ConnectionState.CONNECTING

    def test_set_callback(self) -> None:
        """Test setting callback after construction."""
        sm = ConnectionStateMachine()
        states_received: list[ConnectionState] = []

        def callback(state: ConnectionState, error: Exception | None) -> None:
            states_received.append(state)

        sm.set_callback(callback)
        sm.set_state_sync(ConnectionState.CONNECTING)

        assert states_received == [ConnectionState.CONNECTING]

    def test_remove_callback(self) -> None:
        """Test removing callback."""
        callback_count = 0

        def callback(state: ConnectionState, error: Exception | None) -> None:
            nonlocal callback_count
            callback_count += 1

        sm = ConnectionStateMachine(on_state_change=callback)
        sm.set_state_sync(ConnectionState.CONNECTING)
        assert callback_count == 1

        sm.set_callback(None)
        sm.set_state_sync(ConnectionState.CONNECTED)
        assert callback_count == 1  # No additional callback

    @pytest.mark.asyncio
    async def test_set_state_async(self) -> None:
        """Test async state transitions."""
        sm = ConnectionStateMachine()

        await sm.set_state(ConnectionState.CONNECTING)
        assert sm.state == ConnectionState.CONNECTING

        await sm.set_state(ConnectionState.CONNECTED)
        assert sm.state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_async_callback_invoked(self) -> None:
        """Test callback is invoked on async state transitions."""
        states_received: list[ConnectionState] = []

        def callback(state: ConnectionState, error: Exception | None) -> None:
            states_received.append(state)

        sm = ConnectionStateMachine(on_state_change=callback)

        await sm.set_state(ConnectionState.CONNECTING)
        await sm.set_state(ConnectionState.CONNECTED)

        assert states_received == [ConnectionState.CONNECTING, ConnectionState.CONNECTED]


class TestAgentConnectionState:
    """Tests for AthyrAgent connection state integration."""

    def test_agent_initial_state(self) -> None:
        """Test agent starts in disconnected state."""
        from athyr.client import AthyrAgent

        agent = AthyrAgent("localhost:9090")
        assert agent.state == ConnectionState.DISCONNECTED
        assert not agent.connected()

    def test_agent_state_callback(self) -> None:
        """Test agent state callback is invoked."""
        from athyr.client import AthyrAgent

        states: list[ConnectionState] = []

        def on_change(state: ConnectionState, error: Exception | None) -> None:
            states.append(state)

        agent = AthyrAgent("localhost:9090", on_state_change=on_change)

        # Manually trigger state change for testing
        agent._state_machine.set_state_sync(ConnectionState.CONNECTING)
        agent._state_machine.set_state_sync(ConnectionState.CONNECTED)

        assert states == [ConnectionState.CONNECTING, ConnectionState.CONNECTED]
