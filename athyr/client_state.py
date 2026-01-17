"""Connection state management for Athyr agents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ConnectionState(Enum):
    """Agent connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"

    def __str__(self) -> str:
        return self.value


# Type alias for state change callbacks
# Callback receives: (new_state, error_if_any)
StateChangeCallback = Callable[[ConnectionState, Exception | None], None]


class ConnectionStateMachine:
    """Thread-safe connection state machine with callbacks.

    Example:
        def on_state_change(state: ConnectionState, error: Exception | None) -> None:
            print(f"State changed to {state}")
            if error:
                print(f"Error: {error}")

        state_machine = ConnectionStateMachine(on_state_change=on_state_change)
        state_machine.set_state(ConnectionState.CONNECTING)
        state_machine.set_state(ConnectionState.CONNECTED)
    """

    def __init__(
        self,
        *,
        on_state_change: StateChangeCallback | None = None,
    ) -> None:
        """Initialize the state machine.

        Args:
            on_state_change: Optional callback invoked on state transitions.
        """
        self._state = ConnectionState.DISCONNECTED
        self._on_state_change = on_state_change
        self._lock = asyncio.Lock()

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Whether the agent is currently connected."""
        return self._state == ConnectionState.CONNECTED

    async def set_state(
        self,
        new_state: ConnectionState,
        error: Exception | None = None,
    ) -> None:
        """Transition to a new state.

        Args:
            new_state: The new connection state.
            error: Optional error associated with the transition.
        """
        async with self._lock:
            old_state = self._state
            if old_state == new_state:
                return

            self._state = new_state

        # Invoke callback outside the lock to avoid deadlocks
        if self._on_state_change is not None:
            try:
                self._on_state_change(new_state, error)
            except Exception:
                # Don't let callback errors break state transitions
                pass

    def set_state_sync(
        self,
        new_state: ConnectionState,
        error: Exception | None = None,
    ) -> None:
        """Synchronous state transition (for use in sync contexts).

        Args:
            new_state: The new connection state.
            error: Optional error associated with the transition.
        """
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state

        # Invoke callback
        if self._on_state_change is not None:
            try:
                self._on_state_change(new_state, error)
            except Exception:
                pass

    def set_callback(self, callback: StateChangeCallback | None) -> None:
        """Set or update the state change callback.

        Args:
            callback: The new callback, or None to remove.
        """
        self._on_state_change = callback
