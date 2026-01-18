"""Session operations for Athyr agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc

from athyr._grpc import athyr_pb2
from athyr.exceptions import from_grpc_error
from athyr.models import Session, SessionProfile

if TYPE_CHECKING:
    from athyr._internal.connection import ConnectionManager


class SessionOperations:
    """Handles memory session operations.

    Provides methods for:
    - Creating sessions
    - Retrieving sessions
    - Deleting sessions
    - Adding hints to sessions

    Example:
        sessions = SessionOperations(connection)
        session = await sessions.create(system_prompt="You are helpful.")
        await sessions.add_hint(session.id, "User prefers concise responses")
    """

    def __init__(self, connection: ConnectionManager) -> None:
        """Initialize session operations.

        Args:
            connection: The connection manager for stub access.
        """
        self._connection = connection

    async def create(
        self,
        profile: SessionProfile | None = None,
        system_prompt: str = "",
    ) -> Session:
        """Create a new memory session.

        Creates a conversation session for maintaining context across
        multiple completions. Sessions can be configured with different
        memory profiles (rolling window, summarization, etc.).

        Args:
            profile: Session memory profile configuration. Defaults to
                rolling window with 4000 max tokens.
            system_prompt: Agent personality/instructions for this session.

        Returns:
            Session object with the assigned session ID.

        Raises:
            ConnectionError: If not connected to Athyr.
            AthyrError: If session creation fails.

        Example:
            session = await sessions.create(
                profile=SessionProfile(max_tokens=8000),
                system_prompt="You are a helpful assistant.",
            )
        """
        stub = self._connection.get_stub()

        try:
            request = athyr_pb2.CreateSessionRequest(
                agent_id=self._connection.agent_id,
                profile=profile.to_proto() if profile else None,
                system_prompt=system_prompt,
            )
            response = await stub.CreateSession(request)
            return Session.from_proto(response)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    async def get(self, session_id: str) -> Session:
        """Retrieve an existing session.

        Gets the full session data including messages, summary,
        hints, and profile configuration.

        Args:
            session_id: The session ID returned from create.

        Returns:
            Session object with current state.

        Raises:
            ConnectionError: If not connected to Athyr.
            NotFoundError: If session doesn't exist.
            AthyrError: If retrieval fails.

        Example:
            session = await sessions.get("session-123")
            print(f"Session has {len(session.messages)} messages")
        """
        stub = self._connection.get_stub()

        try:
            request = athyr_pb2.GetSessionRequest(
                agent_id=self._connection.agent_id,
                session_id=session_id,
            )
            response = await stub.GetSession(request)
            return Session.from_proto(response)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    async def delete(self, session_id: str) -> None:
        """Delete a session.

        Permanently removes the session and all its messages.
        This operation cannot be undone.

        Args:
            session_id: The session ID to delete.

        Raises:
            ConnectionError: If not connected to Athyr.
            NotFoundError: If session doesn't exist.
            AthyrError: If deletion fails.

        Example:
            await sessions.delete("session-123")
        """
        stub = self._connection.get_stub()

        try:
            request = athyr_pb2.DeleteSessionRequest(
                agent_id=self._connection.agent_id,
                session_id=session_id,
            )
            await stub.DeleteSession(request)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    async def add_hint(self, session_id: str, hint: str) -> None:
        """Add a hint to a session.

        Hints are persistent notes that influence LLM responses for
        this session. Use hints to provide context that should be
        remembered across conversations.

        Args:
            session_id: The session ID to add the hint to.
            hint: The hint text to add.

        Raises:
            ConnectionError: If not connected to Athyr.
            NotFoundError: If session doesn't exist.
            AthyrError: If the operation fails.

        Example:
            await sessions.add_hint(session.id, "User prefers concise responses")
        """
        stub = self._connection.get_stub()

        try:
            request = athyr_pb2.AddHintRequest(
                agent_id=self._connection.agent_id,
                session_id=session_id,
                hint=hint,
            )
            await stub.AddHint(request)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e
