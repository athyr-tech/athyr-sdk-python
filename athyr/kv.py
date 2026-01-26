"""Key-Value storage operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import grpc

from athyr._grpc import athyr_pb2
from athyr.exceptions import from_grpc_error

if TYPE_CHECKING:
    from athyr._grpc import athyr_pb2_grpc


@dataclass
class KVEntry:
    """Key-value entry returned from get operations.

    Attributes:
        value: The stored value as bytes.
        revision: Version number for optimistic concurrency control.
    """

    value: bytes
    revision: int


class KVBucket(Protocol):
    """Protocol defining the KV bucket interface."""

    async def get(self, key: str) -> KVEntry:
        """Get a value by key."""
        ...

    async def put(self, key: str, value: bytes) -> int:
        """Put a value, returns revision number."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a key."""
        ...

    async def keys(self, prefix: str = "") -> list[str]:
        """List keys with optional prefix filter."""
        ...


class Bucket:
    """Bucket-scoped wrapper for KV operations.

    Provides a cleaner API by scoping all operations to a specific bucket,
    eliminating the need to pass the bucket name to every call.

    Example:
        bucket = agent.kv("user-preferences")
        await bucket.put("theme", b"dark")
        entry = await bucket.get("theme")
        print(entry.value)  # b"dark"
    """

    def __init__(
        self,
        bucket: str,
        agent_id: str,
        stub: athyr_pb2_grpc.AthyrStub,
        connected_check: callable,
    ) -> None:
        """Initialize the bucket wrapper.

        Args:
            bucket: The bucket name.
            agent_id: The agent's ID for API calls.
            stub: The gRPC stub for making calls.
            connected_check: Callable that raises if not connected.
        """
        self._bucket = bucket
        self._agent_id = agent_id
        self._stub = stub
        self._connected_check = connected_check

    @property
    def name(self) -> str:
        """Get the bucket name."""
        return self._bucket

    async def get(self, key: str) -> KVEntry:
        """Get a value by key.

        Args:
            key: The key to retrieve.

        Returns:
            KVEntry with value and revision.

        Raises:
            ConnectionError: If not connected to Athyr.
            NotFoundError: If key doesn't exist.
            AthyrError: If the operation fails.

        Example:
            entry = await bucket.get("user:123:name")
            print(f"Value: {entry.value.decode()}, revision: {entry.revision}")
        """
        self._connected_check()

        try:
            response = await self._stub.KVGet(
                athyr_pb2.KVGetRequest(
                    agent_id=self._agent_id,
                    bucket=self._bucket,
                    key=key,
                )
            )
            return KVEntry(value=response.value, revision=response.revision)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    async def put(self, key: str, value: bytes) -> int:
        """Store a value.

        Args:
            key: The key to store under.
            value: The value as bytes.

        Returns:
            The new revision number.

        Raises:
            ConnectionError: If not connected to Athyr.
            AthyrError: If the operation fails.

        Example:
            revision = await bucket.put("config", b'{"debug": true}')
            print(f"Stored at revision {revision}")
        """
        self._connected_check()

        try:
            response = await self._stub.KVPut(
                athyr_pb2.KVPutRequest(
                    agent_id=self._agent_id,
                    bucket=self._bucket,
                    key=key,
                    value=value,
                )
            )
            return response.revision
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    async def delete(self, key: str) -> None:
        """Delete a key.

        Args:
            key: The key to delete.

        Raises:
            ConnectionError: If not connected to Athyr.
            NotFoundError: If key doesn't exist.
            AthyrError: If the operation fails.

        Example:
            await bucket.delete("old-config")
        """
        self._connected_check()

        try:
            await self._stub.KVDelete(
                athyr_pb2.KVDeleteRequest(
                    agent_id=self._agent_id,
                    bucket=self._bucket,
                    key=key,
                )
            )
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    async def keys(self, prefix: str = "") -> list[str]:
        """List keys in the bucket.

        Args:
            prefix: Optional prefix to filter keys. Empty string returns all keys.

        Returns:
            List of matching key names.

        Raises:
            ConnectionError: If not connected to Athyr.
            AthyrError: If the operation fails.

        Example:
            # Get all keys
            all_keys = await bucket.keys()

            # Get keys starting with "user:"
            user_keys = await bucket.keys("user:")
        """
        self._connected_check()

        try:
            response = await self._stub.KVList(
                athyr_pb2.KVListRequest(
                    agent_id=self._agent_id,
                    bucket=self._bucket,
                    prefix=prefix,
                )
            )
            return list(response.keys)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e
