"""LLM operations for Athyr agents."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import grpc

from athyr.exceptions import from_grpc_error
from athyr.models import CompletionRequest, CompletionResponse, Model, StreamChunk

if TYPE_CHECKING:
    from athyr._internal.connection import ConnectionManager


class LLMOperations:
    """Handles LLM completion operations.

    Provides methods for:
    - Blocking completions
    - Streaming completions
    - Listing available models

    Example:
        llm = LLMOperations(connection)
        response = await llm.complete(request)
    """

    def __init__(self, connection: ConnectionManager) -> None:
        """Initialize LLM operations.

        Args:
            connection: The connection manager for stub access.
        """
        self._connection = connection

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Perform a blocking LLM completion.

        Args:
            request: The completion request with model, messages, and options.

        Returns:
            CompletionResponse with the generated content and metadata.

        Raises:
            ConnectionError: If not connected to Athyr.
            InvalidRequestError: If request parameters are invalid.
            UnavailableError: If the LLM service is unavailable.
        """
        stub = self._connection.get_stub()

        try:
            proto_request = request.to_proto(agent_id=self._connection.agent_id)
            proto_response = await stub.Complete(proto_request)
            return CompletionResponse.from_proto(proto_response)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Perform a streaming LLM completion.

        Args:
            request: The completion request with model, messages, and options.

        Yields:
            StreamChunk objects with incremental content. The final chunk
            has done=True and includes usage statistics.

        Raises:
            ConnectionError: If not connected to Athyr.
            InvalidRequestError: If request parameters are invalid.
            UnavailableError: If the LLM service is unavailable.
            StreamError: If the stream encounters an error.
        """
        stub = self._connection.get_stub()

        try:
            proto_request = request.to_proto(agent_id=self._connection.agent_id)
            stream = stub.CompleteStream(proto_request)
            async for chunk in stream:
                yield StreamChunk.from_proto(chunk)
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e

    async def models(self) -> list[Model]:
        """Get available LLM models.

        Returns:
            List of available Model objects.

        Raises:
            ConnectionError: If not connected to Athyr.
            UnavailableError: If the service is unavailable.
        """
        stub = self._connection.get_stub()

        try:
            from athyr._grpc import athyr_pb2
            response = await stub.ListModels(athyr_pb2.ListModelsRequest())
            return [Model.from_proto(m) for m in response.models]
        except grpc.RpcError as e:
            raise from_grpc_error(e) from e
