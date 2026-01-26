"""Server for hosting Athyr services.

This module provides a Server class that manages multiple services,
handles their lifecycle, and provides graceful shutdown.

Example:
    from athyr.server import Server
    from athyr.service import Service

    async def greet(ctx, req: GreetRequest) -> GreetResponse:
        return GreetResponse(message=f"Hello, {req.name}!")

    server = Server("localhost:9090")
    server.add(Service(
        subject="greet.invoke",
        handler=greet,
        request_type=GreetRequest,
        response_type=GreetResponse,
    ))

    await server.run()
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from athyr.client import AthyrAgent
from athyr.models import AgentCard
from athyr.protocol import Agent
from athyr.service import (
    DefaultServiceContext,
    Middleware,
    RawHandler,
    Service,
    ServiceError,
)
from athyr.subscription import Subscription

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

Req = TypeVar("Req")
Resp = TypeVar("Resp")


@dataclass
class Server:
    """Manages multiple services and their lifecycle.

    The Server connects to Athyr, subscribes all registered services,
    and handles graceful shutdown on signals or context cancellation.

    Example:
        server = Server(
            address="localhost:9090",
            agent_name="my-service",
        )
        server.add(my_service)
        await server.run()
    """

    address: str
    """Athyr server address (host:port)."""

    agent_name: str = "service-agent"
    """Name for agent registration."""

    agent_description: str = ""
    """Description for agent registration."""

    version: str = "1.0.0"
    """Agent version for registration."""

    tls: bool = False
    """Enable TLS for connection."""

    services: list[Service[Any, Any]] = field(default_factory=list)
    """Registered services."""

    middleware: list[Middleware] = field(default_factory=list)
    """Global middleware applied to all services."""

    _agent: Agent | None = field(default=None, repr=False)
    """Internal agent connection."""

    _subscriptions: list[Subscription] = field(default_factory=list, repr=False)
    """Active subscriptions."""

    _handler_tasks: list[asyncio.Task[None]] = field(default_factory=list, repr=False)
    """Background tasks handling messages."""

    _shutdown_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    """Event signaling shutdown."""

    def add(self, service: Service[Any, Any]) -> Server:
        """Add a service to the server.

        Args:
            service: The service to add.

        Returns:
            Self for method chaining.
        """
        self.services.append(service)
        return self

    def handle(
        self,
        subject: str,
        handler: Callable[[Any, Req], Any],
        request_type: type[Req],
        response_type: type[Resp],
        *,
        name: str = "",
        queue_group: str = "",
    ) -> Server:
        """Add a typed handler for a subject.

        Convenience method that creates and adds a Service.

        Args:
            subject: The subject to handle requests on.
            handler: Typed handler function.
            request_type: Type for deserializing requests.
            response_type: Type for serializing responses.
            name: Display name for the service.
            queue_group: Queue group for load balancing.

        Returns:
            Self for method chaining.
        """
        service: Service[Req, Resp] = Service(
            subject=subject,
            handler=handler,
            request_type=request_type,
            response_type=response_type,
            name=name,
            queue_group=queue_group,
        )
        self.services.append(service)
        return self

    def handle_raw(
        self,
        subject: str,
        handler: RawHandler,
        *,
        name: str = "",
        queue_group: str = "",
    ) -> Server:
        """Add a raw handler for a subject.

        Args:
            subject: The subject to handle requests on.
            handler: Raw handler function.
            name: Display name for the service.
            queue_group: Queue group for load balancing.

        Returns:
            Self for method chaining.
        """
        service = Service.raw(
            subject=subject,
            handler=handler,
            name=name,
            queue_group=queue_group,
        )
        self.services.append(service)
        return self

    def use(self, *middleware: Middleware) -> Server:
        """Add global middleware applied to all services.

        Args:
            middleware: Middleware functions to add.

        Returns:
            Self for method chaining.
        """
        self.middleware.extend(middleware)
        return self

    async def run(self) -> None:
        """Start the server and block until shutdown.

        Connects to Athyr, subscribes all services, and handles
        requests until a shutdown signal is received or an error occurs.

        Raises:
            ValueError: If no services are registered.
            ConnectionError: If connection to Athyr fails.
        """
        if not self.services:
            raise ValueError("No services registered")

        # Reset shutdown event
        self._shutdown_event.clear()

        # Create and connect agent
        self._agent = AthyrAgent(
            self.address,
            agent_card=AgentCard(
                name=self.agent_name,
                description=self.agent_description,
                version=self.version,
            ),
            tls=self.tls,
        )

        try:
            await self._agent.connect()
            logger.info(
                f"Server connected as {self._agent.agent_id()} "
                f"with {len(self.services)} service(s)"
            )

            # Subscribe all services
            for service in self.services:
                await self._subscribe_service(service)

            # Set up signal handlers
            self._setup_signal_handlers()

            # Wait for shutdown
            logger.info("Server running, press Ctrl+C to stop")
            await self._shutdown_event.wait()

        finally:
            await self._cleanup()

    async def shutdown(self) -> None:
        """Request graceful shutdown.

        This method can be called from another task to stop the server.
        """
        logger.info("Shutdown requested")
        self._shutdown_event.set()

    async def _subscribe_service(self, service: Service[Any, Any]) -> None:
        """Subscribe to a service's subject and start handling messages."""
        assert self._agent is not None

        handler = service.build_handler(self.middleware)

        # Subscribe with or without queue group
        if service.queue_group:
            sub = await self._agent.queue_subscribe(
                service.subject, service.queue_group
            )
        else:
            sub = await self._agent.subscribe(service.subject)

        self._subscriptions.append(sub)

        # Start background task to handle messages
        task = asyncio.create_task(
            self._handle_messages(sub, handler, service.name),
            name=f"handler:{service.name}",
        )
        self._handler_tasks.append(task)

        logger.info(f"Subscribed service '{service.name}' on {service.subject}")

    async def _handle_messages(
        self,
        sub: Subscription,
        handler: RawHandler,
        service_name: str,
    ) -> None:
        """Handle incoming messages for a subscription."""
        assert self._agent is not None

        try:
            async for msg in sub.messages():
                if self._shutdown_event.is_set():
                    break

                # Create service context
                ctx = DefaultServiceContext(
                    _agent=self._agent,
                    _subject=msg.subject,
                    _reply_subject=msg.reply,
                )

                # Call handler and send response
                try:
                    response = await handler(ctx, msg.data)

                    # Send response if there's a reply subject
                    if msg.reply:
                        await self._agent.publish(msg.reply, response)

                except ServiceError as e:
                    # Send structured error response
                    if msg.reply:
                        await self._agent.publish(msg.reply, e.to_json())
                    logger.warning(f"Service error in {service_name}: {e}")

                except Exception as e:
                    # Send generic error response
                    if msg.reply:
                        error_response = ServiceError(
                            code="internal",
                            message=str(e),
                        ).to_json()
                        await self._agent.publish(msg.reply, error_response)
                    logger.exception(f"Unhandled error in {service_name}: {e}")

        except asyncio.CancelledError:
            logger.debug(f"Handler task cancelled for {service_name}")
        except Exception as e:
            logger.exception(f"Message loop error for {service_name}: {e}")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        def signal_handler(signum: int) -> None:
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating shutdown")
            self._shutdown_event.set()

        # Handle SIGINT (Ctrl+C) and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler, sig)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

    async def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        logger.info("Cleaning up server resources")

        # Cancel handler tasks
        for task in self._handler_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._handler_tasks:
            await asyncio.gather(*self._handler_tasks, return_exceptions=True)
        self._handler_tasks.clear()

        # Unsubscribe all subscriptions
        for sub in self._subscriptions:
            try:
                await sub.unsubscribe()
            except Exception as e:
                logger.debug(f"Error unsubscribing: {e}")
        self._subscriptions.clear()

        # Close agent connection
        if self._agent is not None:
            try:
                await self._agent.close()
            except Exception as e:
                logger.debug(f"Error closing agent: {e}")
            self._agent = None

        logger.info("Server stopped")


# Convenience functions


async def run(
    address: str,
    subject: str,
    handler: Callable[[Any, Req], Any],
    request_type: type[Req],
    response_type: type[Resp],
    *,
    agent_name: str = "",
    queue_group: str = "",
) -> None:
    """Run a single typed service.

    Convenience function for simple single-service deployments.

    Args:
        address: Athyr server address.
        subject: Subject to handle requests on.
        handler: Typed handler function.
        request_type: Type for deserializing requests.
        response_type: Type for serializing responses.
        agent_name: Name for agent registration (defaults to subject).
        queue_group: Queue group for load balancing.

    Example:
        await run(
            "localhost:9090",
            "greet.invoke",
            greet_handler,
            GreetRequest,
            GreetResponse,
        )
    """
    server = Server(
        address=address,
        agent_name=agent_name or subject,
    )
    server.handle(
        subject,
        handler,
        request_type,
        response_type,
        queue_group=queue_group,
    )
    await server.run()


async def run_raw(
    address: str,
    subject: str,
    handler: RawHandler,
    *,
    agent_name: str = "",
    queue_group: str = "",
) -> None:
    """Run a single raw service.

    Convenience function for simple single-service deployments.

    Args:
        address: Athyr server address.
        subject: Subject to handle requests on.
        handler: Raw handler function.
        agent_name: Name for agent registration (defaults to subject).
        queue_group: Queue group for load balancing.
    """
    server = Server(
        address=address,
        agent_name=agent_name or subject,
    )
    server.handle_raw(subject, handler, queue_group=queue_group)
    await server.run()
