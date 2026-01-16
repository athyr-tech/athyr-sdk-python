#!/usr/bin/env python
"""Messaging example for Athyr SDK.

Demonstrates messaging patterns:
- Pub/Sub: Publishing and subscribing to subjects
- Queue Groups: Load-balanced message consumption
- Request/Reply: Synchronous RPC-style communication

Usage:
    # Start Athyr server first
    athyr serve

    # Run the example
    python examples/messaging.py
"""

import asyncio

from athyr import AthyrAgent, AgentCard


async def demo_pubsub() -> None:
    """Demonstrate basic publish/subscribe."""
    print("\n" + "=" * 50)
    print("Demo 1: Basic Pub/Sub")
    print("=" * 50)

    # Create two agents: publisher and subscriber
    pub_card = AgentCard(name="publisher", description="Publishes events")
    sub_card = AgentCard(name="subscriber", description="Receives events")

    async with AthyrAgent("localhost:9090", agent_card=sub_card) as subscriber:
        async with AthyrAgent("localhost:9090", agent_card=pub_card) as publisher:
            print(f"Publisher:  {publisher.agent_id()}")
            print(f"Subscriber: {subscriber.agent_id()}")

            # Subscribe to events
            print("\nSubscribing to 'demo.events'...")
            sub = await subscriber.subscribe("demo.events")

            await asyncio.sleep(0.1)  # Let subscription establish

            # Publish messages
            print("Publishing 3 messages...")
            for i in range(3):
                message = f"Event {i + 1}".encode()
                await publisher.publish("demo.events", message)
                print(f"  Published: Event {i + 1}")

            # Receive messages
            print("\nReceiving messages:")
            received = 0

            async def receive():
                nonlocal received
                async for msg in sub.messages():
                    print(f"  Received: {msg.data.decode()}")
                    received += 1
                    if received >= 3:
                        break

            try:
                await asyncio.wait_for(receive(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

            await sub.unsubscribe()
            print(f"\nTotal received: {received}/3 messages")


async def demo_queue_groups() -> None:
    """Demonstrate queue group load balancing."""
    print("\n" + "=" * 50)
    print("Demo 2: Queue Groups (Load Balancing)")
    print("=" * 50)

    # Create workers in the same queue group
    worker1_card = AgentCard(name="worker-1")
    worker2_card = AgentCard(name="worker-2")
    pub_card = AgentCard(name="task-publisher")

    async with AthyrAgent("localhost:9090", agent_card=worker1_card) as worker1:
        async with AthyrAgent("localhost:9090", agent_card=worker2_card) as worker2:
            async with AthyrAgent("localhost:9090", agent_card=pub_card) as publisher:
                print(f"Worker 1:  {worker1.agent_id()}")
                print(f"Worker 2:  {worker2.agent_id()}")
                print(f"Publisher: {publisher.agent_id()}")

                # Both workers subscribe with same queue group
                print("\nWorkers joining queue group 'workers'...")
                sub1 = await worker1.queue_subscribe("demo.tasks", "workers")
                sub2 = await worker2.queue_subscribe("demo.tasks", "workers")

                await asyncio.sleep(0.1)

                # Track who receives what
                worker1_msgs: list[str] = []
                worker2_msgs: list[str] = []

                # Publish tasks
                print("Publishing 6 tasks...")
                for i in range(6):
                    await publisher.publish("demo.tasks", f"Task {i + 1}".encode())

                # Receive in parallel
                async def collect(sub, msgs: list[str], name: str):
                    try:
                        async def recv():
                            async for msg in sub.messages():
                                msgs.append(msg.data.decode())
                                print(f"  {name} got: {msg.data.decode()}")

                        await asyncio.wait_for(recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        pass

                print("\nDistributing tasks:")
                await asyncio.gather(
                    collect(sub1, worker1_msgs, "Worker 1"),
                    collect(sub2, worker2_msgs, "Worker 2"),
                )

                await sub1.unsubscribe()
                await sub2.unsubscribe()

                print(f"\nDistribution:")
                print(f"  Worker 1: {len(worker1_msgs)} tasks")
                print(f"  Worker 2: {len(worker2_msgs)} tasks")
                print(f"  Total: {len(worker1_msgs) + len(worker2_msgs)}/6 tasks")


async def demo_request_reply() -> None:
    """Demonstrate request/reply RPC pattern."""
    print("\n" + "=" * 50)
    print("Demo 3: Request/Reply (RPC)")
    print("=" * 50)

    # Create service and client
    service_card = AgentCard(name="calculator-service")
    client_card = AgentCard(name="calculator-client")

    async with AthyrAgent("localhost:9090", agent_card=service_card) as service:
        async with AthyrAgent("localhost:9090", agent_card=client_card) as client:
            print(f"Service: {service.agent_id()}")
            print(f"Client:  {client.agent_id()}")

            # Service listens for requests
            print("\nService listening on 'demo.calculator'...")
            sub = await service.subscribe("demo.calculator")

            # Background: service handles requests
            async def handle_requests():
                async for msg in sub.messages():
                    # Parse request (simple format: "add:1:2")
                    parts = msg.data.decode().split(":")
                    if len(parts) == 3 and parts[0] == "add":
                        a, b = int(parts[1]), int(parts[2])
                        result = a + b
                        print(f"  Service: {a} + {b} = {result}")

                        # Reply if inbox provided
                        if msg.reply:
                            await service.publish(msg.reply, str(result).encode())

            service_task = asyncio.create_task(handle_requests())

            await asyncio.sleep(0.1)

            # Client makes RPC calls
            print("\nClient making requests:")
            for a, b in [(10, 5), (7, 3), (100, 200)]:
                request = f"add:{a}:{b}".encode()
                print(f"  Client: Requesting {a} + {b}...")

                try:
                    response = await client.request(
                        "demo.calculator",
                        request,
                        timeout=5.0,
                    )
                    print(f"  Client: Got result = {response.decode()}")
                except Exception as e:
                    print(f"  Client: Error - {e}")

            # Cleanup
            service_task.cancel()
            await sub.unsubscribe()


async def main() -> None:
    """Run all messaging demos."""
    print("Athyr SDK - Messaging Examples")
    print("=" * 50)

    await demo_pubsub()
    await demo_queue_groups()
    await demo_request_reply()

    print("\n" + "=" * 50)
    print("All demos complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())