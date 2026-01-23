# Athyr Python SDK

Python SDK for [Athyr](https://athyr.tech) - a lightweight runtime for AI agents as distributed microservices.

## Installation

```bash
# From GitHub
pip install git+https://github.com/athyr-tech/athyr-sdk-python.git

# From PyPI (coming soon)
pip install athyr
```

## Quick Start

```python
import asyncio
from athyr import AthyrAgent, AgentCard
from athyr.models import CompletionRequest, Message

async def main():
    card = AgentCard(name="my-agent", description="Demo agent")

    async with AthyrAgent("localhost:9090", agent_card=card) as agent:
        # List available models
        models = await agent.models()
        print(f"Available: {[m.name for m in models]}")

        # Make a completion request
        response = await agent.complete(CompletionRequest(
            model="llama3",
            messages=[Message(role="user", content="Hello!")]
        ))
        print(response.content)

asyncio.run(main())
```

## Features

### LLM Completions

```python
# Blocking completion
response = await agent.complete(request)

# Streaming completion
async for chunk in agent.complete_stream(request):
    print(chunk.content, end="")
```

### Messaging (Pub/Sub)

```python
# Publish
await agent.publish("events.user.created", b'{"id": 123}')

# Subscribe
sub = await agent.subscribe("events.>")
async for msg in sub.messages():
    print(f"{msg.subject}: {msg.data}")

# Request/Reply (RPC)
response = await agent.request("services.echo", b"hello", timeout=5.0)
```

### Memory Sessions

```python
# Create session with context
session = await agent.create_session(system_prompt="You are helpful.")

# Use session in completions
response = await agent.complete(CompletionRequest(
    model="llama3",
    messages=[Message(role="user", content="Remember my name is Alice")],
    session_id=session.id,
))

# Add hints for persistent context
await agent.add_hint(session.id, "User prefers concise responses")
```

### Tool Calling

```python
from athyr.tools import tool, ToolRegistry

@tool(description="Get current weather")
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 72°F"

registry = ToolRegistry()
registry.register(get_weather)

# Tools are automatically called in completion loop
response = await run_tool_loop(agent, request, registry)
```

### Key-Value Storage

```python
bucket = agent.kv("user-settings")
await bucket.put("theme", b"dark")
entry = await bucket.get("theme")
print(entry.value)  # b"dark"
```

### Orchestration Patterns

```python
from athyr.orchestration import Pipeline, FanOut, HandoffRouter, GroupChat

# Pipeline: Sequential processing
pipeline = Pipeline("processor").add(step1).add(step2).add(step3)
result = await pipeline.execute(agent, input_data)

# FanOut: Parallel execution
fanout = FanOut("analyzer").add(agent1).add(agent2).add(agent3)
result = await fanout.execute(agent, data, all_must_succeed())

# HandoffRouter: Dynamic routing
router = HandoffRouter("support").triage("triage").route("billing", "billing-agent")
result = await router.handle(agent, request)

# GroupChat: Multi-agent discussion
chat = GroupChat("review").add(reviewer1).add(reviewer2).max_rounds(5)
trace = await chat.discuss(agent, topic)
```

### Auto-Reconnect

```python
agent = AthyrAgent(
    "localhost:9090",
    auto_reconnect=True,
    max_retries=5,
    base_backoff=1.0,
    on_state_change=lambda state, err: print(f"State: {state}"),
)
```

## Examples

See the [examples/](examples/) directory:

| Example | Description |
|---------|-------------|
| `quickstart.py` | Basic agent setup and completion |
| `streaming.py` | Streaming completions |
| `messaging.py` | Pub/sub and request/reply |
| `tool_calling.py` | Function tools with LLM |
| `chat_agent.py` | Memory sessions with tools |
| `resilient_agent.py` | Auto-reconnect handling |
| `pipeline.py` | Sequential orchestration |
| `fanout.py` | Parallel orchestration |
| `handoff.py` | Dynamic routing |
| `groupchat.py` | Multi-agent discussion |

### Running Examples

```bash
# Terminal 1: Start Athyr server
athyr serve

# Terminal 2: Run any example (ATHYR_MODEL required)
ATHYR_MODEL=llama2 python examples/quickstart.py
ATHYR_MODEL=liquid/lfm-2.5-1.2b-instruct:free python examples/streaming.py
```

For orchestration examples that need multiple agents:

```bash
# Terminal 2: Start agents
ATHYR_MODEL=llama2 python examples/agents.py pipeline

# Terminal 3: Run the orchestrator
python examples/pipeline.py
```

## Requirements

- Python 3.10+
- Running Athyr server

## License

MIT
