# Changelog

## [0.1.0] - 2026-01-18

Initial release of the Athyr Python SDK.

### Added

- **Agent Client** (`AthyrAgent`)
  - Async context manager for connection lifecycle
  - TLS support
  - Auto-reconnect with exponential backoff

- **LLM Completions**
  - Blocking completions (`complete`)
  - Streaming completions (`complete_stream`)
  - Model listing (`models`)

- **Messaging**
  - Publish/subscribe (`publish`, `subscribe`)
  - Queue groups for load balancing (`queue_subscribe`)
  - Request/reply RPC pattern (`request`)

- **Memory Sessions**
  - Session creation with system prompts
  - Session hints for persistent context
  - Rolling window memory profile

- **Tool Calling**
  - `@tool` decorator with JSON Schema generation
  - `ToolRegistry` for managing tools
  - `run_tool_loop` for agentic tool execution

- **Key-Value Storage**
  - Bucket-scoped operations
  - Get, put, delete, list keys

- **Orchestration Patterns**
  - `Pipeline`: Sequential agent chaining
  - `FanOut`: Parallel execution with aggregation
  - `HandoffRouter`: Dynamic routing to specialists
  - `GroupChat`: Multi-agent discussion
