"""Orchestration patterns for multi-agent workflows.

This package provides helpers for common agent orchestration patterns
that simplify building multi-agent systems with Athyr.

Patterns:
    - Pipeline: Sequential agent chains (A → B → C)
    - FanOut: Parallel execution with aggregation
    - HandoffRouter: Dynamic routing based on triage
    - GroupChat: Multi-agent collaborative discussion
"""

from athyr.orchestration.fanout import (
    AgentResult,
    AgentTrace,
    FanOut,
    FanOutError,
    FanOutTrace,
    all_must_succeed,
    collect_all,
    first_success,
    require_quorum,
)
from athyr.orchestration.groupchat import (
    ChatError,
    ChatRequest,
    ChatResponse,
    ChatTrace,
    ConsensusFunc,
    GroupChat,
    Manager,
    Message,
    func_manager,
    round_robin,
)
from athyr.orchestration.handoff import (
    HandoffDecision,
    HandoffError,
    HandoffRouter,
    HandoffStep,
    HandoffTrace,
)
from athyr.orchestration.pipeline import (
    Pipeline,
    PipelineError,
    PipelineTrace,
    StepTrace,
)

__all__ = [
    # Pipeline
    "Pipeline",
    "PipelineTrace",
    "StepTrace",
    "PipelineError",
    # FanOut
    "FanOut",
    "FanOutTrace",
    "FanOutError",
    "AgentResult",
    "AgentTrace",
    "all_must_succeed",
    "first_success",
    "collect_all",
    "require_quorum",
    # HandoffRouter
    "HandoffRouter",
    "HandoffDecision",
    "HandoffTrace",
    "HandoffStep",
    "HandoffError",
    # GroupChat
    "GroupChat",
    "ChatTrace",
    "ChatRequest",
    "ChatResponse",
    "ChatError",
    "Message",
    "Manager",
    "ConsensusFunc",
    "round_robin",
    "func_manager",
]
