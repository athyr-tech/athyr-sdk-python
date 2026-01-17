"""Orchestration patterns for multi-agent workflows.

This package provides helpers for common agent orchestration patterns
that simplify building multi-agent systems with Athyr.

Patterns:
    - Pipeline: Sequential agent chains (A → B → C)
    - FanOut: Parallel execution with aggregation
    - HandoffRouter: Dynamic routing based on triage
    - GroupChat: Multi-agent collaborative discussion
"""

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
]
