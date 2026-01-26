"""Tests for Pipeline orchestration pattern."""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from athyr.orchestration.pipeline import (
    Pipeline,
    PipelineError,
    PipelineTrace,
    StepTrace,
)


class TestPipelineCreation:
    """Tests for Pipeline creation and configuration."""

    def test_create_pipeline(self) -> None:
        """Test creating a pipeline with name."""
        pipeline = Pipeline("test-pipeline")

        assert pipeline.name == "test-pipeline"
        assert pipeline.steps == []

    def test_add_step(self) -> None:
        """Test adding a step to the pipeline."""
        pipeline = Pipeline("test").step("first", "agent.first.invoke")

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0] == ("first", "agent.first.invoke")

    def test_add_multiple_steps(self) -> None:
        """Test adding multiple steps."""
        pipeline = (
            Pipeline("test")
            .step("first", "agent.first.invoke")
            .step("second", "agent.second.invoke")
            .step("third", "agent.third.invoke")
        )

        assert len(pipeline.steps) == 3
        assert pipeline.steps[0] == ("first", "agent.first.invoke")
        assert pipeline.steps[1] == ("second", "agent.second.invoke")
        assert pipeline.steps[2] == ("third", "agent.third.invoke")

    def test_fluent_api_returns_self(self) -> None:
        """Test that step() returns self for chaining."""
        pipeline = Pipeline("test")
        result = pipeline.step("first", "agent.first.invoke")

        assert result is pipeline


class TestPipelineExecution:
    """Tests for Pipeline execution."""

    @pytest.mark.asyncio
    async def test_execute_empty_pipeline(self) -> None:
        """Test executing an empty pipeline returns input."""
        pipeline = Pipeline("empty")
        mock_agent = AsyncMock()

        result = await pipeline.execute(mock_agent, b"input data")

        assert result == b"input data"
        mock_agent.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_single_step(self) -> None:
        """Test executing a single-step pipeline."""
        pipeline = Pipeline("single").step("process", "agent.process.invoke")

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=b"processed")

        result = await pipeline.execute(mock_agent, b"input")

        assert result == b"processed"
        mock_agent.request.assert_called_once_with("agent.process.invoke", b"input")

    @pytest.mark.asyncio
    async def test_execute_chains_output_to_input(self) -> None:
        """Test that each step receives previous step's output."""
        pipeline = (
            Pipeline("chain")
            .step("step1", "agent.step1.invoke")
            .step("step2", "agent.step2.invoke")
            .step("step3", "agent.step3.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[b"output1", b"output2", b"output3"]
        )

        result = await pipeline.execute(mock_agent, b"input")

        assert result == b"output3"
        assert mock_agent.request.call_count == 3

        calls = mock_agent.request.call_args_list
        assert calls[0].args == ("agent.step1.invoke", b"input")
        assert calls[1].args == ("agent.step2.invoke", b"output1")
        assert calls[2].args == ("agent.step3.invoke", b"output2")

    @pytest.mark.asyncio
    async def test_execute_raises_on_step_failure(self) -> None:
        """Test that execute raises PipelineError on step failure."""
        pipeline = (
            Pipeline("failing")
            .step("ok", "agent.ok.invoke")
            .step("fail", "agent.fail.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[b"ok", Exception("Step failed")]
        )

        with pytest.raises(PipelineError) as exc_info:
            await pipeline.execute(mock_agent, b"input")

        error = exc_info.value
        assert error.step == "fail"
        assert error.index == 1
        assert error.subject == "agent.fail.invoke"
        assert "Step failed" in str(error.cause)


class TestPipelineTracing:
    """Tests for Pipeline tracing."""

    @pytest.mark.asyncio
    async def test_trace_records_all_steps(self) -> None:
        """Test that tracing records all step executions."""
        pipeline = (
            Pipeline("traced")
            .step("step1", "agent.step1.invoke")
            .step("step2", "agent.step2.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=[b"out1", b"out2"])

        trace = await pipeline.execute_with_trace(mock_agent, b"input")

        assert trace.pipeline == "traced"
        assert len(trace.steps) == 2
        assert trace.error is None

        # Check step traces
        assert trace.steps[0].name == "step1"
        assert trace.steps[0].index == 0
        assert trace.steps[0].input == b"input"
        assert trace.steps[0].output == b"out1"
        assert trace.steps[0].error is None

        assert trace.steps[1].name == "step2"
        assert trace.steps[1].index == 1
        assert trace.steps[1].input == b"out1"
        assert trace.steps[1].output == b"out2"
        assert trace.steps[1].error is None

    @pytest.mark.asyncio
    async def test_trace_records_duration(self) -> None:
        """Test that tracing records execution duration."""
        pipeline = Pipeline("timed").step("slow", "agent.slow.invoke")

        mock_agent = AsyncMock()

        async def slow_request(subject: str, data: bytes) -> bytes:
            await asyncio.sleep(0.05)  # 50ms
            return b"done"

        mock_agent.request = slow_request

        trace = await pipeline.execute_with_trace(mock_agent, b"input")

        assert trace.duration >= timedelta(milliseconds=50)
        assert trace.steps[0].duration >= timedelta(milliseconds=50)

    @pytest.mark.asyncio
    async def test_trace_output_returns_final_output(self) -> None:
        """Test that trace.output() returns final step output."""
        pipeline = (
            Pipeline("test")
            .step("step1", "agent.step1.invoke")
            .step("step2", "agent.step2.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=[b"out1", b"final"])

        trace = await pipeline.execute_with_trace(mock_agent, b"input")

        assert trace.output() == b"final"

    @pytest.mark.asyncio
    async def test_trace_output_returns_none_on_error(self) -> None:
        """Test that trace.output() returns None when pipeline failed."""
        pipeline = Pipeline("test").step("fail", "agent.fail.invoke")

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=Exception("Failed"))

        trace = await pipeline.execute_with_trace(mock_agent, b"input")

        assert trace.output() is None
        assert trace.error is not None

    @pytest.mark.asyncio
    async def test_trace_records_error_on_failure(self) -> None:
        """Test that trace records error details on failure."""
        pipeline = (
            Pipeline("failing")
            .step("ok", "agent.ok.invoke")
            .step("fail", "agent.fail.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[b"ok", Exception("Boom")]
        )

        trace = await pipeline.execute_with_trace(mock_agent, b"input")

        assert trace.error is not None
        assert trace.error.step == "fail"
        assert trace.error.index == 1
        assert len(trace.steps) == 2
        assert trace.steps[1].error is not None


class TestPipelineStepOptions:
    """Tests for Pipeline step options."""

    @pytest.mark.asyncio
    async def test_step_timeout(self) -> None:
        """Test that step timeout works."""
        pipeline = Pipeline("timeout").step(
            "slow", "agent.slow.invoke", timeout=0.05
        )

        mock_agent = AsyncMock()

        async def slow_request(subject: str, data: bytes) -> bytes:
            await asyncio.sleep(1.0)  # Too slow
            return b"done"

        mock_agent.request = slow_request

        trace = await pipeline.execute_with_trace(mock_agent, b"input")

        assert trace.error is not None
        assert isinstance(trace.steps[0].error, asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_step_transform(self) -> None:
        """Test that step transform is applied."""
        pipeline = Pipeline("transform").step(
            "process",
            "agent.process.invoke",
            transform=lambda x: x.upper(),
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=b"lowercase")

        result = await pipeline.execute(mock_agent, b"input")

        assert result == b"LOWERCASE"

    @pytest.mark.asyncio
    async def test_transform_error_is_captured(self) -> None:
        """Test that transform errors are captured in trace."""
        def bad_transform(data: bytes) -> bytes:
            raise ValueError("Transform failed")

        pipeline = Pipeline("bad-transform").step(
            "process",
            "agent.process.invoke",
            transform=bad_transform,
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=b"data")

        trace = await pipeline.execute_with_trace(mock_agent, b"input")

        assert trace.error is not None
        assert "transform failed" in str(trace.error.cause)


class TestPipelineError:
    """Tests for PipelineError."""

    def test_error_string(self) -> None:
        """Test PipelineError string representation."""
        error = PipelineError(
            step="process",
            index=2,
            subject="agent.process.invoke",
            cause=Exception("Network error"),
        )

        error_str = str(error)
        assert "process" in error_str
        assert "index 2" in error_str
        assert "agent.process.invoke" in error_str
        assert "Network error" in error_str

    def test_error_repr(self) -> None:
        """Test PipelineError repr."""
        error = PipelineError(
            step="process",
            index=2,
            subject="agent.process.invoke",
            cause=Exception("Network error"),
        )

        repr_str = repr(error)
        assert "PipelineError" in repr_str
        assert "process" in repr_str


class TestStepTrace:
    """Tests for StepTrace."""

    def test_step_trace_defaults(self) -> None:
        """Test StepTrace default values."""
        trace = StepTrace(
            index=0,
            name="test",
            subject="test.invoke",
            input=b"data",
        )

        assert trace.output == b""
        assert trace.error is None
        assert trace.duration == timedelta(0)


class TestPipelineTrace:
    """Tests for PipelineTrace."""

    def test_pipeline_trace_defaults(self) -> None:
        """Test PipelineTrace default values."""
        trace = PipelineTrace(pipeline="test")

        assert trace.steps == []
        assert trace.error is None
        assert trace.duration == timedelta(0)

    def test_output_empty_pipeline(self) -> None:
        """Test output() returns None for empty pipeline."""
        trace = PipelineTrace(pipeline="empty")

        assert trace.output() is None
