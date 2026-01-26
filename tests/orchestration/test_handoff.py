"""Tests for HandoffRouter orchestration pattern."""

import json
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from athyr.orchestration.handoff import (
    HandoffDecision,
    HandoffError,
    HandoffRouter,
    HandoffStep,
    HandoffTrace,
)


class TestHandoffDecision:
    """Tests for HandoffDecision dataclass."""

    def test_from_json_route_only(self) -> None:
        """Test parsing a simple route decision."""
        data = json.dumps({"route": "billing"}).encode()
        decision = HandoffDecision.from_json(data)

        assert decision.route == "billing"
        assert decision.context == b""
        assert decision.handled is False
        assert decision.response == b""

    def test_from_json_with_context(self) -> None:
        """Test parsing a decision with context."""
        data = json.dumps({
            "route": "technical",
            "context": {"issue_type": "network", "priority": "high"},
        }).encode()
        decision = HandoffDecision.from_json(data)

        assert decision.route == "technical"
        context = json.loads(decision.context)
        assert context["issue_type"] == "network"
        assert context["priority"] == "high"

    def test_from_json_handled_directly(self) -> None:
        """Test parsing a decision where triage handled directly."""
        data = json.dumps({
            "handled": True,
            "response": {"message": "Your balance is $100"},
        }).encode()
        decision = HandoffDecision.from_json(data)

        assert decision.handled is True
        response = json.loads(decision.response)
        assert response["message"] == "Your balance is $100"

    def test_to_json_roundtrip(self) -> None:
        """Test that to_json and from_json are inverses."""
        original = HandoffDecision(
            route="billing",
            context=json.dumps({"key": "value"}).encode(),
            handled=False,
        )

        serialized = original.to_json()
        restored = HandoffDecision.from_json(serialized)

        assert restored.route == original.route
        assert restored.context == original.context
        assert restored.handled == original.handled

    def test_from_json_invalid_raises(self) -> None:
        """Test that invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            HandoffDecision.from_json(b"not json")


class TestHandoffRouterCreation:
    """Tests for HandoffRouter creation and configuration."""

    def test_create_router(self) -> None:
        """Test creating a router with name."""
        router = HandoffRouter("support-triage")

        assert router.name == "support-triage"

    def test_set_triage(self) -> None:
        """Test setting the triage agent."""
        router = HandoffRouter("test").triage("agent.triage.invoke")

        assert router._triage_subject == "agent.triage.invoke"

    def test_add_routes(self) -> None:
        """Test adding routes."""
        router = (
            HandoffRouter("test")
            .route("billing", "agent.billing.invoke")
            .route("technical", "agent.tech.invoke")
        )

        assert router._routes["billing"] == "agent.billing.invoke"
        assert router._routes["technical"] == "agent.tech.invoke"

    def test_set_max_handoffs(self) -> None:
        """Test setting max handoffs."""
        router = HandoffRouter("test").max_handoffs(5)

        assert router._max_handoffs == 5

    def test_set_fallback(self) -> None:
        """Test setting fallback agent."""
        router = HandoffRouter("test").fallback("agent.human.invoke")

        assert router._fallback_subject == "agent.human.invoke"

    def test_fluent_api_returns_self(self) -> None:
        """Test that all methods return self for chaining."""
        router = HandoffRouter("test")

        assert router.triage("triage") is router
        assert router.route("billing", "billing") is router
        assert router.max_handoffs(5) is router
        assert router.fallback("fallback") is router


class TestHandoffRouterExecution:
    """Tests for HandoffRouter execution."""

    @pytest.mark.asyncio
    async def test_handle_no_triage_configured(self) -> None:
        """Test that handle fails without triage configured."""
        router = HandoffRouter("test")
        mock_agent = AsyncMock()

        with pytest.raises(HandoffError) as exc_info:
            await router.handle(mock_agent, b"request")

        assert "no triage agent configured" in str(exc_info.value.cause)

    @pytest.mark.asyncio
    async def test_handle_triage_handles_directly(self) -> None:
        """Test when triage handles the request directly."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
        )

        triage_response = json.dumps({
            "handled": True,
            "response": {"answer": "Done"},
        }).encode()

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=triage_response)

        result = await router.handle(mock_agent, b"request")

        response = json.loads(result)
        assert response["answer"] == "Done"
        mock_agent.request.assert_called_once_with("agent.triage.invoke", b"request")

    @pytest.mark.asyncio
    async def test_handle_routes_to_specialist(self) -> None:
        """Test routing from triage to specialist."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
        )

        triage_response = json.dumps({"route": "billing"}).encode()
        billing_response = b"Your bill is $50"

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[triage_response, billing_response]
        )

        result = await router.handle(mock_agent, b"request")

        assert result == billing_response
        assert mock_agent.request.call_count == 2

    @pytest.mark.asyncio
    async def test_handle_passes_context_to_specialist(self) -> None:
        """Test that context is passed to specialist agent."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
        )

        context = {"customer_id": "123", "issue": "refund"}
        triage_response = json.dumps({
            "route": "billing",
            "context": context,
        }).encode()

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[triage_response, b"Refund processed"]
        )

        await router.handle(mock_agent, b"request")

        # Check that context was passed to billing agent
        calls = mock_agent.request.call_args_list
        billing_call_data = calls[1].args[1]
        passed_context = json.loads(billing_call_data)
        assert passed_context["customer_id"] == "123"

    @pytest.mark.asyncio
    async def test_handle_unknown_route_uses_fallback(self) -> None:
        """Test that unknown route uses fallback agent."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
            .fallback("agent.human.invoke")
        )

        triage_response = json.dumps({"route": "unknown_dept"}).encode()
        fallback_response = b"Transferred to human agent"

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[triage_response, fallback_response]
        )

        result = await router.handle(mock_agent, b"request")

        assert result == fallback_response
        calls = mock_agent.request.call_args_list
        assert calls[1].args[0] == "agent.human.invoke"

    @pytest.mark.asyncio
    async def test_handle_unknown_route_no_fallback_fails(self) -> None:
        """Test that unknown route without fallback raises error."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
        )

        triage_response = json.dumps({"route": "unknown_dept"}).encode()

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=triage_response)

        with pytest.raises(HandoffError) as exc_info:
            await router.handle(mock_agent, b"request")

        assert "unknown route" in str(exc_info.value.cause)
        assert "unknown_dept" in str(exc_info.value.cause)

    @pytest.mark.asyncio
    async def test_handle_non_json_response_treated_as_final(self) -> None:
        """Test that non-JSON response is treated as final output."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
        )

        # Triage returns non-JSON (plain text)
        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=b"Simple answer")

        result = await router.handle(mock_agent, b"request")

        assert result == b"Simple answer"

    @pytest.mark.asyncio
    async def test_handle_specialist_can_hand_off(self) -> None:
        """Test that specialist can hand off to another specialist."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
            .route("refunds", "agent.refunds.invoke")
        )

        triage_response = json.dumps({"route": "billing"}).encode()
        billing_response = json.dumps({"route": "refunds"}).encode()
        refunds_response = b"Refund approved"

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[triage_response, billing_response, refunds_response]
        )

        result = await router.handle(mock_agent, b"request")

        assert result == refunds_response
        assert mock_agent.request.call_count == 3

    @pytest.mark.asyncio
    async def test_handle_max_handoffs_exceeded(self) -> None:
        """Test that max handoffs limit is enforced."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
            .route("a", "agent.a.invoke")
            .route("b", "agent.b.invoke")
            .max_handoffs(3)
        )

        # Create infinite loop between agents
        def cycle_response(subject: str, data: bytes) -> bytes:
            if "triage" in subject or "a" in subject:
                return json.dumps({"route": "b"}).encode()
            return json.dumps({"route": "a"}).encode()

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=cycle_response)

        with pytest.raises(HandoffError) as exc_info:
            await router.handle(mock_agent, b"request")

        assert "max handoffs" in str(exc_info.value.cause)

    @pytest.mark.asyncio
    async def test_handle_triage_error(self) -> None:
        """Test error handling when triage fails."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=Exception("Triage down"))

        with pytest.raises(HandoffError) as exc_info:
            await router.handle(mock_agent, b"request")

        assert exc_info.value.agent == "triage"
        assert "Triage down" in str(exc_info.value.cause)

    @pytest.mark.asyncio
    async def test_handle_specialist_error(self) -> None:
        """Test error handling when specialist fails."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
        )

        triage_response = json.dumps({"route": "billing"}).encode()

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[triage_response, Exception("Billing unavailable")]
        )

        with pytest.raises(HandoffError) as exc_info:
            await router.handle(mock_agent, b"request")

        assert exc_info.value.agent == "billing"
        assert "Billing unavailable" in str(exc_info.value.cause)


class TestHandoffRouterTracing:
    """Tests for HandoffRouter tracing."""

    @pytest.mark.asyncio
    async def test_trace_records_path(self) -> None:
        """Test that tracing records the routing path."""
        router = (
            HandoffRouter("support")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
        )

        triage_response = json.dumps({"route": "billing"}).encode()

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[triage_response, b"Bill details"]
        )

        trace = await router.handle_with_trace(mock_agent, b"request")

        assert trace.router == "support"
        assert len(trace.path) == 2
        assert trace.path[0].agent == "triage"
        assert trace.path[0].route == "billing"
        assert trace.path[1].agent == "billing"
        assert trace.path[1].route == "final"

    @pytest.mark.asyncio
    async def test_trace_route_names(self) -> None:
        """Test trace.route_names() method."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
            .route("billing", "agent.billing.invoke")
            .route("refunds", "agent.refunds.invoke")
        )

        triage_response = json.dumps({"route": "billing"}).encode()
        billing_response = json.dumps({"route": "refunds"}).encode()

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(
            side_effect=[triage_response, billing_response, b"Done"]
        )

        trace = await router.handle_with_trace(mock_agent, b"request")

        assert trace.route_names() == ["triage", "billing", "refunds"]

    @pytest.mark.asyncio
    async def test_trace_records_duration(self) -> None:
        """Test that tracing records duration."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=b"Direct answer")

        trace = await router.handle_with_trace(mock_agent, b"request")

        assert trace.duration >= timedelta(0)
        assert trace.path[0].duration >= timedelta(0)

    @pytest.mark.asyncio
    async def test_trace_records_input_output(self) -> None:
        """Test that trace records input and output."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=b"Response")

        trace = await router.handle_with_trace(mock_agent, b"Input data")

        assert trace.input == b"Input data"
        assert trace.output == b"Response"

    @pytest.mark.asyncio
    async def test_trace_records_error(self) -> None:
        """Test that trace records errors."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
        )

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(side_effect=Exception("Failed"))

        trace = await router.handle_with_trace(mock_agent, b"request")

        assert trace.error is not None
        assert "Failed" in str(trace.error.cause)
        assert len(trace.path) == 1
        assert trace.path[0].error is not None

    @pytest.mark.asyncio
    async def test_trace_handled_directly(self) -> None:
        """Test trace when triage handles directly."""
        router = (
            HandoffRouter("test")
            .triage("agent.triage.invoke")
        )

        response = json.dumps({
            "handled": True,
            "response": {"msg": "Handled"},
        }).encode()

        mock_agent = AsyncMock()
        mock_agent.request = AsyncMock(return_value=response)

        trace = await router.handle_with_trace(mock_agent, b"request")

        assert trace.error is None
        assert len(trace.path) == 1
        output = json.loads(trace.output)
        assert output["msg"] == "Handled"


class TestHandoffStep:
    """Tests for HandoffStep dataclass."""

    def test_step_defaults(self) -> None:
        """Test HandoffStep default values."""
        step = HandoffStep(
            index=0,
            agent="triage",
            subject="agent.triage.invoke",
        )

        assert step.route == ""
        assert step.output == b""
        assert step.error is None


class TestHandoffTrace:
    """Tests for HandoffTrace dataclass."""

    def test_trace_defaults(self) -> None:
        """Test HandoffTrace default values."""
        trace = HandoffTrace(router="test")

        assert trace.path == []
        assert trace.input == b""
        assert trace.output == b""
        assert trace.error is None

    def test_route_names_empty(self) -> None:
        """Test route_names() on empty trace."""
        trace = HandoffTrace(router="test")

        assert trace.route_names() == []


class TestHandoffError:
    """Tests for HandoffError."""

    def test_error_string_with_agent_and_route(self) -> None:
        """Test error string with agent and route."""
        error = HandoffError(
            agent="billing",
            route="refunds",
            cause=Exception("Not found"),
        )

        error_str = str(error)
        assert "billing" in error_str
        assert "refunds" in error_str
        assert "Not found" in error_str

    def test_error_string_with_agent_only(self) -> None:
        """Test error string with agent only."""
        error = HandoffError(
            agent="triage",
            cause=Exception("Timeout"),
        )

        error_str = str(error)
        assert "triage" in error_str
        assert "Timeout" in error_str

    def test_error_string_with_cause_only(self) -> None:
        """Test error string with cause only."""
        error = HandoffError(
            cause=Exception("General failure"),
        )

        error_str = str(error)
        assert "General failure" in error_str

    def test_error_preserves_path(self) -> None:
        """Test that error preserves the path taken."""
        path = [
            HandoffStep(index=0, agent="triage", subject="triage"),
            HandoffStep(index=1, agent="billing", subject="billing"),
        ]
        error = HandoffError(
            agent="billing",
            cause=Exception("Failed"),
            path=path,
        )

        assert len(error.path) == 2
        assert error.path[0].agent == "triage"
        assert error.path[1].agent == "billing"
