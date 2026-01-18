"""Tests for Agent client."""

import pytest

from athyr.client import Agent, AthyrAgent
from athyr.exceptions import ConnectionError
from athyr.models import AgentCard


class TestAgentProtocol:
    """Tests for Agent Protocol."""

    def test_athyr_agent_implements_protocol(self) -> None:
        """Test that AthyrAgent implements Agent protocol."""
        agent = AthyrAgent("localhost:9090")
        assert isinstance(agent, Agent)


class TestAthyrAgent:
    """Tests for AthyrAgent."""

    def test_create_minimal(self) -> None:
        """Test creating agent with minimal config."""
        agent = AthyrAgent("localhost:9090")
        # Access internal attributes through _connection
        assert agent._connection._address == "localhost:9090"
        assert agent._connection._agent_card.name == "python-agent"
        assert agent._connection._tls is False
        assert agent._connection._connect_timeout == 10.0
        assert agent._connection._heartbeat_interval == 30.0

    def test_create_with_card(self) -> None:
        """Test creating agent with custom card."""
        card = AgentCard(name="my-agent", description="Test")
        agent = AthyrAgent("localhost:9090", agent_card=card)
        assert agent._connection._agent_card.name == "my-agent"
        assert agent._connection._agent_card.description == "Test"

    def test_create_with_options(self) -> None:
        """Test creating agent with custom options."""
        agent = AthyrAgent(
            "localhost:9090",
            tls=True,
            connect_timeout=5.0,
            heartbeat_interval=60.0,
        )
        assert agent._connection._tls is True
        assert agent._connection._connect_timeout == 5.0
        assert agent._connection._heartbeat_interval == 60.0

    def test_not_connected_initially(self) -> None:
        """Test agent is not connected initially."""
        agent = AthyrAgent("localhost:9090")
        assert agent.connected() is False
        assert agent.agent_id() == ""

    def test_ensure_connected_raises(self) -> None:
        """Test _connection.get_stub raises when not connected."""
        agent = AthyrAgent("localhost:9090")
        with pytest.raises(ConnectionError):
            agent._connection.get_stub()


class TestAthyrAgentConnection:
    """Tests for agent connection (integration-style, would need mock)."""

    @pytest.mark.asyncio
    async def test_connect_failure_invalid_address(self) -> None:
        """Test connection fails with invalid address."""
        from athyr.exceptions import AthyrError

        agent = AthyrAgent(
            "invalid-host:9999",
            connect_timeout=1.0,  # Short timeout
        )
        # Could be UnavailableError or ConnectionError depending on failure mode
        with pytest.raises(AthyrError):
            await agent.connect()

    @pytest.mark.asyncio
    async def test_context_manager_without_server(self) -> None:
        """Test context manager handles connection failure."""
        from athyr.exceptions import AthyrError

        # Could be UnavailableError or ConnectionError depending on failure mode
        with pytest.raises(AthyrError):
            async with AthyrAgent("invalid:9999", connect_timeout=1.0) as agent:
                pass  # Should not reach here