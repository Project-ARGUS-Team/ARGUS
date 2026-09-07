"""Tests for the LLM gateway and deterministic mock provider."""

from argus.llm import MockLLMProvider
from argus.simulation import ActionType, Goal, Simulation, Vector2


def test_mock_returns_move_for_positional_goal() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]
    agent.goal = Goal(
        goal_id="goal-1",
        description="Reach the destination",
        target_position=Vector2(80.0, 90.0),
        importance=1.0,
    )

    provider = MockLLMProvider()
    result = provider.request_cognitive_update(
        simulation.build_agent_context(agent.agent_id)
    )

    assert result.action is not None
    assert result.action.action_type == ActionType.MOVE
    assert result.action.target_position == Vector2(80.0, 90.0)


def test_mock_returns_wait_without_target() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)

    provider = MockLLMProvider()
    result = provider.request_cognitive_update(
        simulation.build_agent_context("agent-0001")
    )

    assert result.action is not None
    assert result.action.action_type == ActionType.WAIT


def test_mock_is_deterministic() -> None:
    first = Simulation.create(agent_count=1, seed=42)
    second = Simulation.create(agent_count=1, seed=42)
    provider = MockLLMProvider()

    first_result = provider.request_cognitive_update(
        first.build_agent_context("agent-0001")
    )
    second_result = provider.request_cognitive_update(
        second.build_agent_context("agent-0001")
    )

    assert first_result == second_result


def test_mock_counts_calls() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    provider = MockLLMProvider()

    for _ in range(4):
        provider.request_cognitive_update(
            simulation.build_agent_context("agent-0001")
        )

    assert provider.call_count == 4
