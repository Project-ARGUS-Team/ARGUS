"""Tests for explicit cognitive updates."""

from argus.llm import MockLLMProvider
from argus.simulation import ActionType, Goal, Simulation, Vector2


def test_cognitive_update_calls_gateway_and_applies_delta() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]
    agent.goal = Goal(
        goal_id="goal-1",
        description="Reach destination",
        target_position=Vector2(80.0, 90.0),
        importance=1.0,
    )
    gateway = MockLLMProvider()

    result = simulation.request_cognitive_update(agent.agent_id, gateway)

    assert gateway.call_count == 1
    assert result.action is not None
    assert agent.current_action == result.action
    assert agent.current_action.action_type == ActionType.MOVE


def test_cognitive_update_does_not_advance_simulation() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    gateway = MockLLMProvider()

    simulation.request_cognitive_update("agent-0001", gateway)

    assert simulation.current_tick == 0
    assert simulation.simulation_time == 0.0


def test_tick_does_not_invoke_gateway() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    gateway = MockLLMProvider()

    simulation.tick()

    assert gateway.call_count == 0


def test_cognitive_update_then_tick_moves_agent() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]
    agent.goal = Goal(
        goal_id="goal-1",
        description="Reach destination",
        target_position=Vector2(80.0, 90.0),
        importance=1.0,
    )
    gateway = MockLLMProvider()
    start = agent.position

    simulation.request_cognitive_update(agent.agent_id, gateway)
    simulation.tick()

    assert agent.position != start
    assert gateway.call_count == 1


def test_cognitive_update_is_scoped_to_one_agent() -> None:
    simulation = Simulation.create(agent_count=2, seed=42)
    first = simulation.agents["agent-0001"]
    second = simulation.agents["agent-0002"]
    first.goal = Goal(
        goal_id="goal-1",
        description="Reach destination",
        target_position=Vector2(80.0, 90.0),
        importance=1.0,
    )
    gateway = MockLLMProvider()

    simulation.request_cognitive_update(first.agent_id, gateway)

    assert first.current_action is not None
    assert second.current_action is None
    assert gateway.call_count == 1
