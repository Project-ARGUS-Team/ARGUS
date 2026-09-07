"""Tests for cognitive context and state application."""

from argus.simulation import (
    Action,
    ActionType,
    Simulation,
    SimulationState,
    StateDelta,
    Vector2,
)


def test_context_is_a_snapshot_of_agent_state() -> None:
    simulation = Simulation.create(agent_count=2, seed=42)

    context = simulation.build_agent_context("agent-0001")

    assert context.agent_id == "agent-0001"
    assert context.simulation_tick == 0
    assert context.position == simulation.agents["agent-0001"].position
    assert len(context.nearby_agents) == 1


def test_state_delta_move_is_translated_into_velocity() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]

    start = agent.position
    target = Vector2(start.x + 3.0, start.y + 4.0)

    simulation.apply_state_delta(
        agent.agent_id,
        StateDelta(
            action=Action(
                action_type=ActionType.MOVE,
                target_position=target,
            )
        ),
    )

    assert agent.velocity == Vector2(0.6, 0.8)

    simulation.tick()

    assert agent.position == Vector2(start.x + 0.6, start.y + 0.8)


def test_wait_clears_velocity() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]
    agent.velocity = Vector2(2.0, 3.0)

    simulation.apply_state_delta(
        agent.agent_id,
        StateDelta(action=Action(action_type=ActionType.WAIT)),
    )

    assert agent.velocity == Vector2(0.0, 0.0)
