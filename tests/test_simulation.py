"""Tests for the simulation core."""

import pytest

from argus.simulation import (
    Action,
    ActionType,
    AgentState,
    Goal,
    Simulation,
    SimulationState,
    Vector2,
    World,
)


def make_agent() -> AgentState:
    return AgentState(
        agent_id="agent-0001",
        position=Vector2(10.0, 20.0),
        velocity=Vector2(2.0, 3.0),
        goal=Goal(
            goal_id="goal-0001",
            description="Reach the target",
            target_position=Vector2(50.0, 50.0),
            importance=0.8,
        ),
    )


def test_simulation_advances_ticks_and_time() -> None:
    simulation = Simulation.create(agent_count=10, seed=42)
    simulation.run(20)
    assert simulation.current_tick == 20
    assert simulation.simulation_time == 20.0
    assert len(simulation.agents) == 10


def test_simulation_creation_is_deterministic() -> None:
    first = Simulation.create(agent_count=10, seed=42)
    second = Simulation.create(agent_count=10, seed=42)
    assert first.agents == second.agents


def test_movement_advances_by_velocity() -> None:
    agent = make_agent()
    agent.set_action(Action(action_type=ActionType.MOVE, target_position=Vector2(50.0, 50.0)))
    simulation = Simulation(
        state=SimulationState(agents={agent.agent_id: agent}),
        world=World(tick_duration=0.5),
    )
    simulation.tick()
    assert agent.position == Vector2(11.0, 21.5)


def test_movement_is_clamped_to_world_bounds() -> None:
    agent = make_agent()
    agent.position = Vector2(99.0, 99.0)
    agent.velocity = Vector2(5.0, 7.0)
    agent.set_action(Action(action_type=ActionType.MOVE))
    simulation = Simulation(
        state=SimulationState(agents={agent.agent_id: agent}),
        world=World(),
    )
    simulation.tick()
    assert agent.position == Vector2(100.0, 100.0)


def test_negative_tick_count_is_rejected() -> None:
    simulation = Simulation.create()
    with pytest.raises(ValueError):
        simulation.run(-1)


def test_negative_agent_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        Simulation.create(agent_count=-1)
