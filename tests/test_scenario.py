"""Tests for the deterministic baseline demonstration scenario."""

from argus.simulation.scenario import create_baseline_scenario


def test_baseline_scenario_is_deterministic() -> None:
    first = create_baseline_scenario(agent_count=10, seed=42)
    second = create_baseline_scenario(agent_count=10, seed=42)

    for agent_id in first.simulation.agents:
        a = first.simulation.agents[agent_id]
        b = second.simulation.agents[agent_id]
        assert a.position == b.position
        assert a.goal == b.goal
        assert a.social_connections == b.social_connections

    assert first.simulation.state.events == second.simulation.state.events


def test_baseline_scenario_has_meaningful_world_state() -> None:
    scenario = create_baseline_scenario(agent_count=12)

    assert len(scenario.landmarks) == 4
    assert len(scenario.simulation.state.events) == 3
    assert all(
        agent.goal.target_position is not None
        for agent in scenario.simulation.agents.values()
    )
    assert all(
        agent.social_connections
        for agent in scenario.simulation.agents.values()
    )
