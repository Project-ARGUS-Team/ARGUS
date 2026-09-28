"""Tests for the deterministic city-scale baseline scenario."""

from argus.simulation.scenario import create_baseline_scenario


def test_baseline_scenario_is_deterministic() -> None:
    first = create_baseline_scenario(agent_count=20, seed=42)
    second = create_baseline_scenario(agent_count=20, seed=42)

    assert first.landmarks == second.landmarks
    assert first.simulation.world == second.simulation.world

    for agent_id in first.simulation.agents:
        a = first.simulation.agents[agent_id]
        b = second.simulation.agents[agent_id]
        assert a.position == b.position
        assert a.goal == b.goal
        assert a.social_connections == b.social_connections

    assert first.simulation.state.events == second.simulation.state.events


def test_baseline_scenario_has_city_scale_world() -> None:
    scenario = create_baseline_scenario(agent_count=12)

    assert scenario.simulation.world.width == 180.0
    assert scenario.simulation.world.height == 120.0
    assert len(scenario.landmarks) == 12
    assert len(scenario.simulation.state.events) == 10
    assert all(
        agent.goal.target_position is not None
        for agent in scenario.simulation.agents.values()
    )


def test_baseline_scenario_events_are_reproducible_and_spread_out() -> None:
    scenario = create_baseline_scenario(agent_count=40, seed=42)
    starts = [
        event.start_tick
        for event in scenario.simulation.state.events.values()
    ]

    assert len(set(starts)) >= 6
    assert min(starts) >= 20
    assert max(starts) <= 520
    assert all(
        event.end_tick > event.start_tick
        for event in scenario.simulation.state.events.values()
    )


def test_agents_have_distinct_persistent_profiles_and_routines() -> None:
    scenario = create_baseline_scenario(agent_count=20, seed=42)
    profiles = [
        agent.profile
        for agent in scenario.simulation.agents.values()
    ]

    assert all(profile is not None for profile in profiles)
    assert len({profile.name for profile in profiles if profile is not None}) == 20
    assert len({profile.occupation for profile in profiles if profile is not None}) >= 4
    assert all(
        profile.routine
        for profile in profiles
        if profile is not None
    )
    assert all(
        len(profile.routine) >= 7
        for profile in profiles
        if profile is not None
    )


def test_agents_start_at_home_with_home_activity() -> None:
    scenario = create_baseline_scenario(agent_count=10, seed=42)

    for agent in scenario.simulation.agents.values():
        assert agent.profile is not None
        assert agent.position == agent.profile.home_position
        assert agent.current_activity.value == "home"


def test_baseline_scenario_has_connected_pedestrian_roads() -> None:
    scenario = create_baseline_scenario(agent_count=10, seed=42)

    assert len(scenario.roads) >= 10
    assert all(road.start != road.end for road in scenario.roads)
    assert any(road.name == "Central Boulevard" for road in scenario.roads)


def test_trip_is_not_abandoned_when_schedule_moves_ahead() -> None:
    from argus.llm.demo import ScenarioLLMProvider
    from argus.simulation import Action, ActionType

    scenario = create_baseline_scenario(agent_count=1, seed=42)
    agent = scenario.simulation.agents["agent-0001"]
    office = agent.profile.work_position
    assert office is not None

    agent.current_action = Action(
        action_type=ActionType.MOVE,
        target_position=office,
    )
    scenario.simulation.state.tick = 125

    provider = ScenarioLLMProvider(roads=scenario.roads)
    delta = provider.request_cognitive_update(
        scenario.simulation.build_agent_context(agent.agent_id)
    )

    assert delta.action is not None
    assert delta.action.action_type == ActionType.MOVE
    assert delta.action.target_position is not None
