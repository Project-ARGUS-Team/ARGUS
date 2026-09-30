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


def test_baseline_scenario_has_town_scale_population_and_no_events() -> None:
    scenario = create_baseline_scenario()

    assert len(scenario.simulation.agents) == 30
    assert scenario.simulation.world.width == 180.0
    assert scenario.simulation.world.height == 120.0
    assert len(scenario.landmarks) == 12
    assert scenario.simulation.state.events == {}
    assert all(
        agent.goal.target_position is not None
        for agent in scenario.simulation.agents.values()
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

    assert len(scenario.roads) >= 11
    assert all(road.start != road.end for road in scenario.roads)
    assert all(road.pedestrian_allowed for road in scenario.roads)
    assert any(road.name == "Central Boulevard" for road in scenario.roads)
    assert any(road.name == "Park Market Street" for road in scenario.roads)


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


def test_interaction_scenario_is_a_separate_two_agent_village():
    from argus.simulation.scenario import create_interaction_scenario
    from argus.simulation.agent import TransportMode

    scenario = create_interaction_scenario()

    assert len(scenario.simulation.agents) == 2
    assert scenario.simulation.world.width == 100.0
    assert scenario.simulation.world.height == 80.0
    assert len(scenario.landmarks) == 7
    assert scenario.simulation.state.events == {}
    assert all(agent.transport_mode == TransportMode.WALK for agent in scenario.simulation.agents.values())
    assert {landmark.name for landmark in scenario.landmarks} == {
        "Arun's House",
        "Maya's House",
        "Town Hall",
        "Schoolhouse",
        "Village Café",
        "Village Market",
        "Village Park",
    }


def test_interaction_scenario_agents_keep_usual_routines():
    from argus.simulation.scenario import create_interaction_scenario

    scenario = create_interaction_scenario()
    agents = list(scenario.simulation.agents.values())

    assert agents[0].profile is not None
    assert agents[1].profile is not None
    assert agents[0].profile.occupation == "Office worker"
    assert agents[1].profile.occupation == "Student"
    assert all(agent.profile.routine for agent in agents)
    assert all(agent.profile.routine[0].activity.value == "home" for agent in agents)
    assert all(len(agent.profile.routine) >= 7 for agent in agents)
    assert agents[0].profile.work_position.x == 70.0
    assert agents[1].profile.work_position.x == 68.0


def test_interaction_scenario_agents_know_each_other():
    from argus.simulation.scenario import create_interaction_scenario

    scenario = create_interaction_scenario()
    first, second = scenario.simulation.agents.values()

    assert second.agent_id in first.social_connections
    assert first.agent_id in second.social_connections
    assert first.relationships[second.agent_id] == 0.50
    assert second.relationships[first.agent_id] == 0.50
