"""Tests for the pedestrian-only baseline town."""

from argus.llm.demo import ScenarioLLMProvider
from argus.simulation import Action, ActionType, TransportMode, Vector2
from argus.simulation.scenario import create_baseline_scenario
from argus.simulation.world import RoadSegment


def test_baseline_is_pedestrian_only() -> None:
    scenario = create_baseline_scenario()

    assert len(scenario.simulation.agents) == 30
    assert all(
        agent.transport_mode == TransportMode.WALK
        for agent in scenario.simulation.agents.values()
    )


def test_baseline_has_no_random_events() -> None:
    scenario = create_baseline_scenario()

    assert scenario.simulation.state.events == {}


def test_park_market_street_connects_the_two_landmarks() -> None:
    scenario = create_baseline_scenario(agent_count=1, seed=42)
    road = next(
        road for road in scenario.roads if road.road_id == "road-park-market"
    )

    assert road.name == "Park Market Street"
    assert {road.start, road.end} == {
        Vector2(45.0, 94.0),
        Vector2(30.0, 58.0),
    }
    assert road.pedestrian_allowed


def test_park_to_market_uses_new_street() -> None:
    scenario = create_baseline_scenario(agent_count=1, seed=42)
    provider = ScenarioLLMProvider(roads=scenario.roads)

    waypoint = provider._road_waypoint(
        Vector2(45.0, 94.0),
        Vector2(30.0, 58.0),
        "agent-0001",
    )

    assert waypoint == Vector2(30.0, 58.0)


def test_pedestrian_routing_ignores_vehicle_access_flags() -> None:
    roads = (
        RoadSegment(
            "pedestrian-cut",
            "Pedestrian Cut",
            Vector2(0.0, 0.0),
            Vector2(10.0, 0.0),
            vehicle_allowed=False,
            pedestrian_allowed=True,
        ),
        RoadSegment(
            "second",
            "Second Road",
            Vector2(10.0, 0.0),
            Vector2(20.0, 0.0),
            vehicle_allowed=False,
            pedestrian_allowed=True,
        ),
    )
    provider = ScenarioLLMProvider(roads=roads)

    waypoint = provider._road_waypoint(
        Vector2(0.0, 0.0),
        Vector2(20.0, 0.0),
        "agent-0001",
    )

    assert waypoint == Vector2(10.0, 0.0)


def test_route_keeps_current_waypoint_until_reached() -> None:
    roads = (
        RoadSegment(
            "first",
            "First Road",
            Vector2(0.0, 0.0),
            Vector2(20.0, 0.0),
        ),
        RoadSegment(
            "second",
            "Second Road",
            Vector2(20.0, 0.0),
            Vector2(40.0, 0.0),
        ),
    )
    scenario = create_baseline_scenario(agent_count=1, seed=42)
    agent = scenario.simulation.agents["agent-0001"]
    agent.transport_mode = TransportMode.WALK
    agent.position = Vector2(1.0, 0.0)
    agent.travel_destination = Vector2(40.0, 0.0)
    agent.current_action = Action(
        action_type=ActionType.MOVE,
        target_position=Vector2(20.0, 0.0),
    )

    provider = ScenarioLLMProvider(roads=roads, arrival_radius=2.0)
    context = scenario.simulation.build_agent_context(agent.agent_id)
    delta = provider.request_cognitive_update(context)

    assert delta.action is not None
    assert delta.action.target_position == Vector2(20.0, 0.0)


def test_market_road_is_not_part_of_baseline_map() -> None:
    scenario = create_baseline_scenario(agent_count=1, seed=42)

    road_ids = {road.road_id for road in scenario.roads}
    road_names = {road.name for road in scenario.roads}

    assert "Market Road" not in road_names
    assert "road-central-west" not in road_ids
    assert "road-park-market" in road_ids


def test_routines_have_small_deterministic_time_variation() -> None:
    first = create_baseline_scenario(agent_count=10, seed=42)
    second = create_baseline_scenario(agent_count=10, seed=42)

    first_routines = {
        agent.agent_id: agent.profile.routine
        for agent in first.simulation.agents.values()
        if agent.profile is not None
    }
    second_routines = {
        agent.agent_id: agent.profile.routine
        for agent in second.simulation.agents.values()
        if agent.profile is not None
    }

    assert first_routines == second_routines
    departure_times = {
        routine[1].start_tick
        for routine in first_routines.values()
        if len(routine) > 1
    }
    assert len(departure_times) > 1
    for routine in first_routines.values():
        assert all(0 <= entry.start_tick < entry.end_tick <= 720 for entry in routine)
