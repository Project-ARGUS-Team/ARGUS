"""Tests for transport, traffic, and event-aware routing."""

from argus.llm.demo import ScenarioLLMProvider
from argus.simulation import (
    Action,
    ActionType,
    Simulation,
    StateDelta,
    TransportMode,
    Vector2,
)
from argus.simulation.scenario import create_baseline_scenario
from argus.simulation.world import RoadSegment


def test_baseline_assigns_both_transport_modes() -> None:
    scenario = create_baseline_scenario(agent_count=10, seed=42)

    modes = {
        agent.transport_mode
        for agent in scenario.simulation.agents.values()
    }

    assert TransportMode.WALK in modes
    assert TransportMode.CAR in modes


def test_closed_road_changes_car_route() -> None:
    roads = (
        RoadSegment(
            "direct",
            "Direct Road",
            Vector2(0.0, 0.0),
            Vector2(10.0, 0.0),
            vehicle_allowed=False,
        ),
        RoadSegment(
            "detour",
            "Detour Road",
            Vector2(0.0, 0.0),
            Vector2(0.0, 10.0),
        ),
        RoadSegment(
            "detour-east",
            "Detour East",
            Vector2(0.0, 10.0),
            Vector2(10.0, 0.0),
        ),
    )
    provider = ScenarioLLMProvider(roads=roads)

    direct = provider._road_waypoint(
        Vector2(0.0, 0.0),
        Vector2(10.0, 0.0),
        "agent-0001",
    )
    rerouted = provider._road_waypoint(
        Vector2(0.0, 0.0),
        Vector2(10.0, 0.0),
        "agent-0001",
        ("direct",),
    )

    assert direct == Vector2(10.0, 0.0)
    assert rerouted == Vector2(0.0, 10.0)


def test_pedestrian_ignores_road_closure() -> None:
    scenario = create_baseline_scenario(agent_count=1, seed=42)
    agent = scenario.simulation.agents["agent-0001"]
    agent.transport_mode = TransportMode.WALK
    scenario.simulation.state.tick = 170

    closure = next(
        event
        for event in scenario.simulation.state.events.values()
        if event.event_type == "road_closure"
    )
    closure.start_tick = 0
    closure.end_tick = 720
    closure.affected_road_ids = ("road-north-west",)

    context = scenario.simulation.build_agent_context(agent.agent_id)

    assert "road-north-west" in context.closed_road_ids
    assert context.transport_mode == "walk"

    provider = ScenarioLLMProvider(roads=scenario.roads)
    delta = provider.request_cognitive_update(context)

    assert delta.action is not None
    assert delta.action.action_type == ActionType.MOVE


def test_car_speed_is_reduced_by_local_congestion() -> None:
    simulation = Simulation.create(agent_count=2, seed=42)
    first = simulation.agents["agent-0001"]
    second = simulation.agents["agent-0002"]

    first.transport_mode = TransportMode.CAR
    second.transport_mode = TransportMode.CAR
    second.position = first.position

    target = Vector2(first.position.x + 20.0, first.position.y)

    simulation.apply_state_delta(
        first.agent_id,
        StateDelta(
            action=Action(
                action_type=ActionType.MOVE,
                target_position=target,
            )
        ),
    )

    assert first.velocity.x < 6.0


def test_vehicle_routing_does_not_take_shortcut() -> None:
    roads = (
        RoadSegment(
            "direct",
            "Direct Road",
            Vector2(0.0, 0.0),
            Vector2(10.0, 0.0),
        ),
        RoadSegment(
            "detour",
            "Detour Road",
            Vector2(0.0, 0.0),
            Vector2(0.0, 10.0),
        ),
        RoadSegment(
            "detour-east",
            "Detour East",
            Vector2(0.0, 10.0),
            Vector2(10.0, 0.0),
        ),
    )
    provider = ScenarioLLMProvider(roads=roads)

    waypoint = provider._road_waypoint(
        Vector2(0.0, 0.0),
        Vector2(10.0, 0.0),
        "agent-0001",
        vehicle=True,
    )

    assert waypoint == Vector2(0.0, 10.0)


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
    agent.transport_mode = TransportMode.CAR
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

    road_names = {road.name for road in scenario.roads}

    assert "Market Road" not in road_names
    assert "road-central-west" not in {road.road_id for road in scenario.roads}


def test_plaza_to_market_uses_available_route_after_map_simplification() -> None:
    scenario = create_baseline_scenario(agent_count=1, seed=42)
    provider = ScenarioLLMProvider(roads=scenario.roads)

    waypoint = provider._road_waypoint(
        Vector2(88.0, 55.0),
        Vector2(30.0, 58.0),
        "agent-0001",
        vehicle=True,
    )

    # Market Road has deliberately been removed from the baseline. The first
    # valid graph waypoint is therefore Central Station, followed by the
    # eastern/northern connectors to Market.
    assert waypoint == Vector2(145.0, 55.0)


def test_vehicle_route_respects_vehicle_access() -> None:
    roads = (
        RoadSegment(
            "pedestrian-only",
            "Pedestrian Cut",
            Vector2(0.0, 0.0),
            Vector2(10.0, 0.0),
            vehicle_allowed=False,
        ),
        RoadSegment(
            "vehicle-route",
            "Vehicle Route",
            Vector2(0.0, 0.0),
            Vector2(0.0, 10.0),
        ),
        RoadSegment(
            "vehicle-route-east",
            "Vehicle Route East",
            Vector2(0.0, 10.0),
            Vector2(10.0, 0.0),
        ),
    )
    provider = ScenarioLLMProvider(roads=roads)

    waypoint = provider._road_waypoint(
        Vector2(0.0, 0.0),
        Vector2(10.0, 0.0),
        "agent-0001",
        vehicle=True,
    )

    assert waypoint == Vector2(0.0, 10.0)


def test_route_does_not_keep_wrong_previous_waypoint_at_junction() -> None:
    scenario = create_baseline_scenario(agent_count=1, seed=42)
    provider = ScenarioLLMProvider(roads=scenario.roads, arrival_radius=4.0)

    # Simulate an agent reaching Central Plaza while its previous action still
    # points toward Central Station. The destination is Market. The route
    # controller must choose Market Road rather than preserve the old branch.
    agent = scenario.simulation.agents["agent-0001"]
    agent.position = Vector2(88.0, 55.0)
    agent.travel_destination = Vector2(30.0, 58.0)
    agent.current_action = Action(
        action_type=ActionType.MOVE,
        target_position=Vector2(145.0, 55.0),
    )

    context = scenario.simulation.build_agent_context(agent.agent_id)
    delta = provider.request_cognitive_update(context)

    assert delta.action is not None
    assert delta.action.target_position == Vector2(25.0, 58.0)

def test_transport_is_mixed_within_each_occupation() -> None:
    scenario = create_baseline_scenario(agent_count=60, seed=42)

    by_occupation: dict[str, set[TransportMode]] = {}
    for agent in scenario.simulation.agents.values():
        assert agent.profile is not None
        by_occupation.setdefault(agent.profile.occupation, set()).add(
            agent.transport_mode
        )

    assert all(
        modes == {TransportMode.CAR, TransportMode.WALK}
        for modes in by_occupation.values()
    )


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

