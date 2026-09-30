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


def test_central_plaza_to_market_uses_market_road() -> None:
    scenario = create_baseline_scenario(agent_count=1, seed=42)
    provider = ScenarioLLMProvider(roads=scenario.roads)

    waypoint = provider._road_waypoint(
        Vector2(88.0, 55.0),
        Vector2(30.0, 58.0),
        "agent-0001",
        vehicle=True,
    )

    assert waypoint == Vector2(25.0, 58.0)


def test_committed_route_advances_monotonically_from_plaza() -> None:
    scenario = create_baseline_scenario(agent_count=1, seed=42)
    provider = ScenarioLLMProvider(roads=scenario.roads, arrival_radius=4.0)

    first = provider._road_waypoint(
        Vector2(105.0, 34.0),
        Vector2(30.0, 58.0),
        "agent-0001",
        vehicle=True,
    )
    second = provider._road_waypoint(
        Vector2(88.0, 55.0),
        Vector2(30.0, 58.0),
        "agent-0001",
        vehicle=True,
    )

    assert first == Vector2(88.0, 55.0)
    assert second == Vector2(25.0, 58.0)


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
