"""Deterministic city-scale demonstration scenarios for the ARGUS baseline."""

from __future__ import annotations

from dataclasses import dataclass
import random

from argus.simulation.agent import Goal, Vector2
from argus.simulation.events import WorldEvent
from argus.simulation.simulation import Simulation
from argus.simulation.world import World


@dataclass(frozen=True, slots=True)
class Landmark:
    """A named location used by a demonstration scenario."""

    landmark_id: str
    name: str
    position: Vector2
    radius: float = 6.0


@dataclass(frozen=True, slots=True)
class BaselineScenario:
    """A city-scale world containing locations, routines and events."""

    simulation: Simulation
    landmarks: tuple[Landmark, ...]


def create_baseline_scenario(
    agent_count: int = 60,
    seed: int = 42,
) -> BaselineScenario:
    """Create a deterministic but varied small-city baseline scenario."""
    world = World(width=180.0, height=120.0, tick_duration=1.0)
    simulation = Simulation.create(
        agent_count=agent_count,
        seed=seed,
        world=world,
    )

    landmarks = (
        Landmark("homes_north", "North Homes", Vector2(25.0, 18.0), 9.0),
        Landmark("school", "School", Vector2(62.0, 16.0), 7.0),
        Landmark("hospital", "Hospital", Vector2(145.0, 18.0), 8.0),
        Landmark("office", "Office District", Vector2(108.0, 34.0), 9.0),
        Landmark("market", "Market", Vector2(30.0, 58.0), 8.0),
        Landmark("plaza", "Central Plaza", Vector2(88.0, 55.0), 9.0),
        Landmark("station", "Central Station", Vector2(145.0, 55.0), 8.0),
        Landmark("park", "Riverside Park", Vector2(45.0, 94.0), 10.0),
        Landmark("cafe", "Café Row", Vector2(82.0, 88.0), 7.0),
        Landmark("entertainment", "Entertainment", Vector2(128.0, 92.0), 9.0),
        Landmark("south_homes", "South Homes", Vector2(165.0, 100.0), 9.0),
        Landmark("bus_depot", "Bus Depot", Vector2(158.0, 76.0), 7.0),
    )

    agents = list(simulation.agents.values())

    for index, agent in enumerate(agents):
        landmark = landmarks[index % len(landmarks)]
        agent.goal = Goal(
            goal_id=f"goal-{index + 1:04d}",
            description=f"Visit {landmark.name}",
            target_position=landmark.position,
            importance=0.35 + (index % 4) * 0.2,
        )

        if len(agents) > 1:
            agent.social_connections.add(
                agents[(index + 1) % len(agents)].agent_id
            )
        if len(agents) > 2:
            agent.social_connections.add(
                agents[(index + 2) % len(agents)].agent_id
            )

    # Generate a reproducible event schedule from the scenario seed. The
    # event times are random-looking, but a seed makes experiments repeatable.
    rng = random.Random(seed + 10_000)
    event_types = (
        ("market_rush", "Market rush"),
        ("public_gathering", "Public gathering"),
        ("bus_delay", "Bus delay"),
        ("minor_accident", "Minor accident"),
        ("community_fair", "Community fair"),
        ("power_outage", "Power outage"),
        ("sports_event", "Sports event"),
        ("medical_alert", "Medical alert"),
        ("road_closure", "Road closure"),
        ("school_event", "School event"),
    )

    events: list[WorldEvent] = []
    for index, (event_type, _description) in enumerate(event_types):
        landmark = landmarks[rng.randrange(len(landmarks))]
        start_tick = rng.randint(20, 520)
        duration = rng.randint(12, 45)
        participant_count = max(1, agent_count // rng.randint(5, 9))
        participants = {
            agents[rng.randrange(len(agents))].agent_id
            for _ in range(participant_count)
        }
        events.append(
            WorldEvent(
                event_id=f"event-{index + 1:03d}",
                event_type=event_type,
                position=landmark.position,
                start_tick=start_tick,
                end_tick=start_tick + duration,
                participants=participants,
            )
        )

    simulation.state.events.update(
        {event.event_id: event for event in events}
    )

    return BaselineScenario(
        simulation=simulation,
        landmarks=landmarks,
    )
