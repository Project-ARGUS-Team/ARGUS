"""Deterministic demonstration scenarios for the ARGUS baseline."""

from __future__ import annotations

from dataclasses import dataclass

from argus.simulation.agent import Action, ActionType, Goal, Vector2
from argus.simulation.events import WorldEvent
from argus.simulation.simulation import Simulation


@dataclass(frozen=True, slots=True)
class Landmark:
    """A named location used by a demonstration scenario."""

    landmark_id: str
    name: str
    position: Vector2
    radius: float = 8.0


@dataclass(frozen=True, slots=True)
class BaselineScenario:
    """A compact world containing landmarks, goals, relationships and events."""

    simulation: Simulation
    landmarks: tuple[Landmark, ...]


def create_baseline_scenario(
    agent_count: int = 30,
    seed: int = 42,
) -> BaselineScenario:
    """Create a deterministic, visually interesting baseline scenario."""
    simulation = Simulation.create(agent_count=agent_count, seed=seed)

    landmarks = (
        Landmark("market", "Market", Vector2(25.0, 25.0)),
        Landmark("plaza", "Plaza", Vector2(50.0, 50.0)),
        Landmark("park", "Park", Vector2(75.0, 25.0)),
        Landmark("station", "Station", Vector2(50.0, 80.0)),
    )

    agents = list(simulation.agents.values())

    for index, agent in enumerate(agents):
        landmark = landmarks[index % len(landmarks)]
        agent.goal = Goal(
            goal_id=f"goal-{index + 1:04d}",
            description=f"Visit the {landmark.name.lower()}",
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

        if index % 3 == 0:
            agent.set_action(
                Action(
                    action_type=ActionType.MOVE,
                    target_position=landmark.position,
                )
            )
            dx = landmark.position.x - agent.position.x
            dy = landmark.position.y - agent.position.y
            distance = (dx * dx + dy * dy) ** 0.5
            if distance > 0:
                agent.velocity = Vector2(dx / distance, dy / distance)

    events = (
        WorldEvent(
            event_id="market-opening",
            event_type="market_opening",
            position=landmarks[0].position,
            start_tick=20,
            end_tick=80,
            participants={agent.agent_id for agent in agents[::3]},
        ),
        WorldEvent(
            event_id="plaza-gathering",
            event_type="social_gathering",
            position=landmarks[1].position,
            start_tick=50,
            end_tick=140,
            participants={agent.agent_id for agent in agents[1::3]},
        ),
        WorldEvent(
            event_id="station-disruption",
            event_type="transport_disruption",
            position=landmarks[3].position,
            start_tick=100,
            end_tick=125,
            participants=set(),
        ),
    )

    simulation.state.events.update(
        {event.event_id: event for event in events}
    )

    return BaselineScenario(simulation=simulation, landmarks=landmarks)
