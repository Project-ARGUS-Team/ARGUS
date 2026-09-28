"""Cognitive context and state-delta models."""

from dataclasses import dataclass

from argus.simulation.agent import (
    Action,
    ActivityType,
    AgentProfile,
    Goal,
    RoutineEntry,
    Vector2,
)


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """Read-only observation of another agent."""

    agent_id: str
    position: Vector2
    distance: float


@dataclass(frozen=True, slots=True)
class EventObservation:
    """Read-only observation of an active world event."""

    event_id: str
    event_type: str
    position: Vector2
    distance: float
    is_participant: bool = False


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Read-only information supplied to an agent's cognitive process."""

    agent_id: str
    simulation_tick: int
    simulation_time: float
    position: Vector2
    velocity: Vector2
    goal: Goal
    current_action: Action | None
    plan: tuple[Action, ...]
    profile: AgentProfile | None = None
    current_activity: ActivityType = ActivityType.HOME
    current_routine: RoutineEntry | None = None
    nearby_agents: tuple[AgentObservation, ...] = ()
    active_events: tuple[EventObservation, ...] = ()
    social_connections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StateDelta:
    """Structured result of a cognitive update."""

    action: Action | None = None
    plan: tuple[Action, ...] = ()
    goal: Goal | None = None
    activity: ActivityType | None = None
