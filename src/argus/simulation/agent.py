"""Core agent state and action models for the ARGUS simulation."""

from dataclasses import dataclass, field
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class Vector2:
    """A two-dimensional position or velocity."""

    x: float
    y: float

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)


@dataclass(frozen=True, slots=True)
class Goal:
    """A goal an agent is currently pursuing."""

    goal_id: str
    description: str
    target_position: Vector2 | None
    importance: float


class ActivityType(StrEnum):
    """High-level activities used by an agent's daily routine."""

    HOME = "home"
    COMMUTE = "commute"
    WORK = "work"
    STUDY = "study"
    SHOP = "shop"
    EAT = "eat"
    LEISURE = "leisure"
    SOCIAL = "social"


@dataclass(frozen=True, slots=True)
class RoutineEntry:
    """One time-bounded activity in an agent's recurring routine."""

    activity: ActivityType
    description: str
    target_position: Vector2
    start_tick: int
    end_tick: int
    importance: float = 0.5

    def contains(self, tick: int, day_length: int) -> bool:
        """Return whether this entry is active at the given simulation tick."""
        if self.end_tick <= self.start_tick:
            return False
        local_tick = tick % day_length
        return self.start_tick <= local_tick < self.end_tick


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """Persistent identity and preferences for one simulated person."""

    name: str
    occupation: str
    home_position: Vector2
    work_position: Vector2 | None = None
    leisure_position: Vector2 | None = None
    social_preference: float = 0.5
    routine: tuple[RoutineEntry, ...] = ()


class ActionType(StrEnum):
    """Actions understood by the simulation core."""

    MOVE = "move"
    WAIT = "wait"
    INTERACT = "interact"


@dataclass(frozen=True, slots=True)
class Action:
    """A structured action produced by an agent's cognitive process."""

    action_type: ActionType
    target_agent_id: str | None = None
    target_position: Vector2 | None = None
    duration: float = 0.0


@dataclass(slots=True)
class AgentState:
    """Authoritative state maintained for one simulation agent."""

    agent_id: str
    position: Vector2
    velocity: Vector2
    goal: Goal
    profile: AgentProfile | None = None
    current_activity: ActivityType = ActivityType.HOME
    current_action: Action | None = None
    plan: list[Action] = field(default_factory=list)
    social_connections: set[str] = field(default_factory=set)
    active: bool = True

    def set_action(self, action: Action) -> None:
        self.current_action = action

    def clear_action(self) -> None:
        self.current_action = None
        self.velocity = Vector2(0.0, 0.0)
