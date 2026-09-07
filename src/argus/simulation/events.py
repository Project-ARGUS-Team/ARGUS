"""World event models."""

from dataclasses import dataclass, field

from argus.simulation.agent import Vector2


@dataclass(slots=True)
class WorldEvent:
    """An event occurring at a location over a range of simulation ticks."""

    event_id: str
    event_type: str
    position: Vector2
    start_tick: int
    end_tick: int
    participants: set[str] = field(default_factory=set)

    def is_active(self, tick: int) -> bool:
        return self.start_tick <= tick <= self.end_tick
