"""Telemetry data contracts for ARGUS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from argus.scheduling.relevance import RelevanceScore


@dataclass(frozen=True, slots=True)
class SimulationRun:
    """Metadata describing one experiment run."""

    run_id: str
    started_at: datetime
    ended_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class CognitiveUpdateEvent:
    """A single cognitive update executed for an agent."""

    run_id: str
    agent_id: str
    tick: int
    simulation_time: float
    update_kind: str = "full"


@dataclass(frozen=True, slots=True)
class RelevanceScoreRecord:
    """Persisted relevance score for one agent at one simulation tick."""

    run_id: str
    agent_id: str
    tick: int
    simulation_time: float
    score: RelevanceScore
