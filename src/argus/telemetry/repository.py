"""SQLite telemetry repository for ARGUS experiments."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from argus.telemetry.models import (
    CognitiveUpdateEvent,
    RelevanceScoreRecord,
    SimulationRun,
)


class TelemetryRepository:
    """Persist experiment and relevance telemetry in SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._connection = sqlite3.connect(self.database_path)
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS simulation_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT
            );

            CREATE TABLE IF NOT EXISTS agents (
                run_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                PRIMARY KEY (run_id, agent_id),
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS cognitive_update_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                tick INTEGER NOT NULL,
                simulation_time REAL NOT NULL,
                update_kind TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS relevance_score_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                tick INTEGER NOT NULL,
                simulation_time REAL NOT NULL,
                score REAL NOT NULL,
                spatial_relevance REAL NOT NULL,
                interaction_probability REAL NOT NULL,
                goal_importance REAL NOT NULL,
                event_participation REAL NOT NULL,
                social_connectivity REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_cognitive_events_run_tick
            ON cognitive_update_events(run_id, tick);

            CREATE INDEX IF NOT EXISTS idx_cognitive_events_run_agent
            ON cognitive_update_events(run_id, agent_id);

            CREATE INDEX IF NOT EXISTS idx_relevance_run_tick
            ON relevance_score_records(run_id, tick);

            CREATE INDEX IF NOT EXISTS idx_relevance_run_agent
            ON relevance_score_records(run_id, agent_id);
            """
        )
        self._connection.commit()

    def create_run(self, run_id: str) -> SimulationRun:
        """Create and persist a new simulation run."""
        started_at = datetime.now(timezone.utc)
        run = SimulationRun(run_id=run_id, started_at=started_at)
        self._connection.execute(
            "INSERT INTO simulation_runs (run_id, started_at) VALUES (?, ?)",
            (run.run_id, run.started_at.isoformat()),
        )
        self._connection.commit()
        return run

    def complete_run(self, run_id: str) -> None:
        """Mark a run as completed."""
        ended_at = datetime.now(timezone.utc).isoformat()
        self._connection.execute(
            "UPDATE simulation_runs SET ended_at = ? WHERE run_id = ?",
            (ended_at, run_id),
        )
        self._connection.commit()

    def record_agent(self, run_id: str, agent_id: str) -> None:
        """Register an agent as participating in a run."""
        self._connection.execute(
            """
            INSERT OR IGNORE INTO agents (run_id, agent_id)
            VALUES (?, ?)
            """,
            (run_id, agent_id),
        )
        self._connection.commit()

    def record_cognitive_update(self, event: CognitiveUpdateEvent) -> None:
        """Persist one cognitive update event."""
        self._connection.execute(
            """
            INSERT INTO cognitive_update_events (
                run_id, agent_id, tick, simulation_time, update_kind
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.run_id,
                event.agent_id,
                event.tick,
                event.simulation_time,
                event.update_kind,
            ),
        )
        self._connection.commit()

    def record_relevance_score(self, record: RelevanceScoreRecord) -> None:
        """Persist one relevance score record."""
        self._connection.execute(
            """
            INSERT INTO relevance_score_records (
                run_id, agent_id, tick, simulation_time, score,
                spatial_relevance, interaction_probability,
                goal_importance, event_participation, social_connectivity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.agent_id,
                record.tick,
                record.simulation_time,
                record.score,
                record.spatial_relevance,
                record.interaction_probability,
                record.goal_importance,
                record.event_participation,
                record.social_connectivity,
            ),
        )
        self._connection.commit()

    def count_cognitive_updates(self, run_id: str) -> int:
        """Return the number of cognitive updates recorded for a run."""
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM cognitive_update_events
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        return int(row["count"])

    def count_relevance_scores(self, run_id: str) -> int:
        """Return the number of relevance records for a run."""
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM relevance_score_records
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        return int(row["count"])

    def close(self) -> None:
        """Close the SQLite connection."""
        self._connection.close()
