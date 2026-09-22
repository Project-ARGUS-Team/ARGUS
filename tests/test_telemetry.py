"""Tests for Stage 1 SQLite telemetry."""

from argus.llm import MockLLMProvider
from argus.scheduling import BaselineScheduler
from argus.simulation import Simulation
from argus.telemetry import TelemetryRepository


def test_telemetry_records_baseline_cognitive_updates(tmp_path) -> None:
    repository = TelemetryRepository(tmp_path / "argus.db")
    simulation = Simulation.create(agent_count=2, seed=42)
    gateway = MockLLMProvider()
    scheduler = BaselineScheduler(
        simulation,
        gateway,
        telemetry=repository,
        run_id="run-001",
    )

    scheduler.run(ticks=3)

    assert repository.count_cognitive_updates("run-001") == 6

    rows = repository._connection.execute(
        """
        SELECT agent_id, tick, simulation_time, update_kind
        FROM cognitive_update_events
        WHERE run_id = ?
        ORDER BY tick, agent_id
        """,
        ("run-001",),
    ).fetchall()

    assert [(row["agent_id"], row["tick"]) for row in rows] == [
        ("agent-0001", 0),
        ("agent-0002", 0),
        ("agent-0001", 1),
        ("agent-0002", 1),
        ("agent-0001", 2),
        ("agent-0002", 2),
    ]
    assert all(row["update_kind"] == "full" for row in rows)

    repository.close()


def test_telemetry_registers_agents_and_completes_run(tmp_path) -> None:
    repository = TelemetryRepository(tmp_path / "argus.db")
    simulation = Simulation.create(agent_count=2, seed=42)
    scheduler = BaselineScheduler(
        simulation,
        MockLLMProvider(),
        telemetry=repository,
        run_id="run-002",
    )

    scheduler.run(ticks=1)

    agents = repository._connection.execute(
        "SELECT agent_id FROM agents WHERE run_id = ? ORDER BY agent_id",
        ("run-002",),
    ).fetchall()
    run = repository._connection.execute(
        "SELECT started_at, ended_at FROM simulation_runs WHERE run_id = ?",
        ("run-002",),
    ).fetchone()

    assert [row["agent_id"] for row in agents] == ["agent-0001", "agent-0002"]
    assert run["started_at"]
    assert run["ended_at"]

    repository.close()
