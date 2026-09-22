"""Integration tests for Stage 2 relevance logging."""

from argus.llm import MockLLMProvider
from argus.scheduling import BaselineScheduler, RuleBasedRelevanceScorer
from argus.simulation import Simulation
from argus.telemetry import TelemetryRepository


def test_stage_2_logs_relevance_without_changing_baseline_frequency(tmp_path) -> None:
    repository = TelemetryRepository(tmp_path / "argus.db")
    simulation = Simulation.create(agent_count=3, seed=42)
    gateway = MockLLMProvider()
    scheduler = BaselineScheduler(
        simulation,
        gateway,
        telemetry=repository,
        run_id="stage-2",
        relevance_scorer=RuleBasedRelevanceScorer(),
    )

    scheduler.run(ticks=4)

    assert gateway.call_count == 12
    assert scheduler.total_cognitive_updates == 12
    assert repository.count_cognitive_updates("stage-2") == 12
    assert repository.count_relevance_scores("stage-2") == 12

    repository.close()
