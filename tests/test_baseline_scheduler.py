"""Tests for the Stage 1 uniform full-frequency scheduler."""

from argus.llm import MockLLMProvider
from argus.scheduling import BaselineScheduler
from argus.simulation import Goal, Simulation, Vector2


def test_baseline_updates_every_active_agent_every_tick() -> None:
    simulation = Simulation.create(agent_count=3, seed=42)
    gateway = MockLLMProvider()
    scheduler = BaselineScheduler(simulation, gateway)

    updates = scheduler.run(ticks=4)

    assert updates == 12
    assert scheduler.total_cognitive_updates == 12
    assert gateway.call_count == 12
    assert simulation.current_tick == 4


def test_baseline_skips_inactive_agents() -> None:
    simulation = Simulation.create(agent_count=3, seed=42)
    simulation.agents["agent-0002"].active = False
    gateway = MockLLMProvider()
    scheduler = BaselineScheduler(simulation, gateway)

    updates = scheduler.run(ticks=3)

    assert updates == 6
    assert gateway.call_count == 6
    assert simulation.current_tick == 3


def test_baseline_step_applies_cognition_before_tick() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]
    agent.goal = Goal(
        goal_id="goal-1",
        description="Reach destination",
        target_position=Vector2(80.0, 90.0),
        importance=1.0,
    )
    gateway = MockLLMProvider()
    scheduler = BaselineScheduler(simulation, gateway)

    start = agent.position
    updates = scheduler.step()

    assert updates == 1
    assert gateway.call_count == 1
    assert simulation.current_tick == 1
    assert agent.position != start
