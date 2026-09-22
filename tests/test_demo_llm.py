"""Tests for the deterministic baseline cognitive provider."""

from argus.llm.demo import ScenarioLLMProvider
from argus.simulation import ActionType, Simulation
from argus.simulation.scenario import create_baseline_scenario


def test_demo_provider_moves_toward_goal() -> None:
    scenario = create_baseline_scenario(agent_count=1)
    agent = scenario.simulation.agents["agent-0001"]
    provider = ScenarioLLMProvider()

    delta = provider.request_cognitive_update(
        scenario.simulation.build_agent_context(agent.agent_id)
    )

    assert delta.action is not None
    assert delta.action.action_type == ActionType.MOVE
    assert provider.call_count == 1


def test_demo_provider_interacts_with_participating_event() -> None:
    scenario = create_baseline_scenario(agent_count=3)
    agent = scenario.simulation.agents["agent-0001"]
    event = next(iter(scenario.simulation.state.events.values()))
    event.start_tick = 0
    event.end_tick = 10
    event.participants.add(agent.agent_id)
    provider = ScenarioLLMProvider()

    delta = provider.request_cognitive_update(
        scenario.simulation.build_agent_context(agent.agent_id)
    )

    assert delta.action is not None
    assert delta.action.action_type == ActionType.INTERACT
