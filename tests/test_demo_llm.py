"""Tests for the deterministic baseline cognitive provider."""

from argus.llm.demo import ScenarioLLMProvider
from argus.simulation import ActionType
from argus.simulation.scenario import create_baseline_scenario


def test_demo_provider_moves_toward_route_target() -> None:
    scenario = create_baseline_scenario(agent_count=1)
    agent = scenario.simulation.agents["agent-0001"]
    provider = ScenarioLLMProvider(
        tuple(landmark.position for landmark in scenario.landmarks)
    )

    delta = provider.request_cognitive_update(
        scenario.simulation.build_agent_context(agent.agent_id)
    )

    assert delta.action is not None
    assert delta.action.action_type == ActionType.MOVE
    assert delta.action.target_position is not None
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
    assert delta.action.action_type == ActionType.MOVE

    agent.position = event.position
    delta = provider.request_cognitive_update(
        scenario.simulation.build_agent_context(agent.agent_id)
    )

    assert delta.action is not None
    assert delta.action.action_type == ActionType.INTERACT


def test_demo_provider_is_deterministic_and_varied() -> None:
    scenario = create_baseline_scenario(agent_count=12)
    route = tuple(landmark.position for landmark in scenario.landmarks)
    provider = ScenarioLLMProvider(route)

    actions = []
    for agent in scenario.simulation.agents.values():
        delta = provider.request_cognitive_update(
            scenario.simulation.build_agent_context(agent.agent_id)
        )
        actions.append(delta.action)

    move_targets = {
        (action.target_position.x, action.target_position.y)
        for action in actions
        if action is not None
        and action.action_type == ActionType.MOVE
        and action.target_position is not None
    }

    assert len(move_targets) >= 2
    assert provider.call_count == 12
