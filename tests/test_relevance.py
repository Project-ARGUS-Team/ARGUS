"""Tests for Stage 2 relevance scoring."""

from argus.scheduling import RuleBasedRelevanceScorer
from argus.simulation import Goal, Simulation, Vector2, WorldEvent


def test_goal_importance_is_normalized_into_relevance() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]
    agent.goal = Goal(
        goal_id="goal-1",
        description="Critical goal",
        target_position=None,
        importance=1.0,
    )

    score = RuleBasedRelevanceScorer().score(
        simulation.build_agent_context(agent.agent_id)
    )

    assert score.signals.goal_importance == 1.0
    assert 0.0 <= score.score <= 1.0


def test_spatial_relevance_decreases_with_event_distance() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]
    event = WorldEvent(
        event_id="event-1",
        event_type="incident",
        position=Vector2(agent.position.x + 5.0, agent.position.y),
        start_tick=0,
        end_tick=10,
    )
    simulation.state.events[event.event_id] = event
    scorer = RuleBasedRelevanceScorer(spatial_radius=10.0)

    near = scorer.score(simulation.build_agent_context(agent.agent_id))
    event.position = Vector2(agent.position.x + 9.0, agent.position.y)
    far = scorer.score(simulation.build_agent_context(agent.agent_id))

    assert near.signals.spatial_relevance > far.signals.spatial_relevance


def test_event_participation_is_recorded() -> None:
    simulation = Simulation.create(agent_count=1, seed=42)
    agent = simulation.agents["agent-0001"]
    event = WorldEvent(
        event_id="event-1",
        event_type="meeting",
        position=agent.position,
        start_tick=0,
        end_tick=10,
        participants={agent.agent_id},
    )
    simulation.state.events[event.event_id] = event

    score = RuleBasedRelevanceScorer().score(
        simulation.build_agent_context(agent.agent_id)
    )

    assert score.signals.event_participation == 1.0


def test_social_connectivity_uses_population() -> None:
    simulation = Simulation.create(agent_count=3, seed=42)
    agent = simulation.agents["agent-0001"]
    agent.social_connections.add("agent-0002")

    score = RuleBasedRelevanceScorer().score(
        simulation.build_agent_context(agent.agent_id)
    )

    assert score.signals.social_connectivity == 0.5


def test_all_relevance_signals_are_bounded() -> None:
    simulation = Simulation.create(agent_count=3, seed=42)
    agent = simulation.agents["agent-0001"]
    agent.goal = Goal(
        goal_id="goal-1",
        description="Goal",
        target_position=None,
        importance=2.0,
    )
    agent.social_connections.update(
        ["agent-0002", "agent-0003", "unknown-agent"]
    )

    score = RuleBasedRelevanceScorer().score(
        simulation.build_agent_context(agent.agent_id)
    )

    signals = score.signals
    assert 0.0 <= signals.spatial_relevance <= 1.0
    assert 0.0 <= signals.interaction_probability <= 1.0
    assert 0.0 <= signals.goal_importance <= 1.0
    assert 0.0 <= signals.event_participation <= 1.0
    assert 0.0 <= signals.social_connectivity <= 1.0
    assert 0.0 <= score.score <= 1.0
