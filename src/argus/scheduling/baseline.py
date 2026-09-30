"""Uniform full-frequency baseline scheduler."""

from __future__ import annotations

from uuid import uuid4

from argus.llm.gateway import LLMGateway
from argus.scheduling.relevance import IRelevanceScorer
from argus.simulation.simulation import Simulation
from argus.simulation.agent import ActionType, ActivityType
from argus.telemetry.models import CognitiveUpdateEvent, RelevanceScoreRecord
from argus.telemetry.repository import TelemetryRepository


class BaselineScheduler:
    """Run the Stage 1 baseline with optional Stage 2 scoring."""

    def __init__(
        self,
        simulation: Simulation,
        gateway: LLMGateway,
        telemetry: TelemetryRepository | None = None,
        run_id: str | None = None,
        relevance_scorer: IRelevanceScorer | None = None,
    ) -> None:
        self.simulation = simulation
        self.gateway = gateway
        self.telemetry = telemetry
        self.run_id = run_id
        self.relevance_scorer = relevance_scorer
        self.total_cognitive_updates = 0
        self._run_started = False

    @property
    def current_tick(self) -> int:
        """Return the simulation tick managed by this scheduler."""
        return self.simulation.current_tick

    def _ensure_run(self) -> None:
        if self.telemetry is None or self._run_started:
            return

        self.run_id = self.run_id or f"run-{uuid4().hex}"
        self.telemetry.create_run(self.run_id)

        for agent in self.simulation.agents.values():
            self.telemetry.record_agent(self.run_id, agent.agent_id)

        self._run_started = True

    def step(self) -> int:
        """Score agents if configured, then update every active agent."""
        self._ensure_run()
        updates = 0

        for agent in self.simulation.agents.values():
            if not agent.active:
                continue

            tick = self.simulation.current_tick
            simulation_time = self.simulation.simulation_time
            context = self.simulation.build_agent_context(agent.agent_id)

            if self.relevance_scorer is not None:
                score = self.relevance_scorer.score(context)
                if self.telemetry is not None:
                    signals = score.signals
                    self.telemetry.record_relevance_score(
                        RelevanceScoreRecord(
                            run_id=self.run_id,
                            agent_id=agent.agent_id,
                            tick=tick,
                            simulation_time=simulation_time,
                            score=score.score,
                            spatial_relevance=signals.spatial_relevance,
                            interaction_probability=signals.interaction_probability,
                            goal_importance=signals.goal_importance,
                            event_participation=signals.event_participation,
                            social_connectivity=signals.social_connectivity,
                        )
                    )

            previous_action = agent.current_action
            previous_target = (
                previous_action.target_agent_id
                if previous_action is not None
                and previous_action.action_type == ActionType.INTERACT
                else None
            )
            previous_activity = agent.current_activity
            self.simulation.request_cognitive_update(agent.agent_id, self.gateway)
            updates += 1

            current_action = agent.current_action
            if (
                current_action is not None
                and current_action.action_type == ActionType.INTERACT
                and current_action.target_agent_id is not None
            ):
                target_id = current_action.target_agent_id
                if target_id in self.simulation.agents:
                    target = self.simulation.agents[target_id]
                    self.simulation.register_interaction(
                        agent.agent_id,
                        target_id,
                        positive=True,
                    )
                    self.simulation.register_interaction(
                        target_id,
                        agent.agent_id,
                        positive=True,
                    )
                    self.simulation.add_memory(
                        agent.agent_id,
                        kind="interaction",
                        summary=(
                            f"Met {target.profile.name if target.profile else target_id} "
                            f"during {agent.current_activity.value}."
                        ),
                        importance=0.75,
                        related_agent_ids=(target_id,),
                    )
                    self.simulation.add_memory(
                        target_id,
                        kind="interaction",
                        summary=(
                            f"Met {agent.profile.name if agent.profile else agent.agent_id} "
                            f"during {target.current_activity.value}."
                        ),
                        importance=0.75,
                        related_agent_ids=(agent.agent_id,),
                    )

            if (
                current_action is not None
                and current_action.action_type == ActionType.WAIT
                and agent.current_activity != ActivityType.HOME
            ):
                self.simulation.add_memory(
                    agent.agent_id,
                    kind="experience",
                    summary=(
                        f"Spent time at {agent.current_activity.value} "
                        f"during day {self.simulation.current_tick // 720}."
                    ),
                    importance=0.45,
                )

            if self.telemetry is not None:
                self.telemetry.record_cognitive_update(
                    CognitiveUpdateEvent(
                        run_id=self.run_id,
                        agent_id=agent.agent_id,
                        tick=tick,
                        simulation_time=simulation_time,
                    )
                )

        self.simulation.tick()

        if self.simulation.current_tick > 0 and self.simulation.current_tick % 720 == 0:
            self.simulation.end_of_day_reflection()

        self.total_cognitive_updates += updates
        return updates

    def run(self, ticks: int) -> int:
        """Run the baseline for the requested number of ticks."""
        if ticks < 0:
            raise ValueError("ticks must be non-negative")

        self._ensure_run()
        updates = 0

        try:
            for _ in range(ticks):
                updates += self.step()
        finally:
            if self.telemetry is not None and self.run_id is not None:
                self.telemetry.complete_run(self.run_id)

        return updates
