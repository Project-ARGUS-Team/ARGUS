"""Tkinter-based 2D viewer for the ARGUS baseline simulation."""

from __future__ import annotations

import tkinter as tk

from argus.llm.demo import ScenarioLLMProvider
from argus.scheduling.baseline import BaselineScheduler
from argus.simulation.scenario import BaselineScenario


class SimulationViewer:
    """Interactive 2D viewer for a cognitively driven baseline scenario."""

    def __init__(
        self,
        scenario: BaselineScenario,
        pixels_per_unit: float = 7.0,
    ) -> None:
        self.scale = pixels_per_unit
        self.paused = False
        self.running = False
        self._install_scenario(scenario)

        width = int(self.simulation.world.width * self.scale)
        height = int(self.simulation.world.height * self.scale)

        self.root = tk.Tk()
        self.root.title("ARGUS — Baseline Simulation")
        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            background="#111827",
            highlightthickness=0,
        )
        self.canvas.pack()

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X)
        tk.Button(controls, text="Play", command=self.play).pack(
            side=tk.LEFT
        )
        tk.Button(controls, text="Pause", command=self.pause).pack(
            side=tk.LEFT
        )
        tk.Button(controls, text="Step", command=self.step).pack(
            side=tk.LEFT
        )
        tk.Button(controls, text="Reset", command=self.reset).pack(
            side=tk.LEFT
        )
        self.status = tk.Label(controls, text="")
        self.status.pack(side=tk.RIGHT)

        self._draw()

    def _install_scenario(self, scenario: BaselineScenario) -> None:
        self.scenario = scenario
        self.simulation = scenario.simulation
        route_points = tuple(landmark.position for landmark in scenario.landmarks)
        self.gateway = ScenarioLLMProvider(route_points)
        self.scheduler = BaselineScheduler(
            self.simulation,
            self.gateway,
        )

    def _screen(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    def _draw(self) -> None:
        self.canvas.delete("all")
        current_tick = self.simulation.current_tick

        for landmark in self.scenario.landmarks:
            x, y = self._screen(
                landmark.position.x,
                landmark.position.y,
            )
            radius = landmark.radius * self.scale
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                outline="#4b5563",
                width=2,
            )
            self.canvas.create_text(
                x,
                y,
                text=landmark.name,
                fill="#d1d5db",
            )

        for event in self.simulation.state.events.values():
            active = event.is_active(current_tick)
            upcoming = current_tick < event.start_tick
            if not active and not upcoming:
                continue

            x, y = self._screen(event.position.x, event.position.y)
            if active:
                outline = "#f59e0b"
                label = f"{event.event_type} (ACTIVE)"
            else:
                outline = "#6b7280"
                label = f"{event.event_type} (tick {event.start_tick})"

            self.canvas.create_polygon(
                x,
                y - 10,
                x + 10,
                y,
                x,
                y + 10,
                x - 10,
                y,
                outline=outline,
                fill="",
                width=2,
            )
            self.canvas.create_text(
                x,
                y - 18,
                text=label,
                fill=outline,
            )

        for agent in self.simulation.agents.values():
            if not agent.active:
                continue

            x, y = self._screen(agent.position.x, agent.position.y)
            radius = 4.0
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#60a5fa",
                outline="",
            )

            action = agent.current_action
            if action is None or action.action_type.name != "INTERACT":
                continue

            target_id = action.target_agent_id
            if target_id is not None and target_id in self.simulation.agents:
                target = self.simulation.agents[target_id]
                tx, ty = self._screen(target.position.x, target.position.y)
                self.canvas.create_line(x, y, tx, ty, fill="#c084fc", width=2)

        active_events = [
            event.event_type
            for event in self.simulation.state.events.values()
            if event.is_active(current_tick)
        ]
        next_events = [
            event
            for event in self.simulation.state.events.values()
            if event.start_tick > current_tick
        ]
        next_event = min(next_events, key=lambda event: event.start_tick, default=None)

        event_text = (
            f"Active: {', '.join(active_events)}"
            if active_events
            else (
                f"Next: {next_event.event_type} @ {next_event.start_tick}"
                if next_event
                else "Events complete"
            )
        )

        self.status.config(
            text=(
                f"Tick {current_tick}  |  "
                f"Agents {len(self.simulation.agents)}  |  "
                f"Cognitive updates {self.scheduler.total_cognitive_updates}  |  "
                f"{event_text}"
            )
        )

    def step(self) -> None:
        """Run one full-frequency cognitive baseline tick and redraw."""
        self.scheduler.step()
        self._draw()

    def play(self) -> None:
        """Start continuous simulation playback."""
        self.running = True
        self.paused = False
        self._run_frame()

    def pause(self) -> None:
        """Pause continuous simulation playback."""
        self.paused = True
        self.running = False

    def reset(self) -> None:
        """Reset by rebuilding the deterministic scenario."""
        from argus.simulation.scenario import create_baseline_scenario

        self._install_scenario(
            create_baseline_scenario(
                agent_count=len(self.simulation.agents)
            )
        )
        self._draw()

    def _run_frame(self) -> None:
        if not self.running or self.paused:
            return
        self.scheduler.step()
        self._draw()
        self.root.after(50, self._run_frame)

    def run(self) -> None:
        """Start the viewer event loop."""
        self.root.mainloop()
