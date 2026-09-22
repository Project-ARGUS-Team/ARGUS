"""Minimal Tkinter-based 2D simulation viewer."""

from __future__ import annotations

import tkinter as tk

from argus.simulation.scenario import BaselineScenario


class SimulationViewer:
    """Interactive 2D viewer for a baseline scenario."""

    def __init__(
        self,
        scenario: BaselineScenario,
        pixels_per_unit: float = 7.0,
    ) -> None:
        self.scenario = scenario
        self.simulation = scenario.simulation
        self.scale = pixels_per_unit
        self.paused = False
        self.running = False

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

    def _screen(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    def _draw(self) -> None:
        self.canvas.delete("all")

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
            if not event.is_active(self.simulation.current_tick):
                continue
            x, y = self._screen(event.position.x, event.position.y)
            self.canvas.create_polygon(
                x,
                y - 10,
                x + 10,
                y,
                x,
                y + 10,
                x - 10,
                y,
                outline="#f59e0b",
                fill="",
                width=2,
            )
            self.canvas.create_text(
                x,
                y - 18,
                text=event.event_type,
                fill="#fbbf24",
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

        self.status.config(
            text=(
                f"Tick {self.simulation.current_tick}  |  "
                f"Time {self.simulation.simulation_time:.0f}s  |  "
                f"Agents {len(self.simulation.agents)}"
            )
        )

    def step(self) -> None:
        """Advance one simulation tick and redraw."""
        self.simulation.tick()
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

        self.scenario = create_baseline_scenario(
            agent_count=len(self.simulation.agents)
        )
        self.simulation = self.scenario.simulation
        self._draw()

    def _run_frame(self) -> None:
        if not self.running or self.paused:
            return
        self.simulation.tick()
        self._draw()
        self.root.after(50, self._run_frame)

    def run(self) -> None:
        """Start the viewer event loop."""
        self.root.mainloop()
