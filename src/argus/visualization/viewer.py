"""Interactive Tkinter viewer for the ARGUS baseline simulation."""

from __future__ import annotations

import math
import tkinter as tk

from argus.llm.demo import ScenarioLLMProvider
from argus.scheduling.baseline import BaselineScheduler
from argus.simulation.agent import ActionType
from argus.simulation.scenario import BaselineScenario, create_baseline_scenario


class SimulationViewer:
    """Slow, inspectable city-scale viewer for baseline behavior."""

    def __init__(
        self,
        scenario: BaselineScenario,
        pixels_per_unit: float = 6.0,
    ) -> None:
        self.scale = pixels_per_unit
        self.paused = True
        self.running = False
        self.speed = 1.0
        self.selected_agent_id: str | None = None
        self._install_scenario(scenario)

        self.root = tk.Tk()
        self.root.title("ARGUS — Baseline Simulation")

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        map_frame = tk.Frame(main)
        map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        width = int(self.simulation.world.width * self.scale)
        height = int(self.simulation.world.height * self.scale)
        self.canvas = tk.Canvas(
            map_frame,
            width=width,
            height=height,
            background="#111827",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        inspector = tk.Frame(main, width=310, padx=12, pady=12)
        inspector.pack(side=tk.RIGHT, fill=tk.Y)
        inspector.pack_propagate(False)

        tk.Label(
            inspector,
            text="AGENT INSPECTOR",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor=tk.W)

        self.agent_info = tk.Label(
            inspector,
            text="Click an agent to inspect it.",
            justify=tk.LEFT,
            anchor=tk.NW,
        )
        self.agent_info.pack(fill=tk.X, pady=(10, 16))

        tk.Label(
            inspector,
            text="SIMULATION",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor=tk.W)

        self.status = tk.Label(
            inspector,
            text="",
            justify=tk.LEFT,
            anchor=tk.NW,
        )
        self.status.pack(fill=tk.X, pady=(8, 12))

        tk.Label(
            inspector,
            text="SPEED",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor=tk.W)

        self.speed_label = tk.Label(inspector, text="1.00×")
        self.speed_label.pack(anchor=tk.W)

        self.speed_scale = tk.Scale(
            inspector,
            from_=0.25,
            to=4.0,
            resolution=0.25,
            orient=tk.HORIZONTAL,
            command=self._set_speed,
        )
        self.speed_scale.set(self.speed)
        self.speed_scale.pack(fill=tk.X)

        controls = tk.Frame(inspector)
        controls.pack(fill=tk.X, pady=(8, 12))
        tk.Button(controls, text="Play", command=self.play).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        tk.Button(controls, text="Pause", command=self.pause).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        tk.Button(controls, text="Step", command=self.step).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        tk.Button(
            controls,
            text="Reset",
            command=self.reset,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            inspector,
            text="EVENTS",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor=tk.W)

        self.event_info = tk.Label(
            inspector,
            text="",
            justify=tk.LEFT,
            anchor=tk.NW,
        )
        self.event_info.pack(fill=tk.X, pady=(8, 0))

        self._draw()

    def _install_scenario(self, scenario: BaselineScenario) -> None:
        self.scenario = scenario
        self.simulation = scenario.simulation
        route_points = tuple(landmark.position for landmark in scenario.landmarks)
        self.gateway = ScenarioLLMProvider(route_points, roads=scenario.roads)
        self.scheduler = BaselineScheduler(
            self.simulation,
            self.gateway,
        )

    def _screen(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    def _set_speed(self, value: str) -> None:
        self.speed = max(0.25, float(value))
        self.speed_label.config(text=f"{self.speed:.2f}×")

    def _on_canvas_click(self, event: tk.Event) -> None:
        world_x = event.x / self.scale
        world_y = event.y / self.scale
        nearest_id = None
        nearest_distance = float("inf")

        for agent in self.simulation.agents.values():
            dx = agent.position.x - world_x
            dy = agent.position.y - world_y
            distance = (dx * dx + dy * dy) ** 0.5
            if distance < nearest_distance:
                nearest_id = agent.agent_id
                nearest_distance = distance

        if nearest_id is not None and nearest_distance <= 7.0:
            self.selected_agent_id = nearest_id
        else:
            self.selected_agent_id = None
        self._draw()

    def _landmark_name(self, position) -> str:
        if position is None:
            return "None"
        nearest = min(
            self.scenario.landmarks,
            key=lambda landmark: (
                (landmark.position.x - position.x) ** 2
                + (landmark.position.y - position.y) ** 2
            ),
        )
        distance = (
            (nearest.position.x - position.x) ** 2
            + (nearest.position.y - position.y) ** 2
        ) ** 0.5
        return nearest.name if distance <= nearest.radius * 1.5 else (
            f"({position.x:.1f}, {position.y:.1f})"
        )

    def _action_label(self, action) -> str:
        if action is None:
            return "None"
        return action.action_type.value.upper()

    def _agent_draw_positions(self) -> dict[str, tuple[float, float]]:
        """Return visually separated positions without changing simulation state."""
        agents = [agent for agent in self.simulation.agents.values() if agent.active]
        positions: dict[str, tuple[float, float]] = {}
        groups: dict[tuple[int, int], list] = {}
        cell_size = 10.0

        for agent in agents:
            sx, sy = self._screen(agent.position.x, agent.position.y)
            key = (int(sx // cell_size), int(sy // cell_size))
            groups.setdefault(key, []).append(agent)

        for group in groups.values():
            group.sort(key=lambda agent: agent.agent_id)
            if len(group) == 1:
                agent = group[0]
                positions[agent.agent_id] = self._screen(
                    agent.position.x, agent.position.y
                )
                continue

            screen_positions = [
                self._screen(agent.position.x, agent.position.y)
                for agent in group
            ]
            center_x = sum(point[0] for point in screen_positions) / len(group)
            center_y = sum(point[1] for point in screen_positions) / len(group)
            radius = min(18.0, 7.0 + len(group) * 1.5)

            for index, agent in enumerate(group):
                angle = (2.0 * math.pi * index / len(group)) - math.pi / 2.0
                positions[agent.agent_id] = (
                    center_x + math.cos(angle) * radius,
                    center_y + math.sin(angle) * radius,
                )

        return positions

    def _draw_agent_inspector(self) -> None:
        if self.selected_agent_id is None:
            self.agent_info.config(text="Click an agent to inspect it.")
            return

        agent = self.simulation.agents.get(self.selected_agent_id)
        if agent is None:
            self.agent_info.config(text="Agent no longer exists.")
            return

        action = agent.current_action
        target = action.target_position if action else None
        nearby_count = 0
        for other in self.simulation.agents.values():
            if other.agent_id == agent.agent_id or not other.active:
                continue
            dx = other.position.x - agent.position.x
            dy = other.position.y - agent.position.y
            if (dx * dx + dy * dy) ** 0.5 <= 10.0:
                nearby_count += 1

        active_events = [
            event.event_type
            for event in self.simulation.state.events.values()
            if event.is_active(self.simulation.current_tick)
            and (
                agent.agent_id in event.participants
                or (
                    (event.position.x - agent.position.x) ** 2
                    + (event.position.y - agent.position.y) ** 2
                ) ** 0.5 <= 12.0
            )
        ]

        display_name = agent.profile.name if agent.profile else agent.agent_id
        occupation = agent.profile.occupation if agent.profile else "No occupation"

        work_destination = (
            self._landmark_name(agent.profile.work_position)
            if agent.profile is not None and agent.profile.work_position is not None
            else "None"
        )
        current_destination = self._landmark_name(
            agent.travel_destination
            if agent.travel_destination is not None
            else agent.goal.target_position
        )

        text = (
            f"{display_name}\n"
            f"{occupation}\n\n"
            f"WORKPLACE\n"
            f"  {work_destination}\n\n"
            f"POSITION\n"
            f"  ({agent.position.x:.1f}, {agent.position.y:.1f})\n"
            f"  velocity ({agent.velocity.x:.1f}, {agent.velocity.y:.1f})\n\n"
            f"ACTION\n"
            f"  {self._action_label(action)}\n"
            f"  target: {current_destination}\n\n"
            f"ACTIVITY\n"
            f"  {agent.current_activity.value.upper()}\n\n"
            f"GOAL\n"
            f"  {agent.goal.description}\n"
            f"  importance: {agent.goal.importance:.2f}\n\n"
            f"SOCIAL\n"
            f"  connections: {len(agent.social_connections)}\n"
            f"  nearby: {nearby_count}\n\n"
            f"EVENTS\n"
            f"  {', '.join(active_events) if active_events else 'None'}"
        )
        self.agent_info.config(text=text)

    def _time_of_day(self) -> str:
        """Convert the 720-tick simulation day into a readable clock."""
        minutes = (6 * 60 + (self.simulation.current_tick % 720) * 2) % (24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def _draw(self) -> None:
        self.canvas.delete("all")
        current_tick = self.simulation.current_tick

        for road in self.scenario.roads:
            x1, y1 = self._screen(road.start.x, road.start.y)
            x2, y2 = self._screen(road.end.x, road.end.y)
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill="#374151",
                width=12,
            )
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill="#6b7280",
                width=4,
            )

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
            if not event.is_active(current_tick):
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

            if event.event_type == "road_closure":
                for road in self.scenario.roads:
                    if road.road_id not in event.affected_road_ids:
                        continue
                    x1, y1 = self._screen(road.start.x, road.start.y)
                    x2, y2 = self._screen(road.end.x, road.end.y)
                    self.canvas.create_line(
                        x1, y1, x2, y2,
                        fill="#ef4444",
                        width=5,
                    )
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text="X",
                        fill="#fee2e2",
                        font=("TkDefaultFont", 12, "bold"),
                    )

        agent_draw_positions = self._agent_draw_positions()

        for agent in self.simulation.agents.values():
            if not agent.active:
                continue

            x, y = agent_draw_positions[agent.agent_id]
            radius = 6.0
            selected = agent.agent_id == self.selected_agent_id
            if agent.transport_mode.value == "car":
                self.canvas.create_rectangle(
                    x - 5,
                    y - 3,
                    x + 5,
                    y + 3,
                    fill="#f97316",
                    outline="#f8fafc" if selected else "",
                    width=4 if selected else 1,
                )
            else:
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill="#60a5fa",
                    outline="#f8fafc" if selected else "",
                    width=4 if selected else 1,
                )

            action = agent.current_action
            if action is not None and action.action_type == ActionType.INTERACT:
                if action.target_agent_id in self.simulation.agents:
                    target = self.simulation.agents[action.target_agent_id]
                    tx, ty = agent_draw_positions.get(
                        target.agent_id,
                        self._screen(target.position.x, target.position.y),
                    )
                    self.canvas.create_line(
                        x,
                        y,
                        tx,
                        ty,
                        fill="#c084fc",
                        width=2,
                    )

        self._draw_agent_inspector()

        active_events = [
            event.event_type
            for event in self.simulation.state.events.values()
            if event.is_active(current_tick)
        ]
        upcoming_events = sorted(
            (
                event
                for event in self.simulation.state.events.values()
                if event.start_tick > current_tick
            ),
            key=lambda event: event.start_tick,
        )[:4]

        active_text = (
            ", ".join(active_events)
            if active_events
            else "None"
        )
        upcoming_text = (
            "\n".join(
                f"  {event.event_type} @ tick {event.start_tick}"
                for event in upcoming_events
            )
            if upcoming_events
            else "  None"
        )

        self.status.config(
            text=(
                f"Tick {current_tick}\n"
                f"Time of day {self._time_of_day()}\n"
                f"Simulation {self.simulation.simulation_time:.0f}s\n"
                f"Agents {len(self.simulation.agents)}\n"
                f"Cognitive updates {self.scheduler.total_cognitive_updates}"
            )
        )
        self.event_info.config(
            text=(
                f"Active\n  {active_text}\n\n"
                f"Upcoming\n{upcoming_text}"
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
        self.running = False
        self.paused = True
        self.selected_agent_id = None
        self._install_scenario(
            create_baseline_scenario(
                agent_count=len(self.simulation.agents),
                seed=42,
            )
        )
        self._draw()

    def _run_frame(self) -> None:
        if not self.running or self.paused:
            return
        self.scheduler.step()
        self._draw()
        delay = max(40, int(500 / self.speed))
        self.root.after(delay, self._run_frame)

    def run(self) -> None:
        """Start the viewer event loop."""
        self.root.mainloop()
