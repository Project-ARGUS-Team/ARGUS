"""Run the 30-agent ARGUS town baseline viewer."""

from argus.simulation.scenario import create_baseline_scenario
from argus.visualization import SimulationViewer


if __name__ == "__main__":
    scenario = create_baseline_scenario(agent_count=30, seed=42)
    SimulationViewer(scenario, pixels_per_unit=5.0).run()
