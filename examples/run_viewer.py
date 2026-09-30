"""Launch the inspectable ARGUS baseline viewer."""

from argus.simulation.scenario import create_interaction_scenario
from argus.visualization import SimulationViewer


if __name__ == "__main__":
    scenario = create_interaction_scenario()
    SimulationViewer(scenario, pixels_per_unit=5.0).run()
