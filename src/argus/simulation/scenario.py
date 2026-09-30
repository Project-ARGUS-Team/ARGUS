"""Deterministic town-scale demonstration scenarios for the ARGUS baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import random

from argus.simulation.agent import (
    ActivityType,
    AgentProfile,
    Goal,
    RoutineEntry,
    TransportMode,
    Vector2,
)
from argus.simulation.simulation import Simulation
from argus.simulation.world import RoadSegment, World


@dataclass(frozen=True, slots=True)
class Landmark:
    """A named location used by a demonstration scenario."""

    landmark_id: str
    name: str
    position: Vector2
    radius: float = 6.0


@dataclass(frozen=True, slots=True)
class BaselineScenario:
    """A town-scale world containing locations, routines and pedestrian roads."""

    simulation: Simulation
    landmarks: tuple[Landmark, ...]
    roads: tuple[RoadSegment, ...] = ()


NAMES = (
    "Arun", "Maya", "Vivek", "Ananya", "Rahul", "Diya", "Nikhil", "Isha",
    "Aditya", "Meera", "Kiran", "Riya", "Rohan", "Asha", "Sanjay", "Neha",
    "Akhil", "Priya", "Varun", "Sneha", "Arjun", "Nandita", "Manu", "Tara",
    "Amit", "Kavya", "Ravi", "Pooja", "Dev", "Lakshmi", "Vishal", "Anu",
    "Joel", "Sara", "Naveen", "Aditi", "Imran", "Farah", "Sameer", "Zoya",
    "Irfan", "Hana", "Kabir", "Nisha", "Yash", "Sana", "Rakesh", "Mira",
    "Ajay", "Leena", "Suraj", "Nitya", "Rohit", "Alina", "Faisal", "Jaya",
    "Suresh", "Mina", "Karthik", "Reshma",
)

OCCUPATION_TYPES = (
    ("office", "Office worker"),
    ("student", "Student"),
    ("healthcare", "Healthcare worker"),
    ("retail", "Retail worker"),
    ("retired", "Retired"),
)


def _routine(
    *entries: tuple[ActivityType, str, Vector2, int, int, float],
) -> tuple[RoutineEntry, ...]:
    return tuple(
        RoutineEntry(
            activity=activity,
            description=description,
            target_position=target,
            start_tick=start,
            end_tick=end,
            importance=importance,
        )
        for activity, description, target, start, end, importance in entries
    )


def _build_profile(
    index: int,
    home: Vector2,
    office: Vector2,
    school: Vector2,
    hospital: Vector2,
    market: Vector2,
    park: Vector2,
    cafe: Vector2,
    entertainment: Vector2,
    station: Vector2,
    transport_mode: TransportMode,
) -> AgentProfile:
    kind, occupation = OCCUPATION_TYPES[index % len(OCCUPATION_TYPES)]
    leisure = (park, cafe, entertainment)[index % 3]
    social_preference = (0.3, 0.55, 0.7, 0.45, 0.8)[index % 5]

    if kind == "student":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 45, 0.45),
            (ActivityType.COMMUTE, "Travel to school", school, 45, 75, 0.7),
            (ActivityType.STUDY, "Morning classes", school, 75, 195, 0.9),
            (ActivityType.EAT, "Lunch break", cafe, 195, 225, 0.6),
            (ActivityType.STUDY, "Afternoon classes", school, 225, 315, 0.85),
            (ActivityType.COMMUTE, "Travel home", home, 315, 345, 0.65),
            (ActivityType.SOCIAL, "Meet friends", leisure, 345, 435, 0.65),
            (ActivityType.HOME, "Evening at home", home, 435, 720, 0.4),
        )
        work = school
    elif kind == "healthcare":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 45, 0.4),
            (ActivityType.COMMUTE, "Travel to hospital", hospital, 45, 75, 0.75),
            (ActivityType.WORK, "Morning hospital shift", hospital, 75, 195, 1.0),
            (ActivityType.EAT, "Lunch break", market, 195, 225, 0.55),
            (ActivityType.WORK, "Afternoon hospital shift", hospital, 225, 315, 1.0),
            (ActivityType.COMMUTE, "Travel home", home, 315, 345, 0.65),
            (ActivityType.LEISURE, "Evening leisure", leisure, 345, 435, 0.5),
            (ActivityType.HOME, "Evening at home", home, 435, 720, 0.4)
        )
        work = hospital
    elif kind == "retail":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 60, 0.4),
            (ActivityType.COMMUTE, "Travel to market", market, 60, 90, 0.7),
            (ActivityType.WORK, "Morning market shift", market, 90, 195, 0.9),
            (ActivityType.EAT, "Lunch break", cafe, 195, 225, 0.55),
            (ActivityType.WORK, "Afternoon market shift", market, 225, 315, 0.9),
            (ActivityType.COMMUTE, "Travel home", home, 315, 345, 0.65),
            (ActivityType.SOCIAL, "Meet people", leisure, 345, 435, 0.6),
            (ActivityType.HOME, "Evening at home", home, 435, 720, 0.4),
        )
        work = market
    elif kind == "retired":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 60, 0.4),
            (ActivityType.EAT, "Morning outing", cafe, 60, 120, 0.55),
            (ActivityType.LEISURE, "Walk in the park", park, 120, 210, 0.65),
            (ActivityType.SHOP, "Shopping", market, 210, 285, 0.65),
            (ActivityType.HOME, "Afternoon at home", home, 285, 375, 0.4),
            (ActivityType.SOCIAL, "Social visit", leisure, 375, 465, 0.7),
            (ActivityType.HOME, "Evening at home", home, 465, 720, 0.4),
        )
        work = None
    else:
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 45, 0.4),
            (ActivityType.COMMUTE, "Commute to work", office, 45, 75, 0.7),
            (ActivityType.WORK, "Morning work", office, 75, 195, 0.9),
            (ActivityType.EAT, "Lunch break", cafe, 195, 225, 0.55),
            (ActivityType.WORK, "Afternoon work", office, 225, 315, 0.9),
            (ActivityType.COMMUTE, "Commute home", home, 315, 345, 0.65),
            (ActivityType.LEISURE, "Evening outing", leisure, 345, 435, 0.55),
            (ActivityType.HOME, "Evening at home", home, 435, 720, 0.4),
        )
        work = office

    return AgentProfile(
        name=NAMES[index % len(NAMES)],
        occupation=occupation,
        home_position=home,
        work_position=work,
        leisure_position=leisure,
        social_preference=social_preference,
        transport_mode=transport_mode,
        routine=routine,
    )


def _vary_routine(
    routine: tuple[RoutineEntry, ...],
    rng: random.Random,
) -> tuple[RoutineEntry, ...]:
    """Add small deterministic timing variation to a daily routine.

    One simulation tick is two minutes, so offsets of 5, 8, or 15 ticks are
    roughly 10, 16, or 30 minutes. The first HOME block stays anchored while
    the rest of the day moves together, keeping the routine internally valid.
    """
    if len(routine) <= 1:
        return routine

    offset = rng.choice((-15, -8, -5, 0, 5, 8, 15))
    first_home = routine[0]
    home_end = max(
        first_home.start_tick + 1,
        min(720, first_home.end_tick + offset),
    )
    varied = [
        replace(first_home, end_tick=home_end),
    ]
    for entry in routine[1:]:
        start = max(0, min(720, entry.start_tick + offset))
        end = max(start + 1, min(720, entry.end_tick + offset))
        varied.append(replace(entry, start_tick=start, end_tick=end))
    return tuple(varied)


def create_interaction_scenario(seed: int = 84) -> BaselineScenario:
    """Create a tiny two-person village for interaction and memory work.

    This is intentionally a separate scenario from the 30-agent baseline.
    It uses a handful of individual buildings and shared public spaces so
    two agents can be observed closely as interaction mechanics are added.
    """
    world = World(width=100.0, height=80.0, tick_duration=1.0)
    landmarks = (
        Landmark("house_a", "Arun's House", Vector2(16.0, 18.0), 5.0),
        Landmark("house_b", "Maya's House", Vector2(24.0, 18.0), 5.0),
        Landmark("town_hall", "Town Hall", Vector2(70.0, 18.0), 6.0),
        Landmark("schoolhouse", "Schoolhouse", Vector2(68.0, 50.0), 6.0),
        Landmark("cafe", "Village Café", Vector2(45.0, 42.0), 5.0),
        Landmark("market", "Village Market", Vector2(25.0, 55.0), 6.0),
        Landmark("park", "Village Park", Vector2(75.0, 62.0), 8.0),
    )

    roads = (
        RoadSegment("village-main", "Main Street", Vector2(16, 18), Vector2(70, 18)),
        RoadSegment("school-lane", "School Lane", Vector2(70, 18), Vector2(68, 50)),
        RoadSegment("cafe-lane", "Café Lane", Vector2(45, 42), Vector2(68, 50)),
        RoadSegment("market-lane", "Market Lane", Vector2(25, 55), Vector2(45, 42)),
        RoadSegment("park-lane", "Park Lane", Vector2(45, 42), Vector2(75, 62)),
        RoadSegment("south-lane", "South Lane", Vector2(24, 18), Vector2(25, 55)),
    )

    simulation = Simulation.create(
        agent_count=2,
        seed=seed,
        world=world,
        roads=roads,
    )
    landmark_map = {landmark.landmark_id: landmark for landmark in landmarks}
    routine_rng = random.Random(seed + 20_000)
    agents = list(simulation.agents.values())

    # Keep the familiar office-worker/student routine types, but place them
    # in individual village buildings and give both a shared café destination
    # so a later interaction system has a natural meeting point.
    homes = (landmark_map["house_a"].position, landmark_map["house_b"].position)
    for index, agent in enumerate(agents):
        profile = _build_profile(
            index,
            homes[index],
            landmark_map["town_hall"].position,
            landmark_map["schoolhouse"].position,
            landmark_map["schoolhouse"].position,
            landmark_map["market"].position,
            landmark_map["park"].position,
            landmark_map["cafe"].position,
            landmark_map["park"].position,
            landmark_map["cafe"].position,
            TransportMode.WALK,
        )
        profile = replace(
            profile,
            leisure_position=landmark_map["cafe"].position,
            routine=tuple(
                replace(
                    entry,
                    target_position=(
                        landmark_map["cafe"].position
                        if entry.activity in {ActivityType.LEISURE, ActivityType.SOCIAL}
                        else entry.target_position
                    ),
                )
                for entry in _vary_routine(profile.routine, routine_rng)
            ),
        )
        agent.profile = profile
        agent.transport_mode = TransportMode.WALK
        agent.position = homes[index]
        first = profile.routine[0]
        agent.current_activity = first.activity
        agent.asleep = True
        agent.goal = Goal(
            goal_id=f"goal-{index + 1:04d}",
            description=first.description,
            target_position=first.target_position,
            importance=first.importance,
        )

    # The two residents know each other. The interaction mechanism can later
    # decide when proximity becomes an actual conversation or encounter.
    first_id, second_id = agents[0].agent_id, agents[1].agent_id
    agents[0].social_connections.add(second_id)
    agents[1].social_connections.add(first_id)
    agents[0].relationships[second_id] = 0.50
    agents[1].relationships[first_id] = 0.50

    return BaselineScenario(
        simulation=simulation,
        landmarks=landmarks,
        roads=roads,
    )


def create_baseline_scenario(
    agent_count: int = 30,
    seed: int = 42,
) -> BaselineScenario:
    """Create a deterministic but varied Smallville-style town baseline."""
    world = World(width=180.0, height=120.0, tick_duration=1.0)
    landmarks = (
        Landmark("homes_north", "North Homes", Vector2(25.0, 18.0), 9.0),
        Landmark("school", "School", Vector2(62.0, 16.0), 7.0),
        Landmark("hospital", "Hospital", Vector2(145.0, 18.0), 8.0),
        Landmark("office", "Office District", Vector2(108.0, 34.0), 9.0),
        Landmark("market", "Market", Vector2(30.0, 58.0), 8.0),
        Landmark("plaza", "Central Plaza", Vector2(88.0, 55.0), 9.0),
        Landmark("station", "Central Station", Vector2(145.0, 55.0), 8.0),
        Landmark("park", "Riverside Park", Vector2(45.0, 94.0), 10.0),
        Landmark("cafe", "Café Row", Vector2(82.0, 88.0), 7.0),
        Landmark("entertainment", "Entertainment", Vector2(128.0, 92.0), 9.0),
        Landmark("south_homes", "South Homes", Vector2(165.0, 100.0), 9.0),
        Landmark("bus_depot", "Bus Depot", Vector2(158.0, 76.0), 7.0),
    )

    roads = (
        RoadSegment("road-north-west", "North Avenue", Vector2(25, 18), Vector2(62, 16)),
        RoadSegment("road-north-east", "Hospital Avenue", Vector2(62, 16), Vector2(145, 18)),
        RoadSegment("road-central-east", "Station Road", Vector2(88, 55), Vector2(145, 55)),
        RoadSegment("road-south-west", "Park Road", Vector2(45, 94), Vector2(82, 88)),
        RoadSegment("road-park-market", "Park Market Street", Vector2(45, 94), Vector2(30, 58)),
        RoadSegment("road-south-east", "Entertainment Road", Vector2(82, 88), Vector2(128, 92)),
        RoadSegment("road-south-homes", "South Avenue", Vector2(128, 92), Vector2(165, 100)),
        RoadSegment("road-vertical-west", "West Connector", Vector2(25, 18), Vector2(30, 58)),
        RoadSegment("road-vertical-central", "Central Boulevard", Vector2(88, 55), Vector2(82, 88)),
        RoadSegment("road-vertical-east", "East Connector", Vector2(145, 18), Vector2(145, 55)),
        RoadSegment("road-office", "Office Connector", Vector2(108, 34), Vector2(88, 55)),
        RoadSegment("road-depot", "Depot Connector", Vector2(145, 55), Vector2(158, 76)),
    )

    simulation = Simulation.create(
        agent_count=agent_count,
        seed=seed,
        world=world,
        roads=roads,
    )

    landmark_map = {landmark.landmark_id: landmark for landmark in landmarks}
    home_centers = (
        landmark_map["homes_north"].position,
        landmark_map["south_homes"].position,
    )

    agents = list(simulation.agents.values())
    routine_rng = random.Random(seed + 20_000)
    for index, agent in enumerate(agents):
        # The baseline is deliberately pedestrian-only. Keep the transport
        # field for compatibility with the wider architecture, but every
        # baseline resident walks.
        transport_mode = TransportMode.WALK
        home_center = home_centers[index % len(home_centers)]
        home = Vector2(
            home_center.x + ((index * 7) % 13 - 6) * 0.45,
            home_center.y + ((index * 11) % 13 - 6) * 0.45,
        )
        profile = _build_profile(
            index,
            home,
            landmark_map["office"].position,
            landmark_map["school"].position,
            landmark_map["hospital"].position,
            landmark_map["market"].position,
            landmark_map["park"].position,
            landmark_map["cafe"].position,
            landmark_map["entertainment"].position,
            landmark_map["station"].position,
            transport_mode,
        )
        profile = replace(
            profile,
            routine=_vary_routine(profile.routine, routine_rng),
        )
        agent.profile = profile
        agent.transport_mode = TransportMode.WALK
        agent.position = home
        first_routine = profile.routine[0]
        agent.current_activity = first_routine.activity
        agent.asleep = True
        agent.relationships = {
            agents[(index + 1) % len(agents)].agent_id: 0.50
        } if len(agents) > 1 else {}
        agent.goal = Goal(
            goal_id=f"goal-{index + 1:04d}",
            description=first_routine.description,
            target_position=first_routine.target_position,
            importance=first_routine.importance,
        )

        if len(agents) > 1:
            agent.social_connections.add(
                agents[(index + 1) % len(agents)].agent_id
            )
        if len(agents) > 2:
            agent.social_connections.add(
                agents[(index + 2) % len(agents)].agent_id
            )

    # Random world events are intentionally disabled in the baseline. The
    # event system remains available as infrastructure for a later rebuild.
    simulation.state.events.clear()

    return BaselineScenario(
        simulation=simulation,
        landmarks=landmarks,
        roads=roads,
    )
