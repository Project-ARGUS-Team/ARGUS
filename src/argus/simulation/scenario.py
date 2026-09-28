"""Deterministic city-scale demonstration scenarios for the ARGUS baseline."""

from __future__ import annotations

from dataclasses import dataclass
import random

from argus.simulation.agent import (
    ActivityType,
    AgentProfile,
    Goal,
    RoutineEntry,
    Vector2,
)
from argus.simulation.events import WorldEvent
from argus.simulation.simulation import Simulation
from argus.simulation.world import World


@dataclass(frozen=True, slots=True)
class Landmark:
    """A named location used by a demonstration scenario."""

    landmark_id: str
    name: str
    position: Vector2
    radius: float = 6.0


@dataclass(frozen=True, slots=True)
class BaselineScenario:
    """A city-scale world containing locations, routines and events."""

    simulation: Simulation
    landmarks: tuple[Landmark, ...]


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
) -> AgentProfile:
    kind, occupation = OCCUPATION_TYPES[index % len(OCCUPATION_TYPES)]
    leisure = (park, cafe, entertainment)[index % 3]
    social_preference = (0.3, 0.55, 0.7, 0.45, 0.8)[index % 5]

    if kind == "student":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 55, 0.45),
            (ActivityType.COMMUTE, "Travel to school", school, 55, 70, 0.7),
            (ActivityType.STUDY, "Attend school", school, 70, 145, 0.9),
            (ActivityType.EAT, "Lunch break", cafe, 145, 165, 0.6),
            (ActivityType.STUDY, "Afternoon classes", school, 165, 185, 0.85),
            (ActivityType.COMMUTE, "Travel home", home, 185, 200, 0.65),
            (ActivityType.SOCIAL, "Meet friends", leisure, 200, 225, 0.65),
            (ActivityType.HOME, "Evening at home", home, 225, 240, 0.4),
        )
        work = school
    elif kind == "healthcare":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 50, 0.4),
            (ActivityType.COMMUTE, "Travel to hospital", hospital, 50, 70, 0.75),
            (ActivityType.WORK, "Hospital shift", hospital, 70, 150, 1.0),
            (ActivityType.EAT, "Lunch break", market, 150, 165, 0.55),
            (ActivityType.WORK, "Hospital shift", hospital, 165, 190, 1.0),
            (ActivityType.COMMUTE, "Travel home", home, 190, 210, 0.65),
            (ActivityType.LEISURE, "Evening leisure", leisure, 210, 228, 0.5),
            (ActivityType.HOME, "Evening at home", home, 228, 240, 0.4),
        )
        work = hospital
    elif kind == "retail":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 55, 0.4),
            (ActivityType.COMMUTE, "Travel to market", market, 55, 70, 0.7),
            (ActivityType.WORK, "Market shift", market, 70, 145, 0.9),
            (ActivityType.EAT, "Lunch break", cafe, 145, 165, 0.55),
            (ActivityType.WORK, "Market shift", market, 165, 195, 0.9),
            (ActivityType.COMMUTE, "Travel home", home, 195, 212, 0.65),
            (ActivityType.SOCIAL, "Meet people", leisure, 212, 230, 0.6),
            (ActivityType.HOME, "Evening at home", home, 230, 240, 0.4),
        )
        work = market
    elif kind == "retired":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 65, 0.4),
            (ActivityType.EAT, "Morning outing", cafe, 65, 85, 0.55),
            (ActivityType.LEISURE, "Walk in the park", park, 85, 120, 0.65),
            (ActivityType.SHOP, "Shopping", market, 120, 145, 0.65),
            (ActivityType.HOME, "Afternoon at home", home, 145, 180, 0.4),
            (ActivityType.SOCIAL, "Social visit", leisure, 180, 215, 0.7),
            (ActivityType.HOME, "Evening at home", home, 215, 240, 0.4),
        )
        work = None
    else:
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 55, 0.4),
            (ActivityType.COMMUTE, "Commute to work", office, 55, 72, 0.7),
            (ActivityType.WORK, "Workday", office, 72, 145, 0.9),
            (ActivityType.EAT, "Lunch break", cafe, 145, 165, 0.55),
            (ActivityType.WORK, "Afternoon work", office, 165, 190, 0.9),
            (ActivityType.COMMUTE, "Commute home", home, 190, 208, 0.65),
            (ActivityType.LEISURE, "Evening outing", leisure, 208, 230, 0.55),
            (ActivityType.HOME, "Evening at home", home, 230, 240, 0.4),
        )
        work = office

    return AgentProfile(
        name=NAMES[index % len(NAMES)],
        occupation=occupation,
        home_position=home,
        work_position=work,
        leisure_position=leisure,
        social_preference=social_preference,
        routine=routine,
    )


def create_baseline_scenario(
    agent_count: int = 60,
    seed: int = 42,
) -> BaselineScenario:
    """Create a deterministic but varied small-city baseline scenario."""
    world = World(width=180.0, height=120.0, tick_duration=1.0)
    simulation = Simulation.create(
        agent_count=agent_count,
        seed=seed,
        world=world,
    )

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

    landmark_map = {landmark.landmark_id: landmark for landmark in landmarks}
    home_centers = (
        landmark_map["homes_north"].position,
        landmark_map["south_homes"].position,
    )

    agents = list(simulation.agents.values())
    for index, agent in enumerate(agents):
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
        )
        agent.profile = profile
        agent.position = home
        first_routine = profile.routine[0]
        agent.current_activity = first_routine.activity
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

    rng = random.Random(seed + 10_000)
    event_types = (
        ("market_rush", "Market rush"),
        ("public_gathering", "Public gathering"),
        ("bus_delay", "Bus delay"),
        ("minor_accident", "Minor accident"),
        ("community_fair", "Community fair"),
        ("power_outage", "Power outage"),
        ("sports_event", "Sports event"),
        ("medical_alert", "Medical alert"),
        ("road_closure", "Road closure"),
        ("school_event", "School event"),
    )

    events: list[WorldEvent] = []
    for index, (event_type, _description) in enumerate(event_types):
        landmark = landmarks[rng.randrange(len(landmarks))]
        start_tick = rng.randint(20, 520)
        duration = rng.randint(12, 45)
        participant_count = (
            max(1, agent_count // rng.randint(5, 9))
            if agents
            else 0
        )
        participants = {
            agents[rng.randrange(len(agents))].agent_id
            for _ in range(participant_count)
        }
        events.append(
            WorldEvent(
                event_id=f"event-{index + 1:03d}",
                event_type=event_type,
                position=landmark.position,
                start_tick=start_tick,
                end_tick=start_tick + duration,
                participants=participants,
                importance=rng.uniform(0.35, 1.0),
            )
        )

    simulation.state.events.update(
        {event.event_id: event for event in events}
    )

    return BaselineScenario(
        simulation=simulation,
        landmarks=landmarks,
    )
