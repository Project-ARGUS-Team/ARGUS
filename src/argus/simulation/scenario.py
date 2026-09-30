"""Deterministic city-scale demonstration scenarios for the ARGUS baseline."""

from __future__ import annotations

from dataclasses import dataclass
import random

from argus.simulation.agent import (
    ActivityType,
    AgentProfile,
    Goal,
    RoutineEntry,
    TransportMode,
    Vector2,
)
from argus.simulation.events import WorldEvent
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
    """A city-scale world containing locations, routines and events."""

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
) -> AgentProfile:
    kind, occupation = OCCUPATION_TYPES[index % len(OCCUPATION_TYPES)]
    leisure = (park, cafe, entertainment)[index % 3]
    social_preference = (0.3, 0.55, 0.7, 0.45, 0.8)[index % 5]

    if kind == "student":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 60, 0.45),
            (ActivityType.COMMUTE, "Travel to school", school, 60, 135, 0.7),
            (ActivityType.STUDY, "Attend school", school, 135, 375, 0.9),
            (ActivityType.EAT, "Lunch break", cafe, 375, 435, 0.6),
            (ActivityType.STUDY, "Afternoon classes", school, 435, 525, 0.85),
            (ActivityType.COMMUTE, "Travel home", home, 525, 585, 0.65),
            (ActivityType.SOCIAL, "Meet friends", leisure, 585, 660, 0.65),
            (ActivityType.HOME, "Evening at home", home, 660, 720, 0.4),
        )
        work = school
    elif kind == "healthcare":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 54, 0.4),
            (ActivityType.COMMUTE, "Travel to hospital", hospital, 54, 129, 0.75),
            (ActivityType.WORK, "Hospital shift", hospital, 129, 375, 1.0),
            (ActivityType.EAT, "Lunch break", market, 375, 435, 0.55),
            (ActivityType.WORK, "Hospital shift", hospital, 435, 525, 1.0),
            (ActivityType.COMMUTE, "Travel home", home, 525, 585, 0.65),
            (ActivityType.LEISURE, "Evening leisure", leisure, 585, 660, 0.5),
            (ActivityType.HOME, "Evening at home", home, 660, 720, 0.4),
        )
        work = hospital
    elif kind == "retail":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 165, 0.4),
            (ActivityType.COMMUTE, "Travel to market", market, 60, 135, 0.7),
            (ActivityType.WORK, "Market shift", market, 135, 375, 0.9),
            (ActivityType.EAT, "Lunch break", cafe, 435, 495, 0.55),
            (ActivityType.WORK, "Market shift", market, 495, 585, 0.9),
            (ActivityType.COMMUTE, "Travel home", home, 585, 636, 0.65),
            (ActivityType.SOCIAL, "Meet people", leisure, 636, 690, 0.6),
            (ActivityType.HOME, "Evening at home", home, 690, 720, 0.4),
        )
        work = market
    elif kind == "retired":
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 75, 0.4),
            (ActivityType.EAT, "Morning outing", cafe, 75, 150, 0.55),
            (ActivityType.LEISURE, "Walk in the park", park, 150, 285, 0.65),
            (ActivityType.SHOP, "Shopping", market, 285, 375, 0.65),
            (ActivityType.HOME, "Afternoon at home", home, 375, 495, 0.4),
            (ActivityType.SOCIAL, "Social visit", leisure, 495, 615, 0.7),
            (ActivityType.HOME, "Evening at home", home, 615, 720, 0.4),
        )
        work = None
    else:
        routine = _routine(
            (ActivityType.HOME, "Morning at home", home, 0, 165, 0.4),
            (ActivityType.COMMUTE, "Commute to work", office, 165, 216, 0.7),
            (ActivityType.WORK, "Workday", office, 216, 435, 0.9),
            (ActivityType.EAT, "Lunch break", cafe, 435, 495, 0.55),
            (ActivityType.WORK, "Afternoon work", office, 495, 570, 0.9),
            (ActivityType.COMMUTE, "Commute home", home, 570, 624, 0.65),
            (ActivityType.LEISURE, "Evening outing", leisure, 624, 690, 0.55),
            (ActivityType.HOME, "Evening at home", home, 690, 720, 0.4),
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
        RoadSegment("road-central-west", "Market Road", Vector2(25, 58), Vector2(88, 55)),
        RoadSegment("road-central-east", "Station Road", Vector2(88, 55), Vector2(145, 55)),
        RoadSegment("road-south-west", "Park Road", Vector2(45, 94), Vector2(82, 88)),
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
        agent.transport_mode = (
            TransportMode.CAR
            if index % 5 in (0, 1)
            else TransportMode.WALK
        )
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

        affected_road_ids: tuple[str, ...] = ()
        if event_type == "road_closure":
            road = roads[rng.randrange(len(roads))]
            position = Vector2(
                (road.start.x + road.end.x) / 2,
                (road.start.y + road.end.y) / 2,
            )
            affected_road_ids = (road.road_id,)
            participants: set[str] = set()
        else:
            position = landmark.position
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
                position=position,
                start_tick=start_tick,
                end_tick=start_tick + duration,
                participants=participants,
                importance=rng.uniform(0.35, 1.0),
                affected_road_ids=affected_road_ids,
            )
        )

    simulation.state.events.update(
        {event.event_id: event for event in events}
    )

    return BaselineScenario(
        simulation=simulation,
        landmarks=landmarks,
        roads=roads,
    )
