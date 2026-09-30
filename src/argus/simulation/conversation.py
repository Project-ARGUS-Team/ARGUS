"""Small deterministic conversation library for the ARGUS demo."""

from __future__ import annotations

import random

from argus.simulation.agent import ConversationRecord


_CONVERSATIONS: tuple[ConversationRecord, ...] = (
    ConversationRecord(
        topic="work and study",
        opening="How has work been going lately?",
        follow_up="Anything interesting happening there?",
    ),
    ConversationRecord(
        topic="weekend plans",
        opening="Do you have any plans for the weekend?",
        follow_up="Are you going anywhere, or keeping it quiet?",
    ),
    ConversationRecord(
        topic="food",
        opening="Have you tried anything good to eat around here?",
        follow_up="What do you usually order?",
    ),
    ConversationRecord(
        topic="the village",
        opening="Have you noticed anything new around the village?",
        follow_up="What do you think of the changes?",
    ),
    ConversationRecord(
        topic="hobbies",
        opening="What have you been doing in your free time?",
        follow_up="How did you get into that?",
    ),
    ConversationRecord(
        topic="commute",
        opening="How was your trip over here today?",
        follow_up="Is your usual route still the easiest one?",
    ),
    ConversationRecord(
        topic="local news",
        opening="Have you heard any news from around town?",
        follow_up="Do you know what happened?",
    ),
    ConversationRecord(
        topic="plans for the evening",
        opening="What are you doing after this?",
        follow_up="Sounds nice. Do you go there often?",
    ),
)


def choose_conversation(rng: random.Random) -> ConversationRecord:
    """Choose one general topic for a new interaction."""
    return rng.choice(_CONVERSATIONS)
