"""Small scripted policies that see only their own permitted observation.

Each policy returns an action ID. The environment owns the engine and executes
the action, so these policies cannot read the other player's cards or the deck.
"""

from dataclasses import dataclass

import numpy as np
from pokerkit import Deck, Label, StandardHighHand

from .actions import Action


CARDS = tuple(Deck.STANDARD)
RANKS = "23456789TJQKA"
OPPONENT_NAMES = ("random", "check_call", "tight", "aggressive")


def random_legal(mask, rng):
    legal = np.flatnonzero(mask)
    if not legal.size:
        raise ValueError("An opponent needs at least one legal action.")
    return int(rng.choice(legal))


def check_or_fold(mask):
    if mask[Action.CHECK_CALL]:
        return int(Action.CHECK_CALL)
    if mask[Action.FOLD]:
        return int(Action.FOLD)
    raise ValueError("Neither check/call nor fold is available.")


def hand_strength(observation):
    """A deliberately rough score, not an equity or win-probability estimate.

    Preflop: reward pairs, high cards, suited cards, and connected ranks.
    Postflop: use the best made-hand category, with no hidden-card sampling.
    A strong shared board can therefore fool this intentionally simple bot.
    """
    own = [CARDS[i] for i in np.flatnonzero(observation["own_cards"])]
    board = [CARDS[i] for i in np.flatnonzero(observation["board_cards"])]
    if len(own) != 2:
        raise ValueError("Expected two private cards at an opponent decision.")

    ranks = sorted(RANKS.index(card.rank) for card in own)
    if not board:
        if ranks[0] == ranks[1]:
            return 0.60 + 0.35 * ranks[0] / 12
        score = 0.15 + 0.35 * sum(ranks) / 24
        score += 0.08 if own[0].suit == own[1].suit else 0
        score += 0.07 if ranks[1] - ranks[0] == 1 else 0
        return score

    category = StandardHighHand.from_game(own, board).entry.label
    scores = dict(zip(Label, (0.20, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98, 1.0)))
    return scores[category]


class RandomOpponent:
    name = "random"

    def choose_action(self, observation, mask, rng):
        return random_legal(mask, rng)


@dataclass(frozen=True)
class CheckCallOpponent:
    # A value of 1 preserves the original environment's passive opponent.
    call_probability: float = 1.0
    name = "check_call"

    def __post_init__(self):
        if not 0 <= self.call_probability <= 1:
            raise ValueError("call_probability must be between zero and one.")

    def choose_action(self, observation, mask, rng):
        if rng.random() < self.call_probability:
            return check_or_fold(mask)
        return random_legal(mask, rng)


class TightOpponent:
    name = "tight"

    def choose_action(self, observation, mask, rng):
        strength = hand_strength(observation)
        call_bb = observation["table"][2]
        raises = np.flatnonzero(mask[Action.MIN_RAISE:]) + Action.MIN_RAISE

        if strength >= 0.75 and raises.size:
            return int(raises[0])
        if call_bb > 0 and strength < 0.45 and mask[Action.FOLD]:
            return int(Action.FOLD)
        return check_or_fold(mask)


@dataclass(frozen=True)
class AggressiveOpponent:
    raise_probability: float = 0.75
    name = "aggressive"

    def __post_init__(self):
        if not 0 <= self.raise_probability <= 1:
            raise ValueError("raise_probability must be between zero and one.")

    def choose_action(self, observation, mask, rng):
        raises = np.flatnonzero(mask[Action.MIN_RAISE:]) + Action.MIN_RAISE
        if raises.size and rng.random() < self.raise_probability:
            return int(rng.choice(raises))
        return check_or_fold(mask)


def make_opponent(name):
    """Construct one bot; mixture selection belongs to the hand wrapper."""
    factories = {
        "random": RandomOpponent,
        "check_call": CheckCallOpponent,
        "tight": TightOpponent,
        "aggressive": AggressiveOpponent,
    }
    if name not in factories:
        raise ValueError(f"Unknown opponent: {name!r}. Choose from {OPPONENT_NAMES}.")
    return factories[name]()
