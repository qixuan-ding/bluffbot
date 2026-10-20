"""Public observations with nine learner-relative seat slots."""

import gymnasium as gym
import numpy as np
from pokerkit import (
    BetCollection,
    BlindOrStraddlePosting,
    BoardDealing,
    CheckingOrCalling,
    CompletionBettingOrRaisingTo,
    Deck,
    Folding,
)

from .actions import NUM_ACTIONS, action_options


MAX_SEATS = 9
HISTORY_LENGTH = 16
CARD_INDEX = {card: index for index, card in enumerate(Deck.STANDARD)}

TABLE_FEATURES = (
    "street_fraction",
    "pot_bb",
    "call_bb",
    "minimum_raise_to_bb",
    "maximum_raise_to_bb",
    "player_count_fraction",
    "remaining_count_fraction",
    "learner_is_button",
)

SEAT_FEATURES = (
    "stack_bb",
    "street_bet_bb",
    "committed_bb",
    "in_hand",
    "folded",
    "all_in",
    "is_button",
    "is_small_blind",
    "is_big_blind",
)

# History row:
# present flag | actor one-hot (9) | street one-hot (4) |
# kind one-hot (5) | chips added in BB | raise-to total in BB
HISTORY_KINDS = ("blind", "fold", "check", "call", "raise")
HISTORY_WIDTH = 1 + MAX_SEATS + 4 + len(HISTORY_KINDS) + 2

OBSERVATION_SHAPES = {
    "own_cards": (52,),
    "board_cards": (52,),
    "table": (len(TABLE_FEATURES),),
    "seat_mask": (MAX_SEATS,),
    "seats": (MAX_SEATS, len(SEAT_FEATURES)),
    "history": (HISTORY_LENGTH, HISTORY_WIDTH),
    "legal_actions": (NUM_ACTIONS,),
}


def observation_space():
    binary = {"own_cards", "board_cards", "seat_mask", "legal_actions"}

    return gym.spaces.Dict({
        key: gym.spaces.Box(
            0, 1 if key in binary else np.inf, shape, dtype=np.float32
        )
        for key, shape in OBSERVATION_SHAPES.items()
    })


def _public_records(state):
    """Read public operations; never inspect private dealing records."""
    player_count = state.player_count
    street_bets = [0] * player_count
    collected = np.zeros(player_count, dtype=np.float32)
    folded = np.zeros(player_count, dtype=bool)
    records = []
    street = 0
    board_count = 0

    for operation in state.operations:
        if isinstance(operation, BoardDealing):
            board_count += len(operation.cards)
            street = 1 if board_count == 3 else 2 if board_count == 4 else 3
            continue

        if isinstance(operation, BetCollection):
            collected += np.asarray(operation.bets, dtype=np.float32)
            street_bets = [0] * player_count
            continue

        if isinstance(operation, BlindOrStraddlePosting):
            player = operation.player_index
            added = operation.amount
            street_bets[player] += added
            kind = 0
            raise_to = 0

        elif isinstance(operation, Folding):
            player = operation.player_index
            folded[player] = True
            kind, added, raise_to = 1, 0, 0

        elif isinstance(operation, CheckingOrCalling):
            player = operation.player_index
            added = operation.amount
            street_bets[player] += added
            kind = 2 if added == 0 else 3
            raise_to = 0

        elif isinstance(operation, CompletionBettingOrRaisingTo):
            player = operation.player_index
            raise_to = operation.amount
            added = raise_to - street_bets[player]
            street_bets[player] = raise_to
            kind = 4

        else:
            continue

        records.append((player, street, kind, added, raise_to))

    # Collected contributions plus outstanding bets; exclude refunded excess.
    committed = collected + np.asarray(state.bets, dtype=np.float32)

    return records, folded, committed


def encode_observation(state, learner_index, big_blind):
    player_count = state.player_count

    if not 2 <= player_count <= MAX_SEATS:
        raise ValueError("Observations support 2–9 occupied seats.")

    # PokerKit's current heads-up layout: index 1 = button/SB, index 0 = BB.
    # For a later multiplayer wrapper, the button is the final engine index.
    button = player_count - 1
    small_blind = button if player_count == 2 else 0
    big_blind_index = 0 if player_count == 2 else 1

    observation = {
        key: np.zeros(shape, dtype=np.float32)
        for key, shape in OBSERVATION_SHAPES.items()
    }

    for card in state.hole_cards[learner_index]:
        observation["own_cards"][CARD_INDEX[card]] = 1

    for group in state.board_cards:
        for card in group:
            observation["board_cards"][CARD_INDEX[card]] = 1

    records, folded, committed = _public_records(state)

    # Slot 0 is always the learner; subsequent slots follow seat order.
    for player in range(player_count):
        slot = (player - learner_index) % player_count
        active = not folded[player]

        observation["seat_mask"][slot] = 1
        observation["seats"][slot] = (
            state.stacks[player] / big_blind,
            state.bets[player] / big_blind,
            committed[player] / big_blind,
            float(active),
            float(folded[player]),
            float(active and state.stacks[player] == 0),
            float(player == button),
            float(player == small_blind),
            float(player == big_blind_index),
        )

    learner_turn = (
        state.status and state.actor_index == learner_index
    )

    call = (state.checking_or_calling_amount or 0) if learner_turn else 0

    minimum = (
        state.min_completion_betting_or_raising_to_amount
        if learner_turn else None
    )

    maximum = (
        state.max_completion_betting_or_raising_to_amount
        if learner_turn else None
    )

    board_count = int(observation["board_cards"].sum())
    street = 0 if board_count < 3 else min(board_count - 2, 3)

    observation["table"][:] = (
        street / 3,
        state.total_pot_amount / big_blind,
        call / big_blind,
        (minimum or 0) / big_blind,
        (maximum or 0) / big_blind,
        player_count / MAX_SEATS,
        int(np.count_nonzero(~folded)) / MAX_SEATS,
        float(learner_index == button),
    )

    # Oldest to newest; unused rows at the beginning remain zero.
    recent = records[-HISTORY_LENGTH:]

    for row_index, record in enumerate(
        recent, start=HISTORY_LENGTH - len(recent)
    ):
        player, street, kind, added, raise_to = record
        slot = (player - learner_index) % player_count

        row = observation["history"][row_index]
        row[0] = 1
        row[1 + slot] = 1
        row[1 + MAX_SEATS + street] = 1
        row[1 + MAX_SEATS + 4 + kind] = 1
        row[-2] = added / big_blind
        row[-1] = raise_to / big_blind

    mask, _ = action_options(state, learner_index)
    observation["legal_actions"][:] = mask

    return observation