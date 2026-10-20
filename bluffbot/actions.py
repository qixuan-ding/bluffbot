"""Six discrete actions mapped to legal PokerKit raise-to amounts."""

from enum import IntEnum

import numpy as np


class Action(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    MIN_RAISE = 2
    HALF_POT = 3
    POT = 4
    ALL_IN = 5


NUM_ACTIONS = len(Action)


def action_options(state, player_index):
    """Return (legal mask, raise-to amounts), indexed by Action.

    Fold/check/call and unavailable actions have a raise-to value of None.
    Duplicate raise amounts keep only the lowest action ID.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    amounts = [None] * NUM_ACTIONS

    if (
        state is None
        or not state.status
        or state.actor_index != player_index
    ):
        return mask, tuple(amounts)

    mask[Action.FOLD] = state.can_fold()
    mask[Action.CHECK_CALL] = state.can_check_or_call()

    minimum = state.min_completion_betting_or_raising_to_amount
    maximum = state.max_completion_betting_or_raising_to_amount

    # These properties also account for whether raising is permitted.
    if minimum is None or maximum is None:
        return mask, tuple(amounts)

    call = state.checking_or_calling_amount or 0
    bet_after_call = state.bets[player_index] + call

    # total_pot_amount already includes current street bets.
    pot_after_call = state.total_pot_amount + call

    candidates = (
        (Action.MIN_RAISE, minimum),
        # Integer rounding: an odd-chip half-pot rounds upward.
        (Action.HALF_POT, bet_after_call + (pot_after_call + 1) // 2),
        (Action.POT, bet_after_call + pot_after_call),
        (Action.ALL_IN, maximum),
    )

    seen = set()

    for action, desired in candidates:
        amount = int(max(minimum, min(desired, maximum)))

        if amount in seen:
            continue

        if state.can_complete_bet_or_raise_to(amount):
            mask[action] = True
            amounts[action] = amount
            seen.add(amount)

    return mask, tuple(amounts)