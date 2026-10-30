"""Action-menu legality, exact raise sizes, and duplicate sizing regressions."""

import numpy as np
import pytest

from bluffbot.actions import Action, action_options


@pytest.mark.parametrize(
    "small_blind_stack, expected",
    [
        (200, (None, None, 4, None, 6, 200)),
        (5, (None, None, 4, None, 5, None)),
        (3, (None, None, 3, None, None, None)),
    ],
)
def test_raise_candidates_clip_to_stack_and_remove_duplicates(
    small_blind_stack, expected, make_betting_state
):
    state = make_betting_state((200, small_blind_stack))
    mask, amounts = action_options(state, 1)

    assert amounts == expected
    assert mask.dtype == np.bool_
    assert mask[:2].tolist() == [True, True]

    raises = [amount for amount in amounts if amount is not None]
    assert len(raises) == len(set(raises))
    assert mask[2:].tolist() == [amount is not None for amount in amounts[2:]]


def test_raise_sizes_use_pot_after_calling(make_betting_state):
    state = make_betting_state()
    state.complete_bet_or_raise_to(10)

    # BB has bet 2, must call 8, and the current pot is 12.
    # The half-pot raise-to is 2 + 8 + (12 + 8) / 2 = 20.
    mask, amounts = action_options(state, 0)

    assert state.actor_index == 0
    assert state.total_pot_amount == 12
    assert state.checking_or_calling_amount == 8
    assert mask.all()
    assert amounts == (None, None, 18, 20, 30, 200)


def test_half_pot_rounds_an_odd_chip_upward(make_betting_state):
    state = make_betting_state((200, 200, 200))
    state.check_or_call()
    state.fold()
    state.check_or_call()

    # The folded small blind leaves a 5-chip pot on the flop.
    mask, amounts = action_options(state, state.actor_index)

    assert state.total_pot_amount == 5
    assert mask[Action.HALF_POT]
    assert amounts[Action.MIN_RAISE] == 2
    assert amounts[Action.HALF_POT] == 3


def test_all_in_call_is_available_only_as_check_call(make_betting_state):
    state = make_betting_state((6, 200))
    state.complete_bet_or_raise_to(10)

    mask, amounts = action_options(state, 0)

    assert state.checking_or_calling_amount == 4
    assert mask.tolist() == [True, True, False, False, False, False]
    assert amounts == (None,) * 6


def test_short_all_in_does_not_reopen_raising(make_betting_state):
    state = make_betting_state((12, 200, 200))
    state.complete_bet_or_raise_to(10)
    state.complete_bet_or_raise_to(12)
    state.check_or_call()

    # Player 2 already raised; the extra 2 is less than a full raise of 8.
    assert state.actor_index == 2
    mask, amounts = action_options(state, 2)

    assert mask.tolist() == [True, True, False, False, False, False]
    assert amounts == (None,) * 6


def test_full_raise_reopens_raising(make_betting_state):
    state = make_betting_state((200, 200, 200))
    state.complete_bet_or_raise_to(10)
    state.complete_bet_or_raise_to(18)
    state.check_or_call()

    assert state.actor_index == 2
    mask, amounts = action_options(state, 2)

    assert mask[Action.MIN_RAISE]
    assert amounts[Action.MIN_RAISE] == 26
