"""Regression tests for the current heads-up, check/call-opponent wrapper."""

import numpy as np
import pytest
from pokerkit import (
    Automation,
    Card,
    CheckingOrCalling,
    CompletionBettingOrRaisingTo,
    Deck,
    NoLimitTexasHoldem,
)

from bluffbot.actions import Action, action_options
from bluffbot.env import HeadsUpEnv
from bluffbot.observations import (
    CARD_INDEX,
    HISTORY_LENGTH,
    encode_observation,
    observation_space,
)


# Current action IDs. Update these when the six-action menu is introduced.
FOLD = 0
CHECK_CALL = 1
MIN_RAISE = 2
ALL_IN = 5


@pytest.fixture(params=[1, 0], ids=["button", "big_blind"])
def env(request):
    environment = HeadsUpEnv()
    environment.reset(seed=42)

    # Seed once: passing a seed again also resets the position rotation.
    if request.param == 0:
        environment.reset()

    yield environment
    environment.close()


def assert_chips_conserved(env):
    # PokerKit's total_pot_amount already includes outstanding street bets.
    assert sum(env.state.stacks) + env.state.total_pot_amount == 400


def assert_settled(env, result):
    observation, reward, terminated, truncated, info = result

    assert terminated
    assert not truncated
    assert not env.state.status
    assert env.state.total_pot_amount == 0
    assert sum(info["final_stacks"]) == 400
    assert info["final_stacks"] == tuple(env.state.stacks)
    assert reward == (info["final_stacks"][env.learner_index] - 200) / 2
    assert info["profit_bb"] == reward
    assert not env.action_masks().any()
    assert env.observation_space.contains(observation)
    assert_chips_conserved(env)


def test_reset_posts_blinds_and_waits_for_learner(env):
    state = env.state

    assert state.status
    assert state.player_count == 2
    assert state.actor_index == env.learner_index
    assert state.street_index == 0
    assert len(state.hole_cards[0]) == len(state.hole_cards[1]) == 2
    assert not state.board_cards
    assert env.decisions == 0

    if env.learner_index == 1:
        # Button/small blind acts before the opponent.
        assert tuple(state.stacks) == (198, 199)
        assert tuple(state.bets) == (2, 1)
        assert state.checking_or_calling_amount == 1
    else:
        # The scripted small blind has already called automatically.
        assert tuple(state.stacks) == (198, 198)
        assert tuple(state.bets) == (2, 2)
        assert state.checking_or_calling_amount == 0

    assert_chips_conserved(env)


def test_learner_position_alternates_between_hands():
    environment = HeadsUpEnv()
    positions = []

    for hand in range(6):
        _, info = environment.reset(seed=42 if hand == 0 else None)
        positions.append(info["learner_index"])
        assert info["learner_is_button"] == (hand % 2 == 0)
        assert_settled(environment, environment.step(ALL_IN))

    assert positions == [1, 0, 1, 0, 1, 0]
    environment.close()


def test_small_blind_fold_settles_without_another_decision():
    environment = HeadsUpEnv()
    environment.reset(seed=42)

    result = environment.step(FOLD)
    assert_settled(environment, result)

    _, reward, _, _, info = result
    assert reward == -0.5
    assert info["final_stacks"] == (201, 199)
    assert info["learner_decisions"] == 1
    environment.close()


def test_check_call_reaches_showdown_in_correct_action_order(env):
    for street, board_size in enumerate((0, 3, 4, 5)):
        assert env.state.actor_index == env.learner_index
        assert env.state.street_index == street
        assert sum(len(group) for group in env.state.board_cards) == board_size

        result = env.step(CHECK_CALL)
        _, reward, terminated, truncated, info = result

        assert not truncated
        assert info["learner_decisions"] == street + 1
        assert_chips_conserved(env)

        if street < 3:
            assert not terminated
            assert reward == 0.0

    assert_settled(env, result)

    # Each player commits 2 chips; the winner receives the 4-chip pot.
    assert tuple(sorted(info["final_stacks"])) in ((198, 202), (200, 200))
    assert reward in (-1.0, 0.0, 1.0)

    actors = [
        operation.player_index
        for operation in env.state.operations
        if isinstance(operation, CheckingOrCalling)
    ]
    # Button/SB first preflop; BB first on each later street.
    assert actors == [1, 0, 0, 1, 0, 1, 0, 1]


def test_preflop_all_in_runs_out_board_and_settles(env):
    result = env.step(ALL_IN)
    assert_settled(env, result)

    _, reward, _, _, info = result
    assert sum(len(group) for group in env.state.board_cards) == 5
    assert tuple(sorted(info["final_stacks"])) in ((0, 400), (200, 200))
    assert reward in (-100.0, 0.0, 100.0)
    assert info["learner_decisions"] == 1


def test_minimum_raises_preserve_chips_through_every_street(env):
    # Preflop raise-to is 4; each later opening bet is 2.
    for street, expected_pot in enumerate((8, 12, 16, 0)):
        result = env.step(MIN_RAISE)
        _, reward, terminated, truncated, info = result

        assert not truncated
        assert env.state.total_pot_amount == expected_pot
        assert_chips_conserved(env)

        if street < 3:
            assert not terminated
            assert reward == 0.0
            assert tuple(env.state.stacks) == (
                200 - expected_pot // 2,
                200 - expected_pot // 2,
            )
            assert env.state.actor_index == env.learner_index

    assert_settled(env, result)
    assert tuple(sorted(info["final_stacks"])) in ((190, 210), (200, 200))
    assert reward in (-5.0, 0.0, 5.0)


def test_reset_after_finished_hand_restores_stacks_and_counters(env):
    previous_index = env.learner_index
    assert_settled(env, env.step(ALL_IN))

    observation, info = env.reset()

    assert env.state.status
    assert env.learner_index == 1 - previous_index
    assert info["learner_decisions"] == 0
    assert "final_stacks" not in info
    assert "profit_bb" not in info
    assert env.observation_space.contains(observation)

    expected_stacks = (198, 199) if env.learner_index == 1 else (198, 198)
    assert tuple(env.state.stacks) == expected_stacks
    assert_chips_conserved(env)


def test_step_requires_reset_before_first_hand():
    environment = HeadsUpEnv()
    with pytest.raises(RuntimeError, match="reset"):
        environment.step(CHECK_CALL)
    environment.close()


def test_step_requires_reset_after_finished_hand(env):
    assert_settled(env, env.step(ALL_IN))

    with pytest.raises(RuntimeError, match="reset"):
        env.step(CHECK_CALL)


def test_one_thousand_random_legal_hands():
    environment = HeadsUpEnv()
    action_rng = np.random.default_rng(123)
    seat_counts = {0: 0, 1: 0}

    for hand in range(1_000):
        observation, info = environment.reset(
            seed=123 if hand == 0 else None
        )
        seat_counts[info["learner_index"]] += 1

        # Bound the loop so a failure to terminate cannot hang the test.
        for decision in range(1, 101):
            assert environment.observation_space.contains(observation)
            assert environment.state.actor_index == environment.learner_index
            assert_chips_conserved(environment)

            legal_actions = np.flatnonzero(environment.action_masks())
            assert legal_actions.size > 0
            action = int(action_rng.choice(legal_actions))

            result = environment.step(action)
            observation, reward, terminated, truncated, info = result

            assert not truncated
            assert environment.observation_space.contains(observation)
            assert info["learner_decisions"] == decision
            assert_chips_conserved(environment)

            if terminated:
                assert_settled(environment, result)
                break

            assert reward == 0.0
        else:
            pytest.fail(f"Hand {hand} did not terminate within 100 decisions")

    assert seat_counts == {0: 500, 1: 500}
    environment.close()

# These checks cover the six-action menu and public observation contract.


def make_betting_state(stacks=(200, 200)):
    """Create a PokerKit hand paused at the first betting decision."""
    return NoLimitTexasHoldem.create_state(
        tuple(Automation), False, 0, (1, 2), 2, stacks, len(stacks)
    )


class FixedDeckEnv(HeadsUpEnv):
    """Use known cards so payout tests do not depend on random outcomes."""

    def __init__(self, card_order):
        super().__init__()
        self.card_order = list(Card.parse(card_order))
        assert len(set(self.card_order)) == len(self.card_order)

    def _advance(self):
        if self.decisions == 0 and not any(self.state.hole_cards):
            remaining = [
                card for card in Deck.STANDARD if card not in self.card_order
            ]
            self.state.deck_cards.clear()
            self.state.deck_cards.extend(self.card_order + remaining)

        super()._advance()


@pytest.mark.parametrize(
    "small_blind_stack, expected",
    [
        (200, (None, None, 4, None, 6, 200)),
        (5, (None, None, 4, None, 5, None)),
        (3, (None, None, 3, None, None, None)),
    ],
)
def test_raise_candidates_clip_to_stack_and_remove_duplicates(
    small_blind_stack, expected
):
    state = make_betting_state((200, small_blind_stack))
    mask, amounts = action_options(state, 1)

    assert amounts == expected
    assert mask.dtype == np.bool_
    assert mask[:2].tolist() == [True, True]

    raises = [amount for amount in amounts if amount is not None]
    assert len(raises) == len(set(raises))
    assert mask[2:].tolist() == [amount is not None for amount in amounts[2:]]


def test_raise_sizes_use_pot_after_calling():
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


def test_half_pot_rounds_an_odd_chip_upward():
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


def test_all_in_call_is_available_only_as_check_call():
    state = make_betting_state((6, 200))
    state.complete_bet_or_raise_to(10)

    mask, amounts = action_options(state, 0)

    assert state.checking_or_calling_amount == 4
    assert mask.tolist() == [True, True, False, False, False, False]
    assert amounts == (None,) * 6


def test_short_all_in_does_not_reopen_raising():
    state = make_betting_state((12, 200, 200))
    state.complete_bet_or_raise_to(10)
    state.complete_bet_or_raise_to(12)
    state.check_or_call()

    # Player 2 already raised; the extra 2 is less than a full raise of 8.
    assert state.actor_index == 2
    mask, amounts = action_options(state, 2)

    assert mask.tolist() == [True, True, False, False, False, False]
    assert amounts == (None,) * 6


def test_full_raise_reopens_raising():
    state = make_betting_state((200, 200, 200))
    state.complete_bet_or_raise_to(10)
    state.complete_bet_or_raise_to(18)
    state.check_or_call()

    assert state.actor_index == 2
    mask, amounts = action_options(state, 2)

    assert mask[Action.MIN_RAISE]
    assert amounts[Action.MIN_RAISE] == 26


def test_duplicate_action_is_rejected_without_changing_hand(env):
    operations_before = len(env.state.operations)
    stacks_before = tuple(env.state.stacks)

    with pytest.raises(ValueError, match="duplicate"):
        env.step(Action.HALF_POT)

    assert len(env.state.operations) == operations_before
    assert tuple(env.state.stacks) == stacks_before
    assert env.decisions == 0


@pytest.mark.parametrize(
    "action, expected_raise_to",
    [
        (Action.MIN_RAISE, 2),
        (Action.HALF_POT, 6),
        (Action.POT, 12),
        (Action.ALL_IN, 194),
    ],
)
def test_environment_executes_each_mapped_raise(env, action, expected_raise_to):
    env.step(Action.POT)
    result = env.step(action)

    raises = [
        operation
        for operation in env.state.operations
        if isinstance(operation, CompletionBettingOrRaisingTo)
    ]

    assert raises[-1].amount == expected_raise_to
    assert_chips_conserved(env)

    if result[2]:
        assert_settled(env, result)
    else:
        assert result[1] == 0.0


@pytest.mark.parametrize("raise_to", [10, 200])
def test_folding_to_opponent_raise_returns_unmatched_chips(raise_to):
    state = make_betting_state()
    state.complete_bet_or_raise_to(raise_to)

    environment = HeadsUpEnv()
    environment.state = state
    environment.learner_index = 0

    result = environment.step(Action.FOLD)
    assert_settled(environment, result)

    assert result[1] == -1.0
    assert result[4]["final_stacks"] == (198, 202)
    environment.close()


@pytest.mark.parametrize("learner_index", [1, 0], ids=["button", "big_blind"])
@pytest.mark.parametrize("winner_index", [0, 1], ids=["aces_in_seat_0", "aces_in_seat_1"])
@pytest.mark.parametrize("action", [Action.CHECK_CALL, Action.ALL_IN])
def test_known_winner_receives_correct_payout(learner_index, winner_index, action):
    holes = "Ac Kc Ad Kd" if winner_index == 0 else "Kc Ac Kd Ad"
    # Hole cards, burn, flop, burn, turn, burn, river.
    environment = FixedDeckEnv(holes + " 2h 3c 4d 8h 5s 9s 6s Jh")
    environment.reset(seed=42)
    if learner_index == 0:
        environment.reset()

    assert set(environment.state.hole_cards[winner_index]) == set(Card.parse("Ac Ad"))

    for _ in range(10):
        result = environment.step(action)
        if result[2]:
            break
        assert result[1] == 0.0
    else:
        pytest.fail("Known-winner hand did not finish.")

    assert_settled(environment, result)

    if action == Action.ALL_IN:
        expected = (400, 0) if winner_index == 0 else (0, 400)
        profit = 100.0
    else:
        expected = (202, 198) if winner_index == 0 else (198, 202)
        profit = 1.0

    assert result[4]["final_stacks"] == expected
    assert result[1] == (profit if learner_index == winner_index else -profit)
    environment.close()


@pytest.mark.parametrize("learner_index", [1, 0], ids=["button", "big_blind"])
@pytest.mark.parametrize("action", [Action.CHECK_CALL, Action.ALL_IN])
def test_royal_flush_on_board_splits_pot(learner_index, action):
    # Both players must use the same five spades on the board.
    environment = FixedDeckEnv("2c 3c 2d 3d 4c As Ks Qs 4d Js 4h Ts")
    environment.reset(seed=42)
    if learner_index == 0:
        environment.reset()

    for _ in range(10):
        result = environment.step(action)
        if result[2]:
            break
    else:
        pytest.fail("Tied hand did not finish.")

    assert_settled(environment, result)
    assert result[4]["final_stacks"] == (200, 200)
    assert result[1] == 0.0
    environment.close()


def test_observation_reserves_nine_learner_relative_seats(env):
    observation = env._observation()

    assert env.observation_space.contains(observation)
    assert observation["own_cards"].shape == (52,)
    assert observation["board_cards"].shape == (52,)
    assert observation["seats"].shape == (9, 9)
    assert observation["seat_mask"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert not observation["seats"][2:].any()

    # Slot 0 is the learner, even when the engine seat index changes.
    assert observation["seats"][0, 0] == env.state.stacks[env.learner_index] / 2
    assert observation["seats"][0, 6] == float(env.learner_index == 1)
    assert observation["seats"][1, 6] == float(env.learner_index == 0)
    assert observation["seats"][:2, 3].tolist() == [1, 1]
    assert not observation["seats"][:2, 4:6].any()
    np.testing.assert_array_equal(observation["legal_actions"], env.action_masks())


def test_observation_separates_street_bets_from_total_contributions(env):
    observation, reward, done, _, _ = env.step(Action.POT)

    assert not done
    assert reward == 0
    assert observation["board_cards"].sum() == 3
    np.testing.assert_allclose(observation["seats"][:2, 0], [97, 97])
    np.testing.assert_array_equal(observation["seats"][:2, 1], [0, 0])
    np.testing.assert_array_equal(observation["seats"][:2, 2], [3, 3])
    assert observation["table"][1] == 6


def test_hidden_cards_and_future_deck_do_not_change_observation(env):
    before = env._observation()
    opponent = 1 - env.learner_index

    old_cards = list(env.state.hole_cards[opponent])
    new_cards = [env.state.deck_cards.popleft() for _ in range(2)]
    env.state.hole_cards[opponent][:] = new_cards
    env.state.deck_cards.extend(old_cards)
    env.state.deck_cards.reverse()

    after = env._observation()

    for field in before:
        np.testing.assert_array_equal(before[field], after[field])

    expected_own_cards = np.zeros(52, dtype=np.float32)
    for card in env.state.hole_cards[env.learner_index]:
        expected_own_cards[CARD_INDEX[card]] = 1
    np.testing.assert_array_equal(after["own_cards"], expected_own_cards)


def test_public_history_preserves_raise_total_and_chips_added(env):
    starting_bet = env.state.bets[env.learner_index]
    observation, _, _, _, _ = env.step(Action.POT)

    # History columns: present, actor[9], street[4], kind[5], added, raise-to.
    history = observation["history"]
    raise_rows = history[history[:, 18] == 1]
    assert len(raise_rows) == 1

    row = raise_rows[0]
    assert row[0] == 1
    assert row[1] == 1
    assert row[10] == 1
    assert row[-2] == (6 - starting_bet) / 2
    assert row[-1] == 3


def test_public_history_keeps_only_latest_sixteen_actions():
    state = make_betting_state()

    for _ in range(20):
        state.complete_bet_or_raise_to()

    observation = encode_observation(state, state.actor_index, 2)
    history = observation["history"]

    assert observation_space().contains(observation)
    assert history.shape == (HISTORY_LENGTH, 21)
    assert history[:, 0].sum() == HISTORY_LENGTH == 16
    assert history[:, 18].tolist() == [1] * 16
    assert history[-1, -1] == 21


def test_folded_seat_is_distinct_from_unused_seat():
    environment = HeadsUpEnv()
    environment.reset(seed=42)
    observation, _, _, _, _ = environment.step(Action.FOLD)

    assert observation["seat_mask"][0] == 1
    assert observation["seats"][0, 3] == 0
    assert observation["seats"][0, 4] == 1
    assert observation["seat_mask"][2] == 0
    assert not observation["seats"][2].any()
    environment.close()
