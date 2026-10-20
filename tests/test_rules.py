"""Regression tests for the current heads-up, check/call-opponent wrapper."""

import numpy as np
import pytest
from pokerkit import CheckingOrCalling

from bluffbot.env import HeadsUpEnv


# Current action IDs. Update these when the six-action menu is introduced.
FOLD = 0
CHECK_CALL = 1
MIN_RAISE = 2
ALL_IN = 3


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