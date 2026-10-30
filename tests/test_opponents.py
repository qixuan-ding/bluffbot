"""Opponent integration, reproducibility, and hands with no learner decision."""

import numpy as np
import pytest
from pokerkit import Card, CompletionBettingOrRaisingTo

from bluffbot.actions import Action
from bluffbot.env import HeadsUpEnv
from bluffbot.observations import CARD_INDEX
from bluffbot.opponents import (
    AggressiveOpponent,
    CheckCallOpponent,
    OPPONENT_NAMES,
    TightOpponent,
    hand_strength,
)


class FoldOpponent:
    def choose_action(self, observation, mask, rng):
        return Action.FOLD if mask[Action.FOLD] else Action.CHECK_CALL


class AllInOpponent:
    def choose_action(self, observation, mask, rng):
        raises = np.flatnonzero(mask)[np.flatnonzero(mask) >= Action.MIN_RAISE]
        return int(raises[-1]) if raises.size else Action.CHECK_CALL


def play_hand(env, rng):
    actions = []
    for _ in range(200):
        mask = env.action_masks()
        assert mask.any()
        np.testing.assert_array_equal(env._observation()["legal_actions"], mask)
        assert env.observation_space.contains(env._observation())
        action = int(rng.choice(np.flatnonzero(mask)))
        actions.append(action)
        obs, reward, done, truncated, info = env.step(action)
        assert not truncated
        assert sum(env.state.stacks) + env.state.total_pot_amount == 400
        if done:
            assert sum(info["final_stacks"]) == 400
            assert env.state.total_pot_amount == 0
            assert reward == (env.state.stacks[env.learner_index] - 200) / 2
            assert not env.action_masks().any()
            assert not obs["legal_actions"].any()
            return actions, reward, info
        assert reward == 0
    pytest.fail("Opponent hand failed to terminate.")


@pytest.mark.parametrize("opponent", (*OPPONENT_NAMES, "mixture"))
def test_scripted_opponents_complete_random_hands_in_both_positions(opponent):
    env = HeadsUpEnv(opponent=opponent)
    rng = np.random.default_rng(27)
    positions = []
    for hand in range(200):
        env.reset(seed=42 if hand == 0 else None)
        positions.append(env.learner_index)
        selected = env.opponent
        play_hand(env, rng)
        assert env.opponent is selected
    assert positions.count(0) == positions.count(1) == 100


def test_opponent_preflop_fold_returns_reward_exactly_once():
    env = HeadsUpEnv(opponent=FoldOpponent())
    env.reset(seed=42)
    env.step(Action.FOLD)
    obs, info = env.reset()

    assert env.learner_index == 0
    assert not env.state.status
    assert info["awaiting_terminal_acknowledgement"]
    assert info["learner_decisions"] == 0
    assert env.action_masks().tolist() == [False, True, False, False, False, False]
    np.testing.assert_array_equal(obs["legal_actions"], env.action_masks())
    operations = len(env.state.operations)
    with pytest.raises(ValueError, match="Illegal"):
        env.step(Action.FOLD)

    obs, reward, done, truncated, info = env.step(Action.CHECK_CALL)
    assert (reward, done, truncated) == (0.5, True, False)
    assert info["final_stacks"] == (201, 199)
    assert info["learner_decisions"] == 0
    assert not info["learner_acted"]
    assert not info["awaiting_terminal_acknowledgement"]
    assert len(env.state.operations) == operations
    assert not obs["legal_actions"].any()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(Action.CHECK_CALL)


def test_opponent_all_in_uses_shared_raise_amount_and_settles():
    env = HeadsUpEnv(opponent=AllInOpponent())
    env.reset(seed=42)
    env.step(Action.FOLD)
    env.reset()
    raises = [op for op in env.state.operations if isinstance(op, CompletionBettingOrRaisingTo)]
    assert raises[-1].amount == 200
    assert env.action_masks().tolist() == [True, True, False, False, False, False]
    _, reward, done, _, info = env.step(Action.CHECK_CALL)
    assert done
    assert reward in (-100, 0, 100)
    assert sum(info["final_stacks"]) == 400


def test_seed_reproduces_opponent_selection_actions_and_rewards():
    def trace():
        env = HeadsUpEnv(opponent="mixture")
        rng = np.random.default_rng(123)
        result = []
        for hand in range(40):
            obs, info = env.reset(seed=72 if hand == 0 else None)
            cards = tuple(obs["own_cards"])
            actions, reward, final = play_hand(env, rng)
            result.append((info["opponent"], cards, actions, reward, final["final_stacks"]))
        return result

    first = trace()
    assert first == trace()
    assert {row[0] for row in first} == set(OPPONENT_NAMES)


def test_opponent_random_draws_do_not_change_later_deals():
    first = HeadsUpEnv()
    second = HeadsUpEnv()
    for hand in range(10):
        first_obs, _ = first.reset(seed=9 if hand == 0 else None)
        second_obs, _ = second.reset(seed=9 if hand == 0 else None)
        np.testing.assert_array_equal(first_obs["own_cards"], second_obs["own_cards"])
        assert first.state.deck_cards == second.state.deck_cards
        # Simulate strategies consuming different numbers of random draws.
        second._opponent_rng.random(100)


def test_opponent_receives_its_own_cards_and_relative_seat_zero():
    class InspectOpponent:
        def choose_action(self, observation, mask, rng):
            actor = env.state.actor_index
            expected = np.zeros(52, dtype=np.float32)
            for card in env.state.hole_cards[actor]:
                expected[CARD_INDEX[card]] = 1
            np.testing.assert_array_equal(observation["own_cards"], expected)
            assert observation["seats"][0, 0] == env.state.stacks[actor] / 2
            np.testing.assert_array_equal(observation["legal_actions"], mask)
            assert observation["own_cards"].sum() == 2
            return Action.CHECK_CALL

    env = HeadsUpEnv(opponent=InspectOpponent())
    env.reset(seed=42)
    env.step(Action.CHECK_CALL)


def test_invalid_opponent_action_is_rejected_before_engine_execution():
    class InvalidOpponent:
        def choose_action(self, observation, mask, rng):
            return 99

    env = HeadsUpEnv(opponent=InvalidOpponent())
    env.reset(seed=42)
    with pytest.raises(ValueError, match="Opponent selected"):
        env.step(Action.CHECK_CALL)


def test_tight_bot_folds_weak_cards_but_checks_for_free():
    env = HeadsUpEnv()
    obs, _ = env.reset(seed=42)
    obs["own_cards"][:] = 0
    for card in Card.parse("2c 7d"):
        obs["own_cards"][CARD_INDEX[card]] = 1
    rng = np.random.default_rng(1)
    bot = TightOpponent()
    assert bot.choose_action(obs, env.action_masks(), rng) == Action.FOLD
    obs["table"][2] = 0
    assert bot.choose_action(obs, env.action_masks(), rng) == Action.CHECK_CALL
    weak = hand_strength(obs)
    obs["own_cards"][:] = 0
    for card in Card.parse("Ac Ad"):
        obs["own_cards"][CARD_INDEX[card]] = 1
    assert hand_strength(obs) > weak
    assert bot.choose_action(obs, env.action_masks(), rng) == Action.MIN_RAISE


def test_aggressive_bot_falls_back_to_call_when_raising_is_unavailable():
    mask = np.array([True, True, False, False, False, False])
    assert AggressiveOpponent(1).choose_action({}, mask, np.random.default_rng(0)) == Action.CHECK_CALL


@pytest.mark.parametrize("factory", [CheckCallOpponent, AggressiveOpponent])
def test_invalid_behavior_probability_is_rejected(factory):
    with pytest.raises(ValueError, match="between"):
        factory(1.5)
