"""Public observation encoding, seat status, action history, and privacy."""

import numpy as np

from bluffbot.actions import Action
from bluffbot.env import HeadsUpEnv
from bluffbot.observations import (
    CARD_INDEX,
    HISTORY_LENGTH,
    encode_observation,
    observation_space,
)


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


def test_public_history_keeps_only_latest_sixteen_actions(make_betting_state):
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
