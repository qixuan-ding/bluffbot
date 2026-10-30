import gymnasium as gym
import numpy as np
from pokerkit import Automation, Deck, NoLimitTexasHoldem

from .actions import Action, NUM_ACTIONS, action_options
from .observations import encode_observation, observation_space
from .opponents import OPPONENT_NAMES, make_opponent


class HeadsUpEnv(gym.Env):
    """One heads-up hand per episode against a fixed or sampled scripted bot.

    If the opponent folds before the learner acts, reset exposes a forced
    CHECK_CALL acknowledgement. Its step returns the terminal reward without
    executing a poker action; info['learner_acted'] is False for that step.
    """

    STARTING_STACK = 200
    BIG_BLIND = 2

    CARD_IDS = {
        card: i + 1
        for i, card in enumerate(Deck.STANDARD)
    }

    def __init__(self, opponent="check_call"):
        # Calls initialization from the HeadsUp Env
        # Enumeraters actions and creates observation space, actions.py and observations.py
        super().__init__()

        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self.observation_space = observation_space()

        self.state = None
        self.hand_count = 0
        self.learner_index = 1
        self.decisions = 0
        self._opponent_pool = (
            tuple(make_opponent(name) for name in OPPONENT_NAMES)
            if isinstance(opponent, str) and opponent == "mixture"
            else (make_opponent(opponent) if isinstance(opponent, str) else opponent,)
        )
        if not all(
            callable(getattr(bot, "choose_action", None))
            for bot in self._opponent_pool
        ):
            raise TypeError("An opponent must provide choose_action(observation, mask, rng).")
        self.opponent = self._opponent_pool[0]
        self._opponent_rng = None
        self._pending_terminal = False

    def reset(self, *, seed=None, options=None):
        # Starts a fresh hand and stops at the first decision
        super().reset(seed=seed)

        # Keep opponent draws separate so action choices do not change future decks.
        if seed is not None or self._opponent_rng is None:
            self._opponent_rng = np.random.default_rng(
                np.random.SeedSequence([self.np_random_seed, 1])
            )
        self.opponent = self._opponent_pool[
            int(self._opponent_rng.integers(len(self._opponent_pool)))
        ]
        self._pending_terminal = False

        if seed is not None:
            self.hand_count = 0

        # index 0 = big blind, 1 = button / small blind
        self.learner_index = 1 - (self.hand_count % 2)
        self.hand_count += 1
        self.decisions = 0

        # Blinds, bet collection, showdown handling, and payouts are automated
        # Dealing cards (and burning) are not
        automations = tuple(
            automation
            for automation in Automation
            if automation not in (
                Automation.HOLE_DEALING,
                Automation.BOARD_DEALING,
                Automation.CARD_BURNING,
            )
        )

        # Creating the game state
        self.state = NoLimitTexasHoldem.create_state(
            automations,
            False,  # Ante trimming; irrelevant with no antes.
            0,      # No antes.
            (1, 2), # Heads-up mapping: BB=0, SB=1.
            self.BIG_BLIND,
            (self.STARTING_STACK, self.STARTING_STACK),
            2,      # Two players.
        )

        # Shuffling, then replacing PokerKit's generated deck with shuffled
        deck = list(Deck.STANDARD)
        order = self.np_random.permutation(len(deck))

        self.state.deck_cards.clear()
        self.state.deck_cards.extend(
            deck[int(i)] for i in order
        )

        self._advance()

        # Gym reset cannot return a terminal reward. Expose one forced
        # acknowledgement transition for a hand won before any learner action.
        # It settles the episode without executing or counting a poker decision.
        self._pending_terminal = not self.state.status

        return self._observation(), self._info()

    def action_masks(self):
        # Determines which actions are currently legal
        mask, _ = action_options(self.state, self.learner_index)
        if self._pending_terminal:
            mask[Action.CHECK_CALL] = True
        return mask

    def step(self, action):
        # The agent executes an action, then returns an observation
        # RETURNS: reward function (0 until the end),
        #          if program terminated, if program truncated (false after successful step),
        #          info, which contains inspection and logging

        if self.state is None or (not self.state.status and not self._pending_terminal):
            raise RuntimeError(
                "Call reset() before starting a new hand."
            )

        if not self.action_space.contains(action):
            raise ValueError(f"Unknown action: {action}")

        action = int(action)
        mask = self.action_masks()
        _, amounts = action_options(self.state, self.learner_index)

        if not mask[action]:
            raise ValueError(f"Illegal or duplicate action: {action}")

        if self._pending_terminal:
            self._pending_terminal = False
            return self._step_result(learner_acted=False)

        if action == Action.FOLD:
            self.state.fold()

        elif action == Action.CHECK_CALL:
            self.state.check_or_call()

        elif action == Action.MIN_RAISE:
            # No amount means the minimum legal bet/raise.
            # The shared action mapping now supplies that minimum explicitly.
            self.state.complete_bet_or_raise_to(amounts[action])

        else:
            self.state.complete_bet_or_raise_to(amounts[action])

        self.decisions += 1
        self._advance()

        return self._step_result(learner_acted=True)

    def _step_result(self, learner_acted):

        terminated = not self.state.status
        reward = 0.0

        if terminated:
            expected_total = 2 * self.STARTING_STACK

            if sum(self.state.stacks) != expected_total:
                raise RuntimeError(
                    "Chips were not conserved."
                )

            reward = (
                self.state.stacks[self.learner_index]
                - self.STARTING_STACK
            ) / self.BIG_BLIND

        info = self._info()
        info["learner_acted"] = learner_acted
        return (
            self._observation(),
            reward,
            terminated,
            False,  # No time-limit truncation.
            info,
        )

    def _advance(self):
        # Burn card or deal card until next learning step

        state = self.state

        while state.status:
            if state.can_burn_card():
                state.burn_card()

            elif state.can_deal_hole():
                state.deal_hole()

            elif state.can_deal_board():
                state.deal_board()

            elif state.actor_index == self.learner_index:
                return

            elif state.actor_index is not None:
                # Temporary opponent: always check or call.
                # The default still checks/calls; configured bots use the same menu.
                actor = state.actor_index
                mask, amounts = action_options(state, actor)
                observation = encode_observation(state, actor, self.BIG_BLIND)
                action = self.opponent.choose_action(
                    observation, mask.copy(), self._opponent_rng
                )
                if not self.action_space.contains(action) or not mask[int(action)]:
                    raise ValueError(
                        f"Opponent selected an illegal or duplicate action: {action}"
                    )
                action = int(action)
                if action == Action.FOLD:
                    state.fold()
                elif action == Action.CHECK_CALL:
                    state.check_or_call()
                else:
                    state.complete_bet_or_raise_to(amounts[action])

            else:
                raise RuntimeError(
                    "Unhandled PokerKit operation."
                )

    def _observation(self):

        observation = encode_observation(
            self.state, self.learner_index, self.BIG_BLIND
        )
        observation["legal_actions"][:] = self.action_masks()
        return observation

    def _info(self):
        # Misc training info

        _, amounts = action_options(self.state, self.learner_index)
        info = {
            "learner_index": self.learner_index,
            "learner_is_button": self.learner_index == 1,
            "learner_decisions": self.decisions,
            "action_raise_to": amounts,
            "opponent": getattr(self.opponent, "name", type(self.opponent).__name__),
            "awaiting_terminal_acknowledgement": self._pending_terminal,
        }

        if not self.state.status:
            info["final_stacks"] = tuple(
                self.state.stacks
            )

            info["profit_bb"] = (
                self.state.stacks[self.learner_index]
                - self.STARTING_STACK
            ) / self.BIG_BLIND

        return info
