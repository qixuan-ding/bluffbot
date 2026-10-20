import gymnasium as gym
from pokerkit import Automation, Deck, NoLimitTexasHoldem

from .actions import Action, NUM_ACTIONS, action_options
from .observations import encode_observation, observation_space


class HeadsUpEnv(gym.Env):
    """Step 6 scaffold: one heads-up hand per episode."""

    STARTING_STACK = 200
    BIG_BLIND = 2

    CARD_IDS = {
        card: i + 1
        for i, card in enumerate(Deck.STANDARD)
    }

    def __init__(self):
        # Calls initialization from the HeadsUp Env
        # Enumeraters actions and creates observation space, actions.py and observations.py
        super().__init__()

        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self.observation_space = observation_space()

        self.state = None
        self.hand_count = 0
        self.learner_index = 1
        self.decisions = 0

    def reset(self, *, seed=None, options=None):
        # Starts a fresh hand and stops at the first decision
        super().reset(seed=seed)

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

        if not self.state.status:
            raise RuntimeError(
                "Hand ended before the learner could act."
            )

        return self._observation(), self._info()

    def action_masks(self):
        # Determines which actions are currently legal
        mask, _ = action_options(self.state, self.learner_index)
        return mask

    def step(self, action):
        # The agent executes an action, then returns an observation
        # RETURNS: reward function (0 until the end),
        #          if program terminated, if program truncated (false after successful step),
        #          info, which contains inspection and logging

        if self.state is None or not self.state.status:
            raise RuntimeError(
                "Call reset() before starting a new hand."
            )

        if not self.action_space.contains(action):
            raise ValueError(f"Unknown action: {action}")

        action = int(action)
        mask, amounts = action_options(self.state, self.learner_index)

        if not mask[action]:
            raise ValueError(f"Illegal or duplicate action: {action}")

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

        return (
            self._observation(),
            reward,
            terminated,
            False,  # No time-limit truncation.
            self._info(),
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
                state.check_or_call()

            else:
                raise RuntimeError(
                    "Unhandled PokerKit operation."
                )

    def _observation(self):

        return encode_observation(
            self.state, self.learner_index, self.BIG_BLIND
        )

    def _info(self):
        # Misc training info

        _, amounts = action_options(self.state, self.learner_index)
        info = {
            "learner_index": self.learner_index,
            "learner_is_button": self.learner_index == 1,
            "learner_decisions": self.decisions,
            "action_raise_to": amounts,
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
