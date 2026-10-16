import gymnasium as gym
import numpy as np
from pokerkit import Automation, Deck, NoLimitTexasHoldem


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
        # Enumeraters actions and creates observation space which is a 15 number array
        super().__init__()

        # 0~3: fold, check/call, min bet/raise, all-in respectively
        self.action_space = gym.spaces.Discrete(4)

        # Own cards (2), Board cards (5), Stacks (2), Current bets (2)
        # Street, pot, call amount, button flag: 4
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=200.0,
            shape=(15,),
            dtype=np.float32,
        )

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
        # Determines which actions are currently legal, initialized as all false

        mask = np.zeros(4, dtype=bool)

        if self.state is None or not self.state.status:
            return mask

        if self.state.actor_index != self.learner_index:
            return mask

        state = self.state

        mask[0] = state.can_fold()
        mask[1] = state.can_check_or_call()
        mask[2] = state.can_complete_bet_or_raise_to()

        all_in_to = (
            state.bets[self.learner_index]
            + state.stacks[self.learner_index]
        )

        mask[3] = state.can_complete_bet_or_raise_to(
            all_in_to
        )

        return mask

    def step(self, action):
        # The agent executes an action

        if self.state is None or not self.state.status:
            raise RuntimeError(
                "Call reset() before starting a new hand."
            )

        if not self.action_space.contains(action):
            raise ValueError(f"Unknown action: {action}")

        action = int(action)

        if not self.action_masks()[action]:
            raise ValueError(f"Illegal action: {action}")

        if action == 0:
            self.state.fold()

        elif action == 1:
            self.state.check_or_call()

        elif action == 2:
            # No amount means the minimum legal bet/raise.
            self.state.complete_bet_or_raise_to()

        else:
            i = self.learner_index

            self.state.complete_bet_or_raise_to(
                self.state.bets[i] + self.state.stacks[i]
            )

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
        # Convert gamestate data into the 15-number array fed into the agent

        state = self.state
        i = self.learner_index
        other = 1 - i

        def encode(cards, length):
            values = [
                self.CARD_IDS[card] / 52.0
                for card in cards
            ]
            return (values + [0.0] * length)[:length]

        own_cards = encode(state.hole_cards[i], 2)

        board_cards = encode(
            [
                card
                for group in state.board_cards
                for card in group
            ],
            5,
        )

        call_amount = 0

        if state.status and state.actor_index == i:
            call_amount = (
                state.checking_or_calling_amount or 0
            )

        return np.array(
            own_cards
            + board_cards
            + [
                state.stacks[i] / 2,
                state.stacks[other] / 2,
                state.bets[i] / 2,
                state.bets[other] / 2,
                (
                    state.street_index
                    if state.street_index is not None
                    else 0
                ) / 3,
                state.total_pot_amount / 2,
                call_amount / 2,
                float(i == 1),
            ],
            dtype=np.float32,
        )

    def _info(self):
        info = {
            "learner_index": self.learner_index,
            "learner_is_button": self.learner_index == 1,
            "learner_decisions": self.decisions,
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