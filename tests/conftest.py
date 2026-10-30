"""Shared seeded environment and raw betting-state factories for pytest."""

import pytest
from pokerkit import Automation, NoLimitTexasHoldem

from bluffbot.env import HeadsUpEnv


@pytest.fixture(params=[1, 0], ids=["button", "big_blind"])
def env(request):
    environment = HeadsUpEnv()
    environment.reset(seed=42)

    # Seed once: passing a seed again also resets the position rotation.
    if request.param == 0:
        environment.reset()

    yield environment
    environment.close()


def create_betting_state(stacks=(200, 200)):
    """Create a PokerKit hand paused at the first betting decision."""
    return NoLimitTexasHoldem.create_state(
        tuple(Automation), False, 0, (1, 2), 2, stacks, len(stacks)
    )


@pytest.fixture
def make_betting_state():
    """Let rule, action, and observation tests build focused engine states."""
    return create_betting_state
