"""Masked, stochastic checkpoint comparison on fixed independent sessions.

This is an initial heads-up diagnostic. Held-out variants are relatives of the
training bots, not a strong external poker benchmark or evidence of GTO play.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import set_random_seed
import torch

from .actions import Action
from .env import HeadsUpEnv
from .opponents import AggressiveOpponent, CheckCallOpponent, OPPONENT_NAMES, make_opponent


BENCHMARK_NAMES = (*OPPONENT_NAMES, "heldout_passive", "heldout_aggressive")


def benchmark_opponent(name):
    if name == "heldout_passive":
        return CheckCallOpponent(call_probability=0.85)
    if name == "heldout_aggressive":
        return AggressiveOpponent(raise_probability=0.95)
    return make_opponent(name)


def session_interval(values, seed):
    """Percentile bootstrap of complete session results, never individual hands."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def evaluate_checkpoint(model, opponent_name, sessions=5, hands=500, seed=1000):
    if sessions < 1 or hands < 1:
        raise ValueError("sessions and hands must be positive.")
    rates = []
    action_counts = {action.name: 0 for action in Action}
    raise_to_counts = {}
    zero_decision_hands = 0
    positions = {"button": {"hands": 0, "profit_bb": 0.0}, "big_blind": {"hands": 0, "profit_bb": 0.0}}
    for session in range(sessions):
        session_seed = seed + session
        # Reset the policy's sampling RNG per session and checkpoint too.
        set_random_seed(session_seed + 1_000_000)
        env = HeadsUpEnv(opponent=benchmark_opponent(opponent_name))
        profit = 0.0
        try:
            for hand in range(hands):
                observation, info = env.reset(seed=session_seed if hand == 0 else None)
                position = "button" if info["learner_is_button"] else "big_blind"
                for _ in range(200):
                    mask = env.action_masks()
                    if info["awaiting_terminal_acknowledgement"]:
                        # This transition returns a reward but is not a poker action.
                        action = int(Action.CHECK_CALL)
                    else:
                        predicted, _ = model.predict(
                            observation, action_masks=mask, deterministic=False
                        )
                        action = int(predicted)
                        if not env.action_space.contains(action) or not mask[action]:
                            raise RuntimeError("Evaluation policy selected an illegal action.")
                        action_counts[Action(action).name] += 1
                        amount = info["action_raise_to"][action]
                        if amount is not None:
                            amount_bb = str(amount / env.BIG_BLIND)
                            raise_to_counts[amount_bb] = raise_to_counts.get(amount_bb, 0) + 1
                    observation, reward, done, truncated, info = env.step(action)
                    if truncated:
                        raise RuntimeError("Unexpected truncated hand in evaluation.")
                    if done:
                        profit += reward
                        zero_decision_hands += int(info["learner_decisions"] == 0)
                        positions[position]["hands"] += 1
                        positions[position]["profit_bb"] += reward
                        break
                else:
                    raise RuntimeError("Evaluation hand exceeded 200 learner decisions.")
        finally:
            env.close()
        rates.append(100 * profit / hands)
    decisions = sum(action_counts.values())
    return {
        "opponent": opponent_name,
        "player_count": 2,
        "sessions": sessions,
        "hands_per_session": hands,
        "first_session_seed": seed,
        "bb_per_100": float(np.mean(rates)),
        "session_bb_per_100": rates,
        "session_bootstrap_95_interval": session_interval(rates, seed + 2_000_000),
        "learner_decisions": decisions,
        "zero_decision_hands": zero_decision_hands,
        "illegal_action_rate": 0.0,
        "action_counts": action_counts,
        "action_frequencies": {name: count / decisions if decisions else 0 for name, count in action_counts.items()},
        "raise_to_bb_counts": raise_to_counts,
        "positions": positions,
    }


def compare(checkpoints, sessions=5, hands=500, seed=1000, opponents=BENCHMARK_NAMES):
    torch.set_num_threads(1)
    results = {}
    for checkpoint in checkpoints:
        key = str(Path(checkpoint))
        if key in results:
            raise ValueError("Checkpoint paths must be distinct.")
        model = MaskablePPO.load(checkpoint, device="cpu")
        results[key] = [
            evaluate_checkpoint(model, name, sessions, hands, seed)
            for name in opponents
        ]
    report = {"checkpoints": results, "paired_differences": []}
    # With two checkpoints, report the second minus the first on the same seeds.
    if len(results) == 2:
        first, second = results.values()
        for before, after in zip(first, second):
            differences = np.subtract(after["session_bb_per_100"], before["session_bb_per_100"])
            report["paired_differences"].append({
                "opponent": after["opponent"],
                "bb_per_100_change": float(differences.mean()),
                "session_bootstrap_95_interval": session_interval(differences, seed + 3_000_000),
            })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--opponents", nargs="+", choices=BENCHMARK_NAMES, default=BENCHMARK_NAMES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("The output file already exists; choose a new path.")
    report = compare(args.checkpoints, args.sessions, args.hands, args.seed, args.opponents)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(json.dumps(report, indent=2) + "\n")
    for checkpoint, results in report["checkpoints"].items():
        print(checkpoint)
        for result in results:
            print(f"  {result['opponent']}: {result['bb_per_100']:.2f} BB/100; "
                  f"95% session bootstrap interval {result['session_bootstrap_95_interval']}")
    for difference in report["paired_differences"]:
        print(f"Change vs first checkpoint, {difference['opponent']}: "
              f"{difference['bb_per_100_change']:.2f} BB/100; "
              f"95% interval {difference['session_bootstrap_95_interval']}")


if __name__ == "__main__":
    main()
