"""Section 5: a small, reproducible heads-up MaskablePPO experiment."""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch

from .env import HeadsUpEnv


class PokerCounters(BaseCallback):
    """Count actual poker decisions separately from Gym interface steps."""

    def __init__(self):
        super().__init__()
        self.learner_decisions = 0
        self.completed_hands = 0
        self.zero_decision_hands = 0
        self.profit_bb = 0.0

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            self.learner_decisions += int(info["learner_acted"])
            if done:
                self.completed_hands += 1
                self.zero_decision_hands += int(info["learner_decisions"] == 0)
                self.profit_bb += info["profit_bb"]
        return True

    def _on_rollout_end(self):
        self.logger.record("poker/learner_decisions", self.learner_decisions)
        self.logger.record("poker/completed_hands", self.completed_hands)
        self.logger.record("poker/zero_decision_hands", self.zero_decision_hands)
        self.logger.record("poker/profit_bb", self.profit_bb)


def train(
    decisions=10_000,
    seed=7,
    output=None,
    *,
    opponent="mixture",
    policy_layers=(128, 128),
    value_layers=(128, 128),
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=1.0,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
):
    """Train complete rollout batches until at least `decisions` poker decisions.

    The target can be exceeded by one rollout. Keeping the final rollout whole
    ensures its collected transitions are used in an optimizer update.
    """
    if decisions < 1:
        raise ValueError("decisions must be positive.")
    if n_steps < 2 or batch_size < 2 or batch_size > n_steps:
        raise ValueError("Require 2 <= batch_size <= n_steps.")
    if n_steps % batch_size:
        raise ValueError("batch_size must divide n_steps evenly.")
    if n_epochs < 1 or learning_rate <= 0:
        raise ValueError("n_epochs and learning_rate must be positive.")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be between zero and one.")
    if clip_range <= 0 or ent_coef < 0:
        raise ValueError("clip_range must be positive and ent_coef nonnegative.")
    if not policy_layers or not value_layers or not all(
        isinstance(width, int) and width > 0
        for width in (*policy_layers, *value_layers)
    ):
        raise ValueError("Network layers must contain positive integer widths.")
    parameters = {
        "decisions": decisions,
        "seed": seed,
        "opponent": opponent,
        "policy_layers": list(policy_layers),
        "value_layers": list(value_layers),
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
    }
    output = Path(output or f"runs/model_a_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    # A small feedforward network usually needs just one CPU thread here.
    torch.set_num_threads(1)
    (output / "training_config.json").write_text(json.dumps(parameters, indent=2) + "\n")
    env = HeadsUpEnv(opponent=opponent)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={"net_arch": {"pi": list(policy_layers), "vf": list(value_layers)}},
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        # The objective is undiscounted net chips over a finite, complete hand.
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        tensorboard_log=str(output / "tensorboard"),
        seed=seed,
        device="cpu",
        verbose=0,
    )
    # Reuse one logger across learn() calls. Otherwise each rollout's next
    # learn() call replaces the logger before optimizer diagnostics are saved.
    logger = configure(str(output / "tensorboard"), ["tensorboard"])
    model.set_logger(logger)
    model.save(output / "untrained")
    counters = PokerCounters()
    started = perf_counter()
    try:
        while counters.learner_decisions < decisions:
            model.learn(
                total_timesteps=model.n_steps,
                callback=counters,
                reset_num_timesteps=False,
                tb_log_name="model_a",
            )
            model.logger.dump(step=model.num_timesteps)
            if model.num_timesteps % 2048 == 0:
                print(
                    f"{counters.learner_decisions:,} learner decisions; "
                    f"{counters.completed_hands:,} complete hands",
                    flush=True,
                )
        model.save(output / "trained")
        # Verify the saved artifact can produce a legal masked prediction.
        restored = MaskablePPO.load(output / "trained", device="cpu")
        check_env = HeadsUpEnv(opponent="mixture")
        observation, _ = check_env.reset(seed=seed + 1)
        mask = check_env.action_masks()
        action, _ = restored.predict(observation, action_masks=mask)
        if not mask[int(action)]:
            raise RuntimeError("Reloaded policy selected an illegal action.")
        check_env.close()
    finally:
        env.close()
        logger.close()

    elapsed = perf_counter() - started
    summary = {
        "seed": seed,
        "requested_learner_decisions": decisions,
        "learner_decisions": counters.learner_decisions,
        "interface_steps": model.num_timesteps,
        "completed_hands": counters.completed_hands,
        "zero_decision_hands": counters.zero_decision_hands,
        "training_profit_bb": counters.profit_bb,
        "elapsed_seconds": elapsed,
        "learner_decisions_per_second": counters.learner_decisions / elapsed,
        "opponents": [bot.name for bot in env._opponent_pool],
        "network": {"policy": list(policy_layers), "value": list(value_layers)},
        "gamma": gamma,
        "training_parameters": parameters,
        "reload_check_passed": True,
        "versions": {
            name: version(name)
            for name in ("pokerkit", "gymnasium", "stable-baselines3", "sb3-contrib", "torch", "numpy")
        },
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved experiment to {output.resolve()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    train(args.decisions, args.seed, args.output)


if __name__ == "__main__":
    main()
