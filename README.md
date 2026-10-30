# Bluffbot: Model A proof of concept

A heads-up no-limit Texas Hold'em environment and small feedforward
MaskablePPO experiment. Python 3.12, PokerKit rules, Gymnasium interface,
200-chip stacks, 1/2 blinds, and net chip profit divided by 2 as reward.
Stacks reset every hand. Actual 2–9 player gameplay and checkpoint self-play
remain later milestones in the Model A proof-of-concept plan.

## Scripted opponents

`HeadsUpEnv()` preserves the original check/call opponent. Choose another with
`HeadsUpEnv(opponent="random")`, `"tight"`, `"aggressive"`, or `"mixture"`.
The mixture samples uniformly from the four bots once per hand. Custom bots
provide `choose_action(observation, mask, rng)` and receive their own legal
observation, never the engine state. The environment validates their choices
and executes the shared six-action sizing menu.

The tight bot uses a deliberately basic hand-strength score, not estimated
equity. The passive and aggressive bots also accept configurable probabilities
through `CheckCallOpponent(...)` and `AggressiveOpponent(...)`.

If the small-blind opponent folds before the learner acts, the hand is already
settled during `reset()`. The interface exposes one forced `CHECK_CALL`
acknowledgement whose `step()` returns the terminal reward. This executes no
poker action and has `info["learner_acted"] == False`. It preserves every hand,
its profit, and position rotation. Acknowledgements are excluded from poker
decision counts and action frequencies, but included in SB3's interface steps.
The observation mask matches `action_masks()`, including this forced step.

## Run section 5

From the repository root, use the existing virtual environment:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m bluffbot.train --decisions 10000 --seed 7
```

Training uses the four-bot mixture, dictionary observations, separate two-layer
128-unit policy/value networks, CPU execution, and masked actions. The finite
hand objective uses `gamma=1` so it does not discount profit by hand length.
The decision target rounds up to a complete 256-interface-step rollout.

Each run creates a new directory under `runs/` containing `untrained.zip`,
`trained.zip`, TensorBoard logs, `training_config.json`, and
`training_summary.json`. Use `--output`
to choose a new directory explicitly; existing directories are never reused.
The summary records the seed, dependency versions, actual learner decisions,
completed hands, zero-decision hands, interface steps, and throughput. Training
profit is diagnostic because the policy changes throughout the run.

## Check saved policies

Replace `RUN_DIRECTORY` with the path printed by training:

```bash
.venv/bin/python -m bluffbot.evaluate RUN_DIRECTORY/untrained.zip RUN_DIRECTORY/trained.zip --sessions 5 --hands 500 --seed 1000 --output RUN_DIRECTORY/evaluation.json
.venv/bin/tensorboard --logdir RUN_DIRECTORY/tensorboard
```

Evaluation reloads the checkpoints and samples masked actions. It reports
BB/100, session results, session-bootstrap 95% intervals, paired checkpoint
differences, action frequencies, exact raise-to amounts in BB, position totals,
and illegal-action rate. Invalid actions fail the evaluation instead of being
silently replaced. The same hand seeds are used for checkpoint comparisons;
deck and opponent random streams are separate.

The default benchmark includes the training bots plus held-out passive and
aggressive probability variants. These are modest diagnostics from the same
bot families, not a strong independent poker benchmark. Small session counts
give preliminary intervals. Keep comparison seeds fixed and reserve separate
seeds/opponents for a later final test; do not tune on that final test.
A successful short run validates the pipeline. Repeatable improvement still
needs larger evaluations and multiple training seeds before self-play or
multiplayer expansion.

The working dependency snapshot is in `requirements.lock.txt`. The exploratory
notebook remains in `notebooks/environment_and_actions.ipynb`.

## Training notebook

Open `notebooks/training_experiments.ipynb` with the repository's `.venv` kernel.
It opens with saved charts from `runs/notebook_baseline_seed7` and leaves
`RUN_NEW_TRAINING` and `RUN_EVALUATION` disabled. One settings cell controls the
network sizes, learning rate, rollout length, batch size, training passes,
discount, advantage estimation, clipping, entropy incentive, and evaluation.
Enable the switches to run experiments; each new experiment and evaluation
gets a separate timestamped path.

The notebook plots training progress, PPO diagnostics, checkpoint profit with
session confidence intervals, action choices, raise sizes, and position results.
It also lists saved experiments for comparison. The same parameters are
available as keyword arguments to `bluffbot.train.train(...)`; existing command
line defaults still work. New runs retain optimizer metrics as well as poker
counters. Older runs with missing diagnostics are labeled explicitly.
