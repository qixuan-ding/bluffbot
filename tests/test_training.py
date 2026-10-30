"""Small end-to-end checks for saved policies and evaluation accounting."""

import numpy as np
import pytest

from bluffbot.actions import Action
from bluffbot import evaluate
from bluffbot.train import train


def test_evaluation_includes_zero_decision_hands_and_excludes_acknowledgements(monkeypatch):
    class FoldOpponent:
        def choose_action(self, observation, mask, rng):
            return Action.FOLD

    class RaisePolicy:
        def predict(self, observation, action_masks, deterministic):
            assert not deterministic
            assert action_masks[Action.MIN_RAISE]
            return np.array(Action.MIN_RAISE), None

    monkeypatch.setattr(evaluate, "benchmark_opponent", lambda name: FoldOpponent())
    result = evaluate.evaluate_checkpoint(RaisePolicy(), "fold", sessions=2, hands=4)
    assert result["zero_decision_hands"] == 4
    assert result["learner_decisions"] == 4
    assert result["action_counts"]["MIN_RAISE"] == 4
    assert result["positions"]["button"]["hands"] == 4
    assert result["positions"]["big_blind"]["hands"] == 4
    assert result["bb_per_100"] == 75
    assert result["illegal_action_rate"] == 0


def test_small_training_run_saves_reloads_and_compares_checkpoints(tmp_path):
    output = train(decisions=32, seed=17, output=tmp_path / "experiment")
    assert (output / "untrained.zip").is_file()
    assert (output / "trained.zip").is_file()
    import json
    summary = json.loads((output / "training_summary.json").read_text())
    assert summary["reload_check_passed"]
    assert summary["learner_decisions"] >= 32
    assert summary["interface_steps"] == summary["learner_decisions"] + summary["zero_decision_hands"]
    assert summary["completed_hands"] > 0
    report = evaluate.compare(
        [output / "untrained", output / "trained"],
        sessions=2, hands=10, seed=5000, opponents=["check_call", "heldout_aggressive"],
    )
    assert len(report["paired_differences"]) == 2
    for results in report["checkpoints"].values():
        for result in results:
            assert result["illegal_action_rate"] == 0
            assert sum(row["hands"] for row in result["positions"].values()) == 20
            assert sum(result["action_counts"].values()) == result["learner_decisions"]
            assert sum(result["action_frequencies"].values()) == pytest.approx(1)
    with pytest.raises(FileExistsError):
        train(decisions=32, output=output)


def test_session_interval_requires_multiple_sessions():
    assert evaluate.session_interval([10], seed=1) is None
    assert evaluate.session_interval([10, 10, 10], seed=1) == [10, 10]


def test_training_parameters_are_applied_saved_and_logged(tmp_path):
    import json
    from sb3_contrib import MaskablePPO
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    settings = {
        "decisions": 16, "seed": 12, "opponent": "random",
        "policy_layers": [32], "value_layers": [64, 32],
        "learning_rate": 0.001, "n_steps": 32, "batch_size": 16,
        "n_epochs": 2, "gamma": 1.0, "gae_lambda": 0.9,
        "clip_range": 0.15, "ent_coef": 0.01,
    }
    output = train(output=tmp_path / "custom", **settings)
    assert json.loads((output / "training_config.json").read_text()) == settings
    model = MaskablePPO.load(output / "trained")
    assert model.policy.net_arch == {"pi": [32], "vf": [64, 32]}
    assert model.n_steps == 32
    assert model.n_epochs == 2
    assert model.learning_rate == 0.001
    assert model.ent_coef == 0.01
    metrics = EventAccumulator(str(output / "tensorboard"))
    metrics.Reload()
    assert {"poker/learner_decisions", "train/value_loss", "train/entropy_loss",
            "train/approx_kl"}.issubset(metrics.Tags()["scalars"])


@pytest.mark.parametrize("settings", [
    {"n_steps": 1}, {"batch_size": 1}, {"batch_size": 63},
    {"policy_layers": [0]}, {"gamma": 1.5}, {"learning_rate": 0},
])
def test_invalid_training_settings_do_not_create_run_directory(tmp_path, settings):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        train(output=output, **settings)
    assert not output.exists()
