"""A model horizon counts predicted transitions, excluding the logged seed."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

import supermimic.runners.supermimic_runner as runner_module
from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg


class EndOfUpdate(Exception):
  """Stop the runner after inspecting its first training batch."""


def _runner_with_frames(horizon, frames, tmp_path, monkeypatch):
  runner = object.__new__(SuperMimicRunner)
  runner.cfg = SuperMimicRunnerCfg(max_iterations=1, log_interval=1)
  runner.cfg.wm.horizon = horizon
  runner.cfg.wm.batch_size = 1
  runner.cfg.wm.num_batches = 1
  runner.env = SimpleNamespace(num_envs=1, reset=Mock())
  runner._create_log_dir = Mock(return_value=tmp_path)
  runner._init_wandb = Mock()
  runner._seed_sim_history = Mock()
  runner._cleanup = Mock()
  runner.step = Mock(return_value=1)
  runner.logger = SimpleNamespace(log=Mock())
  runner._wm = SimpleNamespace(bundle=lambda: {})
  runner._representation = SimpleNamespace(actor_obs_mode="rep_state")
  runner.world_model_learner = SimpleNamespace(update=Mock())
  runner.policy_learner = SimpleNamespace(
    actor=torch.nn.Linear(1, 1), critic=None, update=Mock(return_value={})
  )
  monkeypatch.setattr(runner_module.wandb, "run", SimpleNamespace(name="test"))
  monkeypatch.setattr(runner_module, "log_model_artifact", Mock())
  schema = {name: FieldSpec(1) for name in ("state", "action", "sim_ctx")}
  schema.update({name: FieldSpec(dtype=torch.bool) for name in ("terminated", "truncated")})
  runner.replay_buffer = ReplayBuffer(schema, buffer_size=16, num_envs=1)
  for i in range(frames):
    runner.replay_buffer.insert(TensorDict({
      "state": torch.tensor([[float(i)]]),
      "action": torch.tensor([[1.0 if i < frames - 1 else 999.0]]),
      "sim_ctx": torch.zeros(1, 1),
      "terminated": torch.zeros(1, dtype=torch.bool),
      "truncated": torch.zeros(1, dtype=torch.bool),
    }, batch_size=[1]))
  return runner


@pytest.mark.parametrize("horizon", [1, 8])
def test_runner_scores_every_transition_through_the_horizon(horizon, tmp_path, monkeypatch):
  runner = _runner_with_frames(horizon, horizon + 1, tmp_path, monkeypatch)

  def inspect_update(data):
    batch = next(data)
    initial = batch["state"][:, 0].clone().requires_grad_(True)
    model = SimpleNamespace(dynamics=lambda state, action, aux, context: (aux, state + action))
    predicted, _ = WorldModelLearner._rollout(
      model, initial, batch["action"], torch.zeros(1, 1), batch["sim_ctx"][:, 0]
    )
    expected = torch.arange(1, horizon + 1, dtype=torch.float32).reshape(1, horizon, 1)
    torch.testing.assert_close(predicted, expected)
    torch.testing.assert_close(batch["state"][:, 1:], expected)
    predicted[:, -1].sum().backward()
    torch.testing.assert_close(initial.grad, torch.ones_like(initial))
    raise EndOfUpdate

  runner.world_model_learner.update.side_effect = inspect_update
  with pytest.raises(EndOfUpdate):
    runner.run()


def test_runner_waits_for_the_horizon_endpoint(tmp_path, monkeypatch):
  runner = _runner_with_frames(8, 8, tmp_path, monkeypatch)
  runner.world_model_learner.update.side_effect = AssertionError("endpoint s8 is absent")
  runner.run()
  runner.world_model_learner.update.assert_not_called()
  runner.policy_learner.update.assert_not_called()
