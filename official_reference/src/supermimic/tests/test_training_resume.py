from types import SimpleNamespace
from unittest.mock import Mock

import torch
import pytest

from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.checkpoint import save_training_checkpoint, restore_training_checkpoint
from supermimic.runners.supermimic_runner import Ratio, SuperMimicRunner, SuperMimicRunnerCfg


def _runner():
  actor = torch.nn.Linear(2, 1)
  critic = torch.nn.Linear(2, 1)
  dynamics = torch.nn.Linear(2, 1)
  opt = lambda model: torch.optim.Adam(model.parameters(), lr=0.01)
  model = SimpleNamespace(
    bundle=lambda: {"dynamics": dynamics.state_dict()},
    load_bundle=lambda state: dynamics.load_state_dict(state["dynamics"]),
  )
  policy = SimpleNamespace(actor=actor, critic=critic, actor_optimizers=[opt(actor)], critic_optimizers=[opt(critic)])
  wm = SimpleNamespace(optimizers=[opt(dynamics)])
  schema = {"value": FieldSpec(1), "terminated": FieldSpec(dtype=torch.bool), "truncated": FieldSpec(dtype=torch.bool)}
  replay = ReplayBuffer(schema, 4, 1)
  return SimpleNamespace(task_id="test", cfg=SuperMimicRunnerCfg(), _wm=model, policy_learner=policy, world_model_learner=wm, replay_buffer=replay, _sampler=None)


def test_training_checkpoint_restores_weights_optimizers_rng_and_ratio(tmp_path):
  runner = _runner()
  actor = runner.policy_learner.actor
  actor(torch.ones(1, 2)).sum().backward()
  runner.policy_learner.actor_optimizers[0].step()
  ratio = Ratio(0.4)
  ratio(0)
  ratio(2)
  torch.manual_seed(123)
  path = tmp_path / "resume.pt"
  save_training_checkpoint(path, runner, 17, ratio)
  expected_random = torch.rand(3)
  expected_param = actor.weight.detach().clone()
  restored = _runner()
  restored_ratio = Ratio(0.4)
  step = restore_training_checkpoint(path, restored, restored_ratio)
  assert step == 17
  assert torch.equal(restored.policy_learner.actor.weight, expected_param)
  assert restored.policy_learner.actor_optimizers[0].state_dict()["state"]
  assert restored_ratio._prev == ratio._prev
  assert torch.equal(torch.rand(3), expected_random)


def test_runner_stop_request_saves_training_checkpoint(tmp_path):
  runner = SuperMimicRunner.__new__(SuperMimicRunner)
  runner.__dict__.update(vars(_runner()))
  runner.cfg.max_iterations = 10
  runner.cfg.replay_ratio = 0
  runner.cfg.stop_file = str(tmp_path / "stop")
  runner.cfg.checkpoint_training = True
  runner.env = SimpleNamespace(num_envs=1, reset=Mock())
  runner._create_log_dir = lambda: tmp_path
  runner._init_wandb = Mock()
  runner._seed_sim_history = Mock()
  runner._cleanup = Mock()
  runner.step = Mock(return_value=1)
  (tmp_path / "stop").touch()
  runner.run()
  saved = torch.load(tmp_path / "training_state.pt", weights_only=False)
  assert saved["next_step"] == 1
  runner.step.assert_called_once()
  runner._cleanup.assert_called_once()


def test_resume_rejects_changed_training_configuration(tmp_path):
  runner = _runner()
  path = tmp_path / "resume.pt"
  save_training_checkpoint(path, runner, 4, Ratio(0.5))
  restored = _runner()
  restored.cfg.replay_ratio += 1
  with pytest.raises(ValueError, match="configuration"):
    restore_training_checkpoint(path, restored, Ratio(0.5))
