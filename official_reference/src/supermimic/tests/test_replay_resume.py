import torch
from tensordict import TensorDict

from supermimic.replay_buffer import FieldSpec, ReplayBuffer


def test_replay_state_restores_wrapped_window_and_position():
  schema = {"value": FieldSpec(1), "terminated": FieldSpec(dtype=torch.bool), "truncated": FieldSpec(dtype=torch.bool)}
  original = ReplayBuffer(schema, buffer_size=4, num_envs=1)
  for i in range(6):
    original.insert(TensorDict({"value": torch.tensor([[float(i)]]), "terminated": torch.tensor([False]), "truncated": torch.tensor([False])}, batch_size=[1]))
  restored = ReplayBuffer(schema, buffer_size=4, num_envs=1)
  restored.load_state_dict(original.state_dict())
  assert restored._step == original._step
  assert restored.num_transitions == original.num_transitions
  assert torch.equal(restored._storage["value"], original._storage["value"])
  restored.insert(TensorDict({"value": torch.tensor([[6.]]), "terminated": torch.tensor([False]), "truncated": torch.tensor([False])}, batch_size=[1]))
  assert restored._storage["value"][0, original._step].item() == 6


def test_resume_boundary_prevents_cross_episode_samples():
  schema = {"value": FieldSpec(1), "terminated": FieldSpec(dtype=torch.bool), "truncated": FieldSpec(dtype=torch.bool)}
  buffer = ReplayBuffer(schema, buffer_size=8, num_envs=1)
  for i in range(3):
    buffer.insert(TensorDict({"value": torch.tensor([[float(i)]]), "terminated": torch.tensor([False]), "truncated": torch.tensor([False])}, batch_size=[1]))
  buffer.mark_resume_boundary()
  for i in range(3, 6):
    buffer.insert(TensorDict({"value": torch.tensor([[float(i)]]), "terminated": torch.tensor([False]), "truncated": torch.tensor([False])}, batch_size=[1]))
  windows = buffer.sample(3, 100)["value"].squeeze(-1)
  assert ((windows[:, 0] == 0) | (windows[:, 0] == 3)).all()
