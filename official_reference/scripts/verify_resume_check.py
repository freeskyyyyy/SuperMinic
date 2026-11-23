"""Inspect saved checkpoints from the isolated real-simulation resume smoke test."""
import json
import sys
from pathlib import Path

import torch

base = Path("/workspace/runs") / sys.argv[1]
before = torch.load(base / "phase-save/training_state.pt", map_location="cpu", weights_only=False)
after = torch.load(base / "phase-resume/training_state.pt", map_location="cpu", weights_only=False)


def changed(first, second):
  return any(not torch.equal(first[key], second[key]) for key in first)


def steps(checkpoint, key):
  return [int(state["step"].item()) for optimizer in checkpoint[key] for state in optimizer["state"].values() if "step" in state]


results = {
  "steps": [before["next_step"], after["next_step"]],
  "actor_changed": changed(before["actor"], after["actor"]),
  "world_model_changed": changed(before["world_model"]["dynamics"], after["world_model"]["dynamics"]),
  "critic_changed": changed(before["critic"], after["critic"]),
  "optimizer_step_ranges": {},
  "resume_boundary_marked": bool(after["replay"]["storage"]["truncated"][:, 15].all()),
}
for key in ("actor_optimizers", "world_optimizers", "critic_optimizers"):
  old, new = steps(before, key), steps(after, key)
  assert old and len(old) == len(new) and all(a < b for a, b in zip(old, new, strict=True))
  results["optimizer_step_ranges"][key] = [[min(old), max(old)], [min(new), max(new)]]
assert results["actor_changed"] and results["world_model_changed"] and results["critic_changed"]
assert results["resume_boundary_marked"]
print(json.dumps(results, indent=2))
