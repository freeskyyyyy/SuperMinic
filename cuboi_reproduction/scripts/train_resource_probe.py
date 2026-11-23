"""Run the planned small training/resource probe using the unchanged upstream Trainer."""

import argparse
import json
from pathlib import Path

import torch

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--resume", type=Path)
parser.add_argument("--run-name", default="20261116_train_mini_seed42_01")
parser.add_argument("--max-hours", type=float, default=0.1)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
cfg = TrainConfig(run_name=args.run_name, log_root=root / "runs", seed=42, resume=args.resume)
cfg.data.npz_dir = root / "data/processed/lafan1"
cfg.data.clips = ["walk1_subject1", "run1_subject5", "dance1_subject1", "jumps1_subject1"]
cfg.env.num_envs = args.num_envs
cfg.env.replay_capacity = 200
cfg.wm_trainer.batch_size = args.batch_size
cfg.policy_trainer.batch_size = args.batch_size
cfg.schedule.total_iterations = 320000
cfg.schedule.log_interval = 50
cfg.schedule.checkpoint_interval = 100
cfg.schedule.max_hours = args.max_hours
trainer = Trainer(cfg)
models = {"world_model": trainer.wm, "policy": trainer.policy, "critic": trainer.critic}
initial = {name: {key: value.detach().cpu().clone() for key, value in model.named_parameters()} for name, model in models.items()}
trainer.save("ckpt_initial.pt")
initial_iteration = trainer.iteration
initial_updates = trainer.num_updates
restoration = None
if args.resume is not None:
    source_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    loaded_checkpoint = torch.load(trainer.log_dir / "ckpt_initial.pt", map_location="cpu", weights_only=False)
    restoration = {}
    for key in ("wm", "policy", "critic"):
        original = source_checkpoint[key]
        loaded = loaded_checkpoint[key]
        restoration[key] = original.keys() == loaded.keys() and all(torch.equal(value, loaded[name]) for name, value in original.items())
    for key in ("bin_return", "bin_memory"):
        restoration[key] = torch.equal(source_checkpoint[key], loaded_checkpoint[key])
    for key in ("wm_opt", "policy_opt", "critic_opt"):
        original = source_checkpoint[key]
        loaded = loaded_checkpoint[key]
        equal = original["param_groups"] == loaded["param_groups"] and original["state"].keys() == loaded["state"].keys()
        for parameter, state in original["state"].items():
            for name, value in state.items():
                other = loaded["state"][parameter][name]
                equal = equal and (torch.equal(value, other) if torch.is_tensor(value) else value == other)
        restoration[key] = equal
    restoration["iteration"] = initial_iteration == source_checkpoint["iteration"]
    restoration["num_updates"] = initial_updates == source_checkpoint["num_updates"]
    (trainer.log_dir / "restoration_before_training.json").write_text(json.dumps(restoration, indent=2) + "\n")
    print("restoration before training", json.dumps(restoration), flush=True)
    assert all(restoration.values())
    del source_checkpoint, loaded_checkpoint
trainer.run()
changes = {}
for name, model in models.items():
    end = {key: value.detach().cpu() for key, value in model.named_parameters()}
    changes[name] = {
        "changed": any(not torch.equal(initial[name][key], value) for key, value in end.items()),
        "all_finite": all(bool(torch.isfinite(value).all()) for value in end.values()),
        "max_abs_parameter_change": max(float((value - initial[name][key]).abs().max()) for key, value in end.items()),
    }
report = {
    "initial_iteration": initial_iteration, "initial_updates": initial_updates,
    "iteration": trainer.iteration, "num_updates": trainer.num_updates,
    "new_iterations": trainer.iteration - initial_iteration, "new_updates": trainer.num_updates - initial_updates,
    "restoration_before_training": restoration,
    "min_300_new_iterations_reached": trainer.iteration - initial_iteration >= 300,
    "models": changes, "resolved_clip_order": [clip.name for clip in trainer.lib.clips],
    "scope": "small engineering/resource probe, not converged motion tracking or published training-scale reproduction",
}
(trainer.log_dir / "probe_result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
trainer.env.close()
assert all(result["changed"] and result["all_finite"] for result in changes.values())
