"""Evaluate local SuperMimic checkpoints on the Figure 3a motion benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.tasks.task_spec import build_actor_obs_td


@dataclass
class EvaluateConfig:
  checkpoint: Path
  motion_file: Path
  output_dir: Path
  task_id: str = "SuperMimic-GenCoord-G1"
  num_envs: int = 1024
  device: str = "cuda:0"


@torch.no_grad()
def main(cfg: EvaluateConfig) -> None:
  configure_torch_backends()
  device = cfg.device
  if cfg.checkpoint.is_dir():
    run_dir = cfg.checkpoint.parent.parent
    training_state = torch.load(
      run_dir / "training_state.pt", map_location="cpu", weights_only=False
    )
    world_model_state = torch.load(
      cfg.checkpoint / "world_model.pt", map_location="cpu", weights_only=False
    )
    actor_state = torch.load(
      cfg.checkpoint / "actor.pt", map_location="cpu", weights_only=False
    )
    completed_iterations = int(cfg.checkpoint.name.rsplit("_", 1)[-1]) + 1
    checkpoint_source = "evaluation_snapshot"
  else:
    training_state = torch.load(
      cfg.checkpoint, map_location="cpu", weights_only=False
    )
    world_model_state = training_state["world_model"]
    actor_state = training_state["actor"]
    completed_iterations = int(training_state["next_step"])
    checkpoint_source = "training_state"

  runner_cfg = load_runner_cfg({"runner_cfg": training_state["config"]})
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.commands["motion"].motion_file = str(cfg.motion_file)
  env_cfg.commands["motion"].sampling_mode = "start"
  env_cfg.commands["motion"].pose_range = {}
  env_cfg.commands["motion"].velocity_range = {}
  env_cfg.commands["motion"].joint_position_range = (0.0, 0.0)
  env_cfg.observations["actor"].enable_corruption = False
  foot_friction = env_cfg.events["foot_friction"]
  foot_friction.params["ranges"] = (1.0, 1.0)
  for axis in env_cfg.events["base_com"].params["ranges"]:
    env_cfg.events["base_com"].params["ranges"][axis] = (0.0, 0.0)
  env_cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  env_cfg.episode_length_s = 1.0e9

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env)
  wm = build_world_model(runner_cfg, env, device)
  wm.load_bundle(world_model_state)
  wm.dynamics.eval()
  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(actor_state)
  actor.set_slot_mix({"motion": 1.0})
  actor.eval()

  command = base_env.command_manager.get_term("motion")
  motion = command.motion
  motion_ends = motion.motion_ends
  starts = torch.cat((torch.zeros_like(motion_ends[:1]), motion_ends[:-1]))
  lengths = (motion_ends - starts).cpu().numpy()
  names = motion.motion_names
  max_step_reward = base_env.step_dt * sum(
    term.weight for term in env_cfg.rewards.values() if term.weight > 0
  )
  motion_task = next(task for task in actor.task_specs if task.name == "motion")
  sim_context = wm.sim_context.read(base_env)
  env_ids = torch.arange(cfg.num_envs, device=device)
  results: list[dict] = []
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  partial_path = cfg.output_dir / "per_motion.partial.csv"
  started = time.monotonic()
  order = np.argsort(lengths, kind="stable")
  total_batch_steps = sum(
    int(lengths[order[start : start + cfg.num_envs]].max()) - 1
    for start in range(0, len(order), cfg.num_envs)
  )
  completed_batch_steps = 0

  for batch_start in range(0, len(names), cfg.num_envs):
    batch_stop = min(batch_start + cfg.num_envs, len(names))
    batch_indices = order[batch_start:batch_stop]
    env_clip_indices = np.resize(batch_indices, cfg.num_envs)
    active = torch.arange(cfg.num_envs, device=device) < len(batch_indices)
    expected_steps = torch.as_tensor(
      lengths[env_clip_indices] - 1, dtype=torch.long, device=device
    )
    steps = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    returns = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
    finished_steps = torch.zeros_like(steps)
    finished_returns = torch.zeros_like(returns)
    finished_failed = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)

    obs, _ = env.reset()
    selected_frames = starts[
      torch.as_tensor(env_clip_indices, dtype=torch.long, device=device)
    ]
    command.reset_to_frames(env_ids, selected_frames)
    base_env.sim.forward()
    command.update_relative_body_poses()
    base_env.sim.sense()
    obs = env.get_observations()

    full_state = obs["full_state"]
    rep_state = wm.representation.represent(full_state)
    aux_state = wm.representation.extract_aux_state(full_state)
    sensor = wm.representation.compute_actor_sensor(
      rep_state,
      sim_context,
      sim_actor_obs=obs["actor"] if runner_cfg.actor_obs == "proprio" else None,
    )
    history = sensor.unsqueeze(1).repeat(1, runner_cfg.history_length, 1)

    for _ in range(int(expected_steps.max())):
      actor_goal = motion_task.extract_training_goal(
        motion, command.time_steps, aux_state
      )
      actor_obs = build_actor_obs_td({"motion": actor_goal}, history)
      actions = actor(actor_obs)
      next_obs, reward, dones, _ = env.step(actions)
      steps[active] += 1
      returns[active] += reward[active]
      terminated = base_env.reset_terminated.bool()
      finished = active & (dones.bool() | terminated | (steps >= expected_steps))
      finished_steps[finished] = steps[finished]
      finished_returns[finished] = returns[finished]
      finished_failed[finished] = terminated[finished]
      active[finished] = False

      next_state = next_obs["full_state"]
      next_rep_state = wm.representation.represent(next_state)
      next_aux_state = wm.representation.extract_aux_state(next_state)
      next_sensor = wm.representation.compute_actor_sensor(
        next_rep_state,
        sim_context,
        sim_actor_obs=next_obs["actor"] if runner_cfg.actor_obs == "proprio" else None,
      )
      if runner_cfg.history_length == 1:
        history = next_sensor.unsqueeze(1)
      else:
        history = torch.cat((history[:, 1:], next_sensor.unsqueeze(1)), dim=1)
      obs = next_obs
      rep_state = next_rep_state
      aux_state = next_aux_state

    batch_steps = finished_steps.cpu().tolist()
    batch_returns = finished_returns.cpu().tolist()
    batch_failed = finished_failed.cpu().tolist()
    for local_index, clip_index in enumerate(batch_indices):
      expected = int(lengths[clip_index] - 1)
      executed = int(batch_steps[local_index])
      episode_return = float(batch_returns[local_index])
      failed_motion = bool(batch_failed[local_index])
      results.append(
        {
          "name": names[clip_index],
          "frames": int(lengths[clip_index]),
          "expected_steps": expected,
          "executed_steps": executed,
          "return": episode_return,
          "normalized_return": episode_return / (max_step_reward * expected),
          "failed": failed_motion,
          "fail_step": executed if failed_motion else -1,
        }
      )
    rows = results[batch_start:batch_stop]
    with partial_path.open("a" if batch_start else "w", newline="", encoding="utf-8") as fh:
      writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
      if batch_start == 0:
        writer.writeheader()
      writer.writerows(rows)
    failed_count = sum(row["failed"] for row in results)
    completed_batch_steps += int(lengths[batch_indices].max()) - 1
    elapsed = time.monotonic() - started
    remaining = elapsed * (total_batch_steps - completed_batch_steps) / completed_batch_steps
    print(
      f"[INFO] Evaluated {len(results)}/{len(names)} motions | "
      f"failed {failed_count} | elapsed {elapsed / 60:.1f} min | "
      f"estimated remaining {remaining / 60:.1f} min",
      flush=True,
    )

  failed = sum(row["failed"] for row in results)
  incomplete = sum(row["executed_steps"] < row["expected_steps"] and not row["failed"] for row in results)
  mean_return = sum(row["normalized_return"] for row in results) / len(results)
  weighted_return = sum(row["return"] for row in results) / sum(
    max_step_reward * row["expected_steps"] for row in results
  )
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  with (cfg.output_dir / "per_motion.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
  summary = {
    "checkpoint": str(cfg.checkpoint),
    "completed_iterations": completed_iterations,
    "checkpoint_source": checkpoint_source,
    "task_id": cfg.task_id,
    "motions": len(results),
    "failed": failed,
    "incomplete": incomplete,
    "mean_per_motion_normalized_return": mean_return,
    "full_length_step_weighted_normalized_return": weighted_return,
    "num_envs": cfg.num_envs,
    "elapsed_seconds": time.monotonic() - started,
    "motion_file": str(cfg.motion_file),
    "motion_names_sha256": hashlib.sha256(
      (cfg.motion_file / "motion_names.json").read_bytes()
    ).hexdigest(),
    "evaluation": {
      "start_frame": "first frame of each motion",
      "action_noise": 0.0,
      "observation_noise": False,
      "domain_randomization": False,
      "foot_friction": 1.0,
      "normalized_return_denominator": "maximum per-step reward × full motion steps",
    },
  }
  (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps(summary, indent=2), flush=True)
  env.close()


if __name__ == "__main__":
  tyro.cli(main)
