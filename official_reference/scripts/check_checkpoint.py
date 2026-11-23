"""Load local training weights in a new process and evaluate from clip start."""
import json
import sys
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import gencoord_env_cfg
from supermimic.tasks.task_spec import build_actor_obs_td


base = Path("/workspace")
checkpoint = Path(sys.argv[1])
assert checkpoint.is_dir(), checkpoint
cfg = load_runner_cfg({"runner_cfg": json.loads(
    (base / "reports/training-smoke-config.json").read_text()
)})
env_cfg = gencoord_env_cfg(play=True)
env_cfg.commands["motion"].motion_file = str(base / "data/run1_subject5_50hz.npz")
env_cfg.commands["motion"].sampling_mode = "start"
env_cfg.observations["actor"].enable_corruption = False
env_cfg.episode_length_s = 240.0
env = create_env(env_cfg, 1, "cuda:0")
wm = build_world_model(cfg, env, "cuda:0")
wm.load_bundle(torch.load(checkpoint / "world_model.pt", map_location="cuda:0"))
actor = build_actor(cfg, wm, "cuda:0")
actor.load_state_dict(torch.load(checkpoint / "actor.pt", map_location="cuda:0"))
actor.eval()
env.reset()
command = env.unwrapped.command_manager.get_term("motion")
motion_spec = actor.task_specs[0]
history = env.get_observations()["actor"].unsqueeze(1).repeat(1, cfg.history_length, 1)
renderer = OffscreenRenderer(env.unwrapped.sim.mj_model, ViewerConfig(
    height=720, width=1280, origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot", distance=3.0, elevation=-15, azimuth=45,
), env.unwrapped.scene)
renderer.initialize()
total_reward = 0.0
full_steps = command.motion.time_step_total - 1
maximum_step_reward = env.unwrapped.step_dt * sum(
    term.weight for term in env.cfg.rewards.values() if term.weight > 0
)
failed = False
try:
    with torch.inference_mode(), media.VideoWriter(
        str(base / "reports/trained-smoke-eval-720p.mp4"), (720, 1280), fps=25
    ) as writer:
        for step in range(full_steps):
            obs = env.get_observations()
            state = obs["full_state"]
            aux = wm.representation.extract_aux_state(state)
            history = torch.cat([history[:, 1:], obs["actor"].unsqueeze(1)], dim=1)
            goal = motion_spec.extract_training_goal(command.motion, command.time_steps, aux)
            action = actor(build_actor_obs_td({"motion": goal}, history))
            assert torch.isfinite(action).all().item()
            if step % 2 == 0:
                renderer.update(env.unwrapped.sim.data)
                writer.add_image(renderer.render())
            _, reward, done, extras = env.step(action)
            total_reward += reward.item()
            if done.item():
                failed = bool(env.unwrapped.termination_manager.terminated.item())
                break
finally:
    renderer.close()
    env.close()
report = {
    "checkpoint": str(checkpoint), "checkpoint_load": "passed",
    "executed_steps": step + 1, "full_clip_steps": full_steps,
    "early_failure": failed, "episode_reward": total_reward,
    "full_clip_normalized_return": total_reward / (maximum_step_reward * full_steps),
    "tracking_reproduction_passed": not failed and step + 1 == full_steps,
    "training_budget": "31 engineering iterations; no convergence claim",
}
(base / "reports/checkpoint-eval.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
