"""Render the reference motion; this is not a trained-policy demonstration."""
import json
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig
from PIL import Image

base = Path("/workspace")
motion = np.load(base / "data/run1_subject5_50hz.npz")
cfg = unitree_g1_flat_tracking_env_cfg()
cfg.scene.num_envs = 1
scene = Scene(cfg.scene, "cuda:0")
sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=scene.compile(), device="cuda:0")
scene.initialize(sim.mj_model, sim.model, sim.data)
robot = scene["robot"]
assert list(robot.joint_names) == motion["joint_names"].tolist()
renderer = OffscreenRenderer(
    sim.mj_model,
    ViewerConfig(
        height=720,
        width=1280,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        distance=3.0,
        elevation=-15,
        azimuth=45,
    ),
    scene,
)
renderer.initialize()
sample_indices = (0, 3000, 6000, 9000, 11890)
frame_count = 0
try:
    with media.VideoWriter(
        str(base / "reports/reference-run1-full-720p.mp4"), (720, 1280), fps=25
    ) as writer:
        for i in range(0, len(motion["joint_pos"]), 2):
            pose = np.concatenate([motion["root_pos_w"][i], motion["root_quat_w"][i]])
            robot.write_root_link_pose_to_sim(torch.tensor(pose[None], device="cuda:0"))
            robot.write_joint_position_to_sim(
                torch.tensor(motion["joint_pos"][i : i + 1], device="cuda:0")
            )
            renderer.update(sim.data)
            frame = renderer.render()
            writer.add_image(frame)
            frame_count += 1
            if i in sample_indices:
                Image.fromarray(frame).save(base / f"reports/reference-full-{i:05d}.png")
finally:
    renderer.close()
(base / "reports/render-full-check.json").write_text(
    json.dumps({"frames": frame_count, "fps": 25, "resolution": [1280, 720],
                "content": "kinematic reference playback, not trained policy",
                "joint_name_order": "passed", "offscreen_render": "passed"}, indent=2)
    + "\n", encoding="utf-8"
)
