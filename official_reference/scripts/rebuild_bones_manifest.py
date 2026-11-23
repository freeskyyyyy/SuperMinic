"""Rebuild a BONES-SEED selection from SuperMimic v2 Appendix E on CPU.

Reads the previously converted 50 Hz clips and writes a manifest to stdout.
The qualitative floating-feet gates use the documented CuboiLeo thresholds.
"""

import argparse
import csv
import multiprocessing as mp
from pathlib import Path
import re
import sys

import numpy as np

# Defaults of NVlabs/GR00T-WholeBodyControl's filter_and_copy_bones_data.py.
SONIC_KEYWORDS = (
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on",
    "seat", "table", "_sit_", "sit_", "ladder", "crutch", "_bed_", "_ride_",
    "scooter", "stepdown", "acrobatics_", "box_hspu", "cartwheel", "50cm_box_",
    "on_box", "fall_from", "handstand_ff_", "on_1m", "form_box", "off_1m",
    "230m", "jump_over_obstacle_", "lift_crate_come_up_", "jump_to_shoulder_roll",
    "kozak_dance", "stair", "handstand", "box_jump", "monkey_jump",
    "safety_roll", "box_dips", "walking_on_edge", "push_obstacle",
)
# The other Appendix E examples already occur in SONIC's list. Bound the
# additions by tokens so that "box", for example, does not exclude boxing.
PAPER_EXAMPLES = re.compile(r"(?:^|_)(box|sit|jump_off_50cm|look_over_fence)(?:_|$)")
FPS = 50


def name_rejection(name):
    normalized = name.casefold()
    for keyword in SONIC_KEYWORDS:
        if keyword in normalized:
            return f"name:{keyword}"
    match = PAPER_EXAMPLES.search(normalized)
    return f"name:{match[1]}" if match else ""


def longest_run(mask):
    edges = np.flatnonzero(np.diff(mask, prepend=False, append=False))
    return int((edges[1::2] - edges[::2]).max(initial=0))


def body_rejection(lowest_z):
    if longest_run(lowest_z > 0.20) >= 30:
        return "elevated_support"
    if (lowest_z < -0.12).any() or (lowest_z < -0.08).mean() > 0.80:
        return "ground_penetration"
    return ""


class FeasibilityFilter:
    def __init__(self, xml_path, first_clip):
        import mujoco

        self.mj = mujoco
        spec = mujoco.MjSpec.from_file(str(xml_path))
        spec.worldbody.add_geom(
            name="figure3a_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.1)
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.floor = self.model.geom("figure3a_floor").id
        self.feet = [
            self.model.geom(f"{side}_foot{index}_collision").id
            for side in ("left", "right")
            for index in range(1, 8)
        ]
        with np.load(first_clip) as clip:
            self.pelvis = clip["body_names"].tolist().index("pelvis")
            joint_names = [self.model.joint(i).name for i in range(1, self.model.njnt)]
            if joint_names != clip["joint_names"].tolist() or float(clip["fps"].item()) != FPS:
                raise ValueError("Expected the G1 joint order and 50 Hz input")

    def rejection(self, path):
        with np.load(path) as clip:
            positions = clip["body_pos_w"]
            reason = body_rejection(positions[:, :, 2].min(axis=1))
            if reason:
                return reason

            pelvis_pos = positions[:, self.pelvis]
            pelvis_quat = clip["body_quat_w"][:, self.pelvis]
            pelvis_vel = clip["body_lin_vel_w"][:, self.pelvis]
            joint_vel = clip["joint_vel"]
            upright = (1 - 2 * np.square(pelvis_quat[:, 1:3]).sum(axis=1)) > 0.9
            standing = (pelvis_pos[:, 2] > 0.60) & (pelvis_pos[:, 2] < 0.85)
            still = (np.linalg.norm(pelvis_vel, axis=1) < 0.10) & (
                np.sqrt(np.square(joint_vel).mean(axis=1)) < 0.50
            )
            gate = upright & standing & still
            if longest_run(gate) < 25:
                return ""
            joint_pos = clip["joint_pos"]

        run = 0
        previous = -2
        for frame in np.flatnonzero(gate):
            if frame != previous + 1:
                run = 0
            self.data.qpos[:3] = pelvis_pos[frame]
            self.data.qpos[3:7] = pelvis_quat[frame]
            self.data.qpos[7:] = joint_pos[frame]
            self.mj.mj_kinematics(self.model, self.data)
            floating = all(
                self.mj.mj_geomDistance(self.model, self.data, self.floor, foot, 1.0, None)
                > 0.02
                for foot in self.feet
            )
            run = run + 1 if floating else 0
            if run >= 25:
                return "floating_feet"
            previous = frame
        return ""


def init_worker(clip_dir, xml_path, first_clip):
    global CLIP_DIR, FEASIBILITY
    CLIP_DIR = clip_dir
    FEASIBILITY = FeasibilityFilter(xml_path, first_clip)


def classify(row):
    name = row["name"]
    reason = name_rejection(name) or FEASIBILITY.rejection(CLIP_DIR / f"{name}.npz")
    return name, row["frames"], int(not reason), reason, row["keep"], row["reason"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Only process the first N rows for a smoke test")
    args = parser.parse_args()
    if args.g1_xml is None:
        import mjlab

        args.g1_xml = Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
    with args.previous_manifest.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))[: args.limit]
    first_clip = args.npz_dir / f'{rows[0]["name"]}.npz'
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("name", "frames", "keep", "reason", "previous_keep", "previous_reason"))
    with mp.Pool(args.workers, init_worker, (args.npz_dir, args.g1_xml, first_clip)) as pool:
        writer.writerows(pool.imap(classify, rows, chunksize=64))


if __name__ == "__main__":
    main()
