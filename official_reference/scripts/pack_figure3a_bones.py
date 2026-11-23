"""Pack the fixed BONES-SEED selection into mjlab's motion-directory format."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


ROOT_FIELDS = {
    "root_pos_w": ("body_pos_w", 3),
    "root_quat_w": ("body_quat_w", 4),
    "root_lin_vel_w": ("body_lin_vel_w", 3),
    "root_ang_vel_w": ("body_ang_vel_w", 3),
}


def pack(npz_dir: Path, manifest: Path, output: Path, expected_sha256: str) -> None:
    with manifest.open(encoding="utf-8", newline="") as source:
        selected = [(row["name"], int(row["frames"])) for row in csv.DictReader(source)
                    if row["keep"] == "1"]
    names = [name for name, _ in selected]
    digest = hashlib.sha256(("".join(f"{name}\n" for name in names)).encode()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Selected motion list hash differs from the fixed list: {digest}")
    ends = np.cumsum([frames for _, frames in selected], dtype=np.int64)
    with np.load(npz_dir / f"{names[0]}.npz") as sample:
        joints = sample["joint_names"].tolist()
        bodies = sample["body_names"].tolist()
        pelvis = bodies.index("pelvis")

    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as temporary:
        packed = Path(temporary) / "packed"
        packed.mkdir()
        widths = {name: width for name, (_, width) in ROOT_FIELDS.items()}
        widths.update(joint_pos=len(joints), joint_vel=len(joints))
        arrays = {
            name: np.lib.format.open_memmap(
                packed / f"{name}.npy", mode="w+", dtype=np.float32,
                shape=(int(ends[-1]), width),
            )
            for name, width in widths.items()
        }
        start = 0
        for index, ((name, _), end) in enumerate(zip(selected, ends, strict=True), start=1):
            with np.load(npz_dir / f"{name}.npz") as clip:
                if (float(clip["fps"].item()) != 50 or clip["joint_names"].tolist() != joints
                        or clip["body_names"].tolist() != bodies):
                    raise ValueError(f"Unexpected frame rate or robot layout in {name}")
                fields = {key: clip[key] for key in ("joint_pos", "joint_vel")}
                fields.update({key: clip[source][:, pelvis] for key, (source, _) in ROOT_FIELDS.items()})
                for key, data in fields.items():
                    if data.shape != (end - start, widths[key]) or not np.isfinite(data).all():
                        raise ValueError(f"Invalid {key} data in {name}")
                    arrays[key][start:end] = data
            start = end
            if index % 25_000 == 0:
                print(f"[pack] {index:,}/{len(names):,} clips", flush=True)
        for array in arrays.values():
            array.flush()
        del array
        arrays.clear()
        np.save(packed / "motion_ends.npy", ends)
        for name, value in (("joint_names", joints), ("motion_names", names)):
            (packed / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
        packed.rename(output)
    print(f"[pack] {len(names):,} clips, {ends[-1]:,} frames, list SHA256 {digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("expected_sha256")
    pack(**vars(parser.parse_args()))
