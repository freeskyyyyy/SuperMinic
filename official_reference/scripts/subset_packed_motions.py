"""Create a compact packed-motion directory from a named clip list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FIELDS = (
  "joint_pos",
  "joint_vel",
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True, help="packed motion directory")
  parser.add_argument("--clip-list", type=Path, required=True, help="one motion name per line")
  parser.add_argument("--output", type=Path, required=True, help="new packed motion directory")
  args = parser.parse_args()

  source_names = json.loads((args.source / "motion_names.json").read_text())
  requested_names = [line.strip() for line in args.clip_list.read_text().splitlines() if line.strip()]
  source_ends = np.load(args.source / "motion_ends.npy", mmap_mode="r")
  if (
    not requested_names
    or len(set(requested_names)) != len(requested_names)
    or set(requested_names).difference(source_names)
    or args.output.exists()
  ):
    raise ValueError("clip list must be nonempty, unique, present in the source, and output must be new")

  source_index = {name: index for index, name in enumerate(source_names)}
  selected = [source_index[name] for name in requested_names]
  starts = np.concatenate(([0], source_ends[:-1]))
  lengths = np.asarray([source_ends[index] - starts[index] for index in selected], dtype=np.int64)
  total_frames = int(lengths.sum())
  args.output.mkdir(parents=True)
  (args.output / "joint_names.json").write_bytes((args.source / "joint_names.json").read_bytes())
  np.save(args.output / "motion_ends.npy", np.cumsum(lengths))
  (args.output / "motion_names.json").write_text(json.dumps(requested_names))

  for field in FIELDS:
    source_array = np.load(args.source / f"{field}.npy", mmap_mode="r")
    target_array = np.lib.format.open_memmap(
      args.output / f"{field}.npy",
      mode="w+",
      dtype=source_array.dtype,
      shape=(total_frames, *source_array.shape[1:]),
    )
    cursor = 0
    for index, length in zip(selected, lengths, strict=True):
      end = int(source_ends[index])
      start = int(starts[index])
      next_cursor = cursor + int(length)
      target_array[cursor:next_cursor] = source_array[start:end]
      cursor = next_cursor
    target_array.flush()
    del target_array
    print(f"{field}: {total_frames} frames, {source_array.dtype}", flush=True)

  summary = {
    "source": str(args.source),
    "clips": len(requested_names),
    "frames": total_frames,
    "max_clip_frames": int(lengths.max()),
    "motion_names": requested_names,
  }
  (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  print(f"Wrote {len(requested_names)} clips / {total_frames} frames to {args.output}", flush=True)


if __name__ == "__main__":
  main()
