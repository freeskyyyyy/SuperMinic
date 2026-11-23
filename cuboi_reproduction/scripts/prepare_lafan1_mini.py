"""Fetch four fixed-revision raw G1 clips; reuse the previously verified run CSV."""

import csv
import hashlib
import json
import math
import shutil
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
revision = "ce1572906efe6157840e8474d5a0d7aa87481e74"
names = ["walk1_subject1", "run1_subject5", "dance1_subject1", "jumps1_subject1"]
destination = root / "data/raw/lafan1/g1"
manifest_dir = root / "data/manifests"
destination.mkdir(parents=True, exist_ok=True)
manifest_dir.mkdir(parents=True, exist_ok=True)
records = []
for name in names:
    path = destination / (name + ".csv")
    url = f"https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/{revision}/g1/{name}.csv"
    if name == "run1_subject5":
        shutil.copyfile("/home/nvidia/research/supermimic-repro/data/run1_subject5.csv", path)
    else:
        with urllib.request.urlopen(url, timeout=90) as response, path.open("wb") as target:
            shutil.copyfileobj(response, target)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if name == "run1_subject5":
        assert digest == "eacc38d7d6f5d7f32bfefcdf5d91510fea5f815000be870f91cc6e3483a126f6"
    with path.open(newline="") as source:
        rows = list(csv.reader(source))
    assert rows and all(len(row) == 36 for row in rows), name
    assert all(math.isfinite(float(value)) for row in rows for value in row), name
    record = {
        "name": name, "url": url, "revision": revision, "sha256": digest,
        "bytes": path.stat().st_size, "frames": len(rows), "columns": 36,
        "input_fps": 30, "target_fps": 50, "raw_quaternion": "xyzw",
    }
    records.append(record)
    print(json.dumps(record), flush=True)
shutil.copyfile("/home/nvidia/research/supermimic-repro/data/LAFAN1_LICENSE", destination.parent / "LICENSE")
shutil.copyfile("/home/nvidia/research/supermimic-repro/data/LAFAN1_README.md", destination.parent / "README.source.md")
(manifest_dir / "lafan1_mini_raw.json").write_text(json.dumps(records, indent=2) + "\n")
(manifest_dir / "lafan1_mini_eval.txt").write_text("\n".join(names) + "\n")
