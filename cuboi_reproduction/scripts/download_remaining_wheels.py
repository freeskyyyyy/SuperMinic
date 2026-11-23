"""Download the seven interrupted wheels separately, then verify original-lock hashes."""

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import urlparse

root = Path(__file__).resolve().parents[1]
lock = tomllib.loads((root / "src/SuperMimic-Reproduced/uv.lock").read_text())
names = {
    "nvidia-cublas", "nvidia-cudnn-cu13", "nvidia-cufft", "nvidia-cusolver",
    "nvidia-cusparselt-cu13", "nvidia-nccl-cu13", "triton",
}
packages = [p for p in lock["package"] if p["name"] in names]
assert len(packages) == len(names)
directory = root / "cache/wheelhouse"
directory.mkdir(parents=True, exist_ok=True)
jobs = []
for package in packages:
    wheel = next(w for w in package["wheels"] if "aarch64" in w["url"] and ("cp312" in w["url"] or "py3-none" in w["url"]))
    path = directory / Path(urlparse(wheel["url"]).path).name
    log = (directory / (package["name"] + ".download.log")).open("ab")
    process = subprocess.Popen([
        "curl", "--http1.1", "--noproxy", "*", "--fail", "--location",
        "--continue-at", "-", "--silent", "--show-error", "--output", str(path), wheel["url"],
    ], stdout=log, stderr=log)
    jobs.append((package, wheel, path, process, log))
    print(f"started {package['name']} {package['version']} size={wheel['size']}", flush=True)
codes = [process.wait() for _, _, _, process, _ in jobs]
for _, _, _, _, log in jobs:
    log.close()
assert all(code == 0 for code in codes), codes
records = []
for package, wheel, path, _, _ in jobs:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    assert path.stat().st_size == wheel["size"], package["name"]
    assert "sha256:" + digest == wheel["hash"], package["name"]
    record = {"name": package["name"], "version": package["version"], "url": wheel["url"], "file": str(path), "sha256": digest, "bytes": path.stat().st_size}
    records.append(record)
    print(f"verified {package['name']} {digest}", flush=True)
(root / "environment/wheelhouse-manifest.json").write_text(json.dumps(records, indent=2) + "\n")
(root / "environment/remaining-wheels.txt").write_text("\n".join(f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}" for r in records) + "\n")
