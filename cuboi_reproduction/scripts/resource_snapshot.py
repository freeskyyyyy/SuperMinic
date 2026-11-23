"""Record the current host/course state; this is not a course peak-load test."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def output(*args):
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


root = Path(__file__).resolve().parents[1]
memory = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    key, value = line.split(":", 1)
    if key in ("MemTotal", "MemAvailable", "SwapTotal"):
        memory[key + "_bytes"] = int(value.split()[0]) * 1024
report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "current occupancy only, not representative course peak or concurrency acceptance",
    "host": output("hostname"),
    "architecture": output("uname", "-m"),
    "logical_cpus": int(output("nproc")),
    "memory": memory,
    "disk": output("df", "-B1", str(root)),
    "gpu": output("nvidia-smi"),
    "containers": output("docker", "ps", "--format", "{{.Names}}|{{.Image}}|{{.Status}}"),
    "container_usage": output("docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}"),
}
destination = root / "reports" / "resource_baseline_20261116.json"
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
