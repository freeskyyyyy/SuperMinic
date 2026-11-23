"""Measure one command in this research container, without changing its limits."""

import argparse
import json
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
assert args.command, "a command is required"
start = time.monotonic()
samples = []
child = subprocess.Popen(args.command)
while child.poll() is None:
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    samples.append({
        "elapsed_seconds": time.monotonic() - start,
        "container_memory_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text()),
        "host_available_bytes": int(memory["MemAvailable"].split()[0]) * 1024,
        "whole_gpu_util_percent": int(gpu.stdout.strip()),
    })
    time.sleep(1)
code = child.wait()
elapsed = time.monotonic() - start
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "command": args.command,
    "exit_code": code,
    "wall_seconds": elapsed,
    "child_user_cpu_seconds": usage.ru_utime,
    "child_system_cpu_seconds": usage.ru_stime,
    "average_child_cpu_cores": (usage.ru_utime + usage.ru_stime) / elapsed,
    "child_peak_rss_bytes": usage.ru_maxrss * 1024,
    "sampled_peak_container_memory_bytes": max(s["container_memory_bytes"] for s in samples),
    "sampled_min_host_available_bytes": min(s["host_available_bytes"] for s in samples),
    "sampled_peak_whole_gpu_util_percent": max(s["whole_gpu_util_percent"] for s in samples),
    "notes": "1-second samples; GPU utilization and host memory include other work; cgroup/RSS are not complete GPU-memory accounting on unified memory",
    "samples": samples,
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2), flush=True)
raise SystemExit(code)
