"""Correct the observed SBSA wheel tag after verifying the binary is AArch64."""

import base64
import csv
import hashlib
import io
import json
import shutil
import struct
from pathlib import Path

root = Path(__file__).resolve().parents[1]
site = root / "src/SuperMimic-Reproduced/.venv/lib/python3.12/site-packages"
info = site / "nvidia_cusparselt_cu13-0.8.1.dist-info"
wheel_path = info / "WHEEL"
record_path = info / "RECORD"
original_tag = "Tag: py3-none-manylinux2014_sbsa"
correct_tag = "Tag: py3-none-manylinux2014_aarch64"
wheel_text = wheel_path.read_text()
tags = [line for line in wheel_text.splitlines() if line.startswith("Tag:")]
with (site / "nvidia/cusparselt/lib/libcusparseLt.so.0").open("rb") as library:
    header = library.read(20)
machine = struct.unpack("<H", header[18:20])[0]
assert tags in ([original_tag], [correct_tag]) and header[:4] == b"\x7fELF" and machine == 183
if tags == [correct_tag]:
    print("cuSPARSELt AArch64 tag already corrected")
else:
    backup = root / "environment/patches/nvidia-cusparselt-cu13-0.8.1"
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel_path, backup / "WHEEL.original")
    shutil.copy2(record_path, backup / "RECORD.original")
    corrected = wheel_text.replace(original_tag, correct_tag).encode()
    wheel_path.write_bytes(corrected)
    rows = list(csv.reader(io.StringIO(record_path.read_text())))
    key = info.name + "/WHEEL"
    for row in rows:
        if row[0] == key:
            row[1] = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(corrected).digest()).decode().rstrip("=")
            row[2] = str(len(corrected))
    with record_path.open("w", newline="") as output:
        csv.writer(output, lineterminator="\n").writerows(rows)
    report = {"package": "nvidia-cusparselt-cu13==0.8.1", "original_tag": original_tag, "corrected_tag": correct_tag, "elf_machine": machine, "binary_modified": False, "scope": "installed WHEEL metadata and its RECORD entry only; original downloaded wheel retained"}
    (backup / "patch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
