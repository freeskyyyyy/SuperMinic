#!/usr/bin/env bash
set -eu
cd /workspace/src/SuperMimic-Reproduced
.venv/bin/python scripts/data/csv_to_npz_batch.py \
  --input-dir /workspace/data/raw/lafan1/g1 \
  --output-dir /workspace/data/processed/lafan1 \
  --input-fps 30 --output-fps 50 --workers 2
mkdir -p data/raw data/npz
ln -s /workspace/data/raw/lafan1 data/raw/lafan1
ln -s /workspace/data/processed/lafan1 data/npz/lafan1
.venv/bin/python -m pytest \
  tests/test_motion_convert.py tests/test_g1_fk.py tests/test_motion_library_compact.py \
  tests/test_analytic_heads.py tests/test_env_smoke.py \
  tests/test_world_model.py::test_state_spec_identity_and_errors \
  tests/test_imagination.py::test_gae_targets_match_hand_computation \
  -q -rA
env -u PYTHONPATH .venv/bin/python -m supermimic.scripts.train --help > /workspace/reports/P1_train_help.txt
