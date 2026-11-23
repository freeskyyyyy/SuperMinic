#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <checkpoint-in-container> <new-output-dir-in-container>" >&2
  exit 2
fi

checkpoint="$1"
output_dir="$2"
if docker exec supermimic-research test -e "$output_dir"; then
  echo "Output directory already exists: $output_dir" >&2
  exit 1
fi
docker exec supermimic-research mkdir -p "$output_dir"

docker exec -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/supermimic-matplotlib \
  -e TRITON_CACHE_DIR=/workspace/cache/triton \
  -w /workspace/src/supermimic supermimic-research \
  .venv/bin/python src/supermimic/scripts/evaluate_figure3a.py \
  --cfg.checkpoint "$checkpoint" \
  --cfg.motion-file /workspace/data/bones_seed_129491_mjlab_fp32 \
  --cfg.output-dir "$output_dir/results" \
  --cfg.num-envs "${FIGURE3A_EVAL_ENVS:-128}" \
  2>&1 | docker exec -i supermimic-research tee "$output_dir/evaluation.log" | awk '
    /^(solver iterations limit reached|linesearch iterations limit reached)/ { next }
    /^To disable the print warning:.*OverflowType\.(LS_ITERATIONS|ITERATIONS)([[:space:]]|$)/ { next }
    { print; fflush() }
  '
