#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <checkpoint-path-in-container> <output-dir-in-container>" >&2
  exit 2
fi

checkpoint="$1"
output_dir="$2"
motion_subset="$output_dir/motions1024"

if docker exec supermimic-research test -f "$motion_subset/summary.json"; then
  echo "Reusing prepared motion subset: $motion_subset"
else
  docker exec -e PYTHONUNBUFFERED=1 \
    -w /workspace/src/supermimic supermimic-research \
    .venv/bin/python /workspace/scripts/tools/subset_packed_motions.py \
    --source /workspace/data/bones_seed_129491_mjlab_fp32 \
    --clip-list /workspace/data/manifests/figure3a_probe1024.txt \
    --output "$motion_subset"
fi

docker exec -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/tmp/supermimic-matplotlib \
  -e TRITON_CACHE_DIR=/workspace/cache/triton \
  -w /workspace/src/supermimic supermimic-research \
  .venv/bin/python src/supermimic/scripts/evaluate_figure3a.py \
  --cfg.checkpoint "$checkpoint" \
  --cfg.motion-file "$motion_subset" \
  --cfg.output-dir "$output_dir/results" \
  --cfg.num-envs "${FIGURE3A_EVAL_ENVS:-1024}" \
  2>&1 | docker exec -i supermimic-research tee "$output_dir/evaluation.log" | awk '
    /^(solver iterations limit reached|linesearch iterations limit reached)/ { next }
    /^To disable the print warning:.*OverflowType\.(LS_ITERATIONS|ITERATIONS)([[:space:]]|$)/ { next }
    { print; fflush() }
  '
