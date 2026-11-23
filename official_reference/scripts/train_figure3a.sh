#!/usr/bin/env bash
set -euo pipefail

run="$(date +%Y%m%d_%H%M%S)"
run_dir="runs/train/$run"
mkdir -p runs/train checkpoints
mkdir "$run_dir"
stop_file="/workspace/$run_dir/STOP"
printf 'Run directory: %s\nStop command: docker exec supermimic-research touch %s\n' "$run_dir" "$stop_file"

docker exec -e WANDB_MODE=offline -e PYTHONUNBUFFERED=1 \
  -e MPLCONFIGDIR=/tmp/supermimic-matplotlib \
  -e TRITON_CACHE_DIR=/workspace/cache/triton \
  -e WANDB_DATA_DIR=/workspace/cache/wandb \
  -w /workspace/src/supermimic supermimic-research \
  .venv/bin/python src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --motion-file /workspace/data/bones_seed_129491_mjlab_fp32 \
  --device cuda:0 --num-envs 512 --env.seed 42 \
  --runner.run-name "$run" \
  --runner.table-a5 True --runner.actor-obs proprio --runner.history-length 10 \
  --runner.wm.horizon 9 --runner.wm.num-batches 10 --runner.wm.batch-size 128 \
  --runner.policy.batch-size 128 --runner.replay-ratio 9 \
  --runner.max-iterations "${FIGURE3A_MAX_ITERATIONS:-320000}" \
  --runner.save-interval 1000 --runner.eval-checkpoint-interval 4000 --runner.log-interval 50 \
  --runner.checkpoint-training True --runner.upload-checkpoint-artifacts False \
  --runner.stop-file "$stop_file" "$@" \
  2>&1 | tee "$run_dir/train.log" | awk -v run="$run" '
    /^\[INFO\] Logging to logs\/SuperMimic-GenCoord-G1\/[0-9-]+_[0-9-]+$/ {
      system("ln -s ../src/supermimic/" $4 " checkpoints/" run)
      print; fflush(); next
    }
    /^(solver iterations limit reached|linesearch iterations limit reached)/ { next }
    /^To disable the print warning:.*OverflowType\.(LS_ITERATIONS|ITERATIONS)([[:space:]]|$)/ { next }
    /^#{40,}$/ { next }
    /^\[INFO\] Training iteration: [0-9]+$/ { iteration = $NF; next }
    /Learning iteration [0-9]+\/[0-9]+/ {
      block = 1
      reward = length_mean = wm = policy = value_loss = "-"
      next
    }
    block && $1 == "Total" && $2 == "steps:" { steps = $NF; next }
    block && $1 == "Steps" && $2 == "per" { fps = $NF; next }
    block && $1 == "Collection" && $2 == "time:" { sim = $NF; next }
    block && $1 == "WM" && $2 == "Learning" { wm_time = $NF; next }
    block && $1 == "Policy" && $2 == "Learning" { policy_time = $NF; next }
    block && $1 == "Mean" && $2 == "reward:" { reward = $NF; next }
    block && $1 == "Mean" && $2 == "episode" { length_mean = $NF; next }
    block && $1 == "Loss/state:" { wm = $NF; next }
    block && $1 == "Policy/reward:" { policy = $NF; next }
    block && $1 == "Policy/value_function:" { value_loss = $NF; next }
    block && $1 == "Time" && $2 == "elapsed:" { elapsed = $NF; next }
    block && $1 == "ETA:" {
      printf "it %s | steps %d | %s | %.1f it/s | train_ret %s len %s | wm_loss %s imag_reward %s value_loss %s | t sim/wm/pi %s/%s/%s\n",
        iteration, steps, elapsed, fps / 512, reward, length_mean, wm, policy, value_loss, sim, wm_time, policy_time
      fflush()
      block = 0
      next
    }
    block && (/^[[:space:]]*$/ || /^-{40,}$/ ||
      /^[[:space:]]*(Run name|Iteration time|Episode |[^[:space:]:]+\/[^[:space:]:]+):/) { next }
    { print; fflush() }
  '
