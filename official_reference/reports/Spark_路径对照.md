# Spark 目录整理路径对照

以 `/home/nvidia/research/supermimic-repro` 为根目录；容器内根目录为 `/workspace`。结果 JSON 中的历史路径保持原样，旧链接已移除。

- `runs/figure3a_full512_a5_seed42_日期_时间` → `runs/train/日期_时间`。
- `runs/eval_probe1024_iteration116000_retry1_20261119` → `runs/eval/step164000_01` → `runs/eval/iter164001_01`。
- `runs/eval_probe1024_iteration116000_retry2_20261119` → `runs/eval/step164000_02` → `runs/eval/iter164001_02`。
- `runs/eval/step319001` → `runs/eval/iter319001`。
- `scripts/train_figure3a.sh` → `scripts/train.sh`；`scripts/evaluate_figure3a.sh` → `scripts/eval.sh`。
- `scripts/check_*.py`、`scripts/verify_resume_check.py`、`scripts/subset_packed_motions.py` → `scripts/tools/`。
- `scripts/figure3a_probe1024.txt` 已去重，唯一清单为 `data/manifests/figure3a_probe1024.txt`，快捷入口为 `data/probe1024.txt`。
- `data/run1_subject5*`、`data/LAFAN1_*` → `data/examples/lafan1/`。
- `reports/20261118_图3a数据与配置核对.md` → `reports/figure3a/`；早期报告文件 → `reports/archive/`。
- `checkpoints/20261119_120302`、`20261119_120932`、`20261119_173127` 分别链接作者模型目录 `2026-11-19_04-03-13`、`04-09-43`、`09-31-50`。

历史报告内的原始路径记录不改写；脚本中的实际数据和报告路径已更新。

计数口径：训练状态 `next_step` 表示已完成迭代数；旧定期快照目录 `iteration_000164000` 表示刚执行的零起点迭代编号，因此已完成 164001 次。历史结果的 `checkpoint_step` 保留当时口径，不改写。
