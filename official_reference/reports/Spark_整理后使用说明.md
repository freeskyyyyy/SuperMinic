# SuperMimic 复现目录

Spark 项目目录：`/home/nvidia/research/supermimic-repro`，容器内对应 `/workspace`。

- `scripts/train.sh`：训练及续训；新日志写入 `runs/train/日期_时间/`，启动时显示停止命令。
- `scripts/eval.sh`：1024 动作评估，参数为容器内检查点及输出目录。
- `data/bones`：训练数据入口；`data/probe1024.txt`：评估清单入口；`data/examples/lafan1/`：早期样例和许可。
- `runs/train/`：训练日志；`runs/eval/`：评估结果。
- `checkpoints/`：模型目录的快捷入口；实际模型保留在作者仓库的 `src/supermimic/logs/`。
- `scripts/tools/`：数据准备及早期检查脚本；`src/`：作者代码和配套依赖；`reports/figure3a/`：复现报告；`reports/archive/`：早期记录；`environment/`：环境配置；`cache/`：缓存；`third_party/`：第三方参考。

## 最近一次训练

最近一次训练已运行到 320000 次，最新保存的训练状态为已完成 319001 次，路径为 `/workspace/checkpoints/20261119_181436/training_state.pt`。

```bash
cd /home/nvidia/research/supermimic-repro
bash scripts/train.sh --runner.resume-path /workspace/checkpoints/20261119_181436/training_state.pt --env.sim.njmax 512
```

## 评估

评估目录按已完成迭代数命名：`runs/eval/iter164001_01`、`iter164001_02` 和 `iter319001`。旧快照 `iteration_000164000` 的数字是从零开始的循环编号，保存时已完成 164001 次迭代；训练状态文件的 `next_step` 即已完成迭代数。新结果输出 `completed_iterations` 与 `checkpoint_source`，历史 JSON 保持原样；对应关系见 `reports/路径对照.md`。新训练启动时按日志中的作者模型目录自动建立 `checkpoints/日期_时间` 链接。

```bash
cd /home/nvidia/research/supermimic-repro
bash scripts/eval.sh /workspace/checkpoints/20261119_120932/eval_checkpoints/iteration_000164000 /workspace/runs/eval/iter164001_01
```

结果在输出目录的 `results/summary.json` 和 `results/per_motion.csv`；同目录重跑会覆盖结果和 `evaluation.log`。评估和训练同时运行可能显存不足。
