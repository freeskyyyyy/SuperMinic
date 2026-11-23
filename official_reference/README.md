# 官方实现阶段参考归档

本路线现作为参考保存。当前工作基础已切换为 CuboiLeo/SuperMimic-Reproduced，见相邻 cuboi_reproduction 目录。

## 文件位置

| 目录或文件 | 内容 |
|---|---|
| src/supermimic/ | 作者官方仓库、原Git历史、reproduction-baseline分支和未提交改动 |
| src/mjlab_research/ | 作者指定依赖及原Git历史 |
| reports/ | 环境与数据核查、CPU/GPU检查、原始日志、截图和参考视频 |
| scripts/ | 此前GPU、仿真、渲染、短训和续训检查脚本 |
| environment/ | Dockerfile、EGL供应商配置、原环境设置记录 |
| HISTORY.md | 此前官方阶段操作记录，历史路径按下面的迁移映射理解 |
| runs/ | 本机的运行结果归档入口；现有大型权重及完整运行结果主要在Spark |

## 迁移映射（本机）

- research/src → research/official_reference/src。
- research/reports → research/official_reference/reports；其中第三方e8检查报告移至 cuboi_reproduction/reports，Dyn兼容性报告移至 comparisons。
- research/check_*.py、verify_resume_check.py → research/official_reference/scripts。
- research/Dockerfile、10_nvidia.json、环境设置与运行.md → research/official_reference/environment。
- 历史记录中的原路径只用于溯源；当前文件位置以本索引为准。原有数据、代码改动与结果未删除。

## Spark的位置及路径约束

- 主机根目录仍为 /home/nvidia/research/supermimic-repro，容器 supermimic-research 将其挂载到 /workspace。
- src、data、cache、runs及源码仓库中的 .venv、logs 保持原位，避免破坏已安装路径。
- 根目录检查脚本已移到 scripts；Dockerfile 和 10_nvidia.json 已移到 environment。重建镜像应使用 environment 作为构建上下文。
- 31迭代短训权重保留在 src/supermimic/logs/SuperMimic-GenCoord-G1/2026-11-16_02-55-38。
- 16→20步续训验证权重保留在 runs/resume-check-20261116a/phase-save 和 phase-resume。
- reports 保存了本机未同步的大型完整参考视频与运行日志；这些文件并不表示已完成动作跟踪复现。

## 保存的状态

独立环境、物理步进、参考回放和短训工程检查已执行；31迭代策略评估约0.64s提前终止，动作跟踪未收敛。后来续训短验证通过，但仿真重启采用新回合。课程联合负载验证已按用户要求停止，尚未通过。k窗口计数的默认训练逻辑未修改；续训相关修改属于本地工程补丁，不能称为作者原始代码。
