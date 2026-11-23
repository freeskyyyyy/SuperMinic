# 独立研究环境操作记录

> 2026-11-16：当前实现基础已切换为 CuboiLeo/SuperMimic-Reproduced，工作入口见 [cuboi_reproduction/README.md](cuboi_reproduction/README.md)。以下内容是此前官方实现阶段的操作记录，原目录和容器保留作参考。

> 2026-11-16：当前实现基础已切换为 CuboiLeo/SuperMimic-Reproduced，工作入口见 [cuboi_reproduction/README.md](cuboi_reproduction/README.md)。以下内容是此前官方实现阶段的操作记录，原目录和容器保留作参考。

用户于2026-11-16明确授权执行计划。该记录只描述实际操作；计划见../../plans/SuperMimic_复现详细计划_20261116.md。

- Spark工作目录：/home/nvidia/research/supermimic-repro。
- 本机源码：research/src/supermimic、research/src/mjlab_research，分支reproduction-baseline。
- 独立基础镜像：supermimic-research:bootstrap；构建日志在Spark reports/bootstrap-build.log。
- 源码审计：reports/P0_源码审计.md；依赖版本：reports/dependency-audit.json。
- 研究容器计划使用4CPU/16GiB上限，64GiB主机余量保护线；课程GPU重任务时研究退出。尚未通过课程联合验收。
- 截至本记录首次创建：基础镜像仍构建中，尚未创建研究容器或安装Python依赖，未训练。
