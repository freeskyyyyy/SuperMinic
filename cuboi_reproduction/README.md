# 当前复现入口：SuperMimic-Reproduced

2026-11-16 起，用户选择以 CuboiLeo/SuperMimic-Reproduced 为实现基础。先复现发布权重的加载、单动作物理闭环与评估，再决定续训、速度接口和研究扩展。

## 目录约定

| 目录 | 用途 |
|---|---|
| src/SuperMimic-Reproduced/ | 独立 Git 仓库；修改在 reproduction-baseline 分支，原始标签保留 |
| environment/ | ARM64 环境构建、实际依赖版本和启动说明；不复用官方项目 .venv |
| data/raw/ | 原始动作及来源清单，保留原文件 |
| data/processed/ | 按本项目转换器生成的数据；带来源、频率、坐标和关节映射记录 |
| checkpoints/upstream/e8-boneseed-100k/ | 发布者检查点、原始配置及已完成的文件检查记录；不覆盖 |
| runs/ | 每次回放、评估、训练的独立目录；不把新权重写回 upstream |
| reports/ | 本项目的对照、兼容性与验收报告 |
| scripts/ | 本地适配、检查和运行入口；通用算法修改留在源码仓库 |

每次实验命名采用 YYYYMMDD_任务_seedN_序号，内部按需保存 config、logs、metrics、checkpoints、videos。失败结果保留，不用新实验覆盖旧目录。上游自带 logs 默认入口在实际运行前指向 runs，并在报告中记录生效路径。

## 本机和 Spark

- 本机：D:/Coding Program/复现具身智能论文/research/cuboi_reproduction。
- Spark：/home/nvidia/research/supermimic-cuboi，通过 spark-company 连接。
- 两端使用相同的相对目录布局。环境和大规模运行在 Spark，源码、清单与报告在本机保留。
- 上游权重两端各保留一份，SHA256 见 workspace.json；数据集和运行缓存不无差别双向同步。

## 已固定的基线

- 源码：https://github.com/CuboiLeo/SuperMimic-Reproduced。
- 标签：e8-boneseed-100k；提交：ba6105349069bf040f1fdad60873d13138d505a2。
- 权重：e8_boneseed_ckpt_0100000.pt，223470223 字节。
- Python3.12、上游依赖锁及 mjlab 固定版本先按原样检查；ARM64 所需实际调整另存 environment，不能直接复制官方项目的依赖补丁。
- 附件 config.json 与检查点内 compact_library 不同，回放以配套加载代码读取的嵌入配置为准，并记录采用的数据模式。

## 旧目录如何处理

- research/official_reference/ 和现有官方研究容器保留官方实现阶段的源码及证据，不移入本项目、不自动继续其训练。Spark官方根目录仍为 /home/nvidia/research/supermimic-repro。
- research/third_party/CuboiLeo_e8/ 仅保留迁移指引，原文件已移到本目录 checkpoints/upstream。
- research/third_party/SuperMimic_Dyn/ 及 research/comparisons/ 是第二个项目的兼容性参考，不属于本次实现基线。
- output/ 和 tmp/ 中的论文译本及排版材料保持独立。

## 执行边界

具体步骤见[第三方复现详细计划R2](../../plans/CuboiLeo_第三方复现详细计划_20261116.md)，用户已批准开始执行；阶段状态见根目录task_plan.md。课程优先，4CPU/16GiB仅为首次检查设置，按实测调整，不表示GPU隔离或已测课程需求；轻量课程与研究可按实际表现并行，不自动恢复已叫停的联合验证。训练资源测量前移到首个训练闭环可运行后。
