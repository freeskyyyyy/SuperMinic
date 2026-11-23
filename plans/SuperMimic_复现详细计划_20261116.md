# SuperMimic 复现详细计划

版本：2026-11-16 v2。用户已验收并明确授权执行。实施状态见根目录progress.md与research/reports；计划中的命令示例不代表每项已通过。

## 1. 已确认目标与完成定义

用户确认：复现动作跟踪与图7下游速度接口；优先使用持续可用的DGX Spark；希望尽早得到可验收结果，性能不足时再决定其他算力。本计划不假定可使用H100或真实G1，不产生租卡或购买承诺。

新增硬约束：提供可执行的环境设置步骤；课程项目优先，研究训练必须为课程保留资源。Spark持续可用不等于研究可持续独占GPU。

实施原则（用户2026-11-16补充）：入口只校验一次，拒绝重复判空和滥用try/catch；单次逻辑不乱抽函数，低概率防御先征求同意。

- 在输入边界集中校验，内部沿用已成立的约定，避免层层重复校验或判空。
- 异常处理仅用于有明确恢复办法的实际故障，不用宽泛捕获掩盖错误或假装成功。
- 单次使用、职责明确的逻辑直接表达；只在确有复用或独立职责时抽取函数。
- 为低概率假想风险新增防御代码前，先说明场景和代价并征求同意；已观察到的实际故障按证据修复。

实施原则（用户2026-11-16补充）：入口只校验一次，拒绝重复判空和滥用try/catch；单次逻辑不乱抽函数，低概率防御先征求同意。

- 在输入边界集中校验，内部沿用已成立的约定，避免层层重复校验或判空。
- 异常处理仅用于有明确恢复办法的实际故障，不用宽泛捕获掩盖错误或假装成功。
- 单次使用、职责明确的逻辑直接表达；只在确有复用或独立职责时抽取函数。
- 为低概率假想风险新增防御代码前，先说明场景和代价并征求同意；已观察到的实际故障按证据修复。

分三级交付：

1. 第一份工程结果：自己训练出的基础动作跟踪策略，在MuJoCo Warp中从检查点独立加载、完成预选动作，并有视频和数值评价。只展示官方权重或模型内想象回报不算完成。
2. 主要复现结果：固定数据与评估协议，完成动作跟踪基线、Distill速度接口、RL速度接口的训练及图7阶跃测试，报告与原文的差距。
3. 研究用基线：在可承受配置下重复至少3个训练随机种子，保存原始结果和统计，建立日后算法改进的对照。先用一个种子排通流程，再扩展重复实验。

当前不以真机迁移、双H100吞吐、全129514段零失败、论文全图全部重跑为首轮完成要求。缩小数据规模或改变训练配置的结果必须标明“缩小规模复现”；不能宣称完整数值复现。若性能没有达到目标，交付应标记差距与未完成项。

## 2. 本轮已查证事实

- 本地尚未有SuperMimic训练Git仓库；已有英文论文、译本和排版工具。已解析规划目录并沿用根目录规划文件；git diff --stat因非Git目录失败，无训练源码改动可供验收。
- 英文PDF在 `tmp/pdfs/supermimic_original.pdf`，19页。用户原Downloads路径当前不存在，后续直接用工作区副本。
- 官方SuperMimic页面本次显示提交 `abc124c1bd5976341034668af3d12c087f2f1f55`。README要求supermimic与mjlab_research同级，提供GenCoord/MaxCoord G1任务、train/play入口。
- 固定提交pyproject要求Python >=3.13,<3.14，mjlab来自相邻可编辑路径。还需选择兼容的mjlab_research提交及依赖锁文件，不能仅把两个仓库最新main拼起来就视为已验证。
- 固定提交的MultiTaskWorldModelEnv为各槽位使用motion tracking reward；Runner默认slot_mix为motion=1、velocity=0。速度编码器结构存在，不等于图7 RL速度任务已经完整实现。
- Runner基础配置含history_length=1、actor_obs=rep_state、max_iterations=1000000；任务配置可能覆盖。必须保存最终生效配置再与论文对照，不能单凭基类默认断言正式任务错误，也不能照默认直接宣布论文复现。
- ActorCriticPolicyLearner含H=32、gamma=0.99、lambda=0.95；策略更新时固定世界模型权重，critic另拟合GAE目标。实际配置和执行路径仍需运行核对。
- 论文图4使用LAFAN1 40段、2.45小时；图7使用LAFAN1+BONES-SEED筛选出的走跑库。两种数据协议不能混用。

以上是网页源码和PDF核对，不是Spark兼容性或训练性能测试。

## 3. 阶段总览与预计顺序

| 阶段 | 工作 | 通过后才能推进的事项 | 产物 |
|---|---|---|---|
| P0 | 固定源码、论文—实现对照 | 正式配置与缺口清楚 | 版本清单、差异表 |
| P1 | 独立研究容器与性能预检 | GPU仿真、PyTorch反向、课程联合运行通过 | 环境报告、短计时、资源限额 |
| P2 | 动作数据与G1映射 | 数据有效且可追溯 | 数据manifest、逐段检查 |
| P3 | 世界模型/策略/critic计算检查 | 三阶段参数更新符合论文 | 梯度与损失检查记录 |
| P4 | 小规模动作跟踪自训 | 检查点可独立在物理仿真评估 | 第一份可验收结果 |
| P5 | 固定动作库正式基线 | 全库评价与跨种子记录完整 | 跟踪基线报告 |
| P6 | 图7速度任务实现与对齐 | RL/Distill差异和接口路由正确 | 速度任务补丁、测试 |
| P7 | 两种速度接口训练及阶跃评估 | 协议一致、失败记录完整 | 图7复现与差距表 |
| P8 | 整理交付、研究问题筛选 | 基线可重跑、退化现象可复现 | 复现包、研究问题清单 |

P0/P1以及P2的资料整理可交错推进；P3/P4/P5/P6/P7存在依赖，不同时堆上未经验证的改动。Spark上的重任务先单作业串行，避免资源竞争使计时失真。

## 4. P0：源码与论文配置审计

### 工作步骤

1. 复现优先采用作者原生MuJoCo Warp/mjlab路线；官方代码、独立复现、部署仓库分开标注。第三方实现仅作为交叉参考，不混装配置和检查点。
2. 固定supermimic、mjlab_research提交；保存环境锁文件与dirty diff。选择GenCoord或MaxCoord前核对论文实验所用状态表示与任务覆盖，不凭名称决定等价。
3. 建立论文—代码对照表：物理步长/控制周期、G1关节与驱动、动作缩放、观测历史、未来参考帧、域随机化、噪声、奖励/终止、状态表示、reset采样、k/H、critic、优化器、训练日程、批量与更新数据比。
4. 特别核对论文k=8、H=32、10帧本体观测、K=10未来参考帧、32维latent、gamma=0.99、lambda=0.95，以及表A.5训练日程；记录实际任务覆盖后的值。
5. 核查三类缺口：作者代码现成支持、配置即可复现、必须按论文补实现。检查PPO-Sim/PPO-WM/阶跃评估入口是否存在；不预先承诺所有脚本已公开。
6. 核查保存与恢复语义：actor、world model、critic、归一化器、优化器、调度器、迭代计数、随机状态、buffer分别保存了什么。当前Runner可见的权重保存不能直接当作完整断点续训。

### 验收

能够说明选定提交对应论文哪些实验、哪些配置不同、哪些需要补齐；留出原始基线分支。无须为了审计先改算法。

## 5. P1：环境设置步骤与兼容性验收

### 5.1 本次只读检查结果

2026-11-16通过spark-company读取：Ubuntu 24.04.3、aarch64、20个逻辑CPU、GB10、驱动580.82.09；nvidia-smi显示CUDA支持版本13.0，这不等于已经安装CUDA 13工具链。系统内存121GiB、当时MemAvailable约114GiB、无swap，根分区可用约1.8TiB。系统Python为3.12.3，未发现uv。

当时GPU利用率0%，Docker默认上下文没有运行中的容器；robot-course1、course_x1_isaacsim、robot_g1、x1_robot均已停止。本轮未启动、停止或改动任何课程。该快照不能推导课程运行峰值，也不能证明课程当前可正常开展实验。

### 5.2 环境路线

使用独立ARM64研究容器supermimic-research，内部建立Python 3.13项目虚拟环境；宿主机目录建议为/home/nvidia/research/supermimic-repro。源码、数据、输出和缓存均在该独立目录下。保持作者MuJoCo Warp/mjlab路线，不往课程容器里安装论文依赖，不升级宿主驱动、系统Python或课程镜像。

下面命令是待执行操作手册，本轮未运行。先在Windows连接，再在Spark的Bash中执行。兼容的mjlab提交、数据文件、最终训练参数必须通过对应检查后填入，不用未验证的值冒充已可用环境。

### 5.3 第一步：记录机器与课程基线

```powershell
ssh spark-company
```

进入Spark后：

```bash
uname -m
nvidia-smi
free -h
df -h /home/nvidia
docker ps -a
docker stats --no-stream
```

先记录课程单独运行时的代表性实验耗时、可交互画面表现及内存峰值，见第13节。现有容器已停止，不能在此阶段用零占用当课程基线；实际开展课程负载验收时沿用原课程入口与启动办法。

### 5.4 第二步：建立独立目录和基础镜像

```bash
mkdir -p /home/nvidia/research/supermimic-repro/{src,data,runs,reports,cache}
cd /home/nvidia/research/supermimic-repro
```

在该目录保存Dockerfile：

```dockerfile
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential pkg-config \
    libegl1 libgl1 libglib2.0-0 libglfw3 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*
ENV UV_CACHE_DIR=/workspace/cache/uv \
    UV_LINK_MODE=copy \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    MAX_JOBS=2
WORKDIR /workspace
CMD ["bash"]
```

```bash
docker build --platform linux/arm64 -t supermimic-research:bootstrap .
docker image inspect supermimic-research:bootstrap > reports/bootstrap-image.json
```

这是基础环境候选，不是已经验证的CUDA训练镜像；Python基础镜像及uv均须确有ARM64清单。首次构建成功后记录基础镜像摘要，正式环境改用摘要固定。构建与依赖编译安排在课程不繁忙时；遇到CUDA开发头文件/编译工具缺失时，仅在研究镜像补齐匹配工具链，不改宿主驱动。不能因基础镜像构建成功就声称PyTorch/Warp已支持GB10。

### 5.5 第三步：启动带CPU与内存限额的研究容器

```bash
docker run -it --name supermimic-research \
  --gpus device=0 \
  --cpus=4 --memory=16g --memory-swap=16g --shm-size=2g \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v /home/nvidia/research/supermimic-repro:/workspace \
  -w /workspace \
  supermimic-research:bootstrap
```

16GiB是首次冒烟的容器内存候选上限，不是已测足够的训练需求；限制也适用于容器内安装和编译。共享内存用量计入容器内存，2GiB不是额外赠送内存。--gpus只是暴露GPU，不能保留固定比例GPU算力。后续提高上限须满足第13节联合验收。停止后再次进入使用docker start -ai supermimic-research，不重复创建同名容器。缓存、源码、数据与结果位于独立持久化目录，容器退出不删除它们。

### 5.6 第四步：获取并固定作者源码

以下在研究容器中执行：

```bash
cd /workspace/src
git clone https://github.com/HybridRobotics/supermimic.git
git clone https://github.com/HybridRobotics/mjlab_research.git
cd /workspace/src/supermimic
git checkout --detach abc124c1bd5976341034668af3d12c087f2f1f55
git switch -c reproduction-baseline
git rev-parse HEAD > /workspace/reports/supermimic-commit.txt
git -C ../mjlab_research rev-parse HEAD > /workspace/reports/mjlab-candidate-commit.txt
```

两个仓库必须同级。此处mjlab main只是候选，正式提交尚未确定：核对其pyproject与supermimic/uv.lock，根据兼容性选择提交并以完整SHA固定；保存正式mjlab-commit.txt。若最新main与锁文件不一致，先检查历史匹配版本，不直接升级整套依赖。

### 5.7 第五步：安装锁定依赖

```bash
cd /workspace/src/supermimic
python --version
uv --version
uv sync --python 3.13 --locked
uv run --locked python --version
uv pip freeze > /workspace/reports/python-packages.txt
```

应显示Python 3.13.x，并在该项目建立.venv。--locked检查锁文件是否适用；失败时保留错误并定位相邻mjlab版本、ARM64轮子或CUDA依赖问题，不删除锁文件强行绕过。若确须为ARM64修改依赖，保留原锁文件、生成独立补丁并报告版本偏差。uv sync成功只说明安装成功，下一步才确认GPU计算可用。

### 5.8 第六步：分层验证GPU、物理步进与画面

```bash
uv run --locked python - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
assert torch.cuda.is_available()
print('device:', torch.cuda.get_device_name(0))
x = torch.randn(128, 128, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print('CUDA forward/backward passed')
PY

uv run --locked python - <<'PY'
import warp as wp
import mujoco
wp.init()
assert wp.is_cuda_available()
print('MuJoCo:', mujoco.__version__, 'Warp:', wp.__version__)
print(wp.get_device('cuda:0'))
PY

uv run --locked src/supermimic/scripts/train.py --help
uv run --locked src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --help
```

以上还不等于Warp kernel和MuJoCo Warp已实际运行。接着依据固定版本入口完成：Warp CUDA小kernel运行、G1加载、64环境无窗口物理步进至少1000控制步并反复reset、最小世界模型更新、actor/critic更新；最后另测离屏渲染与回放。具体任务/CLI以--help输出和论文对照表为准，MaxCoord也须单独确认。遇到GB10架构不支持，定位PyTorch/Warp版本与实际kernel报错，不把CUDA可见当兼容性通过。

### 5.9 第七步：数据与训练入口绑定

先取得一段有效的G1动作文件，按第6节验证。训练入口支持本地motion_file，避免默认W&B registry权限问题。按实际--help确认环境数、动作文件、外层迭代数、batch及日志目录的参数名，保存首个run的精确命令和最终生效配置。先64环境、单动作、短运行；通过后才尝试128/256，不照4096默认值直接启动。

此时分别测首次编译、稳态采集、世界模型更新、actor/critic更新、评估、检查点I/O；同时记录物理步数、想象步数、峰值主机内存和课程表现。速度接口需要第10节的任务核对/补齐，环境安装完成不会自动补齐论文图7实现。

### 5.10 环境验收

- Python/依赖/两个提交/镜像摘要都有记录；GPU矩阵反向、Warp kernel、MuJoCo Warp物理步进、reset、策略更新分别通过。
- 无窗口训练与可视回放分别验收，初次编译与稳态计时分开；未发生持续OOM、NaN或驱动错误。
- 第13节课程联合负载通过后，才允许长训；不把统一内存等同H100显存/速度。
- 依赖阻断先形成最小诊断；吞吐不足先据实测调整，再讨论外部GPU，不直接迁移到Isaac Sim重写方法。

## 6. P2：动作数据与机器人接口

### 工作步骤

1. 先核查可合法取得的G1重定向数据和W&B artifact权限；优先获取用于冒烟的单段/少量基础动作，再准备LAFAN1 40段基线库。
2. 依据固定源码的数据加载器决定实际文件格式、采样率、坐标、四元数顺序、关节顺序和单位。不把人体原始LAFAN1文件等同可直接用于G1的重定向动作。
3. 记录文件来源、许可、hash、clip ID、时长、帧数、采样率和筛选规则。若无法取得论文完全相同的库，明确偏差并单独命名数据版本。
4. 每段进行数值和运动学检查：有限值、关节范围、根部高度、身体尺寸、地面穿透、脚接触；用原生G1逐段回放，不仅看一张截图。
5. 动作跟踪正式集与开发调试集冻结manifest；将来宣称未见动作泛化时另建按clip划分的留出集。论文训练动作覆盖与未见动作泛化分开报告。
6. 图7阶段再按附录G建立LAFAN1+BONES-SEED走跑库；获取不到精确清单时记录近似过滤差异，不能称严格同数据复现。

### 验收

同一clip能重复加载与回放；索引、单位和关节映射明确；目标动作与实际仿真轨迹分开保存。数据不可靠时先修数据，不靠延长训练掩盖。

## 7. P3：三阶段训练计算正确性

### 世界模型

- 从已记录的同一条轨迹取8步转移，核对9个状态、8个实际执行动作和相应物理上下文。
- 从真实起点初始化，之后使用预测状态连续展开；比较对应记录。区分回合reset边界，禁止把reset后的状态错误当作连续动力学。
- 在固定开发样本上检查可学习损失下降；用独立验证窗口检查1步/8步误差；小训练损失不能代表所有状态下都可靠。
- 核对奖励、观测、终止哪些是解析计算、哪些需要学习，以及状态误差的归一化定义；不把通用公式四项强行实现成四个神经网络。

### 策略与critic

- 策略阶段冻结world-model参数但保留对状态/动作的梯度；检查actor梯度有限且非零、更新前后actor参数变化、同阶段world-model参数不变。
- 检查H=32、折扣、生存权重、终止步奖励下标和末端价值bootstrap。
- critic目标从末端向前递推，目标固定后拟合；critic更新不得连带更新actor/world model。
- 用小规模标量例子核对GAE递推、hard done、soft done、真正终止与时间截断的不同处理。

### 验收

每个优化器只更新应更新的参数；模型与策略梯度链完整。此处做有明确故障检出能力的检查，不为了数量写与实现完全重复的测试。

## 8. P4：尽快得到第一份自训动作跟踪结果

1. 先1段易于验收的基础动作，再扩展到5—10段走跑/转身等开发动作；不先下载全264小时动作库。
2. 保留论文核心k/H与网络结构，优先缩小并行环境、batch及数据量。若仍无法运行，再单独记录网络缩小实验，不与论文规模混称。
3. 先运行300—1000个外层迭代作为工程检查，具体取值依据代码warm-up与实测时间决定；它只检验训练闭环，不作为收敛保证。
4. 验证模型误差、策略/critic损失、梯度、回放库填充和归一化统计；保存一份可加载检查点。
5. 用新的独立进程，在MuJoCo Warp物理仿真中评估自训actor。逐段从首帧到末帧记录完成/失败、归一化回报、跟踪误差和视频。
6. 同条件运行零动作/随机策略作工程对照；不将其代替正式PPO算法基线。
7. 测试重载后的动作数值一致性、评估一致性；若恢复优化器/调度状态缺失，标为仅加载权重或warm start，补齐后才称连续续训。

第一份结果应包含：配置与数据版本、自训检查点、训练前后指标、物理仿真视频、失败日志、复现命令。选定基础动作必须真实完成；未通过则继续定位，不用模型内回报上涨替代。

## 9. P5：动作跟踪正式基线

1. 固定LAFAN1 40段库、有效配置和评估脚本；定期对所有clip从首帧评估，保存每个checkpoint的全库结果。
2. 按论文定义计算完整回合的归一化回报，提前失败不能只按已经执行步数平均后掩盖失败。
3. 报告完成率、失败数、根部/身体位置和朝向误差、速度误差、回合长度，按动作类别分组；明确哪些指标是补充指标。
4. 同时保存训练回报和固定起点评估回报，图4中的二者不是同一统计量。
5. 首先单seed达到稳定可评估状态；再重复至少3个独立训练seed，每个seed有独立训练结果，不能用同一权重重复回放冒充训练seed。
6. 推荐机制检查：同数据/预算下有critic与无critic比较，保持gamma、H和其他训练条件一致。当前BpttCfg默认gamma=1.0，而ActorCriticCfg为0.99，不能仅切换类名就声称只消融了critic；短H对照可随后补。PPO-Sim/PPO-WM、图3全量扩展和H全扫描列为扩展，不阻塞本轮速度接口主目标。

验收是“基线完整且性能结论有证据”。论文0.82仅在对应LAFAN1协议下作数值参照；未达到时报告差距、预算和偏差，不改指标定义制造达标。Spark计时只与本机同条件对照，不能以H100的1.2小时/3小时作为Spark工期。

## 10. P6：补齐并核验速度任务

在原始动作跟踪基线外建立独立实现分支；先补论文原方案，再考虑任何创新。

### 必须逐项核对

1. 速度编码器输入：航向系根部平面速度指令vx/vy、世界坐标偏航角速度指令wz，加最近10帧观测的特征；共享32维latent和decoder。
2. 参考速度指令按附录G使用中心25帧平均及正确坐标系。
3. Distill接口：从参考动作读取指令，使用motion tracking reward及附录A终止；仍通过模型训练，不改成另一个教师动作回归算法。
4. RL接口：实现表A.6各项速度奖励、正则和fall-only终止。所有项乘控制周期0.02s；保留线速度项分母0.5平方与偏航项分母0.5的区别。
5. RL指令源50%参考、50%均匀随机；随机范围vx[-1.5,2]m/s、vy[-1.32,1.32]m/s、wz[-3.35,3.35]rad/s；随机分支10%置零。reset或每250控制步重新选择指令源/随机命令。
6. 在参考指令轨迹上，速度latent对齐motion latent，权重0.1，motion latent停止梯度。
7. 每个接口自己的物理仿真slot采集数据，想象轨迹从该接口记录状态开始；两类环境/想象batch按附录G各占一半。缩小环境总量时保留比例并记录差异。
8. 下游奖励只更新速度encoder；梯度经过decoder，但decoder权重不接受该任务更新。motion目标仍更新motion encoder与decoder；各接口critic及其目标按附录B核验。
9. 做单步/短窗口路由检查：两个奖励、done条件、命令源、起始状态、梯度参数组都确实按接口分流；不能仅把velocity slot比例改成0.5就算完成。

验收：可用构造状态验证各奖励项、坐标变换与终止边界；梯度开关符合论文；所有与公开代码不同的补丁均有依据及版本记录。

## 11. P7：图7训练与评估

### 训练

- Distill与RL encoder使用独立训练run，从头初始化各自encoder，并与motion接口并行训练；不把同一检查点继续微调误称原文从头比较。
- 首先小规模开发运行；正式目标按附录G每种100000迭代，若Spark预算不足则报告缩小协议。记录真实物理步数、想象步数和各优化器更新数，环境变少后不能单靠迭代数宣称等预算。
- 两组保持相同数据库和可比预算；参考指令比例、终止规则等按原协议差异保留。后续研究若要隔离奖励因素，另做单变量控制实验，不把原文多处变化的比较全归因于奖励。
- 完成单seed两组验证后，正式结论扩展到至少3个训练seed，报告均值/离散程度与逐seed结果。

### 原始阶跃测试协议

1. 在三个指令平面构造15x15网格，另一个维度置零，排除各平面的站立点，共672个测试条件；保留原协议的跨平面条件，不随意去重改变数量。
2. 每条条件从站立开始、固定指令5s、4次试验。每个encoder每个训练seed需2688条试验；两种encoder、3个seed共16128条试验，便于提前估算评估量。
3. 取最后2s平均速度误差的范数，再除指令范数；每个命令取未跌倒试验的中位数。不要换成“逐时刻误差范数的平均”而沿用同一指标名。
4. 所有试验都保存fall标志；额外报告跌倒率、4次全跌倒条件数、成功试验数，避免仅统计存活试验造成误导。
5. 原始误差保留；0.9上限只用于复刻图中显示，不能用截断后的数据替代全部分析。
6. 保存命令、实际速度时序、终止时刻、训练seed、试验seed、checkpoint hash，能重建图而不必再训练。

### 报告与验收

- 输出三个平面的误差/失败图、逐条件表、总体中位数、两方法胜负与相等计数、典型快转/稀缺指令视频。
- 论文0.16与0.49、650与15作为原协议参照，不强行要求小规模Spark结果逐点相等；7个全跌倒指令不能直接假定就是剩余7个平局，按原始数据分别统计。
- 方法趋势与精确数值分别判定；若RL未显示优势，先核对实现、训练预算、模型误差、数据覆盖与终止协议，再报告未复现，不能先改算法掩盖。

## 12. P8：交付与研究入口

交付必须包含：

- 固定提交、依赖锁定、启动/训练/评估/恢复说明；原方案分支和复现补丁。
- 数据manifest与获取办法，避免未经许可再分发数据。
- actor/world model/critic权重、必要归一化状态；恢复训练所需状态的完整性说明。
- 原始日志、按clip/command的评价、训练曲线、视频、失败案例和论文差距表。
- 独立重启研究环境后的一次加载和评估验证。

研究点只从确认可复现的现象中筛选：冻结decoder是否限制下游性能；直接开放是否损害原有跟踪；模型不可靠区域是否对应异常策略梯度；困难指令采样是否改善样本利用率。先做有限的诊断对照，再检索相关工作，随后决定研究题目。PCGrad、adapter或门控都只是候选工具，不预先当作新颖性已成立的方法。

## 13. 时间与算力安排

### 13.1 课程优先的资源规则

不能把“降低训练环境数”当作GPU算力硬隔离。该Spark只有一块GB10，本次nvidia-smi显示MIG为N/A；本计划不依赖GPU硬分区。Docker的CPU/内存限制不提供GPU算力比例保证，尤其统一内存下还要观察宿主MemAvailable，不能把容器内存限制解释为完整的GPU内存隔离。

| 项目 | 初始研究设置 | 对课程的保护与判定 |
|---|---|---|
| CPU | 4个CPU配额，OMP/MKL线程4，编译MAX_JOBS=2 | 当前20个逻辑CPU，仅对研究容器限额；不修改课程配额 |
| 容器内存 | 首次16GiB，memory-swap同值 | 防止普通容器内存无限增长；不足先诊断，不能自动放开限制 |
| 主机内存余量 | 暂定MemAvailable至少64GiB | 作为课程和系统的初始保护线，待课程峰值实测后修订，不代表保证所有课程可同时满载 |
| 并行仿真 | 从64环境开始，依次128/256 | 同时缩小模型/策略batch，保持k/H及接口比例；测过才扩，不使用默认4096 |
| GPU | 同时最多1个研究GPU任务 | 课程GPU仿真、渲染或训练需要大量资源时，研究保存后退出 |
| 数据与磁盘 | 先单动作/小库，研究数据输出独立 | 大量下载、解压、编译与视频编码避开上课高负载时段；不清理课程文件 |

64GiB余量和16GiB容器上限是待验证的起始策略，不是已测最优值。课程单独运行时若已经低于余量线，应按课程实际需求重新划线，并停止研究GPU训练。正式研究内存预算按“课程可能增加的占用＋系统安全余量”保留，不能只扣除课程闲置占用；当依赖或batch改变时重新测峰值。未通过联合验收，不提高至24/32GiB或更大批量。

### 13.2 哪些工作可以同跑

1. 课程仅浏览Notebook/查看已有结果时：可以尝试受限小训练，联合验收通过后再长训。
2. 课程开展Isaac Sim/Isaac Lab GPU仿真、画面展示或短训练时：默认课程独占重GPU时段；研究先保存并退出，释放GPU上下文与内存。
3. 课程无GPU需求时：在CPU、内存与余量限制内推进研究长训和评估，仍不同时启动多个研究训练。
4. 大量图7测试同样计入研究GPU任务，不能以“只是评估”为由与课程重GPU任务抢占。

这里的“退出”是完成可恢复检查点后结束研究进程；单纯docker pause或SIGSTOP仍保留GPU内存，不能用于给课程释放资源。现有代码的完整恢复能力尚未核验，因此P0先验证保存/恢复，补齐必要状态后再长训。真正中断恢复与仅加载权重分别标注。

### 13.3 联合负载验收步骤

1. 沿用课程原入口，选日常需要的课程组合及代表性GPU章节，先课程单独运行；记录实验完成时间、画面帧率/交互延迟、MemAvailable、容器CPU/内存和失败情况。课程容器运行中但没有执行实验，不算峰值测试。
2. 加入64环境研究短训；保持课程实验与参数一致，至少重复3次对比并观察10—15分钟稳态。首次CUDA编译单列，不能混进稳态统计。
3. 建议初始通过线：课程无新增失败/OOM、同实验耗时增加不超过20%、交互画面无明显卡顿，且主机余量达线。20%是工程容忍上限，课程对实时画面有更严格要求时以课程要求为准；不能只凭GPU利用率判断。
4. 不通过则减小研究batch/环境数，必要时禁止同跑。128/256环境及正式数据规模分别重测，不能用64环境结果替代。
5. 保存课程单跑/联合运行的对照记录和最终允许配置，再开始长训。需要启动课程重GPU实验时提前结束研究训练，不等课程先卡住。

后续实施时为研究作业提供资源检查、检查点保存退出和恢复入口，检查对象限定研究作业；课程优先与余量规则写入运行说明。当前仅提出该运行机制，尚未部署自动监测或自动停止程序，不能宣称已自动保护课程。

### 13.4 目标窗口与训练预算

以下为工作量估算与目标窗口，不是已测训练耗时或固定完成承诺：

| 里程碑 | 建议目标窗口 | 依赖 |
|---|---|---|
| 源码审计和环境第一轮结果 | 第1—2天 | 源码/依赖可取得，GPU健康 |
| 最小数据、梯度链、短训计时 | 第2—3天 | 环境及数据通过 |
| 第一份自训动作跟踪验收 | 争取第3—7天 | 兼容性无重大阻断，实测吞吐允许 |
| 正式动作库基线、速度实现 | 第一份结果后约1—2周工程工作 | 训练与补实现工作量 |
| 两种速度正式训练和多seed评估 | 之后按实测GPU时确定 | 100000迭代成本、数据规模 |

先得到真实训练计时，再估算N次迭代的采集/模型/策略/评估/I/O总成本；首次编译单列。较大动作库、内存带宽、batch与编译配置改变后重新计时，不能无条件线性外推。每次长训前明确预算、评估间隔、存储需求与中断恢复办法。

持续可用的Spark用于在课程允许时段串行推进基线与重复实验，不同时开多个高占用训练。日历工期需按实测研究可用时段计算；研究跑时不足不能转化为抢占课程。性能不足时优先检查瓶颈、适当缩batch；必要时提交一份额外GPU需求与费用估计再决定。无额外算力批准前不租卡。

## 14. 阶段停止条件

| 现象 | 处理 |
|---|---|
| 依赖或ARM64/CUDA链路不通 | 固定问题到包/版本，形成最小诊断，不先开长期训练 |
| 数据关节/坐标/地面接触错误 | 修正映射或数据清单，作废受影响的比较 |
| k步误差异常或梯度断链 | 返回P3，检查记录动作、reset边界、归一化与detach |
| 模型内回报高但物理仿真差 | 检查模型误差利用、奖励/观测一致性与覆盖，不把想象成功当真实仿真成功 |
| 仅加载权重可用，训练无法等价恢复 | 明确warm start与resume差别，补齐状态后验证 |
| 课程进入重GPU实验，或联合验收不通过 | 研究保存退出、释放GPU；不停止课程或调低课程配置 |
| 主机内存余量低于验收保护线 | 结束研究作业，定位峰值来源；不因课程闲置快照而放宽限额 |
| 图7指标差距明显 | 先审协议和实现，再判断预算/方法差异，不修改评价口径 |

## 15. 依据

- 用户提供英文论文v2，工作区副本 `tmp/pdfs/supermimic_original.pdf`：§3.1—3.3、图4/5/7、表A.4—A.6、附录B/D/F/G。
- [官方SuperMimic](https://github.com/HybridRobotics/supermimic)，[核查提交](https://github.com/HybridRobotics/supermimic/commit/abc124c1bd5976341034668af3d12c087f2f1f55)。
- [pyproject](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/pyproject.toml)。
- [Runner](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/runners/supermimic_runner.py)、[多任务环境](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/envs/multi_task_world_model_env.py)。
- [训练入口](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/scripts/train.py)、[策略学习器](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/policy_learner.py)。
- [mjlab_research](https://github.com/HybridRobotics/mjlab_research)。
- [uv项目管理](https://docs.astral.sh/uv/guides/projects/)、[uv容器安装与锁定](https://docs.astral.sh/uv/guides/integration/docker/)。
- [Docker资源限制](https://docs.docker.com/engine/containers/resource_constraints/)：CPU/内存配额语义；不据此声称GPU硬隔离。
- 2026-11-16 spark-company只读SSH检查：机器规格、nvidia-smi、free、df、Docker容器/镜像与监听端口；未执行课程实验或课程峰值验收。
- 本计划采用GrillMe grilling审查与Planning with Files持续记录；用户确认的是范围/资源/优先级。本轮仅制定计划，不据此宣称复现实验已执行。

