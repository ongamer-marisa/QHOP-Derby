# r14 实现与验收记录

日期：2026-10-13。验证 Unity 6000.0.45f1、Inference Engine 2.2.1、PyTorch 2.3.1+cu121。所有日志位于项目 `Builds/BuildLogs`；可用训练 Player 为 `Builds/Training/QHopTraining.exe`。保留已有实验及工作区改动，没有启动正式长训练。

## 实现契约

| 项目 | 当前行为 |
|---|---|
| 观测/动作 | schema 4，1285 / 28 维；原 1106 维地形、本体四帧、对手和步态保留 |
| 终止 | 训练、评估、训练预览进入 Fallen 即 fall DNF；优先于同帧完赛；一次终局，禁止 bootstrap；最终游戏仍辅助复位 |
| 奖励 | 保留阶段摔倒与 DNF 负项，fall 豁免剩余时间罚款，仍计当帧正常奖励 |
| 扫描 | 57 根，前方 -70°～70° 每10°×3高，±90°/±135°×3高；25/10米；水平稳定坐标，局部 PhysicsScene，自身/Trigger 过滤，缓冲区饱和时扩容 |
| 扫描数据 | 171维距离/最大距离/命中，3维扫描间运动、2维朝向 sin/cos、间隔、年龄、编号，共179维；未命中 `(max,max,0)` |
| 时间 | 首帧强制扫描，100Hz物理下每20tick扫描；20Hz策略，无新扫描时不运行encoder/GRU，隐藏状态不变；bootstrap只查询不消费 |
| 网络 | 177→256→128 ELU encoder，GRU hidden128；Actor/Critic 各自原128维特征与视觉128+年龄融合，257→128 ELU；28维动作/标量价值 |
| PPO | 共享视觉参数仅注册一次；≤128决策序列块，每minibatch8块；短块仅提取有效位置参与损失，相当于掩码去除padding；本回合扫描前缀按当前权重无梯度重放，块内截断反传；更新后刷新活动回合记忆 |
| 状态 | worker/马匹/回合隔离，当前策略与冻结对手独立；终局、重启、模型更换与游戏复位清空 |
| 导出 | 动作 ONNX、Vision ONNX、Metadata JSON；GRU 展开为基础算子，无 GRU/Loop/If；部署同时携带三文件 |
| 兼容 | r13及更早checkpoint明确拒绝；禁用迁移及旧对手池；新版本从头训练 |

默认训练batch仍为16384。此次短程联调临时用128以限制工程验证规模，不改变默认配置。原动作分布、PPO/GAE、熵和课程学习率设置保留；没有增加动作变化率限制。

物理指纹：`qh-100hz-5step-19rb-28a-lidar-gru-terminal-r14`。网络版本：`grouped-mlp-shared-lidar-gru-r14`。最终 Player 契约 SHA256：`037bd7d5273d39f14fc875f56b44e05b5b5cea2d45906af8a66ec53168a0bb24`；`doctor` 同时校验二进制、资源和课程/奖励契约。

## 承重诊断与修复

使用零角度、12个强度均0.7的控制，在20Hz持续送入；记录触地、目标/实测角、误差、姿态、速度和碰撞体。旧参数的真实 Player 标准马104回合出现32次摔倒，证明静息承重失败无需随机策略即可复现。

修复三处：出生时整体刚性旋转使四蹄支撑面一致（特别针对前后腿长度不同），主动关节弹簧由60000提高至240000，躯干局部重心向前移0.2×bodyLength；求解迭代最终64/16。阻尼3000、驱动力上限40000、被动关节5000/350保持原值。初始化即提交静息控制，避免首物理帧驱动目标未设置。

保持总质量、19刚体/18关节、28动作及原角度限位。出生对齐保持骨架相对姿态和关节锚点关系；没有扶正力、额外支撑或放宽摔倒阈值。训练与最终游戏共用这些参数，序列化HorseRig同步更新。

| 验收 | 实际规模 | 结果 |
|---|---:|---|
| Unity Editor 独立冷启动 | 标准100次 + 全81体型，每次20秒 | 181次，0失败 |
| 真实训练 Player 标准马 | 104回合，每次20秒 | 0摔倒；最低相对高度0.997144；最大倾角0.531555°；最大漂移0.016754米 |
| 真实训练 Player 体型网格 | 全81唯一组合；并行完成88回合，每次20秒 | 0摔倒；最低相对高度0.985549；最大倾角17.124973°；最大漂移0.041191米 |
| 原关节到达/限位 | 96动态探针 | 全通过 |

观测/奖励与物理状态检查无非有限值。标准马达到高度≥80%、倾角≤20°、漂移≤0.5米门槛。证据：`r14-standing-player-baseline.json`、`r14-solver-acceptance.log`、`r14-standing-final64.json`、`r14-morph-final64.json`、`r14-final-build.log`。

## 终止、扫描和循环记忆

真实 Unity PhysicsScene 契约探针通过：57方向/高度/射程、未命中、自身及Trigger过滤、最近命中、70个重叠碰撞体下缓冲扩容、扫描周期及水平稳定性；同帧摔倒/终点的失败优先、摔倒计数一次、第二阶段 -2/-5且无剩余时间罚款、其他马继续完成、最终游戏恢复。训练区域各自 PhysicsScene 隔离。

真实宿主 PolicyRunner 使用双ONNX模型连续12次决策，验证每4次决策更新一次GRU及恢复后清空。Python测试覆盖同编号冻结、bootstrap不消费、不同马和新回合隔离、不同历史形成不同状态、跨扫描梯度、PPO共同更新与更新后前缀重放。序列/终局GAE测试通过。测试不是对长期策略表现的评价。

实际Player倍速1和20的完整1285维观测最大差异为0，扫描次数和时间元数据一致，见 `r14-engine-check.log`。训练预览正常启动/退出，画面显示扫描编号/年龄和Python实际GRU计数，支持射线/命中点开关；缓存射线位置不会随旧扫描移动。截图 `r14-preview.png` 是连接初始化时的画面，不能作为步态表现证据。

## 部署与短程联调

对两匹马20步完整序列，每4步新扫描，包含冻结及第12步记忆清空，比较 PyTorch、ONNX Runtime CPU 与真实 Unity Inference Engine CPU：

| 比较 | 最大绝对误差 |
|---|---:|
| Unity 隐藏状态 / PyTorch | 1.490116×10⁻⁷ |
| Unity 动作 / PyTorch | 2.980232×10⁻⁷ |
| ONNX Runtime / PyTorch，动作与状态 | 1.490116×10⁻⁷ |

低于1e-4门槛；ONNX checker通过。真实宿主另验证事件消费与恢复语义。可用 `training/verify_r14_artifacts.py` 重新核对保存的序列及参数变化，不会启动训练。

CUDA短程训练仅运行 **2次PPO更新，共256样本**：第1次保存，重新创建环境续训至第2次。Adam内部step=10是两次PPO的minibatch/epoch优化次数。与同seed初始化相比，Actor/Critic/Vision最大参数变化分别为1.529150e-5、1.521781e-5、1.460314e-5；证明视觉分支实际参与优化。随后2回合评估、5秒预览、双模型导出通过。实验保存在 `Runs/r14-validation-20261013`，状态idle，无worker重启或运行时错误。

44项Python自动测试全部通过，当前构建doctor通过。构建日志有既有诊断路径的一个preview scene清理警告，退出码0；不影响交付Player。证据：`r14-python-tests.log`、`r14-artifact-verification.json`、`r14-doctor-final.log`、`r14-evaluate-smoke.json`、`r14-preview-smoke.log`、`r14-export-smoke.json`。

本次未启动长训练，未将短程模型部署为正式游戏策略，也未证明自然步态已经学会。承重验收针对已定义的静息动作和体型网格；任意学习动作下的摔倒率、视觉避障能力和自然步态需要新实验的正式训练/课程评估。

## 使用

```powershell
$env:PYTHONPATH='training'
.venv-qhop/Scripts/python.exe -m qhop_train.cli doctor
# 新实验从头训练；本次交付没有执行此命令
.venv-qhop/Scripts/python.exe -m qhop_train.cli train --output Runs/r14-new --workers 4 --batch 16384
```

详见 `training/README_cn.md` 的保存、续训、预览和双模型导出命令。旧结构原因分析保留在 `docs/TRAINING_STRUCTURE_AUDIT_20261013_cn.md`，明确标为r13快照。


## 仓库复核（2026-10-20）

重新运行 44 项 Python 测试全部通过；`doctor` 确认 schema 4、1285/28 协议、CUDA 与上文 Player SHA256 一致。同步多输入推理接口桩及核心模式排除列表后，C# `--core` / `--stubs` 均通过；真实外部程序集检查因本机缺少导入后的 ML-Agents／Inference Engine DLL 未能执行，不能把静态检查当作 Unity 真实编译。本次未重跑 Unity 物理、推理序列、可见预览或训练；上文 2026-10-13 结果保留为历史证据。

`Runs/`、`Builds/` 和所有 checkpoint／ONNX 均不随源码仓库分发。本文的日志文件名是本地证据定位，不是仓库下载链接；`verify_r14_artifacts.py` 需要这些原始夹具。新环境先按[训练说明](README_cn.md)安装依赖、重建 Player 并运行测试。
