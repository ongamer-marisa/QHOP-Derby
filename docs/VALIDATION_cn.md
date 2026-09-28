# 展示与 PPO 训练验证记录

记录日期：2026-09-28。工程目录：`D:\dev\QHOP Derby`。

| 检查 | 结果 | 证据或限制 |
|---|---|---|
| Blender 源文件读取与 FBX 导出 | 通过 | `tools/export_horse.py` 在 Blender 2.83 factory startup 模式下成功运行 |
| FBX 往返导入 | 通过 | `tools/verify_asset.py`：40 根骨骼、964 个三角面、4 个材质槽、10 个有效修形端点 |
| C# 核心语法与程序集引用 | 通过 | `python tools/compile_check.py --core` 使用 Unity 6 本机程序集编译项目核心脚本 |
| 含训练／推理源码的 C# 静态检查 | 通过 | `python tools/compile_check.py --stubs`，并由真实 Unity 包完成 Player 脚本编译 |
| 场景文件与 BuildSettings 引用 | 通过静态检查 | 两个场景文件及唯一 GUID 存在，均已列入 `EditorBuildSettings.asset` |
| Unity Editor 导入与脚本重编译 | 通过 | 沙盒外调用 Editor 许可证通信成功，见 `Builds/BuildLogs/training-build-r19.log` |
| 四马物理站立 20 秒 | 部分通过 | 训练 Player 的 Stage 5／6 已在两个四马场地各运行 10 秒；展示场景和最终模型行为待测 |
| 81 组体型、1000 组属性、100 次重置 | 部分通过 | Unity Editor 探针已验证 81 组体型的 19 刚体、18 关节、总质量及最低马蹄碰撞面 y=-0.02，另检查 1000 组生成属性范围；见 `Builds/BuildLogs/morph-probe-r1.log`。旧 r9 构建的重复重置观察到摔倒，新 r10 尚未完成 100 次重置稳定性验收 |
| 碰撞、摩擦、摔倒和扶起 | 未执行 | 需要 Unity 物理运行和画面观察 |
| 动漫渲染、中文 UI、音效与 60 FPS | 未执行 | 需要 Editor／Windows 构建及测试硬件 |
| Windows x64 训练与展示构建 | 通过 | 两个 Player 均已用更新后的共享物理代码重新构建；训练 Player 还加入终局原因日志，见 `Builds/BuildLogs/training-build-r19.log` 和 `Builds/BuildLogs/presentation-build-r9.log`。展示画面和帧率待测 |

展示场景的视觉质量、渲染性能与大批量体型稳定性尚未验证。物理参数在训练构建中经过真实 Player 冒烟测试；后续按 `docs/UNITY_PRESENTATION_PLAN_cn.md` 验收展示场景。

## 六阶段训练实现检查（2026-09-28）

| 检查 | 结果 | 证据或限制 |
|---|---|---|
| Python 包依赖一致性 | 通过 | 项目虚拟环境执行 `pip check`，无断裂依赖 |
| Python 单元测试 | 通过 | 观测协议、课程及 Stage 6 shaping 评估门槛、余弦学习率、GAE、罚停 Actor 掩码、tanh 概率、CPU／CUDA checkpoint、过期构建拒绝及 ONNX/PyTorch 动作对齐，共 15 项 |
| ML-Agents Python 环境导入 | 通过 | `mlagents_envs.environment.UnityEnvironment` 可导入 |
| CUDA 数值更新 | 通过 | PyTorch 2.3.1+cu121 识别 RTX 4060 Laptop；真实 Unity 采样后完成 2 次 PPO 更新，checkpoint 从 128 步恢复到 256 步，Adam 状态保留，见 `Runs/smoke-r6/latest.pt` |
| 最新物理构建上的训练／恢复／导出 | 通过系统冒烟测试 | r10 物理构建上使用 1 worker、128 步 batch 完成一次 PPO 更新，再从 `Runs/smoke-r10/latest.pt` 恢复到第 2 次更新／256 步，无 worker 重启；`training_metrics.jsonl` 记录每轮 PPO、采样时间和吞吐。ONNX 已导出到 `Runs/smoke-r10/QHopPolicy.onnx`，元数据物理指纹为 r10。该模型仅用于链路验证，未部署到游戏 |
| Stage 6 评估门槛与训练链路 | 通过系统冒烟测试 | 课程单测覆盖每档至少 10% 预算、完赛率相对上一档下降不超过 5 个百分点及零 shaping 至少 20% 预算。用明确标记为合成的 Stage 5 基线生成源 checkpoint，在真实 Unity 四马场景完成 1 次 Stage 6 PPO 更新（130 个当前策略步、0 次 worker 重启），见 `Runs/stage6-system-smoke/training_metrics.jsonl`；未声称通过真实晋级评估 |
| Unity 训练场景与构建 | 通过 | 沙盒外执行 `BuildTraining.Build` 后，真实 ML-Agents／Inference Engine 程序集编译成功，Windows Player 生成于 `Builds/Training/QHopTraining.exe`；见 `Builds/BuildLogs/training-build-r19.log` |
| 物理与动作通道 | 部分通过 | 新 r10 Player 中 28 个通道逐项应用并从观测回读；极端动作摔倒到恢复间隔曾测得 60 个决策／3 秒。新构建最低马蹄碰撞面设为与平地 y=-0.02 重合，扶正按地面高度重建；反复重置稳定性及体型组合仍未通过验收 |
| Python↔Unity 通信和并行采样 | 通过初步测试 | 新 r10 Player 的 Stage 4 全难度起伏／障碍场地取得 400 个有效决策；Stage 5 和 Stage 6 全难度四马场地分别取得 1000 个当前策略决策，Stage 6 使用零 shaping；无 NaN。Stage 1 的 1／2／4 进程、32 场地、15 秒独立基准约为 343／556／626 马步／秒，峰值 worker 内存约 272／488／916 MB；该基准来自旧构建，r10 吞吐待复测 |
| 四马逐马终局与超时 | 通过未训练策略场景 | Stage 5 两场、8 匹马运行到终局；Python 收到 5 个当前策略和 3 个冻结对手的终止事件。日志中 5 匹提前 DNF，另 3 匹在 90.00 秒超时 DNF；均未得到名次。真实赛道冲线仍待训练模型测试 |
| 冲线、排名与奖励防重复 | 通过规则探针 | Unity Editor 使用简化刚体直接验证第一匹完赛后其他马继续、物理步内插值计时、同一步冲线顺序、倒地越线无效、扶正前后已取得进度不重复奖励，以及超时 DNF；见 `Builds/BuildLogs/race-probe-r2.log`。仍需真实赛道上的训练模型测试 |
| 分阶段奖励结算 | 通过规则探针 | Unity Editor 验证持续奖励对时间步拆分不敏感、合法新增进度获得收益、Stage 6 零 shaping 只保留比赛目标、完赛和 DNF 奖惩各只结算一次；见 `Builds/BuildLogs/reward-probe-r1.log` |
| 四马并行吞吐 | 通过短基准 | 新训练构建 Stage 5，固定 8 场地、每配置采样 10 秒：1／2／4 进程分别约 379／403／403 当前策略步／秒，峰值 worker 内存约 258／474／901 MB。2 与 4 进程的差异在短基准波动范围内，默认 4 进程仍待更长基准确认 |
| Unity 内 ONNX 动作对齐 | 通过数值测试 | 临时 2 马模型由 Unity Inference Engine CPU 导入运行；相同输入与 ONNX Runtime 的最大绝对动作误差 `2.98e-7`，小于 `1e-4`。当次临时日志后来被 Unity 清理；临时模型已从资源目录清理，正式模型导出后需要重跑对齐 |
| 六阶段正式训练与 500 场独立比赛 | 未执行 | 不存在已收敛 checkpoint、最终 ONNX 或比赛成绩报告 |

当前可确认训练 Player 的基础通信、28 通道逐项、短时并行采样、四马逐马终局、完整 3 秒罚停、CUDA 更新、checkpoint 恢复和 Unity ONNX 数值对齐已运行。81 组体型与 1000 组属性的静态物理检查、冲线规则探针已覆盖；旧构建的反复重置稳定性未通过，新马蹄高度尚未完成 100 次重置测试。真实赛道冲线、展示帧率和正式模型行为仍未验收。短训 checkpoint 仅供系统冒烟测试；尚无经过六阶段正式训练的最终模型。

## Unity 训练菜单与人工晋级短测（2026-09-28）

本节只验证训练控制系统。测试使用 128 步 batch、短时 Player 启动或**明确标记为合成**的自动评估报告；没有启动正式六阶段训练，未训练模型的 DNF 不视为训练器故障。

| 检查 | 结果 | 证据 |
|---|---|---|
| Unity 真实脚本编译与训练 Player | 通过 | `Builds/BuildLogs/training-menu-release-editor-compile.log` 和 `Builds/BuildLogs/training-menu-release-build.log` 均成功；`doctor` 检查依赖、RTX 4060、1078/28 协议与构建契约 |
| Python 安全停止、审核和 PPO 数值测试 | 通过 | `training/tests/test_training.py` 共 22 项，包括采样中停止、更新中停止、更新异常不覆盖 checkpoint、哈希门槛、人工批准与拒绝、学习率恢复 |
| Editor 菜单启动与安全停止 | 通过 | `Builds/BuildLogs/training-menu-probe.log` 报告 `QHOP_MENU_PROBE_PASS updates=0`；菜单实际启动 1 worker，采样中停止并保存初始完整 checkpoint，所属进程退出 |
| 新构建短训与分支恢复 | 通过 | `Runs/menu-smoke-final/latest.pt` 完成 1 次 128 步更新；`Runs/menu-smoke-branch/latest.pt` 从前者完整恢复到 2 次更新，原 checkpoint 哈希未变化，阶段学习率无重复热身 |
| 可见 Stage 1 完整回合 | 通过功能检查 | 最终构建的 `Runs/menu-release-smoke/preview-status.json` 为 `closed, episodes=1`；Player 日志显示 Direct3D 11／RTX 4060 和正常 timeout DNF；验证前后 checkpoint SHA-256 一致 |
| Stage 2～6 可见环境启动 | 通过短启动检查 | `Runs/menu-preview-fixtures/stage*/preview-status.json` 均为 `closed`；各运行 4 秒，Player 日志显示 RTX 4060，checkpoint 哈希一致。使用标记为 `synthetic_preview_fixture` 的未训练模型；0 个完整回合，不代表模型行为达标 |
| 合成自动评估后的人工门槛 | 通过系统检查 | `Runs/menu-review-fixture` 的报告标为 `synthetic_test_data`。未完成可见回合时 `approve-stage` 被拒；完成 1 个 DNF 回合后证据绑定候选哈希，批准使 Stage 1→2 且新阶段学习率回到 0。该报告不代表真实达标 |

真实自动评估连续三次达标、所有阶段的完整可见回合以及最终 500 场独立比赛，仍须在正式训练与人工观察时验证。

## Stage 1 前进奖励与短距离课程修复（2026-09-28）

旧实验 `Runs/editor-20260928-181613` 保留不变。其 16 次更新／262,144 步的 640 个回合均超时；平均有效前进奖励约 +0.326、摔倒惩罚约 −6.672。以下检查只验证新训练配置与启动链路，没有评估模型收敛或步态。

| 检查 | 结果 | 证据或限制 |
|---|---|---|
| Python 课程、checkpoint 和诊断测试 | 通过 | `training/tests/test_training.py` 共 26 项；覆盖四档任务、25 次更新评估门槛、最终档连续三次达标、Unity/Python 契约一致、诊断汇总与子任务 checkpoint 学习率保留 |
| C# 静态与 Unity 真实构建 | 通过 | `python tools/compile_check.py --stubs`；训练 Player 构建日志 `Builds/BuildLogs/stage1-reward-build.log` 成功。`doctor` 核对 1078／28 协议、奖励哈希 `c035eb7a…`、课程哈希 `f62898f3…` 与构建哈希 `9a76c7d3…` |
| 奖励与进度规则 | 通过 Unity 探针 | `Builds/BuildLogs/stage1-reward-probe.log` 验证站定无正收益、有效前进、短距离完赛和提前 DNF 时间成本；`Builds/BuildLogs/stage1-progress-probe.log` 验证越界位移不补发、回退重走不重复奖励以及原有冲线规则 |
| 新 Player 启动与安全停止 | 通过 | `Runs/stage1-reward-startup-20260928/session-status.json` 到达 `sampling` 后正常停止；`latest.pt` 记录 Stage 1 子任务 0、0 次 PPO 更新、0 个训练步及新契约哈希。仅 1 worker，未进行正式训练 |

独立回合统计通道已编译并随 Player 启动；由于这次启动检查在完整回合前停止，尚未通过真实回合验证 `episode_metrics.jsonl` 的内容。新奖励能否改变站定倾向需由后续训练和可见验证判断。
