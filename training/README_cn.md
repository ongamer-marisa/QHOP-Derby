# 六阶段 PPO 训练

训练环境通过 ML-Agents 低层 API 与 Unity 通信，PyTorch 在单 GPU 上更新 Actor/Critic。物理频率 100 Hz，控制频率 20 Hz。第 1～4 阶段默认 4 个进程、每进程 8 个单马场地；第 5～6 阶段默认 4 个进程、每进程 2 个四马场地。阶段配置和验收阈值在 `qhop_train/curriculum.py`，观测协议在 `qhop_train/schema.py` 和 Unity 的 `HorseTrainingObservation.cs`。

奖励日志逐项记录原始量与加权贡献；其中 `effort` 是驱动力指令平方的**控制努力代理**，不等于实测机械能。每匹马终局时 Unity worker 日志写出一条 `QHOP_EPISODE` JSON 记录。

## 安装与构建

使用 Python 3.10.1～3.10.12。项目的 `pyproject.toml` 锁定了已测试的核心包版本；PyTorch 2.3.1 默认安装方式未指定 CUDA wheel，需按本机驱动安装可用的 CUDA wheel 并执行 `doctor` 确认 CUDA。建议在项目内创建虚拟环境：

```powershell
py -3.10 -m venv .venv-qhop
.venv-qhop/Scripts/python.exe -m pip install --no-deps torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
.venv-qhop/Scripts/python.exe -m pip install -e training
```

在 Unity Editor 菜单执行 **QHOP → Build training environment**，或在 Unity 许可证可用时运行：

```powershell
$project = (Get-Location).Path
New-Item -ItemType Directory -Force Builds/BuildLogs | Out-Null
$buildLog = Join-Path $project 'Builds/BuildLogs/training-build.log'
& 'C:\Program Files\Unity\Hub\Editor\6000.0.45f1\Editor\Unity.exe' -batchmode -nographics -quit -projectPath $project -executeMethod BuildTraining.Build -logFile $buildLog
```

构建产物为 `Builds/Training/QHopTraining.exe`。初次运行先用 `doctor` 校验依赖、GPU 与构建文件，并在 Unity 里完成站立、动作通道、3 秒扶正及四马结算物理测试。常规沙盒内 Unity 启动会遇到许可证 IPC 超时；沙盒外批处理已构建训练 Player。

构建完成后，先运行短程通信检查，再运行吞吐基准：

```powershell
$env:PYTHONPATH='training'; .venv-qhop/Scripts/python.exe training/smoke_unity.py --decisions 40
$env:PYTHONPATH='training'; .venv-qhop/Scripts/python.exe training/physics_checks.py --areas 2 --episodes 0
$env:PYTHONPATH='training'; .venv-qhop/Scripts/python.exe training/smoke_unity.py --stage 6 --areas 2 --difficulty 1 --shaping 0 --decisions 200
```

`physics_checks.py --episodes 0` 逐项检查 28 个动作通道。将 `--episodes` 设为 100 才执行完整重置稳定性验收；截至目前旧构建的重复重置测试观察到摔倒，新马蹄高度尚未通过这项验收。`smoke_unity.py` 可用 `--expect-terminals` 要求指定数量的回合终局，适合检查四马比赛超时与逐马结算。

## 启动与继续训练

以下命令均在**项目根目录的 PowerShell** 中执行。先完成上面的安装和 Unity Player 构建，再设置模块路径、检查环境：

```powershell
$env:PYTHONPATH = 'training'
.venv-qhop\Scripts\python.exe -m qhop_train.cli doctor
.venv-qhop\Scripts\python.exe -m qhop_train.cli benchmark --stage 1 --seconds 20
```

`doctor` 应显示构建契约匹配、观测为 1078 维、动作为 28 维，并检查 CUDA。`benchmark` 在相同总场地数下比较 1／2／4 个 Unity 进程的吞吐和内存。下面的命令只作操作示例，不会自动执行。

**新建完整实验。** 使用新的输出目录；默认从 Stage 1 开始，4 个 Unity 进程、总 batch 16,384 个当前策略马匹决策步。Stage 1 每 25 次更新评估当前子任务，最终档连续三次达标后保存候选并停在人工审核；Stage 2～6 每 100 次更新评估。阶段预算用尽但未达标时也会停止。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli train --output Runs/first
```

**先跑短程链路测试。** 此命令只做 1 次 PPO 更新，正常退出并保存 `latest.pt`；短训产物不能当作比赛模型。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli train --output Runs/smoke-example --workers 1 --batch 128 --max-updates 1
```

**从 checkpoint 继续同一实验。** 使用原输出目录和 `latest.pt`；若最初指定过 `--batch`、`--workers` 或 `--seed`，继续时必须指定**相同**值。`resume` 恢复 Actor、Critic、Adam、观测归一化、随机数、课程和阶段内学习率计数，从新的 Unity episode 开始，不恢复中途 PhysX 接触状态。Unity 构建必须与 checkpoint 记录的构建一致。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli resume --checkpoint Runs/first/latest.pt --output Runs/first
.venv-qhop\Scripts\python.exe -m qhop_train.cli resume --checkpoint Runs/smoke-example/latest.pt --output Runs/smoke-example --workers 1 --batch 128 --max-updates 2
```

`--max-updates` 指**实验累计**更新次数：上例短训先到第 1 次，再继续到第 2 次，而非额外训练 2 次。也可以通过控制台“结束训练并保存”安全停止。`latest.pt` 在每次完整 PPO 更新后原子保存；采样中的未提交 rollout 会丢弃，更新中的停止请求会在完成更新后生效。强制结束只能恢复上一个完整更新。

**阶段预算已用尽但未达标。** 训练器会保存 `stageN-needs-work.pt`。从该文件继续时可用 `--extra-steps` 给当前阶段增加当前策略时间步预算；学习率保持原阶段曲线进度，不重新热身。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli resume --checkpoint Runs/first/stage2-needs-work.pt --output Runs/first --extra-steps 1000000
```

**另开微调实验。** `finetune` 继承指定 checkpoint 的网络和观测统计，新建 Adam 与课程状态，不等同于 `resume`。例如已有 Stage 3 晋级模型时，从它启动一个独立的 Stage 4 实验：

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli finetune --checkpoint Runs/first/stage3-passed.pt --output Runs/stage4-finetune --stage 4
```

训练期间查看 `Runs/<实验>/training_metrics.jsonl`、`episode_metrics.jsonl` 和 `tensorboard/`。Stage 1 每 25 次更新评估，Stage 2～6 每 100 次更新评估，报告写入 `evaluations.jsonl`。`latest.pt` 每次完整更新保存，`step-*.pt` 每 50 次更新保存并保留最近 5 份。Stage 6 的 shaping 档位按 `1 → .75 → .5 → .25 → 0` 前进：每档至少训练本阶段预算的 10%，且评估完赛率相对上一档最佳下降不超过 5 个百分点才晋级；零 shaping 至少训练 20% 的阶段预算才可参加最终晋级评估。

Stage 1 先学习 2 米／10 秒，再依次尝试 5 米／15 秒、10 米／20 秒，最终验收仍为 20 米／20 秒。前三档各需一次至少 100 回合的评估达到完赛率 80%、无摔倒完成率 60%；最终档需连续三次达到 90%／80%，随后等待人工验证。子任务切换保留 Adam、观测统计和阶段学习率进度。最新奖励将站定的持续收益改为时间成本，并提高合法前进的相对价值；这只是新的训练配置，实际步态仍需观察。

控制台显示最近 100 个完整回合的完赛、有效前进、摔倒、静止比例与奖励分项。`episode_metrics.jsonl` 每回合记录 DNF 原因、前进位移因姿态／接地／罚停／越界被过滤的量；`training_metrics.jsonl` 记录 Actor 有效样本比例及动作标准差。诊断经 ML-Agents 独立通道传输，不改变 1078 维策略观测。

## Unity 训练控制台与人工晋级

先构建最新的训练 Player，再在 Unity Editor 的 **QHOP → 训练** 菜单打开控制台。默认使用项目 `.venv-qhop`、4 个 Unity 进程及 batch 16,384；窗口可修改实验目录、seed、进程数和 batch。开始训练会创建 Stage 1 实验；若当前目录已有 `latest.pt`，会自动选用新的时间戳实验目录，保留原实验。结束训练并保存会请求停止，等待当前 PPO 更新完成并关闭本会话 worker；窗口显示 `session-status.json`，不依赖解析终端输出。Editor 脚本重载后保留会话 PID 并重新关联；关闭 Editor 时会请求安全停止。

“断点继续”读取当前实验 `latest.pt`；“从 checkpoint 继续…”通过文件选择读取完整 Actor、Critic、Adam、学习率、随机数和课程状态。选择较早 checkpoint 时自动创建新的实验分支目录，原实验文件仍在原位置。旧 checkpoint 没有审核字段时视为未审核，不能借此跳过人工确认。恢复要求相同 Unity 构建指纹、奖励与课程哈希、batch、worker 数和 seed。奖励修复之前的 `Runs/editor-20260928-181613` checkpoint 应保留作对照，不能直接在新奖励下恢复；请使用“开始训练”新建实验。若已达阶段预算但未通过，可在控制台填写追加阶段步数后继续本阶段。

Stage 1 前三档各通过一次评估后升级子任务；最终 20 米任务及 Stage 2～6 连续三次达标时保存 `stageN-candidate.pt` 并停止。点击“启动阶段验证”会打开**独立可见的训练 Player**，按当前阶段和子任务的距离、时限、地形、体型、马数与规则运行；策略使用固定 checkpoint 的 CPU 确定性动作，验证不更新模型。训练还在运行时，该操作先请求安全停止，之后打开 Player。验证 HUD 显示目标、进度、速度、比赛时间、罚停和结局，可切换观察马、重试或退出。正常完赛与 DNF 都算一次完整观察，是否通过仍由你判断。候选 checkpoint 内容哈希、验证 seed 和终局证据保存在 `preview-evidence.jsonl`；至少一回合完成后才可点击“人工通过并进入下一阶段”。批准仅准备下一阶段，仍需手动点击“断点继续”；Stage 6 批准后流程结束，正式部署还需 500 场独立验收。

如果观察不满意，点击“继续本阶段训练”会清除候选资格与连续达标次数，保留模型、Adam、归一化和学习率进度，然后点击“断点继续”。六阶段目标与阈值来自 `Assets/Resources/QHopStageGoals.json`，控制台和验证 HUD 共用。

CLI 也能操作同一审核流程；以下命令仅为示例，不会自动执行：

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli checkpoint-info --checkpoint Runs/first/latest.pt
.venv-qhop\Scripts\python.exe -m qhop_train.cli stop --output Runs/first
.venv-qhop\Scripts\python.exe -m qhop_train.cli preview --checkpoint Runs/first/stage1-candidate.pt --output Runs/first --seed 92500
.venv-qhop\Scripts\python.exe -m qhop_train.cli approve-stage --output Runs/first
.venv-qhop\Scripts\python.exe -m qhop_train.cli resume --checkpoint Runs/first/latest.pt --output Runs/first
```

`preview` 必须使用已重建且构建契约匹配的训练 Player；若 checkpoint 还不是自动评估候选，它只供中途观察，不提供人工晋级资格。短训、合成评估和未训练模型的 DNF 不代表训练器故障，也不代表模型已达到赛马指标。
只检查各阶段 Player 能否启动时，可给 `preview` 增加 `--max-seconds 4`；这种短启动检查不会生成完整回合证据，不能用于人工晋级。

## 评估与导出

只有完成相应阶段并取得 checkpoint 后才运行这些命令。以下示例假设 `Runs/first/final.pt` 已存在：

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli evaluate --checkpoint Runs/first/final.pt --stage 6 --races 500 --seed 82500
.venv-qhop\Scripts\python.exe -m qhop_train.cli export --checkpoint Runs/first/final.pt
```

`export` 把确定性 Actor 和归一化导出到 `Assets/Resources/QHopPolicy.onnx`，并写出 `QHopPolicyMetadata.json`；在 Unity 中按 M 切换模型。部署前还需用同一模型验证 Unity／Python 动作最大绝对误差不超过 `1e-4`，并完成独立 seed 的至少 500 场比赛评估。不要把短训或合成基线 checkpoint 导出为正式游戏模型。

## 离线检查

```powershell
python tools/compile_check.py --core
python tools/compile_check.py --stubs
$env:PYTHONPATH='training'; .venv-qhop/Scripts/python.exe -m unittest discover -s training/tests -v
```

Unity Editor 批处理规则探针（均不依赖已训练模型）：`QHopRaceProbe.Run` 检查冲线和 DNF，`QHopMorphProbe.Run` 检查 81 组体型、1000 组属性及马蹄高度，`QHopRewardProbe.Run` 检查奖励积分与事件只结算一次。用 `-executeMethod` 分别运行，日志保存在 `Builds/BuildLogs/`。

`--stubs` 只检查 C# 语法及项目内程序集引用，不能代替真实 ML-Agents／Inference Engine 导入、PhysX 验证或 Unity 内 ONNX 对齐。逐轮训练指标在 `Runs/<实验>/training_metrics.jsonl` 和 TensorBoard，评估报告在 `evaluations.jsonl`。

