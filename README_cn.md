# QHOP Derby — 四马赛道与 PPO 训练

[日本語](README.md)

Unity 6000.0.45f1 / URP 17.0.1。打开根目录作为 Unity 项目，选择 `Assets/Scenes/DerbyStraight.unity` 或 `Assets/Scenes/AttributeLab.unity`，按 Play。六阶段训练的安装、构建和命令见 [training/README_cn.md](training/README_cn.md)。

## 训练快速入口

日常训练可在 Unity Editor 选择 **QHOP → 训练 → 打开训练控制台**。控制台提供开始、结束并保存、断点继续、从指定 checkpoint 完整恢复、可见阶段验证和人工晋级；每阶段自动评估连续三次达标后会停在“待人工验证”，完成至少一回合可视化观察并由你确认后才进入下一阶段。打开验证窗口时，正在运行的训练先安全停止。完整操作见[训练说明](training/README_cn.md#unity-训练控制台与人工晋级)。

Stage 1 现按 2／5／10／20 米逐档学习。新训练 Player 使用奖励及课程哈希校验 checkpoint；旧实验保留在原目录，点击“开始训练”会选用新的实验目录。控制台可查看最近 100 回合的前进、摔倒和静止比例。

在项目根目录打开 PowerShell，先按 [训练说明](training/README_cn.md#安装与构建)安装依赖并构建 `Builds/Training/QHopTraining.exe`。以下命令仅在你主动执行时启动训练：

```powershell
$env:PYTHONPATH = 'training'
.venv-qhop\Scripts\python.exe -m qhop_train.cli doctor
.venv-qhop\Scripts\python.exe -m qhop_train.cli train --output Runs/first
```

训练退出后，从同一实验的完整 checkpoint 继续：

```powershell
$env:PYTHONPATH = 'training'
.venv-qhop\Scripts\python.exe -m qhop_train.cli resume --checkpoint Runs/first/latest.pt --output Runs/first
```

`resume` 恢复模型、Adam、学习率和课程状态；需要使用与 checkpoint 相同的 Unity 构建、batch、worker 数和实验 seed。自动评估不会直接跨阶段。如何分段运行、增加阶段预算、评估与导出，见[完整训练命令](training/README_cn.md#启动与继续训练)。

场景以可提交的基础场景文件和 `PresentationBootstrap` 自动搭建。在菜单 **QHOP → Generate presentation scenes** 可以把场景固化为含赛道、灯光和管理器的普通 Unity 场景。菜单 **QHOP → Build Windows** 生成 Windows x64 程序至 `Builds/Windows/`。

## 操作

- 空格：开始／暂停物理演示；R：重置；C：全体／单马／自由镜头。
- 1～4：选择马匹。左侧修改八项属性后点击“应用并重置”。
- 输入整数 seed 后点击“全部随机”；配置可保存和读取，文件位于 Unity 的 `Application.persistentDataPath/horses.json`。
- 右下角选择冲量方向及强度，点击“施加冲量”测试碰撞。
- M：在站立控制器和已导出的 PPO 模型间切换；模型缺失或协议不兼容时使用站立控制器。

未安装模型时，展示控制器只负责站立。比赛逐马记录完赛时间和名次；导出的模型通过 `IHorseController` 输出 28 维 `HorseAction`。

## 工程内容

- `Assets/Art/Horse/Source/`：未修改的 Blender 原文件和附带 CC0 许可证页面。
- `Assets/Resources/HorseGame.fbx`：去除 Blender 控制器、约束和驱动后的低面数显示骨架。
- `Assets/Scripts/`：属性随机生成、19 刚体骨架、关节控制、观测接口、四马场景与展示特效。
- `Assets/Editor/`：模型导入设置和场景／Windows 构建菜单。
- `tools/`：可重复的模板拷贝、Blender 导出和静态编译检查。
- `training/`：PyTorch PPO、并行采样、课程、checkpoint、评估与 ONNX 导出。
- `docs/`：实施计划与已执行验证记录。

模型原作者的 IK/FK 切换依赖 Blender 驱动器；Unity 中使用独立物理关节。体型变更通过显示骨架尺度、腿部骨段与每马独立蒙皮网格处理，物理碰撞体由同一体型生成。

## 当前限制

训练器的短程采样、PPO 更新、checkpoint 恢复和 ONNX 数值对齐已验证；尚无经过六阶段正式训练及 500 场独立验收的比赛模型，详见 [docs/VALIDATION_cn.md](docs/VALIDATION_cn.md)。
