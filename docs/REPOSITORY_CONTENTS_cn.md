# 仓库内容管理

[日本語](REPOSITORY_CONTENTS.md) · [项目说明](../README_cn.md)

## 保留并推送

- `Assets/` 内的源码、美术源文件、许可证、场景、材质、资源配置及配套 `.meta`。`Assets/QHopGenerated/HorseRig.asset` 是当前序列化物理配置，名称虽含 Generated，仍是工程所需资源。
- `Packages/manifest.json`、`Packages/packages-lock.json` 与 `ProjectSettings/`，用于重建 Unity 依赖和项目配置。`.vsconfig` 是共享开发组件清单。
- `training/` 和 `tools/` 的源码、测试及诊断工具；旧工具保留用于理解历史，但 r14 禁止旧 checkpoint 迁移。
- 中日文说明、结构审计及精简验证记录。历史结果标注日期和版本，不把短程工程检查作为步态收敛证明。

## 仅保留在本地

| 内容 | 原因 |
|---|---|
| `Runs/`、`backup/`、`*.pt`、`*.pth` | checkpoint、优化器、实验指标和个人备份；体积较大且绑定实验契约 |
| `Builds/` | 可执行构建、原始日志、截图与验证夹具；由源码构建或单独归档 |
| `Library/`、`Temp/`、`Logs/`、`Obj/`、`UserSettings/` | Unity 导入缓存、临时文件与用户偏好 |
| `.venv-qhop/`、`.venv/`、`venv/`、pip/Python 缓存、Python 构建产物 | 本机环境，可按依赖清单重建 |
| `.vs/`、`.idea/`、`*.csproj`、`*.sln` | 编辑器本地状态或 Unity 自动生成的工程文件 |
| `.env`、`.env.*`（保留 `.env.example`） | 本机环境配置可能含凭据；不要将真实密钥写入源码或示例 |
| `*.onnx` 及配套 `.meta`、默认 `QHopPolicyMetadata.json` 及 `.meta` | 导出的实验模型，不作为源码发布；自定义导出路径优先放在 `Runs/` 或 `Builds/` |

忽略文件不会删除本地内容，也不会自动取消已跟踪文件；必要时应逐项检查后取消跟踪。正式模型如需分发，应单独发布带版本和验证报告的产物，并同时携带 `QHopPolicy.onnx`、`QHopPolicyVision.onnx` 与 `QHopPolicyMetadata.json`。

## 推送前复核

在项目根目录检查 `git status --short`、`git diff --cached --stat` 和 `git diff --cached --check`，确认只包含预期源码和文档。大文件、凭据和个人路径还需检查具体内容，不能仅依赖扩展名忽略。原始验证日志不随 Git 分发；克隆项目后按训练说明安装依赖并重建 Player，不能假定本机历史夹具已存在。
