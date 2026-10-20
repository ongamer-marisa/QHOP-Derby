# リポジトリ管理方針

[简体中文](REPOSITORY_CONTENTS_cn.md) · [プロジェクト概要](../README.md)

## Gitで管理するもの

- `Assets/` のソース、美術原本、ライセンス、シーン、マテリアル、設定、対応する `.meta`。`Assets/QHopGenerated/HorseRig.asset` は現行のシリアライズ済み物理設定であり、必要な資産として管理します。
- `Packages/manifest.json`、`Packages/packages-lock.json`、`ProjectSettings/`。`.vsconfig` は共有する開発コンポーネント一覧です。
- `training/` と `tools/` のソース、テスト、診断ツール。旧ツールは履歴の理解用に残しますが、r14では旧checkpointの移行を拒否します。
- 日中の説明、監査、検証記録。旧結果には日付とバージョンを示し、短時間の動作確認を歩様の収束とは扱いません。

## ローカルだけに保存するもの

| 対象 | 理由 |
|---|---|
| `Runs/`、`backup/`、`*.pt`、`*.pth` | checkpoint、最適化状態、実験指標、個人バックアップ |
| `Builds/` | 実行ファイル、元ログ、スクリーンショット、検証データ。ビルドまたは別途保管で取得 |
| `Library/`、`Temp/`、`Logs/`、`Obj/`、`UserSettings/` | Unityキャッシュ、一時ファイル、ユーザー設定 |
| Python仮想環境、pip/Pythonキャッシュ、ビルド出力 | 依存関係一覧から再作成可能なローカル環境 |
| `.vs/`、`.idea/`、`*.csproj`、`*.sln` | ローカルIDE状態、Unityが自動生成するプロジェクトファイル |
| `.env`、`.env.*`（`.env.example`を除く） | ローカル環境・認証設定。実際の秘密情報はソースにも例にも記載しない |
| `*.onnx` と対応する `.meta`、既定の `QHopPolicyMetadata.json` と `.meta` | 実験モデル。独自の出力先は `Runs/` または `Builds/` を推奨 |

無視設定はローカルファイルを削除せず、既に追跡中のファイルも自動で追跡解除しません。正式なモデルは、バージョンと検証結果を付けて別途配布します。`QHopPolicy.onnx`、`QHopPolicyVision.onnx`、`QHopPolicyMetadata.json` を同時に配布してください。

## push前の確認

プロジェクトルートで `git status --short`、`git diff --cached --stat`、`git diff --cached --check` を確認します。大容量ファイル、秘密情報、個人のパスは内容も確認してください。元ログとローカル検証データはGitに同梱しません。新しい環境では学習ガイドに従って依存関係を導入し、Playerを再ビルドします。
