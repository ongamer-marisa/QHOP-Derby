# QHOP Derby — 4頭立てレースと PPO 学習

[简体中文](README_cn.md) · [学習ガイド](training/README.md) · [検証記録](docs/VALIDATION.md)

Unity 6000.0.45f1 / URP 17.0.1。ルートディレクトリを Unity プロジェクトとして開き、`Assets/Scenes/DerbyStraight.unity` または `Assets/Scenes/AttributeLab.unity` を選んで Play を押してください。6段階学習のセットアップ、ビルド、コマンドは[学習ガイド](training/README.md)を参照してください。

## 学習を始める

通常の学習は Unity Editor の **QHOP → 训练 → 打开训练控制台** から操作できます。コンソールでは、開始、安全な停止と保存、途中からの再開、指定 checkpoint からの完全な復元、可視化による段階検証、手動承認を行えます。各段階では自動評価を3回連続で通過すると「手動検証待ち」で停止します。可視化で少なくとも1エピソードを観察し、利用者が承認して初めて次段階に進めます。検証ウィンドウを開く際、進行中の学習は安全に停止します。詳しくは[学習コンソールと手動承認](training/README.md#unity-学習コンソールと手動承認)を参照してください。

Stage 1 は 2／5／10／20 m の順に学習します。新しい学習 Player では報酬とカリキュラムのハッシュを用いて checkpoint を検査します。古い実験は元のディレクトリに残り、「开始训练」を押すと新しい実験ディレクトリが選ばれます。コンソールには直近100エピソードの前進、転倒、静止の割合が表示されます。

プロジェクトルートで PowerShell を開き、[インストールとビルド](training/README.md#インストールとビルド)に従って依存関係を入れ、`Builds/Training/QHopTraining.exe` をビルドします。次のコマンドは利用者が実行した場合にのみ学習を開始します。

```powershell
$env:PYTHONPATH = 'training'
.venv-qhop\Scripts\python.exe -m qhop_train.cli doctor
.venv-qhop\Scripts\python.exe -m qhop_train.cli train --output Runs/first
```

終了後、同じ実験の完全な checkpoint から再開します。

```powershell
$env:PYTHONPATH = 'training'
.venv-qhop\Scripts\python.exe -m qhop_train.cli resume --checkpoint Runs/first/latest.pt --output Runs/first
```

`resume` はモデル、Adam、学習率、カリキュラム状態を復元します。checkpoint と同じ Unity ビルド、batch、worker 数、実験 seed が必要です。自動評価だけでは次段階へ進みません。分割実行、段階予算の追加、評価とエクスポートは[学習コマンド](training/README.md#学習の開始と再開)を参照してください。

シーンはコミット可能な基本シーンと `PresentationBootstrap` から自動構築されます。**QHOP → Generate presentation scenes** で、コース、照明、マネージャーを含む通常の Unity シーンに固定できます。**QHOP → Build Windows** は Windows x64 アプリを `Builds/Windows/` に出力します。

## 操作

- Space：物理デモの開始／一時停止。R：リセット。C：全馬／単独馬／自由カメラの切り替え。
- 1～4：馬を選択。左側で8つの属性を変更し、「应用并重置」を押します。
- 整数の seed を入力して「全部随机」を押すと全馬をランダム生成できます。設定の保存と読み込み先は Unity の `Application.persistentDataPath/horses.json` です。
- 右下で力積の方向と強さを選び、「施加冲量」で衝突を試せます。
- M：起立コントローラーとエクスポート済み PPO モデルを切り替えます。モデルがない場合やプロトコルに互換性がない場合は起立コントローラーを使います。

モデル未導入時、展示用コントローラーは起立のみを担当します。レースでは馬ごとの完走時間と順位を記録します。エクスポートしたモデルは `IHorseController` を介して28次元の `HorseAction` を出力します。

## プロジェクトの構成

- `Assets/Art/Horse/Source/`：未改変の Blender 原本と付属する CC0 ライセンスのページ。
- `Assets/Resources/HorseGame.fbx`：Blender の制御ボーン、制約、ドライバーを除いた低ポリゴン表示用骨格。
- `Assets/Scripts/`：属性生成、19剛体の骨格、関節制御、観測インターフェース、4頭立てシーン、演出。
- `Assets/Editor/`：モデルのインポート設定とシーン／Windows ビルド用メニュー。
- `tools/`：テンプレート複製、Blender エクスポート、静的コンパイル検査の再現可能なツール。
- `training/`：PyTorch PPO、並列サンプリング、カリキュラム、checkpoint、評価、ONNX エクスポート。
- `docs/`：[実装計画](docs/UNITY_PRESENTATION_PLAN.md)と[検証記録](docs/VALIDATION.md)。

元モデルの IK/FK 切り替えは Blender ドライバーに依存するため、Unity では独立した物理関節を使用します。体形は表示骨格のスケール、脚部の長さ、馬ごとの独立したスキンメッシュで変更し、同じ体形から物理衝突体を生成します。

## 現在の制約

短時間のサンプリング、PPO 更新、checkpoint 復元、ONNX の数値一致は検証済みです。6段階の正式な学習と独立した500レースの受け入れ検証を終えたレース用モデルはまだありません。詳細は[検証記録](docs/VALIDATION.md)を参照してください。
