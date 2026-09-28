# 6段階 PPO 学習

[简体中文](README_cn.md) · [プロジェクト概要](../README.md)

学習環境は ML-Agents の低レベル API で Unity と通信し、PyTorch が単一 GPU 上で Actor/Critic を更新します。物理周波数は100 Hz、制御周波数は20 Hzです。Stage 1～4 の既定値は4プロセス、各プロセス8つの単独馬エリアです。Stage 5～6 は4プロセス、各プロセス2つの4頭立てエリアです。段階設定と合格基準は `qhop_train/curriculum.py`、観測プロトコルは `qhop_train/schema.py` と Unity の `HorseTrainingObservation.cs` にあります。

報酬ログには生の値と重み付き寄与を項目別に記録します。`effort` は駆動指令の二乗を使った**制御努力の代理指標**であり、実測の機械エネルギーではありません。各馬のエピソード終了時、Unity worker のログに `QHOP_EPISODE` JSON が1行出力されます。

## インストールとビルド

Python 3.10.1～3.10.12 を使用してください。`pyproject.toml` は検証済みの主要パッケージを固定しています。PyTorch 2.3.1 の通常インストールでは CUDA wheel が指定されないため、利用する GPU ドライバーに適合する CUDA wheel を入れ、`doctor` で CUDA を確認してください。プロジェクト内に仮想環境を作る例です。

```powershell
py -3.10 -m venv .venv-qhop
.venv-qhop/Scripts/python.exe -m pip install --no-deps torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
.venv-qhop/Scripts/python.exe -m pip install -e training
```

Unity Editor の **QHOP → Build training environment** を実行します。Unity ライセンスが使用可能な環境では、次のバッチコマンドも使えます。

```powershell
$project = (Get-Location).Path
New-Item -ItemType Directory -Force Builds/BuildLogs | Out-Null
$buildLog = Join-Path $project 'Builds/BuildLogs/training-build.log'
& 'C:\Program Files\Unity\Hub\Editor\6000.0.45f1\Editor\Unity.exe' -batchmode -nographics -quit -projectPath $project -executeMethod BuildTraining.Build -logFile $buildLog
```

出力は `Builds/Training/QHopTraining.exe` です。初回は `doctor` で依存関係、GPU、ビルドファイルを検査し、Unity で起立、アクションチャンネル、3秒の起こし動作、4頭立てレースの精算を物理テストしてください。通常のサンドボックス内では Unity ライセンス IPC がタイムアウトします。サンドボックス外のバッチ実行では学習 Player をビルドできました。

ビルド後、短い通信確認とスループット基準測定を実行します。

```powershell
$env:PYTHONPATH='training'; .venv-qhop/Scripts/python.exe training/smoke_unity.py --decisions 40
$env:PYTHONPATH='training'; .venv-qhop/Scripts/python.exe training/physics_checks.py --areas 2 --episodes 0
$env:PYTHONPATH='training'; .venv-qhop/Scripts/python.exe training/smoke_unity.py --stage 6 --areas 2 --difficulty 1 --shaping 0 --decisions 200
```

`physics_checks.py --episodes 0` は28個のアクションチャンネルを個別に調べます。`--episodes 100` とした場合にのみ100回のリセット安定性を完全に検証します。旧ビルドではリセットの反復中に転倒が見られ、新しい蹄の高さでの受け入れ検証は未完了です。`smoke_unity.py --expect-terminals` ではエピソード終了数を指定でき、4頭立てレースのタイムアウトと馬ごとの精算を確認できます。

## 学習の開始と再開

以下のコマンドはすべて**プロジェクトルートの PowerShell**で実行します。依存関係の導入と Unity Player のビルド後、モジュールパスを設定して環境を確認します。

```powershell
$env:PYTHONPATH = 'training'
.venv-qhop\Scripts\python.exe -m qhop_train.cli doctor
.venv-qhop\Scripts\python.exe -m qhop_train.cli benchmark --stage 1 --seconds 20
```

`doctor` はビルド契約の一致、観測1078次元、アクション28次元、CUDA の状態を表示します。`benchmark` は同じ総エリア数で Unity の1／2／4プロセスのスループットとメモリを比較します。以下は操作例で、自動実行されません。

**新しい実験を始める。** 新しい出力ディレクトリを使用します。既定では Stage 1、Unity 4プロセス、現在の方策を使う馬の意思決定ステップを合計16,384とする batch から始まります。Stage 1 は25更新ごとに現在の小課題を評価し、最終課題で3回連続して合格すると候補を保存して手動審査待ちになります。Stage 2～6 は100更新ごとに評価します。段階の予算を使い切っても不合格なら停止します。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli train --output Runs/first
```

**短い経路テストを先に実行する。** PPO を1回だけ更新し、正常終了時に `latest.pt` を保存します。短時間学習の成果物はレース用モデルとして扱えません。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli train --output Runs/smoke-example --workers 1 --batch 128 --max-updates 1
```

**同じ実験を checkpoint から再開する。** 元の出力ディレクトリと `latest.pt` を指定します。最初に `--batch`、`--workers`、`--seed` を指定していた場合、再開時にも**同じ値**が必要です。`resume` は Actor、Critic、Adam、観測の正規化、乱数、カリキュラム、段階内の学習率カウンターを復元します。新しい Unity エピソードから始まり、途中の PhysX 接触状態は復元しません。Unity ビルドも checkpoint の記録と一致する必要があります。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli resume --checkpoint Runs/first/latest.pt --output Runs/first
.venv-qhop\Scripts\python.exe -m qhop_train.cli resume --checkpoint Runs/smoke-example/latest.pt --output Runs/smoke-example --workers 1 --batch 128 --max-updates 2
```

`--max-updates` は実験の**累計**更新回数です。例では1回の短い学習の後、累計2回目まで進めます。コンソールの「结束训练并保存」でも安全に停止できます。`latest.pt` は PPO 更新完了ごとにアトミックに保存されます。サンプリング中の未確定 rollout は破棄され、更新中の停止要求は更新完了後に適用されます。強制終了した場合は最後に完了した更新からのみ復元できます。

**段階予算を使い切っても合格しなかった場合。** `stageN-needs-work.pt` が保存されます。そこから再開するとき `--extra-steps` で現在段階の方策ステップ予算を増やせます。学習率は元の段階曲線上の進捗を保ち、ウォームアップをやり直しません。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli resume --checkpoint Runs/first/stage2-needs-work.pt --output Runs/first --extra-steps 1000000
```

**別の微調整実験を始める。** `finetune` は checkpoint のネットワークと観測統計を引き継ぎますが、Adam とカリキュラムを新規作成します。`resume` とは異なります。Stage 3 を通過したモデルから独立した Stage 4 の実験を始める例です。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli finetune --checkpoint Runs/first/stage3-passed.pt --output Runs/stage4-finetune --stage 4
```

学習中は `Runs/<実験>/training_metrics.jsonl`、`episode_metrics.jsonl`、`tensorboard/` を確認します。評価は Stage 1 で25更新ごと、Stage 2～6 で100更新ごとに行われ、`evaluations.jsonl` に記録されます。`latest.pt` は毎回、`step-*.pt` は50更新ごとに保存され、直近5個を保持します。Stage 6 の shaping は `1 → .75 → .5 → .25 → 0` と進みます。各水準で段階予算の10%以上を学習し、評価完走率が直前水準の最良値から5パーセントポイントを超えて下がらない場合のみ進級します。shaping 0 では段階予算の20%以上を学習してから最終進級評価を受けられます。

Stage 1 はまず2 m／10秒、次に5 m／15秒、10 m／20秒を試し、最終的に20 m／20秒で判定します。最初の3課題は各100エピソード以上の評価を1回行い、完走率80%、転倒せずに完走した割合60%が必要です。最終課題は90%／80%を3回連続で達成し、手動検証を待ちます。課題の切り替え後も Adam、観測統計、段階の学習率進捗を保持します。新しい報酬設定では静止し続ける利益を時間コストに変更し、妥当な前進の相対的価値を上げました。歩様の改善は実際の観察が必要です。

コンソールには直近100の完了エピソードについて完走、有効な前進、転倒、静止の割合と報酬項目が表示されます。`episode_metrics.jsonl` は DNF の理由と、姿勢／接地／ペナルティ停止／コース外によって除外された前進距離を記録します。`training_metrics.jsonl` は Actor の有効サンプル率とアクション標準偏差を記録します。診断には ML-Agents の別チャンネルを使うため、1078次元の方策観測は変わりません。

## Unity 学習コンソールと手動承認

まず最新の学習 Player をビルドし、Unity Editor の **QHOP → 训练** メニューでコンソールを開きます。既定値はプロジェクト内の `.venv-qhop`、Unity 4プロセス、batch 16,384です。ウィンドウでは実験ディレクトリ、seed、プロセス数、batch を変更できます。開始時に Stage 1 の実験を作ります。選択したディレクトリに `latest.pt` があれば、元の実験を残して日時付きの新しい実験ディレクトリを自動選択します。「结束训练并保存」は停止を要求し、現在の PPO 更新と worker の終了を待ちます。ウィンドウは端末出力の解析ではなく `session-status.json` を表示します。Editor のスクリプト再読み込み後もセッション PID を保持して再接続し、Editor 終了時は安全な停止を要求します。

「断点继续」は実験の `latest.pt` を読みます。「从 checkpoint 继续…」はファイル選択で Actor、Critic、Adam、学習率、乱数、カリキュラムを完全復元します。古い checkpoint を選ぶと新しい分岐ディレクトリを作り、元の実験ファイルを残します。審査フィールドを持たない旧 checkpoint は未承認として扱い、手動確認を省略できません。復元には Unity ビルドの指紋、報酬とカリキュラムのハッシュ、batch、worker 数、seed の一致が必要です。報酬修正前の `Runs/editor-20260928-181613` は比較用に残し、新しい報酬では「开始训练」から実験を作ってください。予算を使い切って不合格だった段階には、コンソールで追加ステップ数を入力してから再開できます。

Stage 1 の最初の3課題は評価1回の合格で次の小課題へ進みます。最終20 m課題と Stage 2～6 では3回連続合格で `stageN-candidate.pt` を保存し、停止します。「启动阶段验证」は**別ウィンドウの可視学習 Player**を開き、現在の段階・課題の距離、時間制限、地形、体形、馬数、規則で実行します。方策には固定 checkpoint の CPU 決定論的アクションを用い、検証中はモデルを更新しません。学習中なら先に安全停止します。検証 HUD には目標、進捗、速度、レース時間、ペナルティ停止、結果が表示され、観察する馬の切り替え、再試行、終了ができます。完走も DNF も完全な観察1回として数えますが、合否は利用者が判断します。候補 checkpoint の内容ハッシュ、検証 seed、終了時の証拠は `preview-evidence.jsonl` に保存されます。少なくとも1エピソードを終えると「人工通过并进入下一阶段」を押せます。承認後は次段階の準備のみ行い、「断点继续」を手動で押して学習を始めます。Stage 6 の承認でこの流れは終了しますが、正式配備には別途500レースの独立検証が必要です。

観察結果が不十分なら「继续本阶段训练」で候補資格と連続合格回数を消去し、モデル、Adam、正規化、学習率進捗を保ったまま「断点继续」で再開します。6段階の目標と閾値は `Assets/Resources/QHopStageGoals.json` にあり、コンソールと検証 HUD が共用します。

CLI でも同じ審査を操作できます。次のコマンドは例であり、自動実行されません。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli checkpoint-info --checkpoint Runs/first/latest.pt
.venv-qhop\Scripts\python.exe -m qhop_train.cli stop --output Runs/first
.venv-qhop\Scripts\python.exe -m qhop_train.cli preview --checkpoint Runs/first/stage1-candidate.pt --output Runs/first --seed 92500
.venv-qhop\Scripts\python.exe -m qhop_train.cli approve-stage --output Runs/first
.venv-qhop\Scripts\python.exe -m qhop_train.cli resume --checkpoint Runs/first/latest.pt --output Runs/first
```

`preview` には再ビルド済みで契約が一致する学習 Player が必要です。checkpoint が自動評価後の候補でない場合、途中観察に使えても手動進級の資格は得られません。短時間学習、合成評価、未学習モデルによる DNF は学習器の故障やレース指標の達成を意味しません。段階ごとの Player 起動だけを調べるには `preview --max-seconds 4` を使用できますが、完全なエピソード証拠は生成されず、手動進級には使えません。

## 評価とエクスポート

該当段階を終え checkpoint を得てから実行します。以下は `Runs/first/final.pt` が存在する場合の例です。

```powershell
.venv-qhop\Scripts\python.exe -m qhop_train.cli evaluate --checkpoint Runs/first/final.pt --stage 6 --races 500 --seed 82500
.venv-qhop\Scripts\python.exe -m qhop_train.cli export --checkpoint Runs/first/final.pt
```

`export` は決定論的 Actor と正規化を `Assets/Resources/QHopPolicy.onnx` に出力し、`QHopPolicyMetadata.json` も書きます。Unity では M キーでモデルを切り替えます。配備前には同一モデルの Unity／Python アクションの最大絶対誤差が `1e-4` 以下であること、独立した seed で500レース以上の評価を終えることが必要です。短時間学習や合成ベースラインの checkpoint を正式なゲーム用モデルとして出力しないでください。

## オフライン検査

```powershell
python tools/compile_check.py --core
python tools/compile_check.py --stubs
$env:PYTHONPATH='training'; .venv-qhop/Scripts/python.exe -m unittest discover -s training/tests -v
```

Unity Editor バッチのルール検査は学習済みモデルを必要としません。`QHopRaceProbe.Run` はゴールと DNF、`QHopMorphProbe.Run` は81種類の体形・1000組の属性・蹄の高さ、`QHopRewardProbe.Run` は報酬の積算とイベントの一度きりの精算を調べます。`-executeMethod` で個別に実行し、ログは `Builds/BuildLogs/` に保存します。

`--stubs` が調べるのは C# 構文とプロジェクト内アセンブリ参照のみです。実際の ML-Agents／Inference Engine のインポート、PhysX 検証、Unity 内 ONNX の一致確認を代替しません。各学習ステップの指標は `Runs/<実験>/training_metrics.jsonl` と TensorBoard、評価報告は `evaluations.jsonl` に記録されます。
