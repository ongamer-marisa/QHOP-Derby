# 展示と PPO 学習の検証記録

[简体中文](VALIDATION_cn.md) · [プロジェクト概要](../README.md)

記録日：2026-09-28。プロジェクトディレクトリ：`D:\dev\QHOP Derby`。

| 検査項目 | 結果 | 根拠・制約 |
|---|---|---|
| Blender 原本の読込と FBX 出力 | 合格 | `tools/export_horse.py` が Blender 2.83 の factory startup で正常終了 |
| FBX の往復インポート | 合格 | `tools/verify_asset.py`：ボーン40本、三角面964個、マテリアルスロット4個、有効な変形端点10個 |
| C# 中核構文とアセンブリ参照 | 合格 | `python tools/compile_check.py --core` で Unity 6 のローカルアセンブリを使い中核スクリプトをコンパイル |
| 学習／推論ソースを含む C# 静的検査 | 合格 | `python tools/compile_check.py --stubs` および実際の Unity パッケージによる Player スクリプトのコンパイル |
| シーンファイルと BuildSettings の参照 | 静的検査に合格 | 2つのシーンと一意な GUID が存在し、`EditorBuildSettings.asset` に登録済み |
| Unity Editor インポートとスクリプトの再コンパイル | 合格 | サンドボックス外で Editor のライセンス通信に成功。`Builds/BuildLogs/training-build-r19.log` 参照 |
| 4頭が物理的に20秒間起立 | 一部合格 | 学習 Player の Stage 5／6 は2つの4頭立てエリアで各10秒動作。展示シーンと最終モデルの挙動は未検査 |
| 体形81組、属性1000組、リセット100回 | 一部合格 | Unity Editor の検査で体形81組の19剛体、18関節、総質量、蹄の最下部衝突面 y=-0.02 と、生成属性1000組の範囲を確認。`Builds/BuildLogs/morph-probe-r1.log` 参照。旧 r9 ビルドで反復リセット中の転倒を観測。新 r10 は100回の安定性検査が未完了 |
| 衝突、摩擦、転倒、起こし動作 | 未実施 | Unity 物理実行と画面観察が必要 |
| トゥーン描画、中国語 UI、音、60 FPS | 未実施 | Editor／Windows ビルドと試験用ハードウェアが必要 |
| Windows x64 学習・展示ビルド | 合格 | 両 Player を更新した共通物理コードで再ビルド。学習 Player には終了理由ログを追加。`Builds/BuildLogs/training-build-r19.log` と `Builds/BuildLogs/presentation-build-r9.log` 参照。展示画面とフレームレートは未検査 |

展示シーンの見た目、描画性能、多数の体形での安定性は未確認です。物理パラメータは実際の学習 Player で短時間テスト済みです。展示シーンの受け入れは[実装・受け入れ計画](UNITY_PRESENTATION_PLAN.md)に従って進めます。

## 6段階学習の実装検査（2026-09-28）

| 検査項目 | 結果 | 根拠・制約 |
|---|---|---|
| Python パッケージの依存関係 | 合格 | プロジェクト仮想環境の `pip check` で依存関係の破損なし |
| Python ユニットテスト | 合格 | 観測プロトコル、カリキュラム、Stage 6 shaping の評価基準、余弦学習率、GAE、停止ペナルティ時の Actor マスク、tanh 確率、CPU／CUDA checkpoint、古いビルドの拒否、ONNX/PyTorch アクションの一致について計15件 |
| ML-Agents Python 環境インポート | 合格 | `mlagents_envs.environment.UnityEnvironment` をインポート可能 |
| CUDA 数値更新 | 合格 | PyTorch 2.3.1+cu121 が RTX 4060 Laptop を認識。実際の Unity サンプルから PPO を2回更新し、checkpoint を128ステップから256ステップへ再開、Adam の状態を保持。`Runs/smoke-r6/latest.pt` 参照 |
| 最新の物理ビルドで学習／再開／出力 | システム短時間テストに合格 | r10 物理ビルドで1 worker・128ステップ batch の PPO 更新を1回実行。`Runs/smoke-r10/latest.pt` から第2回／256ステップへ worker の再起動なしに再開。`training_metrics.jsonl` に更新、サンプリング時間、スループットを記録。ONNX は `Runs/smoke-r10/QHopPolicy.onnx` に出力し、メタデータの物理指紋は r10。経路確認専用のモデルで、ゲームには配備していない |
| Stage 6 の評価基準と学習経路 | システム短時間テストに合格 | ユニットテストで各 shaping 水準の予算10%以上、直前最良完走率からの低下5パーセントポイント以内、shaping 0 の予算20%以上を検査。合成と明記した Stage 5 ベースラインから checkpoint を作り、実際の Unity 4頭シーンで Stage 6 PPO を1回更新（現在方策130ステップ、worker 再起動0回）。`Runs/stage6-system-smoke/training_metrics.jsonl` 参照。実際の進級評価に合格したという意味ではない |
| Unity 学習シーンとビルド | 合格 | サンドボックス外の `BuildTraining.Build` で実際の ML-Agents／Inference Engine アセンブリのコンパイルに成功し、Windows Player を `Builds/Training/QHopTraining.exe` に生成。`Builds/BuildLogs/training-build-r19.log` 参照 |
| 物理とアクションチャンネル | 一部合格 | 新 r10 Player で28チャンネルを個別に適用し、観測から読み返し。極端なアクションによる転倒から復帰まで60回の意思決定／3秒を測定。新ビルドでは蹄の最下部衝突面を平地 y=-0.02 に合わせ、起こし動作は地面の高さから再構築。反復リセットと体形の組み合わせの受け入れは未完了 |
| Python↔Unity 通信と並列サンプリング | 初期検査に合格 | 新 r10 Player の Stage 4 の最大難度の起伏／障害物エリアで有効な意思決定400回。Stage 5／6 の最大難度の4頭立てエリアで、現在方策の意思決定を各1000回取得。Stage 6 は shaping 0。NaN なし。Stage 1 の1／2／4プロセス・32エリア・15秒の独立ベンチマークは約343／556／626馬ステップ毎秒、worker のピークメモリは約272／488／916 MB。これは旧ビルドの結果で、r10 のスループット再測定は未実施 |
| 4頭の個別終了とタイムアウト | 未学習方策シーンで合格 | Stage 5 の2レース・8頭が終了。Python は現在方策5頭と固定対戦相手3頭の終了イベントを受信。5頭は早期 DNF、残り3頭は90.00秒でタイムアウト DNF。順位はついていない。実際のゴール到達は学習済みモデルで未検査 |
| ゴール、順位、報酬の二重計上防止 | ルール検査に合格 | Unity Editor の簡易剛体で、最初の1頭が完走しても他馬が続行すること、物理ステップ内補間による計時、同一ステップのゴール順、転倒中のゴール無効、復帰前後の進捗報酬の重複防止、タイムアウト DNF を確認。`Builds/BuildLogs/race-probe-r2.log` 参照。実際のコースでの学習済みモデルは未検査 |
| 段階別の報酬精算 | ルール検査に合格 | Unity Editor で継続報酬が時間ステップの分割に左右されないこと、有効な新規進捗への報酬、Stage 6 の shaping 0 ではレース目標のみを残すこと、完走と DNF の報酬・ペナルティを各1回だけ計上することを確認。`Builds/BuildLogs/reward-probe-r1.log` 参照 |
| 4頭立て並列スループット | 短い基準測定に合格 | 新学習ビルドの Stage 5 で8エリアを固定し各設定10秒測定。1／2／4プロセスは約379／403／403現在方策ステップ毎秒、worker のピークメモリは約258／474／901 MB。2と4プロセスの差は短時間測定の揺らぎ内。既定の4プロセスは長時間の測定が必要 |
| Unity 内 ONNX アクション一致 | 数値試験に合格 | 一時的な2頭モデルを Unity Inference Engine CPU にインポート。入力を揃えた ONNX Runtime とのアクションの最大絶対誤差は `2.98e-7` で、`1e-4` 以下。当時の一時ログは後に Unity が消去。一時モデルはリソースから削除済みで、正式モデルの出力後に再検査が必要 |
| 6段階の正式学習と独立した500レース | 未実施 | 収束済み checkpoint、最終 ONNX、レース成績報告は存在しない |

学習 Player の基本通信、28チャンネルの個別試験、短時間の並列サンプリング、4頭の個別終了、完全な3秒の停止ペナルティ、CUDA 更新、checkpoint 復元、Unity の ONNX 数値一致は実行済みです。体形81組と属性1000組の静的な物理検査、ゴールのルール検査も実施しました。旧ビルドの反復リセット安定性は未合格で、新しい蹄の高さでの100回リセットも未実施です。実コースのゴール、展示のフレームレート、正式モデルの挙動は未検査です。短時間学習 checkpoint はシステム確認専用で、6段階を正式に学習した最終モデルはまだありません。

## Unity 学習メニューと手動進級の短時間試験（2026-09-28）

この節で検証したのは学習制御システムのみです。128ステップの batch、短時間の Player 起動、または**合成と明示した**自動評価報告を使いました。6段階の正式学習は開始しておらず、未学習モデルの DNF は学習器の故障とみなしません。

| 検査項目 | 結果 | 根拠 |
|---|---|---|
| Unity の実スクリプトコンパイルと学習 Player | 合格 | `Builds/BuildLogs/training-menu-release-editor-compile.log` と `Builds/BuildLogs/training-menu-release-build.log` が成功。`doctor` は依存関係、RTX 4060、1078/28プロトコル、ビルド契約を確認 |
| Python の安全停止・審査・PPO 数値試験 | 合格 | `training/tests/test_training.py` の22項目。サンプリング中・更新中の停止、更新エラー時の checkpoint 保護、ハッシュ基準、手動承認と拒否、学習率復元を含む |
| Editor メニューからの起動と安全停止 | 合格 | `Builds/BuildLogs/training-menu-probe.log` は `QHOP_MENU_PROBE_PASS updates=0` を報告。メニューが実際に1 worker を起動し、サンプリング中に停止して初期状態の完全な checkpoint を保存。子プロセスは終了 |
| 新ビルドの短時間学習と分岐復元 | 合格 | `Runs/menu-smoke-final/latest.pt` で128ステップを1回更新。`Runs/menu-smoke-branch/latest.pt` はそこから第2回更新へ完全復元。元 checkpoint のハッシュは不変で、段階の学習率を重複ウォームアップしていない |
| 可視 Stage 1 の完全エピソード | 機能検査に合格 | 最終ビルドの `Runs/menu-release-smoke/preview-status.json` は `closed, episodes=1`。Player ログには Direct3D 11／RTX 4060 と通常の timeout DNF。検証前後で checkpoint の SHA-256 は一致 |
| Stage 2～6 の可視環境起動 | 短い起動検査に合格 | `Runs/menu-preview-fixtures/stage*/preview-status.json` はすべて `closed`。各4秒実行し、Player ログに RTX 4060 を確認、checkpoint ハッシュは不変。`synthetic_preview_fixture` と記した未学習モデルを使用し、完全エピソードは0件。モデルの成績達成を意味しない |
| 合成自動評価後の手動承認基準 | システム検査に合格 | `Runs/menu-review-fixture` の報告は `synthetic_test_data`。可視エピソード完了前の `approve-stage` は拒否。DNF エピソード1件後、証拠を候補ハッシュに結び付けて Stage 1→2 の承認ができ、新段階の学習率は0に戻った。実際の成績達成を意味しない |

実際の自動評価での3回連続合格、全段階の完全な可視エピソード、独立した最終500レースは、正式学習と手動観察の中で検証する必要があります。

## Stage 1 の前進報酬と短距離カリキュラムの修正（2026-09-28）

旧実験 `Runs/editor-20260928-181613` はそのまま残しました。16更新／262,144ステップの640エピソードはすべてタイムアウトしました。有効前進報酬の平均は約 +0.326、転倒ペナルティの平均は約 −6.672 でした。以下は新設定と起動経路のみの検査で、モデルの収束や歩様は評価していません。

| 検査項目 | 結果 | 根拠・制約 |
|---|---|---|
| Python のカリキュラム、checkpoint、診断試験 | 合格 | `training/tests/test_training.py` の26項目。4課題、25更新ごとの評価基準、最終課題の3回連続合格、Unity/Python 契約一致、診断集計、課題 checkpoint の学習率保持を含む |
| C# 静的検査と Unity 実ビルド | 合格 | `python tools/compile_check.py --stubs`。学習 Player の `Builds/BuildLogs/stage1-reward-build.log` は成功。`doctor` は1078／28プロトコル、報酬ハッシュ `c035eb7a…`、カリキュラムハッシュ `f62898f3…`、ビルドハッシュ `9a76c7d3…` を照合 |
| 報酬と進捗のルール | Unity 検査に合格 | `Builds/BuildLogs/stage1-reward-probe.log` で静止時に正の報酬がないこと、有効前進、短距離完走、早期 DNF の時間コストを検証。`Builds/BuildLogs/stage1-progress-probe.log` でコース外の移動に後から報酬を与えないこと、後退後の再前進を二重計上しないこと、既存のゴール規則を検証 |
| 新 Player の起動と安全停止 | 合格 | `Runs/stage1-reward-startup-20260928/session-status.json` が `sampling` になった後、正常停止。`latest.pt` に Stage 1 課題0、PPO 更新0回、学習ステップ0回、新しい契約ハッシュを記録。1 worker のみで正式学習は実施していない |

エピソード統計の独立チャンネルはコンパイルされ、Player とともに起動しました。ただし今回は完全なエピソードの前に停止したため、実エピソードでの `episode_metrics.jsonl` の内容は未確認です。新しい報酬が静止傾向を変えるかは、今後の学習と可視検証で判断します。
