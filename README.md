# Yaruo AA Studio

グレースケール画像から、AA表示用フォントSaitamaarの字形を使った構造型Shift-JIS AAの下書きを生成するローカルWebアプリです。

現在の色画像向け既定は、accepted-v2 0500から自己教師学習した線・面分離LSモデルの暫定版です。P6Rの回収線と索引付き候補面を入力し、学習済み開始headと文字head、Saitamaar実幅DPで編集可能な下書きを作ります。安全条件に失敗した場合はP6R-G1へ戻り、累積改善OFFでは従来DeepAAを再現できます。完成品を無修正で保証する段階ではありません。

## ドキュメント

- [開発ロードマップ](docs/ROADMAP.md): 現在地、残作業、各バージョンの目標と完了条件
- [P1〜P6 方向性レビューの採否](docs/P1_P6_DIRECTION_REVIEW.md): P6.1打ち切り、面単位への変更、評価条件と停止分岐
- [外部設計書 v1.1 検討結果](docs/EXTERNAL_DESIGN_V1_1_REVIEW.md): 採用点、成立しない前提、改訂後の実装順
- [P2 構造代理画像ベースライン](docs/STRUCTURE_PROXY_P2.md): 生AAラスタと4倍描画候補のドメインAUC・芯線Chamfer比較
- [P3 線1ch下流文字分類](docs/STRUCTURE_PROXY_P3.md): accepted-v1でのC=1再学習、非空白・頻度帯別charAcc
- [P4 塗り段1](docs/FILL_STAGE1_P4.md): ch2内部の可変幅保持上書き、32条件比較、単独不採用の実測根拠
- [P5 構造augment・ch1窓幅](docs/STRUCTURE_AUGMENT_P5.md): 線を削らない混合、下流追試、ch1量子化崩壊とgamma候補
- [P6 線＋濃度2ch学習](docs/MULTICHANNEL_P6.md): C2三条件、固定実画像の誤塗り、濃度段階の範囲外問題
- [P6R-1 AA塗り面](docs/AA_FILL_SURFACES_P6R1.md): 水平runの二次元面化、accepted-v1全100件集計、面積整合ゲート
- [P6R-2 面指標](docs/SURFACE_METRICS_P6R2.md): 一対一面対応、過剰・欠落・split・merge、未対応集合の拒否契約
- [P6R-2S 修正対応対](docs/CORRECTION_PAIRS_P6R2S.md): 元画像・設定・草案・修正AA・座標・差分のハッシュ付きローカル保存
- [P6R-3 実画像候補面](docs/SURFACE_PROPOSALS_P6R3.md): Lab色領域と線画閉領域の複数粒度候補、P1参照30件の境界漏れ診断
- [P6R-4 対応対と塗り判定基盤](docs/SURFACE_DECISIONS_P6R4.md): 修正AA面との同一座標対応、透明な規則基準、0件時の未評価契約
- [Random Forest実験](docs/RANDOM_FOREST_EXPERIMENT.md): 411文字での再現方法、4ケース比較、採用判断
- [操作ガイド](docs/USER_GUIDE.md): 入力画像の選び方、画面操作、設定の意味、症状別調整、CLI利用
- [処理構造](docs/ARCHITECTURE.md): 実行時パイプライン、前処理、デフォルメ、DeepAA推論、学習・評価構造
- [DeepAA初期データ監査](docs/DATA_AUDIT.md): 500作品・899文字の分布、重複、411文字しきい値、系列漏洩
- [DeepAA 500作品の質的レビュー](docs/BOOTSTRAP_QUALITY_REVIEW.md): 全件目視、題材・完成度・教師適性、今後の利用範囲
- [やる夫AA録2の取込手順](docs/YARUYOMI_IMPORT.md): MLT索引、分類、重複、個別TXT・サムネイル抽出
- [採用コーパスと疑似線画](docs/CURATED_CORPUS.md): accepted-v1固定、MLT分割、weak-v1/v2の失敗結果と実ペアへの移行
- [やる夫AA録2 v32.1監査](docs/YARUYOMI_V32_1_AUDIT.md): 15,023 MLT・約125万AA候補の初回実測
- [教師AAレビューガイド](docs/DATA_REVIEW_GUIDE.md): 採否、0～5点、カテゴリ修正、問題タグの判断基準
- [品質評価](evaluation/README.md): 固定ケース、機械評価、人手受入条件
- [追加学習](training/README.md): データ再構成、モデル学習、開始位置モデル
- [モデル台帳](models/README.md): ONNXモデルの由来、学習条件、評価値、SHA-256
- [リリース確認](docs/RELEASE_CHECKLIST.md): ソースコミットとWindows版公開の確認項目

## 現在できること

- PNG/JPEGなどの画像読込
- 全体／顔・上半身クロップ
- サンプル向けの顔クロップ
- 出力幅、最大行数、細部、デフォルメ強度、輪郭しきい値、ノイズ除去の調整
- 元画像、抽出輪郭、Saitamaarレンダリングの比較
- 人物用の微細輪郭抽出／背景用の長直線・構造抽出／人物線画・背景線画の直接入力プロファイル
- Shift-JISで表現可能な候補文字によるAA生成
- DeepAA学習済み軽量モデルによる411文字分類
- 頻出文字の事前確率補正、字形密度評価、線字形の過剰反復抑制（`.`、`:`、`;`など塗り文字の連続は許可）
- 人物・線画は文字表現の多様性、背景は長直線の幾何一致を優先するプロファイル別デコード
- AAセルより細かい近接線を統合・骨格化し、人物の顔領域を保護する適応的デフォルメ
- 色画像ではP6R回収線と索引付き面を別chで受ける1,187文字LSモデルを暫定既定とし、失敗時は従来のP6R-G1累積生成へfallback（累積改善OFFで旧DeepAAへ切替可能）
- Saitamaar 16px・18px行ピッチでの再描画
- DeepAAとSaitamaar実字形評価による可変幅AA生成
- AAテキストの直接編集、Unicodeコピー、CP932/NCRコピー、UTF-8 TXT保存
- CLIでの一括変換

## 動作環境の目安

通常のAA生成はONNX RuntimeのCPU推論で完結し、GPUやCUDAは不要です。Windows 10/11の64bit環境を対象とします。

| | 実用下限の目安 | 推奨 |
| --- | --- | --- |
| CPU | 比較的新しいx64 4コアCPU | 近年のCore i5／Ryzen 5以上、6～8コア |
| メモリ | 8GB | 16GB |
| ストレージ | 1GB以上の空き容量 | SSD |
| GPU | 不要 | 不要（現在の生成処理では使用しません） |

LSモデルを使う現在の色画像向け既定生成は、開発環境で人物72列×32～43行が約10～12秒、背景120列×30～47行が約15～20秒でした。処理量はおおむね出力幅と行数に応じて増え、設定上限付近の140列×100行では大幅に遅くなる可能性があります。この時間は固定性能を保証するベンチマークではなく、端末性能や入力内容によって変動します。

モデルファイル3点の合計は約23MBです。4GBメモリでも起動できる可能性はありますが、Windows、ブラウザー、Python実行環境を含めると余裕がないため、公開時の最低目安は8GBとします。ソースからの再学習は通常利用とは別要件で、Python 3.12の学習環境とNVIDIA GPUを推奨します。

## 起動

この作業フォルダでは依存関係と画面のビルドが済んでいます。

```powershell
.\run.ps1
```

表示された `http://127.0.0.1:8000/` をブラウザで開きます。

開発モードは次のコマンドです。

```powershell
.\start.ps1
```

## 初回セットアップ

Windows、Python、Node.js LTSが必要です。現在の確認環境はPython 3.13、Node.js 24です。学習専用環境だけはPyTorchの対応に合わせてPython 3.12を使います。AA表示用の`assets/fonts/Saitamaar.ttf`はリポジトリに同梱します。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
cd frontend
npm install
npm run build
cd ..
```

## CLI

```powershell
.\.venv\Scripts\python.exe -m backend.cli .\input.png `
  --columns 72 `
  --abstraction 35 `
  --crop 0.2 0.015 0.64 0.61
```

`--abstraction`は0で無効、100で最強です。人物・線画では顔がクロップ内の中央上部にある前提で、目・口を含む領域の元線を保護します。背景プロファイルでは長直線を崩さないよう弱く適用します。

実験用の全幅Viterbi/DPデコーダは `--decoder viterbi` で選べます。`--decoder viterbi-pruned` は軽量な行開始モデルで候補を絞ってからDPを実行します。通常は高速な `beam` を使用します。

`output` フォルダへ以下を保存します。

- `input.txt`: 編集可能なAA
- `input-edges.png`: 抽出した輪郭
- `input-aa.png`: Saitamaarで再描画した確認画像

## テスト

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 品質評価と追加学習データ

人物・背景のローカルサンプルを同じ条件で再評価するには、`samples/README.md`どおり画像を配置してから次を実行します。

```powershell
.\.venv\Scripts\python.exe -m evaluation.run
```

DeepAAの文字座標から500作品の正解AA画像をローカルで再構成するには、Git対象外の`datasets/bootstrap/deepaa-500.csv`を[来歴文書](datasets/bootstrap/PROVENANCE.md)どおり配置してから、次を実行します。

```powershell
.\.venv\Scripts\python.exe -m training.reconstruct_dataset
```

詳細は `evaluation/README.md` と `training/README.md` にあります。生成物はそれぞれ `output` と `.tmp` に置かれ、Git管理には含めません。

やる夫AA録2のまとめMLTをローカル索引へ取り込み、選別対象だけを個別TXTとSaitamaar画像へ書き出す手順は[MLT取込手順](docs/YARUYOMI_IMPORT.md)を参照してください。配布原本と生成索引は`datasets/incoming/`へ置き、Gitには含めません。

## Windowsポータブル版の作成

フロントエンドをビルドしてからPyInstallerを実行します。

```powershell
cd frontend
npm run build
cd ..
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm .\YaruoAAStudio.spec
```

`dist\YaruoAAStudio\YaruoAAStudio.exe`を起動すると、空いているローカルポートでアプリが立ち上がり、既定ブラウザーで画面が開きます。変換処理とモデル推論はすべてローカルで実行されます。

公開用バイナリを作る場合は、依存ライブラリのライセンス通知を含む[リリースチェックリスト](docs/RELEASE_CHECKLIST.md)も実施してください。

## DeepAA

初期学習済みモデル、文字対応表、ローカル実験で使用する500作品の初期データは [OsciiArt/DeepAA](https://github.com/OsciiArt/DeepAA) に由来します。DeepAAの配布物にはMITライセンス表示がありますが、500作品のAA本文は第三者作品を含むため、このリポジトリには収録しません。固定リビジョン、ハッシュ、ローカル配置方法は `THIRD_PARTY.md` と `datasets/bootstrap/PROVENANCE.md` に記録しています。

原本のソース、旧Kerasファイル、字形pickle、サンプル画像、500作品CSVは取り込んでいません。Git管理する実行時資産は `models/`、来歴とライセンスは `datasets/bootstrap/` に限定しています。学習データはローカル専用とし、追加データセットも利用条件と再配布条件を別々に確認してください。

## ライセンス

このプロジェクト自身のコードは[MIT License](LICENSE)です。取り込んだモデル、フォント、その他の依存物は[THIRD_PARTY.md](THIRD_PARTY.md)を参照してください。開発・評価用画像はGitへ含めず、扱いは[samples/README.md](samples/README.md)に記録します。
