# 固定品質評価

`cases.json` は人物原画、背景原画、人物線画、背景線画の4つの固定入力・変換条件です。画像本体はGit管理しないため、先に `samples/README.md` に記載したローカル画像を配置します。次のコマンドで、AAテキスト、抽出輪郭、再描画画像、機械評価を `output/evaluation-baseline` に保存します。

```powershell
.\.venv\Scripts\python.exe -m evaluation.run
```

実験デコーダを同じ条件で比較する場合は、`--decoder viterbi-pruned` のように上書きします。

線方向と文字境界の不一致スコアは `--structure-strength 1` で比較できます。Random Forest分類器の比較は学習専用環境からモデルパスを指定して実行します。

```powershell
.\.venv-training\Scripts\python.exe -m evaluation.run `
  --classifier random-forest `
  --random-forest-model .\.tmp\training-runs\random-forest\deepaa-random-forest.joblib `
  --structure-strength 1 `
  --output .\output\evaluation-rf-structured
```

`report.json` の距離コストとF1は、抽出輪郭に字形が幾何的に合っているかを追跡します。`text_usage` は空白以外の使用文字数、異なる文字数、上位2文字・10文字への集中率を記録し、少数の頻出文字への縮退を検出します。意味のある線を抽出できたか、顔や建物として読めるかは測れないため、次も固定の人手受入条件とします。

- 人物（80桁前後）: 両目・口が判別でき、顔輪郭・顎・主要な髪束が連続している。
- 背景（120桁前後）: 建物種別と構図が分かり、塔・入口・屋根・主要な遠近線が判別できる。
- 各カテゴリ10枚以上の未学習画像で確認し、局所的な手直しで仕上げられる。

デフォルメ強度を比較するときは、同じ入力・クロップ・桁数を維持します。抽出線の画素数低下だけでなく、顔の目・口や背景の主要線が残っているかを必ず画像で確認します。

## P6R-2 面指標

`evaluation.surface_metrics`は、同一座標の正解面・予測面を一対一対応し、面IoU、被覆率、precision / recall、過剰、欠落、split、merge、面内線損失を計算します。両集合に同じ`correspondence_id`と`coordinate_space_id`がない場合は正解評価を拒否します。P1参照30枚とaccepted-v1 100作品は未対応なので、両者間の正解IoUには使いません。

```powershell
.\.venv\Scripts\python.exe -m training.evaluate_surface_metrics_p6r2
```

定義、合成ケース、固定結果は[P6R-2資料](../docs/SURFACE_METRICS_P6R2.md)を参照してください。

## P6R-3 実画像候補面

P1参照30枚へ、原画のLab色領域と線画の閉領域を複数粒度で提案します。この集合は`accepted-v1`と画素対応しないため、正解IoU、過剰、欠落は計算しません。候補数・面積・被覆、外部背景除外率、粒度間split / mergeと目視用overlayだけを診断します。

```powershell
.\.venv\Scripts\python.exe -m training.evaluate_surface_proposals_p6r3
```

固定結果とゲート判断は[P6R-3資料](../docs/SURFACE_PROPOSALS_P6R3.md)を参照してください。

## P6R-4 対応対と塗り判定

保存済み修正対応対の実クロップ・AAキャンバス上で、入力候補面と修正AA反復塗り面を対応させ、透明な規則ベースv1を同じP6R-2面指標で評価します。

```powershell
.\.venv\Scripts\python.exe -m training.evaluate_surface_decisions_p6r4
```

現時点は修正対応対0件のため品質ゲート未評価です。合成テストは対応・保存契約だけを検証します。条件と未達項目は[P6R-4資料](../docs/SURFACE_DECISIONS_P6R4.md)を参照してください。
