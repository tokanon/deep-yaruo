# Random Forest文字分類実験

## 1. 目的

CoumarとKingston（2025年）の比較結果を、Yaruo AA Studioの条件へ移植して検証します。標準ASCIIの固定10×10タイルではなく、DeepAAと同じ64×64文脈、411文字、Saitamaarの可変文字幅、既存のBeamデコーダを使用します。

この実験はRandom Forestへの全面移行を前提にせず、CNNと同じ入力・評価条件で文字分類器だけを交換する比較です。CoumarらのGPL-3.0コードは取り込まず、scikit-learnを使って独自実装しています。

## 2. 実装

- 学習器: `training/train_random_forest.py`
- 学習・層化抽出: `training/random_forest.py`
- 実行時アダプター: `backend/random_forest.py`
- CLI切替: `--classifier cnn|random-forest`
- 構造スコア: `--structure-strength 0..2`
- 固定評価: 人物原画、背景原画、人物線画、背景線画の4ケース

学習データはDeepAAの500作品をSaitamaarで再構成した画像です。作品単位の400/50/50分割を維持し、trainから各クラス最大32例、validationから各クラス最大16例を層化抽出しました。学習画像には1回の線幅・ぼかし・ノイズ拡張を加えています。

比較用モデルの条件は次のとおりです。

| 項目 | 値 |
| --- | ---: |
| 木の本数 | 64 |
| 最大葉数 | 512 |
| 学習サンプル | 19,848 |
| 検証サンプル | 2,817 |
| 検証accuracy | 0.940362 |
| 検証macro-F1 | 0.932285 |
| 検証Top-5 accuracy | 0.993965 |
| モデルサイズ | 約12.3MB |

モデルは `.tmp/training-runs/random-forest/deepaa-random-forest.joblib` に生成し、Gitや配布版には含めません。

## 3. 4ケース比較

2026年8月18日に、同じ前処理、Beamデコーダ、桁数で比較しました。数値は抽出線と再描画AAの `F1@2px` で、高い方が幾何的に一致しています。

| ケース | CNN・従来スコア | CNN・構造スコア1 | Random Forest・構造スコア1 |
| --- | ---: | ---: | ---: |
| 人物原画 | 0.770 | **0.773** | 0.734 |
| 背景原画 | **0.826** | 0.825 | 0.755 |
| 人物線画 | 0.804 | **0.810** | 0.760 |
| 背景線画 | 0.811 | **0.817** | 0.750 |

Random Forestは4ケースすべてでCNNを下回りました。人物線画の異なる使用文字数はRandom Forestが337、構造スコア付きCNNが180でしたが、Random Forestでは黒丸や局所線に不適切な複雑文字が増えました。文字種の増加だけではAA品質が上がらず、overmatchingを悪化させることを確認しました。

構造スコアは、線方向のヒストグラムと字形セル上下左右の出入口を比較します。4ケース中3ケースでF1が改善しましたが、背景原画では僅かに低下したため、現時点では既定値を0のままにして実験オプションとして扱います。

## 4. 解釈と制限

今回のRandom Forestは、再構成した完成AA画像とその摂動から学習しています。DeepAA原研究で利用された、既存AAから別CNNで推定した元線画は含まれません。そのため、Random Forest方式そのものを否定する結果ではなく、「現在利用できる500作品の再構成データではCNNを置き換えられない」という判定です。

Coumarらの結果と異なる主な条件は次のとおりです。

- 標準ASCIIではなく411種類のShift-JIS文字
- 固定10×10タイルではなく64×64の周辺文脈
- 固定幅ではなくSaitamaarの可変文字幅
- 単純なタイル連結ではなくBeamによる文字開始位置探索
- 合成単字ではなく、完成AA全体から切り出した文脈

## 5. 採用判断

- Random Forestへの置換: **不採用**
- 比較用分類器と学習コード: **保持**
- 方向・境界不一致スコア: **実験継続**
- 次の優先事項: 隣接文字間・上下行間の実接続評価と、人手修正対の収集

次回Random Forestを再評価する条件は、元画像または元線画と人手完成AAの対応データが蓄積された場合です。現在の再構成データを単純に増量するだけの再学習は優先しません。

## 6. 再現手順

学習します。

```powershell
.\.venv-training\Scripts\python.exe -m training.train_random_forest
```

Random Forestと構造スコアで4ケースを評価します。

```powershell
.\.venv-training\Scripts\python.exe -m evaluation.run `
  --classifier random-forest `
  --random-forest-model .\.tmp\training-runs\random-forest\deepaa-random-forest.joblib `
  --structure-strength 1 `
  --output .\output\evaluation-rf-structured
```

同じ条件でCNNを評価します。

```powershell
.\.venv\Scripts\python.exe -m evaluation.run `
  --classifier cnn `
  --structure-strength 1 `
  --output .\output\evaluation-cnn-structured
```

参考文献は[Coumar・Kingston（2025年）](https://arxiv.org/abs/2503.14375)、日本式プロポーショナルAAの不一致スコアは[Chung・Kwon（2022年）](https://doi.org/10.1109/ACCESS.2022.3167567)です。
