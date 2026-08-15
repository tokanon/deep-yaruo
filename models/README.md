# 実行時ONNXモデル

このフォルダにはアプリ実行時に必要なONNXモデルと文字対応表だけを置きます。PyTorchチェックポイント、学習履歴、中間出力はGit管理対象外の `.tmp/training-runs/` に保存します。

実験用Random Forestも `.tmp/training-runs/random-forest/` に生成し、通常配布へは含めません。2026年8月18日の比較ではCNNを置き換えられなかったためです。詳細は[実験資料](../docs/RANDOM_FOREST_EXPERIMENT.md)を参照してください。

## `deepaa-light.onnx`

- 用途: 64×64文脈窓から411文字を分類
- 由来: OsciiArt/DeepAAの軽量KerasモデルをONNXへ形式変換
- 変換元: 固定したDeepAAリビジョンの軽量Keras重み（原本はリポジトリへ同梱しない）
- SHA-256: `11E41523C6208B87CFDDDB5FED97F48324009A396ED72AFF9E155B1DB13A169F`

## `deepaa-charset.csv`

- 用途: `deepaa-light.onnx` の411出力を文字へ対応付ける
- 由来: 固定したDeepAAリビジョンの文字一覧
- SHA-256: `35D39653CCA5E3D5E5933319BA58CFE398241342EC2EF5FE60A12E2E354FE3DA`

## `deepaa-start-locator.onnx`

- 用途: `viterbi-pruned`で行内の文字開始候補を絞る
- 由来: DeepAAの座標データから再構成した400作品で学習し、50作品でvalidation、50作品でtest
- 実装: `training/train_start_locator.py`
- 学習条件: seed 42、12 epochs、batch size 64、segment width 256、AdamW、learning rate 0.0003
- 選択チェックポイント: epoch 12、validation loss 0.278745
- 固定test 50作品、951行、CUDA評価時の0.2しきい値: candidate rate 0.603400、precision ±1px 0.457908、recall ±1px 0.989204
- SHA-256: `DC72BFE987A649D71574F988C3735DCB9239A1211BD5F2BC528ABD4BB1B4F27E`

## `deepaa-surface-v0-ls.onnx`

- 用途: 色画像の線1chと面membership・boundary・tone 3chから1,187文字、開始位置、線/塗りroleを推論
- 由来: accepted-v2 0500の分離逆生成Bを使い、独立した線CNN・面CNNを後段融合して3 epoch学習
- 実行: 16位相のshift-and-stitch全畳み込み走査と、開始/非開始尤度を含むSaitamaar実幅DP
- 元checkpoint SHA-256: `687DD5167E2286FCB50B21AF171BF88E756731993D9B0BD3F23A2D9411BB7832`
- ONNX SHA-256: `3E4F702A2E106654A4ADF2C4875EB09A65259F58BEC4F44184570D5D661BB304`

## `deepaa-surface-v0-vocabulary.json`

- 用途: LSの1,187出力と文字、基準フォント、元checkpointを対応付ける
- SHA-256: `71548DED84946EC2C6CEB92BEA3BDD3F50185D0A9ABB134D4EA7C50A0DBBD857`

開始位置モデルの再学習・評価・実行用ファイルへの反映方法は[追加学習資料](../training/README.md)を参照してください。ライセンスと元データの表示は[第三者コンポーネント台帳](../THIRD_PARTY.md)にあります。
