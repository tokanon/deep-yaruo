# P2 構造代理画像ベースライン

実行日: 2026-08-26

## 目的

完成AAから本当の元画像を復元するのではなく、AAラスタから実入力の線チャンネルへ分布を近づける`structure proxy E`を比較します。P2では候補をドメイン判別AUCと芯線Chamferの二軸で診断し、最終採否はP3の下流文字分類と併せて決めます。

## 入力

- 実画像側: P1を通過したローカル参照30枚。人物16・背景14、原画15・線画15、写真2・イラスト13・線画15。
- AA側: `accepted-v1`の採用100件から、12カテゴリを巡回してseed 42で選んだ30件。
- 権利上の扱い: 画像、AA本文、構造代理画像、評価レポートはローカル専用で、`.tmp/training-runs/structure-proxy-p2/`からGitへ入れない。

## 候補

| ID | 処理 |
| --- | --- |
| `raw_raster` | Saitamaar 16px・18px行送りで1倍描画した生AAラスタを共通`X`へ渡す。 |
| `supersampled_blur_downsample` | Saitamaarを4倍で描画し、高解像度上でsigma 2.0pxのGaussian Blurを掛け、1倍描画と同じキャンバスへ`INTER_AREA`縮小して共通`X`へ渡す。出力相当sigmaは0.5px。 |

両候補とも`E`を`ch0`だけへ使います。`ch1`はAAラスタから直接作る設計を維持し、P2のAUCへ混ぜません。

## 指標

### 画像単位ドメインAUC

P1実画像の`ch0`と各代理候補の`ch0`から、入力寸法を直接特徴にしない305次元の画像記述子を作ります。16×16占有率、縦横投影、8方向ヒストグラム、連結成分統計、端点・分岐比率を含みます。

`StandardScaler + LogisticRegression`を5-fold・5反復の`RepeatedStratifiedKFold`で評価し、同じ画像の特徴が学習と評価へ同時に入らないよう画像単位で分割します。95%区間は実画像・代理画像を各ドメイン内で1,000回ブートストラップして求めます。0.5が判別不能、1.0が完全分離です。

### 芯線Chamfer

正解芯線は`raw_raster`を共通`X`へ通した`ch0`に固定します。これは意味上の正解元画像ではなく、「候補変換が生AAラスタの共通構造を何px動かしたか」を測る基準です。候補`ch0`との双方向距離平均を512px幅の画素単位で記録します。

AA本文の反復塗りラベルは単一文字連続しか完全には扱えないため、P2の完全な芯線正解には使いません。塗り補助ラベルはP4/P6の用途を維持します。

## 固定結果

| 候補 | domain AUC | 95%区間 | Chamfer平均 | 中央 | p95 | 最大 | 破滅的失敗 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_raster` | 0.950 | 0.888–0.996 | 0.000px | 0.000 | 0.000 | 0.000 | 0/30 |
| `supersampled_blur_downsample` | 0.940 | 0.868–0.991 | 0.919px | 0.889 | 1.349 | 1.461 | 0/30 |

4倍描画候補はAUCを0.01下げましたが、信頼区間は大きく重なり、実画像分布へ明確に近づいたとは判定できません。一方で芯線を平均0.92px動かします。最大Chamferの複数人物例も構図崩壊や空出力ではなく、細線位置と接続の小さな変化でした。

## 判断

- `raw_raster`は構造保存の基準として保持する。ただしAUC 0.95で実画像とは容易に判別でき、最終候補ではない。
- `supersampled_blur_downsample`もP3比較へ保持する。現時点ではAUC改善が不確かで、構造距離では生ラスタより不利である。
- AUCだけを下げる強いぼかしは追加しない。情報を消してドメイン判別だけを難しくする候補を避ける。
- 両候補についてP3の`C=1`文字accuracyと頻度帯別accuracyを測るまでは、勝者や混合比を確定しない。
- 学習済みの`E`候補を追加する場合も、同じ実画像30枚、同じ記述子、同じChamfer基準へ通す。

## P3追試

同じ2候補を`accepted-v1`の作品分離testでC=1再学習した結果、生ラスタは全体57.41%・非空白62.88%、4倍候補は55.88%・60.44%でした。4倍候補はP2のAUCを0.01下げた一方、P3でも有用性を改善しなかったため、現在設定は暫定混合から外します。詳細は[P3資料](STRUCTURE_PROXY_P3.md)を参照してください。

## 再現

先にP1参照チャンネルを生成し、学習専用環境からP2を実行します。

```powershell
.\.venv\Scripts\python.exe -m training.prepare_reference_channels
.\.venv-training\Scripts\python.exe -m training.evaluate_structure_proxies
```

既存のローカル出力を明示的に更新する場合だけ`--force`を付けます。

```powershell
.\.venv-training\Scripts\python.exe -m training.evaluate_structure_proxies --force
```

数値の正本は`.tmp/training-runs/structure-proxy-p2/report.json`、画像別のChamferと出力先は同フォルダの`records.jsonl`です。
