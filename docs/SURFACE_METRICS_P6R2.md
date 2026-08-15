# P6R-2 面指標と対応条件

実行日: 2026-08-27

## 目的

同一座標にある正解面集合と予測面集合を一対一対応させ、面IoU、被覆率、塗り面precision / recall、過剰、欠落、split、merge、面内線損失を区別します。同時に、画素対応しない集合へ正解指標を出さない契約を固定します。

## 対応契約

`evaluation.surface_metrics.SurfaceSet`は次を必須情報として持ちます。

- 0を背景、正の整数を面IDとする二次元ラベルマスク
- 同じピクセル座標を表す`coordinate_space_id`
- 正解対であることを明示する`correspondence_id`
- 任意の同一形状の線マスク

正解評価には、両集合で同じ非空の`correspondence_id`、同じ`coordinate_space_id`、同じラベル形状が必要です。未対応集合、対応ID不一致、座標系不一致、形状不一致は評価せず例外にします。P1参照30枚とaccepted-v1 100作品はこの契約上の対応IDを持たないため、正解面IoUを計算できません。

## 指標

- 一対一対応: IoU 0.5以上の候補をIoU、交差面積、面IDの安定順で貪欲に対応する。
- 対応面: IoU、正解面被覆率、予測面被覆率。
- 面検出: true positive、precision、recall、F1、過剰面、欠落面。
- 全画素: 塗り画素のIoU、precision、recall、F1。
- split / merge: 交差面積が小さい側の面積の10%以上を占める重なりグラフで、一つの正解面から複数予測面をsplit、一つの予測面へ複数正解面が入る場合をmergeとする。
- 面内線損失: 正解面内の基準線画素について、予測線の既定1px近傍に残った割合と損失を出す。

split / mergeは一対一対応の成否と別に記録します。例えば1面を正確に半分へ分けた予測は、一方がIoU 0.5で対応してもsplit 1、過剰面1として残ります。

## 固定結果

合成ケースは次を区別しました。

| ケース | TP | precision | recall | 過剰 | 欠落 | split | merge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 完全一致 | 2 | 1.0 | 1.0 | 0 | 0 | 0 | 0 |
| 過剰 | 1 | 0.5 | 1.0 | 1 | 0 | 0 | 0 |
| 欠落 | 1 | 1.0 | 0.5 | 0 | 1 | 0 | 0 |
| split | 1 | 0.5 | 1.0 | 1 | 0 | 1 | 0 |
| merge | 1 | 1.0 | 0.5 | 0 | 1 | 0 | 1 |

面内線損失の合成例では、正解面内2画素の基準線に対し1画素を残し、retention 0.5、loss 0.5を得ました。

P6R-1のaccepted-v1全100件・656面を同じ座標の自己回帰へ通した結果、100件すべてで面precision / recallと画素IoUが1.0、split / mergeが0でした。同じ100件から`correspondence_id`を外した評価は100件すべて拒否しました。これは評価実装と可変幅AA座標の回帰確認であり、実画像候補面の正しさを示す値ではありません。

## ゲート判断

P6R-2は**合格**とします。

- 完全一致、過剰、欠落、split、mergeを個別に区別できる。
- 面内線損失を同一座標で計算できる。
- accepted-v1全100件の可変幅座標マスクを決定的に自己対応できる。
- 未対応集合から正解IoUを出す経路を契約で拒否できる。

次はロードマップどおり、前倒ししたP6R-2S「修正対応対の保存基盤」へ進みます。

## 再現

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_surface_metrics.py
.\.venv\Scripts\python.exe -m training.evaluate_surface_metrics_p6r2
```

数値と作品別契約確認の正本は、Git対象外の`.tmp/training-runs/surface-metrics-p6r2/report.json`です。
