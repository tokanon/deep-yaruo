# 開発・評価用サンプル

このフォルダには、CLIの操作確認と固定品質評価に使うローカル入力画像を置きます。サンプル画像本体はGitへ登録せず、`.gitignore`で除外します。生成されたAA、抽出線、評価レポートもここへ置かず、Git管理対象外の `output/` を使います。

| ファイル | 用途 | 主なプロファイル |
| --- | --- | --- |
| `person-01-source.png` | 人物固定評価 | `person` |
| `person-01-lineart.png` | 人物の完成線画 | `lineart` |
| `background-01-source.png` | 背景固定評価 | `background` |
| `background-01-lineart.png` | 建築の完成線画 | `background_lineart` |

## 配置方法

手元の画像を上表の名前で配置すると、`evaluation/cases.json`と操作資料のコマンドをそのまま使えます。別名や別の場所を使う場合は、CLIへ直接パスを渡すか、ローカル用に評価ケースの`source`を変更します。

画像はリポジトリや配布バイナリへ同梱されません。各利用者が、自分に利用権のある画像だけを配置してください。

## P1参照集合

共通入力抽出器のドメイン差を確認する参照集合は、画像本体とローカル用`p1-reference.json`をこのフォルダへ置きます。追跡対象の`p1-reference.example.json`をコピーし、各画像に重複しない`id`、題材、原画・線画、写真・イラスト、濃淡量、権利状態を記録してください。画像とローカル用マニフェストはどちらもGit対象外です。

```powershell
Copy-Item .\samples\p1-reference.example.json .\samples\p1-reference.json
.\.venv\Scripts\python.exe -m training.prepare_reference_channels
```

P1の参照ゲートは30～50枚で、人物・背景、原画・線画、写真・イラスト・線画、濃淡low・medium・highの各層が少なくとも1枚あることです。この集合はドメイン診断用で、`training_allowed`は既定で`false`です。生成結果とマニフェストは`.tmp/input-channels/`へ保存されます。

2026-08-25のローカル集合は、人物8組・背景7組の原画と線画、計30枚です。内訳は人物16・背景14、原画15・線画15、写真2・イラスト13・線画15、濃淡low 19・medium 5・high 6で、P1層化ゲートを通過しています。各原画と線画は構図差や細部差を許した独立参照であり、画素対応する`gold_pair`ではありません。
