# accepted-v2 0500 線・塗り分離逆変換 v1

## 目的

凍結済み`yaruyomi-accepted-v2-0500`の完成AAを、元本文を正解に保ったまま線と整数面IDへ分離する。これは本当の元画像を復元する処理ではなく、次世代DeepAA型生成器の自己教師入力を作る版付き逆変換である。weak-v1/v2のように線と塗りを一枚へ再合成せず、P4～P6の画素濃度方式も使用しない。

## 候補

- `a-p6r1`: P6R-1で合格した単一文字反復runを隣接行の実幅重なりで面化する基準。
- `b-periodic`: Aを必ず包含し、1～4文字の周期模様を追加する。各文字はSaitamaarの自然advanceを使い、8pxや16pxによる文字幅除外は行わない。

両候補とも、線mask、塗りグリフmask、`int32`面ID、面内の字形濃度属性、文字コード、行、開始x、advance、行幅を保存する。面IDの整数値を濃淡としてモデルへ入力しない。元AAは一度だけ行単位で正本描画し、その画素を自然advanceセルの所有関係で分割するため、線と塗りの論理和は結合文字を含め元描画へ完全一致する。

生成先はローカル専用の次の場所である。

```text
datasets/incoming/yaruyomi/v32.1/accepted-v2/derived/0500/reverse-channels-v1/
├─ manifest.json
├─ records.jsonl
├─ audit.json
├─ works/<entry-id>/
│  ├─ target.txt
│  ├─ channels.npz
│  └─ surfaces.json
└─ previews/*.png
```

## 500件監査結果

| 指標 | A | B |
| --- | ---: | ---: |
| 塗り検出作品 | 438 | 476 |
| 塗り所有文字 | 337,312 | 524,041 |
| run | 26,219 | 36,144 |
| 面 | 5,693 | 7,308 |
| 複数行面 | 4,287 | 5,845 |
| 元描画再構成失敗 | 0 | 0 |
| 面ID整合・連結失敗 | 0 | 0 |

全2,265,274文字のうち、A/Bの所有差は186,729文字、8.243109%で、378/500作品に存在した。Bは400種類の観測モチーフを記録し、現行の20 run・5作品という集計フラグを満たすものは30種類だった。このフラグはカタログの信頼度情報であり、低支持モチーフを無断で線または塗りへ固定するためには使わない。

train 394、validation 51、test 55の全splitで可逆性と面整合性が合格した。カテゴリ別差分率はメカ3.003085%、エフェクト3.287343%から自然18.719901%まで分布する。背景と自然の件数不足は0500 snapshot固有の制約として維持し、750収集方針で補う。

同一処理を別ディレクトリへもう一度実行し、1,500個のwork artifact、manifest、records、24 previewを含むpayload SHA-256が双方とも`e594f72f12d4e28907e44808aa83ea88321144ddf4dcf075fbedf658389ff3cc`で一致した。

全12カテゴリについてA/B差分が最大の2作品、計24 previewを目視した。Bの追加領域は主に空、水面、衣服、髪、建築面の網掛けに対応し、主要輪郭や台詞を大量に面へ移した例はなかった。この監査により、次のモデル設計へ渡す逆変換候補は`b-periodic`とする。ただし、400種類すべてを同じ重みで学習することまでは承認していない。30対応モチーフ、低支持モチーフ、面なし領域をどう入力・損失へ渡すかは次の設計チェックポイントで決める。

この合格は派生データの完全性と逆変換候補の選択であり、学習済みモデルや通常生成の品質合格ではない。通常生成器は変更していない。

## 再現

```powershell
.\.venv\Scripts\python.exe -m training.prepare_reverse_channels_v1
.\.venv\Scripts\python.exe -m training.prepare_reverse_channels_v1 --output .tmp\training-runs\reverse-channels-v1-repeat
.\.venv\Scripts\python.exe -m training.audit_reverse_channels_v1 --visual-status passed --selected-candidate b-periodic
```
