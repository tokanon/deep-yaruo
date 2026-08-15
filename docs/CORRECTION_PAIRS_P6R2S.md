# P6R-2S 修正対応対の保存基盤

実行日: 2026-08-28

## 目的

画像から生成した草案AAを人が編集した時点で、元画像、実際に使った変換条件、抽出線、修正前後のAA、再描画、座標変換、文字差分を一組のローカル記録として固定します。P6R-3の候補面方式を待たずに直接対応対を蓄積し、後から候補面と面対応を同じ記録へ追加できることが目的です。

この記録は`accepted-v1`のAA単体や廃止済み`weak-v1/v2`とは別の`correction_pair`です。UIは草案と編集結果が同一の場合を保存しません。

## 保存形式

既定の保存先はGit対象外の`datasets/incoming/correction-pairs/v1/`です。レコードIDは元画像・来歴・変換条件・草案・修正AA・生成資産ハッシュから決定的に作ります。同じ内容の再保存は同じレコードを返し、既存スナップショットを上書きしません。

```text
datasets/incoming/correction-pairs/v1/<record-id>/
├─ record.json
├─ extensions.json
├─ source-image.bin
├─ processed.png
├─ draft.txt
├─ corrected.txt
├─ draft-rendered.png
└─ corrected-rendered.png
```

`record.json`はスキーマ版1の不変スナップショットです。次を保存します。

- 元画像のファイル名、media type、由来、権利状態、SHA-256
- 正規化後の`ConversionOptions`、実クロップ、beam/CNN識別子
- DeepAAモデル、開始位置モデル、文字表、SaitamaarのパスとSHA-256
- 元画像座標とAAキャンバス座標の順逆変換、フォントサイズ、行送り
- 抽出線、草案AA、修正AA、修正前後のSaitamaar再描画
- 置換・挿入・削除の文字範囲と文字数集計
- 全artifactのbyte数、media type、SHA-256とmanifest自身のSHA-256

`extensions.json`だけは独立した版・revision・内容ハッシュを持ち、原子的に更新できます。P6R-3/4向けに`surface_proposals`、`surface_correspondence`、`surface_metrics`の3スロットを予約しています。拡張更新後も`record.json`と既存artifactは変更しません。

## APIとUI

- `POST /api/correction-pairs`: multipartの一式を検証して保存する。
- `GET /api/correction-pairs`: ローカル記録を新しい順に列挙する。
- `GET /api/correction-pairs/{record_id}`: ハッシュ検証後にmanifest、修正前後テキスト、拡張を読む。
- `GET /api/correction-pairs/{record_id}/assets/{asset_name}`: manifestに登録済みのartifactだけを返す。
- `PATCH /api/correction-pairs/{record_id}/extensions`: 予約済み面スロットを原子的に更新する。
- `POST /api/correction-pairs/{record_id}/verify`: 保存元画像と正規化設定で草案を再生成し、テキスト、抽出線、再描画、クロップ、生成資産を照合する。
- `POST /api/correction-pairs/{record_id}/surface-analysis`: 保存済み実クロップとAAキャンバスへ候補面・修正AA面・規則ベース指標を合わせ、三拡張スロットを一回で更新する。

`POST /api/convert`は、画面のスライダー現在値ではなく、変換時に実際に使った正規化済み`options`も返します。AAエディタで草案を変更すると「修正対を保存」が有効になり、元画像とその変換結果を一式で保存します。UIの権利状態は安全側に`user-provided; local use only; redistribution not granted`と記録します。

## ゲート判断

P6R-2Sは**合格**とします。

- 保存直後と再読込後でmanifest、全artifact、テキスト、設定、座標変換、拡張ハッシュが一致する。
- artifactまたはmanifestの改変を読込時に拒否する。
- 同一内容の再保存が同じIDとなり、不変スナップショットを増殖・上書きしない。
- 保存元と同じ入力・設定・生成資産から草案テキスト、抽出線、草案再描画、実クロップを再照合できる。
- 拡張更新後もベースmanifestがbyte単位で不変である。
- フロントエンドの型検査と本番ビルドを通過する。

固定実行ではP6R-2S対象9テストが通過し、既存機能を含む全回帰は112 passed、1 skippedでした。フロントエンドもTypeScript検査とVite本番ビルドを通過しました。

P6R-2S完了時点では候補面と面対応の値自体は未作成で、後続工程が予約スロットへ追加する設計でした。また、UIの固定権利表示は利用許諾を与えるものではなく、外部公開や学習済み重みの配布前に個別確認が必要です。

2026-08-28のP6R-4基盤で、三スロットを同一revisionへ原子的に保存する経路を追加しました。更新上限10 MBは差分だけでなく更新後の拡張全体へ適用します。実修正対はまだ0件なので品質値は未評価です。詳細は[P6R-4資料](SURFACE_DECISIONS_P6R4.md)を参照してください。

## 再現

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_correction_pairs.py tests\test_app.py

cd frontend
npm run build
cd ..
```
