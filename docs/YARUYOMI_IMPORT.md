# やる夫AA録2・MLT取り込み

## 方針

`https://aa.yaruyomi.com/` で配布されているまとめZIPを、完成AA候補の収集・仕分け元として使います。v32.1は15,023個のMLT、約138万チャンク、展開時約2.41 GBです。全チャンクを個別ファイルへ展開するとNTFSとGitの負荷が大きいため、次の二段階に分けます。

1. 配布ZIPを変更せず保存し、SQLiteへ出典・位置・分類・ハッシュだけを索引化する。
2. 人が確認または採用する候補だけ、1 AA＝1 UTF-8テキストまたはCP932-NCRテキストとして書き出す。

元のMLTはCP932、区切りは`[SPLIT]`です。ZIP内の日本語ファイル名もCP932として読みます。v32.1には3ファイルだけCP932本文へUTF-8の罫線文字またはWindows-1252の引用符が混入しているため、その断片を復元し`cp932+foreign-fragments`として記録します。成人向けを含む題材は品質で優劣を付けず、すべて索引・書き出し対象に含めます。ただし分布確認と評価セット作成のため、該当パスには`sensitive`タグを付けます。

文字参照は例外的な破損ではなく、CP932外の字形をMLTへ保存するための主要な表現です。全件走査では5,306 MLTに数値参照1,345,672個、1,506コードポイントがありました。セミコロンなしの10進参照が235,391個、標準の名前付き参照が2,304個あります。数値参照のうち1,238,577個は復元後の文字をCP932へ直接保存できません。

原本同一性を確認するSHA-256は文字参照を含む未加工本文から計算します。分類、寸法、プレビュー、レビュー表示、UTF-8書き出しは、ハッシュ検証後に数値参照、標準の名前付き参照、MLTで使われる`&x200A;`形式をUnicodeへ復元して扱います。単独の`&`と未知の名前付き文字列は変更しません。マニフェストには`source_normalized_sha256`、出力後の`exported_sha256`、`text_transforms`、`output_transforms`を併記します。

UTF-8出力は復元済みUnicodeをそのまま保存します。MLTなどの旧来CP932テキストへ渡す場合は`--output-encoding cp932-ncr`を指定し、CP932で表現できない文字だけを10進数値参照へ戻します。HTMLソースへ埋め込む場合の`&`や`<`のエスケープは別工程です。

配布元が公開していることと、機械学習利用・再配布の許諾は同じではありません。権利状態は`unknown; local curation only`として保存し、ZIP、SQLite、書き出したAAはGitの対象外にします。外部公開や学習済み重みの配布前に別途確認します。

## 保存場所

```text
datasets/incoming/yaruyomi/v32.1/
├─ source.zip       配布原本。Git対象外
├─ index.sqlite3    全件の索引。Git対象外
└─ export/
   ├─ manifest.jsonl
   └─ aa/0000001.txt
```

## 索引作成

```powershell
.\.venv\Scripts\python.exe -m training.import_yaruyomi index
```

既存DBを明示的に作り直すときだけ`--force`を付けます。形式確認用には`--max-files 10`を指定できます。

```powershell
.\.venv\Scripts\python.exe -m training.import_yaruyomi stats
```

SQLiteには次を保存します。

- ZIPの版、SHA-256、取得時刻、配布元URL、権利状態
- MLTの相対パス、CRC32、文字コード、チャンク番号、見出し
- 原文バイト列のSHA-256と、改行だけLFへ正規化したSHA-256
- AA候補、見出し、メタデータ、テキストの軽い判定
- 人物、背景、建築、自然、小物、メカ、効果、複数人物などの暫定分類
- 行数、CP932基準の最大列数、文字数、AAらしさの補助値
- 完全重複の代表IDと出現数
- `sensitive`タグ。既定では除外しない

分類はフォルダ名・MLT名・見出しのキーワードによる初期仕分けで、正解ラベルではありません。`背景無し`や`武器なし`などの否定語は分類根拠から除きます。人物・作品別フォルダでは一文字の人名が`山`・`森`・`海`などに偶然一致しないよう、建築・自然・小物の推定を原則として`汎用AA`または明示的な背景ファイルに限定します。`汎用AA`では、ロボットMLTの見出しに「学校」がある場合などに題材が逆転しないよう、MLTパスを見出しより優先します。学習へ採用する前に人が確認します。

分類規則だけを更新した場合は、ハッシュを作り直さず再分類できます。

```powershell
.\.venv\Scripts\python.exe -m training.import_yaruyomi recategorize
```

## 候補の書き出し

既定では完全重複を一つにまとめ、成人向けを含む先頭100件を書き出します。

```powershell
.\.venv\Scripts\python.exe -m training.import_yaruyomi export --category background --limit 100
```

MLTなどの旧来CP932テキストへ渡す場合は、次のようにCP932外文字を数値参照へ戻して書き出します。

```powershell
.\.venv\Scripts\python.exe -m training.import_yaruyomi export --category background --limit 100 --output-encoding cp932-ncr
```

- `--limit 0`: 条件に合う全件。大量の小ファイルを作るため通常は使わない
- `--include-duplicates`: 完全重複も個別に出力
- `--exclude-sensitive`: 必要な評価セットに限って該当タグを除外
- `--sensitive-only`: 該当タグだけを抽出。`--exclude-sensitive`とは同時指定不可
- `--thumbnails`: 各TXTをSaitamaarで最大512pxのPNGへ描画
- `--record-type section`: AA以外の見出し等を点検
- `--output-encoding utf-8`: 復元済みUnicodeを保存する既定値
- `--output-encoding cp932-ncr`: CP932外文字だけを`&#番号;`へ戻し、CP932で保存

`manifest.jsonl`には元MLT、チャンク番号、見出し、分類、両ハッシュ、文字コード、権利状態を残します。TXTだけを移動して来歴を失わないよう、仕分け時はマニフェストも一緒に扱います。

## この段階で分かること・分からないこと

索引から、題材分布、文字・行数分布、完全重複、成人向けを含むフォルダ偏りを測れます。フォルダ名だけでは、AAの完成度、作者、元画像との対応、機械学習利用の可否、人物の全身・上半身などを確定できません。次段階ではカテゴリごとのランダム標本をサムネイル化し、人が採否と詳細タグを付けます。
