# リリースチェックリスト

この資料は、ソースコードのコミットとWindowsポータブル版の公開を区別して確認するためのものです。

## 1. ソースコミット前

- `git status --short` で、`.venv*`、`.tmp`、`output`、`build`、`dist`、`release`、`frontend/node_modules`が含まれていないことを確認する。
- `samples/` のローカル画像がステージされていないことを確認する。
- `THIRD_PARTY.md` のモデル・フォントのパスとSHA-256が実ファイルに一致することを確認する。
- `python -m pytest -q` が成功することを確認する。
- `npm run build` が成功することを確認する。
- READMEと`docs/`内のローカルリンクが解決できることを確認する。
- `git diff --check` で空白エラーがないことを確認する。
- ステージ後に `git diff --cached --stat` と `git diff --cached` を確認する。

## 2. モデル更新時

- 元データ、学習条件、seed、評価結果を記録する。
- ONNXモデルをCPUExecutionProviderで読み込めることを確認する。
- 固定評価を実行し、従来結果と比較する。
- `THIRD_PARTY.md` のSHA-256を更新する。
- 学習に追加した画像・AA対の利用条件と再配布条件を記録する。

## 3. Windowsポータブル版の公開前

- クリーン環境でフロントエンドをビルドする。
- `YaruoAAStudio.spec` にフォント、推論モデル、文字辞書が含まれることを確認する。
- PyInstallerで生成したEXEを、PythonやNode.jsを入れていないWindows環境で起動確認する。
- 画像変換、抽出線表示、Saitamaar表示、TXT保存を確認する。
- 配布物にルート`LICENSE`、`THIRD_PARTY.md`、`licenses/DeepAA/LICENSE`のDeepAAライセンスを含める。
- PyInstallerに同梱されたPython依存関係について、各パッケージのライセンスと必須NOTICEを収集して配布物へ含める。
- フロントエンド依存関係について、`frontend/package-lock.json`を基準にライセンスを確認する。
- 配布アーカイブに学習用データ、仮想環境、キャッシュ、評価出力が混入していないことを確認する。

依存関係のライセンス一覧はパッケージ更新のたびに変わるため、固定の手書き一覧だけで済ませず、リリース対象環境から再生成して人が確認します。
