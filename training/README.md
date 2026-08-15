# 学習データ基盤

`datasets/bootstrap/deepaa-500.csv` は、DeepAA由来の500作品について、作品名・開始座標・文字・ラベルを保持するローカル専用の初期データです。第三者AA本文を含むためGitには収録しません。必要な場合だけ、`datasets/bootstrap/PROVENANCE.md`に記載した固定リビジョンから取得して同パスへ配置します。元の線画画像がなくても、`assets/fonts/Saitamaar.ttf` の字形を座標どおりに配置することで、正解AA画像とテキストを再構成できます。

P0のレンダラ逆検証は、ローカル配置したCSVについて、文字ラベル、x座標のadvance累積、18px行送り、文字列からの再構成、一括文字列描画、Windows GDI幅を独立経路で照合します。不一致時は最初の作品、文字、座標、画素をJSONへ記録します。

```powershell
.\.venv\Scripts\python.exe -m training.verify_renderer
```

出力は`.tmp/renderer-verification/report.json`です。2026-08-24の固定実行では500作品、11,208行、512,145配置、899文字を検査し、411クラス対象の510,910配置を含めて不一致は0件でした。全11,208行で、CSV右端、Pillow一括advance、Windows GDI advanceが文字幅加算値と一致しました。GDIのtext extent高さは19pxですが、これはフォントセルのextentであり、データと学習キャンバスの行送りは18pxとして別に固定します。

旧再構成は各グリフをadvance幅の小画像へ切り取っていたため、`f`などインクがセル右端を越える文字を含む102作品で一括描画と差がありました。正本を文字列一括描画へ変更し、検証側は各x座標へグリフを直接描画する独立実装にして解消しています。

P1の共通入力抽出器は、同じ寸法で線`ch0`、低周波濃度`ch1`、暗部面`ch2`を生成します。学習専用の複製を作らず、実行時と共有できる`backend.input_channels`を正本にします。

```powershell
.\.venv\Scripts\python.exe -m training.prepare_reference_channels
```

4固定サンプルだけの場合は`.tmp/input-channels/`へpilotを生成し、参照ゲートは未達として記録します。30～50枚のローカル集合を使う場合は`samples/README.md`に従って`samples/p1-reference.json`を作成します。各画像のch0/ch1/ch2、圧縮NPZ、線・塗り被覆率、濃度4段階の分布、層化不足を`manifest.json`へ保存します。

2026-08-25の固定実行では、人物16枚・背景14枚、原画15枚・線画15枚の計30枚を処理し、写真・イラスト・線画と濃淡low・medium・highの全層を満たしました。線画15枚のch2被覆率は0.08～19.36%（平均2.74%）、原画15枚は21.35～88.92%（平均55.60%）です。暗い写真・夜景で高い被覆率になることは入力診断として保持し、最終AAの塗り文字採用率とは見なしません。各原画と線画は独立した参照画像であり、画素対応する正解対には使用しません。

初回目視では、制限なしの膨張再構成が太い暗部から接続細線全体へ伝播する問題を検出しました。核の侵食半径内だけ境界を復元する制約と、太い矩形につながる細線の回帰テストを追加しています。固定線画でのch2被覆率は人物0.19%、背景4.14%で、原画は実際の暗部として人物70.84%、背景49.25%でした。これは最終塗り文字の採用率ではなく、P4で境界セルと誤塗りを検証するための入力マスクです。

P2では、生AAラスタと4倍描画・ブラー・縮小の構造代理画像を、P1参照30枚に対する画像単位ドメインAUCと、共通`X`後の芯線Chamferで比較します。scikit-learnを使うため学習専用環境から実行します。

```powershell
.\.venv-training\Scripts\python.exe -m training.evaluate_structure_proxies
```

2026-08-26の固定実行では、生ラスタはAUC 0.950・Chamfer 0px、4倍候補はAUC 0.940・Chamfer平均0.919pxでした。AUCの95%区間は重なり、4倍候補の明確な改善とは判定しません。両候補をP3の`C=1`下流文字分類へ残します。条件、指標の意味、結果は[P2資料](../docs/STRUCTURE_PROXY_P2.md)に固定しています。ローカル出力は`.tmp/training-runs/structure-proxy-p2/`です。

P3では`accepted-v1`の100作品をMLT単位の80/10/10 splitのまま使い、各候補の共通`ch0`から411文字を再学習します。

```powershell
.\.venv-training\Scripts\python.exe -m training.train_structure_proxy_p3
```

固定実行は226,146文字例、411語彙被覆99.582%、正規化座標衝突0でした。testでは生ラスタが全体57.41%・非空白62.88%、4倍候補が55.88%・60.44%でした。旧頻度20未満の30例は両方0%で、全体accuracyだけでは希少文字性能を説明できません。条件、頻度帯、作品単位bootstrap、採用判断は[P3資料](../docs/STRUCTURE_PROXY_P3.md)に固定しています。データセットとチェックポイントは`.tmp/training-runs/structure-proxy-p3/`へ保存します。

P4では現行草案Mを保持し、`ch2`内部だけを実測塗り文字へ置き換える学習不要の段1を4固定ケースで評価します。Saitamaarの可変幅を維持し、境界と構造セルを保護します。

```powershell
.\.venv\Scripts\python.exe -m training.evaluate_fill_stage1
.\.venv\Scripts\python.exe -m training.tune_fill_stage1
```

32条件中16条件が機械ゲートを通り、共通選択設定は人物・背景の主要線損失を最大1.35%、ch2外追加を最大0.69%、線画の置換を0に抑えました。一方、人物の黒髪、暗い服、暗い背景を同じ`;`の水平帯として塗り、髪の流れと黒目を表現できませんでした。段1単独は実行時既定へ採用せず、P6のC=2/3学習との比較下限として保持します。詳細は[P4資料](../docs/FILL_STAGE1_P4.md)にあります。ローカル出力は`output/evaluation-fill-stage1/`でGit対象外です。

P5では、生ラスタch0へ座標を動かさない弱い描画劣化だけを混ぜ、作品単位group AUC、Chamfer、P3と同じheld-out文字分類で確認します。線のランダム欠落は再利用しません。

```powershell
.\.venv-training\Scripts\python.exe -m training.evaluate_structure_augment_p5
.\.venv-training\Scripts\python.exe -m training.train_structure_augment_p5
.\.venv-training\Scripts\python.exe -m training.evaluate_tone_windows_p5
```

生ラスタ75/50%と弱いGaussian・JPEG・縮小往復の混合はChamfer平均0.03/0.07pxで、P3 test文字精度の悪化を0.13pt未満に保ちました。1px太線化は1作品でChamfer 13.23pxとなったため不採用です。暫定train混合は`processing_50`です。

ch1診断では、従来のW48・4段階・gamma 1.0がAA側で平均1.03段階しか使わず、ほぼ全面0になることを検出しました。共通抽出器へ量子化前gammaを追加し、P6第一候補をW48×H54、sigma 12、gamma 0.5、4段階としました。W64・W80もC=2比較へ残します。正解AAのない実画像ではcharAccを計算せず、ch1-only AUCと濃淡変動だけを測ります。詳細は[P5資料](../docs/STRUCTURE_AUGMENT_P5.md)にあります。

P6ではW48・W64・W80のC=2を6 epoch学習し、P3 C1と同じtestへ反復塗りrun内外のaccuracyを追加しました。

```powershell
.\.venv-training\Scripts\python.exe -m training.train_multichannel_p6
.\.venv-training\Scripts\python.exe -m training.evaluate_multichannel_p6
```

W48は合成testの塗りrun内を44.36%から45.70%へ改善しましたが、固定人物原画で反復塗りが19文字から2,428文字へ暴増しました。AA trainのch1は段階0/1だけ、人物・背景原画はほぼ段階2/3だけでした。正解と無関係な広域toneをtrainへ50%混ぜるP6bも採用しません。この記録は、塗りを面IDへ索引化せず画素濃度で扱った当時の経緯を再現する監査履歴に限ります。数値や原因分類をaccepted-v2の分離逆生成方式の設計・採否に使いません。現在の主経路は[ロードマップ](../docs/ROADMAP.md)のaccepted-v2線・塗り分離逆生成を正とします。

P3〜P6のモデル・augment・後処理は、品質改善としてはすべて不採用です。P3生ラスタC1は歴史的なDeepAA比較基準に限ります。P4〜P6は塗りを面IDへ索引化せず画素マスク・濃度チャンネルで解こうとした不適切な問題設定なので、コードと結果を監査履歴としてだけ保持し、現在の方式選択、パラメータ、失敗診断、accepted-v2逆生成設計の参考にはしません。

追加の完成AA候補は、やる夫AA録2のまとめMLTを全展開せずSQLiteへ索引化します。取得元、権利上の扱い、成人向けを含むタグ方針、仕分け用書き出し手順は `docs/YARUYOMI_IMPORT.md` を参照してください。

採用100件へ到達したレビューを、原本ハッシュとMLT単位split付きで固定します。採用100件はtrain 80、validation 10、test 10、不採用217件は品質順位用`negative_pool`になります。

```powershell
.\.venv\Scripts\python.exe -m training.freeze_review_corpus
.\.venv\Scripts\python.exe -m training.audit_review_snapshot
```

採用・不採用317件の本文構造だけを使う品質順位Random Forestは、学習専用環境で実行します。source MLT単位の未見評価が弱いため、現在は比較基準でありレビューキューには接続しません。

```powershell
.\.venv-training\Scripts\python.exe -m training.train_quality_ranker
```

採用AAから20作品×4方式の疑似線画候補を再現生成します。

```powershell
.\.venv\Scripts\python.exe -m training.generate_weak_pairs
```

weak-v1評価後は、同一20作品について方式名を伏せる6候補のweak-v2を生成しました。

```powershell
.\.venv\Scripts\python.exe -m training.generate_weak_pairs_v2
```

weak-v1/v2の評価は完了済みで、字形を崩すほど構造と塗りも失われるため、全方式を1枚の画像・AA対応教師として不採用にしました。現役のレビューUI、API、保存実装、再集計コマンドは削除済みです。既存の生成物、評価DB、結果文書は失敗実験の監査証拠として保持します。ただしweak-v1の`dropout`は線品質平均3.95であり、比較基準としてのみ残します。この結果が棄却したのは旧1ch候補です。その後、`E`を線ch0、AAラスタ濃度をch1に分けたP6 C2も実測し、実画像の誤塗りにより不採用としました。現在はAA本文の反復runを二次元の塗り面へまとめ、元画像候補面と分離して扱います。

固定値、出力構造、評価結果、weak pairの制約は[採用コーパスと疑似線画](../docs/CURATED_CORPUS.md)にあります。

`dropout`線へ別方式の濃淡・領域塗りを1枚の画像として合成する実験は、必要線の欠落と意味領域でない誤塗りにより不採用になりました。この生成コードと比較画面は現行機能から削除しています。次の自己教師では、accepted-v2の完成AAから線chと塗り・面chを別々に逆生成し、元AAを正解として保持します。実際の元画像も同じチャンネル定義へ通してP6R-G1と比較し、人手修正対応対は正解対応が必要な未解決例だけの任意監査にします。

accepted-v2 0500の逆変換データは次で生成・監査します。

```powershell
.\.venv\Scripts\python.exe -m training.prepare_reverse_channels_v1
.\.venv\Scripts\python.exe -m training.prepare_reverse_channels_v1 --output .tmp\training-runs\reverse-channels-v1-repeat
.\.venv\Scripts\python.exe -m training.audit_reverse_channels_v1 --visual-status passed --selected-candidate b-periodic
```

P6R-1単一文字面Aと、1～4文字周期模様を実advanceで追加するBを別々に保存します。元本文、文字開始位置、advance、行幅、元描画、線mask、塗りグリフmask、`int32`面ID、面濃度とrun metadataを保持し、面ID値自体を濃淡入力にはしません。固定結果は500/500件で可逆性・面整合性、二重生成のbyte決定性に合格し、24件の層化目視後にBを次段候補としました。詳細は[0500逆変換監査](../docs/REVERSE_CHANNELS_V1.md)です。

承認済みのL/LSモデル比較は学習専用環境から実行します。

```powershell
.\.venv-training\Scripts\python.exe -m training.train_deepaa_surface_v0
.\.venv-training\Scripts\python.exe -m training.evaluate_deepaa_surface_v0 --split validation
.\.venv-training\Scripts\python.exe -m training.evaluate_deepaa_surface_v0 --split test
```

Lは分離線だけ、LSは同じ線CNNと独立した面CNNを後段融合します。面入力はmembership、ID値に依存しないboundary、toneの3枚だけです。学習文字に頻度足切りやUNK統合は行いません。評価契約と停止条件は[DeepAA line/surface v0](../docs/DEEPAA_SURFACE_V0.md)に固定します。

合格したLS checkpointをCPU実行用ONNXへ書き出し、固定15件の実画像安全ゲートを再実行するには次を使います。

```powershell
.\.venv-training\Scripts\python.exe -m training.export_deepaa_surface_v0
.\.venv\Scripts\python.exe -m training.evaluate_deepaa_surface_real_v0
```

実画像decoderは正解文字数や開始座標を使わず、学習済み開始headと文字headを全xで走査して実幅DPします。レポートは`.tmp/training-runs/deepaa-surface-real-v0/report.json`です。

```powershell
.\.venv\Scripts\python.exe -m training.reconstruct_dataset
```

生成先は `.tmp/reconstructed-deepaa` です。固定seed 42で作品単位に400件/50件/50件の train/validation/test へ分割します。同じ作品の文字が複数splitへ漏れることはありません。

ただし、監査の結果、500作品はファイル名から判別できる5系列だけに集中し、通常の作品単位splitでは5系列すべてが複数splitをまたぐことが分かりました。系列漏洩への感度を確認する再構成では次を使います。比率は418件/56件/26件となり、均衡した最終評価ではなく漏洩確認用です。

```powershell
.\.venv\Scripts\python.exe -m training.reconstruct_dataset --split-strategy series-holdout
```

元データの分布、頻度しきい値、系列splitの詳細は[DeepAA初期データ監査](../docs/DATA_AUDIT.md)にあります。監査JSONと文書を再生成するには次を実行します。

```powershell
.\.venv\Scripts\python.exe -m training.audit_bootstrap
```

この500作品は本番教師データではありません。全件の質的確認、題材の偏り、採用範囲は[DeepAA 500作品の質的レビュー](../docs/BOOTSTRAP_QUALITY_REVIEW.md)に記録しています。レビュー用コンタクトシートと採点CSVは次で生成できます。

```powershell
.\.venv\Scripts\python.exe -m training.prepare_quality_review
```

元CSVには完全に同一の行が291,634件あります。再構成時に重複を除去し、文字の開始座標が直前の字形幅と一致することを全行で検証します。生成物の `manifest.json` には分割、寸法、配置データのSHA-256を記録します。

この段階で得られる画像は「正解AAから作った構造代理画像」であり、実写・イラスト風の本当の元絵ではありません。weak-v1/v2で不採用になったのは、輪郭劣化・密度塗りを1枚へ合成した候補です。P6では`E`を線ch0だけに使い、濃度ch1をAAラスタから作るC2まで実測しましたが、実画像へ誤塗りを起こしたため終了しました。これは線・塗り分離の逆生成経路全体の棄却ではありません。次はaccepted-v2から線、反復塗り面、濃淡、文字位置を別出力にし、実画像の色領域・線画閉領域と同じチャンネル契約へ合わせます。実画像A/Bを先に行い、人手修正版AAは正解対応が不可欠な例だけに使います。

`training.examples` は、従来モデルと同じ64×64文脈窓を正解座標から抽出します。正解座標は411文字の分類ラベルと文字開始ラベル1を持ち、既知の開始点から2px以上離れた位置を開始ラベル0として生成できます。これにより、次のモデルでは文字種分類だけでなく、DPで探索すべき開始位置も学習できます。

### AA本文からの線・塗り補助ラベル

完成AAでは`.`、`:`、`;`など同一の濃淡文字を横方向へ連続させ、面の塗りを表現します。`training.aa_fill_layers`はAA本文を直接解析し、罫線・格子・斜線を除外したうえで、反復文字を線レイヤーと塗りレイヤーへ分離します。ぼかしやランダムな線削除は行いません。

```powershell
.\.venv\Scripts\python.exe -m training.analyze_aa_fill_structure
.\.venv\Scripts\python.exe -m training.analyze_aa_fill_surfaces
.\.venv\Scripts\python.exe -m training.generate_text_fill_preview `
  --output .tmp\training-runs\text-fill-preview
```

`accepted-v1`全100作品では78作品から反復塗りを検出し、22作品は検出ゼロでした。後者には複数文字を混ぜた網掛けが含まれるため、単一文字連続だけで全ての塗りを説明しません。この分離画像を実在元画像の代用品にはせず、今後の元画像・修正AA対応対に付ける出力側の補助ラベルとして使います。

P6R-1では隣接行で短い側の25%以上が水平に重なるrunを`FillSurface`へ連結します。全100作品の2,535 runから656面を得て、2,347 run（92.584%）が複数行面へ入りました。複数行面は468、孤立した1行面は188、反復runあり78作品中72作品が複数行面を持ちます。全件で面ラベルマスクの画素数と面積集計が一致したためP6R-1は合格です。結果は[P6R-1資料](../docs/AA_FILL_SURFACES_P6R1.md)、作品別のローカル正本は`.tmp/training-runs/aa-fill-surface-p6r1/report.json`にあります。

P6R-2の面対応・契約テストは次で実行します。

```powershell
.\.venv\Scripts\python.exe -m training.evaluate_surface_metrics_p6r2
```

完全一致、過剰、欠落、split、mergeの合成ケースを区別し、accepted-v1全100件・656面の自己回帰は全件一致しました。対応IDのない評価は全100件を拒否し、未対応集合へ正解IoUを出さない契約も通過しています。結果は[P6R-2資料](../docs/SURFACE_METRICS_P6R2.md)、ローカル正本は`.tmp/training-runs/surface-metrics-p6r2/report.json`です。

P6R-2Sでは、画面で編集した草案を元画像・正規化設定・実クロップ・抽出線・両AA・両再描画・座標変換・文字差分と一緒に`datasets/incoming/correction-pairs/v1/`へ保存します。ベース記録は不変で、P6R-3/4の候補面、面対応、面指標はハッシュ付き拡張へ追記します。保存形式とAPIは[P6R-2S資料](../docs/CORRECTION_PAIRS_P6R2S.md)を参照してください。

P6R-3では色保持読込を追加し、原画はLab量子化連結領域、線画は検出専用bufferだけをClosingした閉領域から、共通`SurfaceProposal`を作ります。P1参照30件は未対応集合なので正解IoUを出さず、候補数・面積・被覆、外部背景除外、粒度間split / mergeと診断画像だけを保存します。

```powershell
.\.venv\Scripts\python.exe -m training.evaluate_surface_proposals_p6r3
```

固定実行では全30件に候補があり、色層の候補数中央値は48/73、線画層は34/39でした。外部背景率は中央値61.7665%で、候補から除外したまま診断値へ残しました。主要な顔・髪・服・背景色面と線画閉領域が候補集合へ入るため条件付き合格です。写真の断片化、淡色面の融合、開いた輪郭の欠落は、まず逆生成学習後のP6R-G1比較へ引き継ぎ、正解対応が必要な場合だけP6R-4の任意修正対で監査します。詳細は[P6R-3資料](../docs/SURFACE_PROPOSALS_P6R3.md)、ローカル正本は`.tmp/training-runs/surface-proposals-p6r3/report.json`です。

P6R-4基盤は、各修正対応対について候補面、修正AA反復塗り面との重なり、規則ベースv1の判断とP6R-2面指標を生成します。既定は読取集計だけで、記録へ保存する場合だけ`--write-extensions`を指定します。

```powershell
.\.venv\Scripts\python.exe -m training.evaluate_surface_decisions_p6r4
.\.venv\Scripts\python.exe -m training.evaluate_surface_decisions_p6r4 --write-extensions
```

2026-08-28時点の修正対応対は0件で、厳密な対応付き面品質の`quality_gate_passed`は`null`です。この値は保存基盤の未評価状態を示すだけで、accepted-v2からの線・塗り分離逆生成やDeepAA型再学習を妨げません。正解対応なしでは実画像の失敗原因を分離できない場合だけ任意収集します。基盤の内容は[P6R-4資料](../docs/SURFACE_DECISIONS_P6R4.md)に固定します。

## 学習環境

Windows版PyTorchの対応範囲に合わせて、アプリ実行環境とは別のPython 3.12仮想環境を使います。このPCではCUDA 13.0版を使用します。

```powershell
py -3.12 -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install -r requirements-training.txt
```

`.venv-training` は学習・ONNX書き出し専用です。配布アプリは引き続き軽量なONNX Runtimeのみで動作します。

## Random Forest比較実験

DeepAAと同じ64×64文脈、411文字、作品単位分割を使う比較用Random Forestを学習できます。

```powershell
.\.venv-training\Scripts\python.exe -m training.train_random_forest
```

モデルと検証レポートは `.tmp/training-runs/random-forest/` に保存します。Random Forestは通常アプリの依存関係や配布モデルには含めません。実画像でCNNを下回ったため置換はせず、元画像・元線画と人手完成AAの対応データが集まった場合の再実験用として保持します。条件と結果は[Random Forest実験資料](../docs/RANDOM_FOREST_EXPERIMENT.md)を参照してください。

既存ONNXからの重み移植が数値的に一致することを確認します。

```powershell
.\.venv-training\Scripts\python.exe -m training.model
```

文字分類と文字開始位置を同時に追加学習します。最初は `--max-train-samples` を指定したスモークテストを推奨します。

```powershell
.\.venv-training\Scripts\python.exe -m training.train_multitask `
  --epochs 1 `
  --max-train-samples 4096 `
  --max-validation-samples 1024
```

チェックポイント、学習履歴、2ヘッドONNXは `.tmp/training-runs/multitask` に保存されます。

開始ヘッドは、未使用作品の全X座標を走査して評価します。候補率と±1pxでのprecision/recallを確認し、DPの候補削減へ使えるしきい値を決めます。

```powershell
.\.venv-training\Scripts\python.exe -m training.evaluate_start_head `
  --checkpoint .tmp\training-runs\multitask\best.pt
```

64×64分類CNNを全Xへ適用すると開始判定自体が遅いため、行全体を一度で処理する軽量な全畳み込みモデルも学習します。

```powershell
.\.venv-training\Scripts\python.exe -m training.train_start_locator
```

入力は `1×64×任意幅`、出力は入力幅と同じ開始logit列です。これで候補位置を先に絞り、411文字分類CNNは候補だけへ適用できます。

全validation作品を走査して、しきい値ごとの候補率と真の開始座標のrecallを測定します。

```powershell
.\.venv-training\Scripts\python.exe -m training.evaluate_start_locator
```

評価後に採用するモデルを実行用パスへ配置し、ハッシュと学習条件を `THIRD_PARTY.md` に記録します。

```powershell
Copy-Item -LiteralPath .\.tmp\training-runs\start-locator\model.onnx `
  -Destination .\models\deepaa-start-locator.onnx
Get-FileHash -Algorithm SHA256 .\models\deepaa-start-locator.onnx
```
