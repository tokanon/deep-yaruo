# DeepAA line/surface v0

## Approved comparison

The frozen accepted-v2 0500 `b-periodic` reverse decomposition is the only training input for this comparison. `L` receives the separated line mask. `LS` uses the same line encoder plus a separate surface encoder and late feature fusion. The surface encoder receives three channels: binary surface membership, value-independent surface boundary, and normalized per-surface tone. Integer surface labels, motif names, motif frequency, category, and the target character are not inputs.

Both variants use the same train/validation/test works, the same train-derived vocabulary, the same character/start/line-or-fill targets, and the same width-constrained evaluation decoder. Every character observed in train receives its own class. There is no frequency cutoff or unknown class; validation/test characters absent from train are counted as incorrect in overall accuracy and reported separately.

The auxiliary confidence strata do not erase training examples. `supported fill` means at least one owning B-run motif meets the frozen cross-work support flag. All other B-owned glyphs are `low-support fill`; non-owned glyphs are `line`. These labels are used for reporting, while the line/fill role is also an auxiliary prediction head.

The line encoder starts from the historical DeepAA C1 convolution weights. The new character, start, and role heads and the LS surface encoder are trained on accepted-v2. P4-P6 model parameters, augmentations, tone definitions, metrics, and failure diagnoses are not reused.

The real-image adapter uses the P6R-4L grayscale line result as the normalized line tensor and converts P6R-3/4 integer proposal IDs and tone attributes through the same membership/boundary/tone function. Thus the ordinal ID itself still cannot leak into inference. This establishes tensor compatibility only; whether the synthetic model transfers usefully is evaluated later on unseen real images.

## Validation and stopping

The best checkpoint is selected by validation loss. The synthetic decoder receives probabilities at held-out true character starts and must choose one character per start while matching the target Saitamaar row width exactly. It may consider the top eight classes plus the most probable class for every advance present in the train vocabulary. It never receives the target character identity. Start localization is measured separately with deterministic non-start examples.

Report character accuracy overall and for line, all fill, supported fill, low-support fill, and train-frequency buckets; start precision/recall; role accuracy; exact row widths; whole-render, line-region, and fill-region pixel mismatch; ink omission and excess. Test is evaluated only after the training and decoder specification are fixed on validation.

Stop rather than advance LS if either fill-character accuracy or fill-render mismatch fails to improve, if line character accuracy or line-render mismatch degrades by more than one absolute percentage point, if either model loses exact width, or if LS cannot consume the corresponding real-image contract. The one-point threshold is fixed before opening test. Even a synthetic pass only authorizes the subsequent unseen-real-image comparison with P6R-G1; it does not change the normal generator.

## Commands

```powershell
.\.venv-training\Scripts\python.exe -m training.train_deepaa_surface_v0
.\.venv-training\Scripts\python.exe -m training.evaluate_deepaa_surface_v0 --split validation
.\.venv-training\Scripts\python.exe -m training.evaluate_deepaa_surface_v0 --split test
.\.venv-training\Scripts\python.exe -m training.export_deepaa_surface_v0
.\.venv\Scripts\python.exe -m training.evaluate_deepaa_surface_real_v0
```

Outputs are local experiment artifacts under `.tmp/training-runs/deepaa-surface-v0/`.

## 0500 result

Training used 1,187 train-observed classes, 1,826,165 train character starts, one deterministic non-start for every four starts, three epochs, and the same 394/51/55 work split for L and LS. L has 2,535,942 parameters and LS has 5,070,982; the additional capacity is the explicitly separate surface encoder and widened late-fusion heads, so this experiment establishes the usefulness of the approved LS system, not a capacity-matched ablation of each individual surface channel.

Best validation checkpoints were epoch 3 for both variants. Test was opened once after the validation decoder was fixed.

| test metric | L | LS | LS - L |
| --- | ---: | ---: | ---: |
| all character accuracy | 75.69% | 88.13% | +12.44 points |
| line character accuracy | 90.14% | 94.36% | +4.22 points |
| fill character accuracy | 28.82% | 67.94% | +39.12 points |
| supported-fill accuracy | 29.50% | 69.46% | +39.96 points |
| low-support-fill accuracy | 6.27% | 17.75% | +11.48 points |
| start precision | 87.19% | 89.04% | +1.85 points |
| start recall | 98.72% | 98.32% | -0.40 points |
| line/fill role accuracy | 82.16% | 99.80% | +17.63 points |
| fill-region render mismatch | 6.91% | 5.89% | -1.02 points |
| line-region render mismatch | 3.22% | 1.70% | -1.51 points |
| target-ink omission | 55.24% | 31.12% | -24.13 points |
| exact-width rows | 2,391/2,391 | 2,391/2,391 | equal |

There were 35 test character occurrences absent from train; they were counted as incorrect overall and reported as unseen. Characters with only 1-9 train occurrences remained poor and declined from 7.69% in L to 5.98% in LS even though the separate low-support-fill stratum improved. This is direct evidence that 0500 is a first learning checkpoint rather than corpus saturation.

All predeclared synthetic stop checks passed. The result does not authorize ordinary generation: the width-constrained synthetic decoder deliberately uses the held-out true start count, while the learned start head is measured separately. The next gate must run LS from real P6R line/surface inputs and compare its visible output with P6R-G1 on images not used to construct the reverse corpus.

Final verification passed with 179 tests and 2 optional skips in the normal environment, 6/6 Torch-specific L/LS tests in the training environment, all 500 B fill/support label checks, Python compilation, and `git diff --check`. The training-environment full-suite attempt could not create pytest fixtures under its Windows temporary-directory ACL; the same non-Torch suite passed in the normal environment, and the Torch-specific suite was therefore run separately.

## Provisional real-image integration

The user approved a provisional default without another human A/B gate after the synthetic LS improvement, provided that end-to-end decoding and the fixed mechanical gate passed. The runtime decoder evaluates the learned LS start and character heads at every horizontal pixel. Its exact-width DP charges a start cost at each selected boundary and non-start costs inside the selected glyph advance, so it does not receive the target character count or target starts. Sixteen shift phases share convolutional work across a row; a single 64px window is numerically checked against the original LS graph during export.

Color profiles now send the P6R-4L recovered line channel and P6R-3/4 selected Lab surface membership, ID-boundary, and source-darkness tone channels to `deepaa-surface-v0-provisional-v1`. Line-art profiles remain on the tone-free P6R-G1 path. The LS result is not passed through the old output fill/information/texture stages a second time. Any model/inference error, exact-width failure, or non-empty input row decoded entirely as whitespace falls back for that whole generation to `p6r-cumulative-g1-v1`.

The fixed 15 difficult real-image cases passed: exact widths 15/15, deterministic text/PNG/pipeline 15/15, zero non-empty rows decoded entirely as whitespace, no use of target start count, and zero P6R-G1 fallbacks. The local report is `.tmp/training-runs/deepaa-surface-real-v0/report.json`. The exported runtime graph SHA-256 is `3E4F702A2E106654A4ADF2C4875EB09A65259F58BEC4F44184570D5D661BB304`; its vocabulary SHA-256 is `71548DED84946EC2C6CEB92BEA3BDD3F50185D0A9ABB134D4EA7C50A0DBBD857`, and it records source checkpoint SHA-256 `687DD5167E2286FCB50B21AF171BF88E756731993D9B0BD3F23A2D9411BB7832`.

This is a safety/integration pass, not proof that LS is visually superior on real images. CPU generation on the fixed cases is noticeably slower than the legacy path (roughly 10–20 seconds per generation on the development machine, varying with rows and width). Corpus expansion and later preference/reward learning remain necessary; they must not be described as already completed reinforcement learning.
