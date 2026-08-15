# Third-party components

## DeepAA

- Source: https://github.com/OsciiArt/DeepAA
- Revision: `ce5d17a83b6c1b11b045ca04761bf197ab41645d`
- License: MIT
- Copyright: Copyright (c) 2017 OsciiArt
- Upstream Keras weight SHA-256 (not vendored): `58DD557DE5956C71C07098F26075A3C4E299319E30ABE97D6861ADDE72BFEA64`
- Converted ONNX SHA-256: `11E41523C6208B87CFDDDB5FED97F48324009A396ED72AFF9E155B1DB13A169F`
- Derived start-locator ONNX SHA-256: `DC72BFE987A649D71574F988C3735DCB9239A1211BD5F2BC528ABD4BB1B4F27E`
- Project-trained accepted-v2 LS ONNX SHA-256: `3E4F702A2E106654A4ADF2C4875EB09A65259F58BEC4F44184570D5D661BB304`
- Character map SHA-256: `35D39653CCA5E3D5E5933319BA58CFE398241342EC2EF5FE60A12E2E354FE3DA`
- Bootstrap dataset SHA-256: `925F563B8251A6CAA083FE28FB12433530A1B887E4B02FA5D48FD64472A9200C`

`models/deepaa-light.onnx` is a format conversion of the pretrained light model.
`models/deepaa-charset.csv` maps its 411 outputs to characters.
`models/deepaa-start-locator.onnx` is trained by this project from the bootstrap
coordinate data that may be placed locally at
`datasets/bootstrap/deepaa-500.csv` using the pipeline under `training/`. The
dataset contains third-party ASCII-art works and is not tracked or distributed
with this repository. Dataset provenance is recorded in
`datasets/bootstrap/PROVENANCE.md`; the MIT notice is retained as
`datasets/bootstrap/LICENSE`.

`models/deepaa-surface-v0-ls.onnx` is trained by this project from the local
accepted-v2 0500 AA review corpus and initialized in part from the converted
DeepAA convolution weights. The accepted-v2 source texts are not vendored.
Review source rights and derived-model redistribution separately before a
public binary release; local integration does not itself grant redistribution
rights to the source AA works.

The bootstrap dataset, upstream source snapshot, Keras files, glyph dictionary,
IDE metadata, and sample images are not vendored. Retrieve the fixed revision
above only for local historical inspection. Do not add the bootstrap dataset to
Git. The retained MIT notice must be included in redistributions containing the
converted or derived model assets.

## Saitamaar

- Source: https://github.com/keage/Saitamaar
- Author: YAMASINA Keage
- License: Public domain
- Bundled path: `assets/fonts/Saitamaar.ttf`
- Bundled TTF SHA-256: `64FED56DCD5A1C64B5E35C92E06B422B71821205E23EFD14C8B1772A43A9D7C5`

Saitamaar is bundled as the canonical runtime and browser-preview font. Its
16px advances match all 411 characters used by the imported DeepAA model, so
the converter no longer depends on an installed Microsoft font.

## Package dependencies

Python and npm dependencies are declared in `requirements*.txt` and
`frontend/package-lock.json`; their source packages are not vendored in this
repository. A PyInstaller distribution does contain Python runtime dependencies.
Before publishing such a binary, collect and bundle the applicable license and
notice files as described in `docs/RELEASE_CHECKLIST.md`.
