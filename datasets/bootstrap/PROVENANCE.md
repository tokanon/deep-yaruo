# DeepAA bootstrap dataset

`deepaa-500.csv` is the initial coordinate dataset used to reconstruct the
500 ASCII-art works inherited from DeepAA. It may be placed in this directory
for local bootstrap experiments, but it is intentionally excluded from Git
because the file contains third-party ASCII-art works. It is not the target
dataset size or the final character-set specification for this project.

- Source: https://github.com/OsciiArt/DeepAA
- Revision: `ce5d17a83b6c1b11b045ca04761bf197ab41645d`
- Original path: `data/data_500.csv`
- License: MIT (`LICENSE` in this directory)
- Copyright: Copyright (c) 2017 OsciiArt
- SHA-256: `925F563B8251A6CAA083FE28FB12433530A1B887E4B02FA5D48FD64472A9200C`

The dataset itself, upstream source snapshot, original Keras files, glyph
pickle, IDE files, and sample images are intentionally not vendored. If local
historical inspection is required, retrieve the exact revision from the source
repository and place `data/data_500.csv` at
`datasets/bootstrap/deepaa-500.csv`. Do not add it to Git.
