from __future__ import annotations


# Repeated punctuation is a conventional AA tone field, not accidental text
# duplication. Keep this set shared by decoding and output-side annotation.
TONE_FILL_CHARACTERS = frozenset(".,:;．，：；・･·'\"`｀´")
DENSE_FILL_CHARACTERS = frozenset("█▓▒░■◆●◎#@")
FILL_REPEAT_CHARACTERS = TONE_FILL_CHARACTERS | DENSE_FILL_CHARACTERS

# These glyphs commonly form rules, borders, grids, or hatching. Repetition
# alone must not turn them into a filled surface.
OUTLINE_RUN_CHARACTERS = frozenset(
    "-_=＝─━═＿￣―—ｰ┼┿╂╋/\\|｜()（）[]［］{}｛｝"
)
