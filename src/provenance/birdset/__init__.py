"""BirdSet corpus tooling (download + eval-set construction).

Standalone, like the Watkins provenance tools in the parent package: these modules
do **not** import from the benchmark package (``src/``). They turn the BirdSet
HuggingFace dataset into the same ``clips/ + gt.tsv + metadata.csv`` eval-set layout
the benchmark already scores. See ``README.md`` here.
"""
