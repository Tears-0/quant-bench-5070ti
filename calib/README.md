Calibration corpora and evaluation text files.

Place the WikiText-2 raw test split here as `wikitext-2-raw-test.txt` — the path is
configured in `configs/eval/wikitext2.yaml`, and its bytes are hashed into every quality
record. Calibration sets used for AWQ/GPTQ/imatrix/EXL3 must be disjoint from it (§6.7);
`calib/*.dat` importance matrices are git-ignored.
