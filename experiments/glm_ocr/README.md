# GLM-OCR vs Tesseract on MeteoSaver cell crops

Tests whether fine-tuning GLM-OCR (0.9B VLM) on the labelled cell crops in
`~/Documents/Visual Code Data/tesseract` beats MeteoSaver's Tesseract setup.

Everything runs with the BelHisFirm GLM venv (transformers>=5 is needed for GLM-OCR):

```
PY="$HOME/Documents/Visual Code Repos/BelHisFirm---HisTableFinder/.venv-glm/bin/python"
cd experiments/glm_ocr
```

Do not pass `-I` to these scripts: it strips the script directory from `sys.path`, so `import common` fails.

## 1. Split (seeded, stems only; writes `split.json`)
```
$PY make_split.py
```
32,107 train / 714 val / 2,854 test. Note that 99.2% of test labels also occur in train
(labels are short numerics), so there is no label-level held-out. Sheet-level leakage
cannot be checked because stems carry no source-sheet id.

## 2. Train (LoRA, one epoch over the first N train cells)
```
$PY train.py --n-train 500   --name n500
$PY train.py --n-train 2000  --name n2000
$PY train.py --n-train 8000  --name n8000
$PY train.py --n-train 0     --name full      # all train cells
```
Writes `runs/<name>/merged` and prints a quick val CER.

## 3. Evaluate on the test split
```
$PY evaluate.py --glm zeroshot=zai-org/GLM-OCR --glm n2000=runs/n2000/merged ...
```
Tesseract systems are scored with all three models in `MeteoSaver/OCR_HTR_models`,
each with MeteoSaver's digit whitelist (`digits`) and with `,-=` added (`punct`).
Outputs `results.csv` (per cell) and `summary.json`; the console shows CER, exact
match, empty rate, seconds per cell, and the worst errors per system.
Use `--limit N` for a quick check and `--no-tesseract` to skip the baseline.

## Caveats
- `cobecore-V9_405` (the model `configuration.ini` / `transcription.py` use) returns empty
  or wrong output on most of these 128-px-high crops, while `cobecore-V6` reads most of them.
  Check which model the crops were meant for before reading the baseline as Tesseract's best.
- GLM-OCR zero-shot returns `.` where labels use `,`; fine-tuning is expected to fix this.
- A previous fine-tune leaked `<think></think>` into outputs; `evaluate.py` strips it and
  reports how many raw outputs had it.

## 4. Whole-sheet comparison (real pipeline, same point: right before QA/QC)
```
.venv-easyocr/bin/python compare_sheet.py --station 508 --file 508_198802_SF.JPG                       # baseline + GLM
.venv-easyocr/bin/python compare_sheet.py --station 508 --file 508_198802_SF.JPG --skip-baseline --resize-h 128 --label glm_h128
.venv-easyocr/bin/python sheet_report.py 508_198802_SF.JPG glm_h128
```
Runs MeteoSaver's own detection and `transcription()` twice, with `pytesseract.image_to_string`
monkeypatched to a GLM worker (`glm_worker.py`, runs in `.venv-glm`) for the second run. `src/` is
not modified. Outputs go to `sheet_runs/<file>/{baseline,glm*}/pre_QA_QC/`.
Sheet 508, 123 manually transcribed temperature cells (existing pipeline output: 75.6%):

| GLM input | correct |
|---|---|
| binarized ROIs, as fed to Tesseract (`--roi-source binarized`) | 37.4% |
| binarized ROIs, rescaled to 128 px high (`--resize-h 128`) | 74.0% |
| original colour ROIs (`--roi-source original`) | 83.7% |
| original colour ROIs, rescaled to 128 px high | **91.1%** |

The training crops are colour/anti-aliased (median 144 gray levels, not binary) and 128 px high, so
the model should see the same kind of input. MeteoSaver's `transcription()` cuts ROIs from the
binarized, line-removed table (`detected[2]`); `--roi-source original` swaps in the original clip
(`detected[3]`, same coordinates).

## 5. Row grouping fix and speed
- `row_grouping.py` / `--row-grouping columnwise`: replaces the KMeans row grouping in
  `organize_contours_top/_midpoint` (it merges/splits rows on sheets with uneven row pitch, e.g. 601).
  All ten sheets, GLM, original ROIs +128: 93.9% -> **96.4%** (existing pipeline output: 89.8%).
  Sheet 601: 68.2% -> 91.5%. Run `batch_sheets.py --row-grouping columnwise`.
- `--fast-glm`: collect all ROIs, run GLM once in batches (`glm_batch.py`, dedupes identical ROIs,
  pre-fills the `<think></think>` prefix), then replay. GLM stage on a sheet: ~142 s -> ~5 s with
  identical output. `bench_glm.py`: up to 17.9x throughput, same predictions and CER.
- `--hoist-borders`: `transcription()` re-applies borders to the whole sheet for every OCR'd cell
  (~2M openpyxl style assignments). Running that loop once gives identical workbooks and cuts the
  non-OCR time of a pass from ~57 s to ~20 s. Applied via a runtime-transformed copy; `src/` untouched.
  Sheet 508 end to end: 123 s -> 48 s.
