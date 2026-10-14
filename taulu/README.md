# taulu cell detection

Settings for the `taulu` cell detection engine (`cell_detection_engine = taulu` in [configuration.ini](../configuration.ini)), which detects the ruled grid of the table with [taulu](https://github.com/GhentCDH/taulu).

## 1. Annotate the header (once per sheet format)

In the dashboard (`uv run python src/tui.py`): select the `taulu` engine in the CELL DETECTION panel, optionally fill in the sheet to annotate (by default the first sheet in IMAGES), and press **ANNOTATE HEADER**. The header is written to `template_path` of the taulu configuration file. Without the dashboard:

```bash
uv run src/annotate_taulu_header.py <image of a sheet>        # writes taulu/header.png and taulu/header.json
```

The annotation opens a window, so it needs a desktop session. The existing header is only replaced once the new one is saved.

A window opens:
1. Click the four corners of the header of the table to crop it.
2. Click the ruled lines of the header, two clicks (the two ends) per line, in any order:
   - the line above the header;
   - the line below it, i.e. the line above the first row of data;
   - every vertical line that continues into the table, including the outer borders and the columns of the row labels.

Right-click removes the last line, `n` saves, `q` quits without saving. The header is then found on all the other sheets of the same format.

## 2. Settings in configuration.ini (`[TableAndCellDetection]`)

| Key | Meaning |
|---|---|
| `taulu_config` | This [taulu.toml](taulu.toml). The paths in it are relative to the TOML file. |
| `taulu_skip_rows` | Rows of the grid between the annotated header and the first row with data (0 if the annotated header ends at the first row of data). |
| `taulu_skip_columns_left`, `taulu_skip_columns_right` | Columns of the grid left and right of the data, e.g. the pentad no. and the date (2) and the date on the right (1). |

`no_of_rows` and `no_of_columns` are the number of rows and columns of data that are transcribed. A warning is printed when the detected grid has a different number of rows or columns.

## 3. Tune the grid detection (taulu.toml)

The sheets have rows of different heights: the days, and the taller Tot. and Moy. rows. `auto_row_heights` detects the height of each row from the image, within `min_row_height_factor` and `max_row_height_factor` (relative to the height of the annotated header). Check the result in `taulu_grid_<image>.jpg`, written to the transient transcription output directory of the station. The detected grid is also saved there as `taulu_grid_<image>.json`.

To tune the parameters interactively:

```python
from taulu import Taulu, TauluConfig
taulu = Taulu.from_config(TauluConfig.from_toml("taulu/taulu.toml"))   # run from the taulu/ directory, or use absolute paths in the TOML
taulu.segment_table("<image of a sheet>", debug_view=True)
```

All parameters are described in [TauluConfig](https://github.com/GhentCDH/taulu).
