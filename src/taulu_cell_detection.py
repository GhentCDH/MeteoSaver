#!/usr/bin/env python
# Table and cell detection with taulu (https://github.com/GhentCDH/taulu): the second cell detection engine, next to the one in
# table_and_cell_detection_model.py. Select it with cell_detection_engine = taulu in configuration.ini.
#
# Where the default engine detects the handwritten text in the table, taulu follows the ruled lines of the table, starting
# from an annotated image of the header (see annotate_taulu_header.py). It returns every cell of the table with its row and
# column, so the cells do not have to be grouped into rows and columns afterwards (see transcription.py).
import os
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_taulu_per_config = {}  # The Taulu instance per configuration file: the header template is only loaded once per process


def _get_taulu(taulu_config):
    '''
    Returns the Taulu instance for a taulu configuration file (TOML, see taulu/taulu.toml in this repository).
    The paths in the configuration file (template_path, annotation_path) are relative to the directory of the configuration file.
    '''
    from taulu import Taulu, TauluConfig

    # OpenCV 5 moved AKAZE (taulu's default feature_detector, to find the header) to the contrib module xfeatures2d
    if not hasattr(cv2, 'AKAZE_create') and hasattr(cv2, 'xfeatures2d'):
        cv2.AKAZE_create = cv2.xfeatures2d.AKAZE_create

    taulu_config = os.path.abspath(taulu_config)
    if taulu_config not in _taulu_per_config:
        config = TauluConfig.from_toml(taulu_config)
        config_dir = os.path.dirname(taulu_config)
        for field in ('template_path', 'annotation_path'):
            value = getattr(config, field)
            if isinstance(value, str) and not os.path.isabs(value):
                setattr(config, field, os.path.join(config_dir, value))
        _taulu_per_config[taulu_config] = Taulu.from_config(config)
    return _taulu_per_config[taulu_config]


def taulu_table_and_cell_detection(binarized_image, original_image, station, month_filename, transient_transcription_output_dir, taulu_config,
                                   skip_rows, skip_columns_left, skip_columns_right, no_of_rows, no_of_columns, margin=10):
    '''
    Detects the grid of the table with taulu, and returns the cells of the data part of the table (without headers and row labels).

    Parameters
    --------------
    binarized_image, original_image : numpy.ndarray
        The binarized and the original image of the sheet (see image_preprocessing).

    station, month_filename, transient_transcription_output_dir : str
        Used for saving the detected grid, for inspection.

    taulu_config : str
        Path to the taulu configuration file (TOML).

    skip_rows : int
        Number of rows of the detected grid above the first row with data (rows of the header below the annotated header).

    skip_columns_left, skip_columns_right : int
        Number of columns of the detected grid left and right of the data (row labels, e.g. the pentad no. and the date).

    no_of_rows, no_of_columns : int
        Expected number of rows and columns with data.

    margin : int
        Margin in pixels around the data part of the table, in the clipped images of the table.

    Returns
    --------------
    detected_table_cells : list or None
        As table_and_cell_detection in table_and_cell_detection_model.py, with one more element: the cells per row.
        - [0]: contours. One 4-point contour per cell, in the coordinates of the clipped table images below.
        - [1]: clipped original image of the table (only its size is used).
        - [2]: clipped binarized image of the table (RGB).
        - [3]: clipped original image of the table.
        - [4]: original image with all the detected cells drawn on it.
        - [5]: cell_grid. The contours per row (cell_grid[row][column]), in the same order as the rows and columns of the sheet.
        None if no table with data was detected.
    '''

    grid = _get_taulu(taulu_config).segment_table(original_image)
    logger.info(f"taulu detected a grid of {grid.rows} rows and {grid.cols} columns")

    # The data part of the grid
    first_row = skip_rows
    last_row = min(skip_rows + no_of_rows, grid.rows) - 1
    first_column = skip_columns_left
    last_column = grid.cols - skip_columns_right - 1
    if last_row < first_row or last_column < first_column:
        print(f"[ WARN ] taulu detected no table with data in {month_filename} ({grid.rows} rows, {grid.cols} columns).", flush=True)
        return None
    if last_row - first_row + 1 < no_of_rows:
        print(f"[ WARN ] taulu detected {last_row - first_row + 1} rows with data in {month_filename}, {no_of_rows} were expected.", flush=True)
    if last_column - first_column + 1 != no_of_columns:
        print(f"[ WARN ] taulu detected {last_column - first_column + 1} columns with data in {month_filename}, {no_of_columns} were expected "
              f"(check taulu_skip_columns_left and taulu_skip_columns_right).", flush=True)
        last_column = min(last_column, first_column + no_of_columns - 1)  # The transcription places at most no_of_columns columns

    # Clip the data part of the table from the image
    corners = np.array(grid.region((first_row, first_column), (last_row, last_column)))
    image_height, image_width = original_image.shape[:2]
    x0, y0 = np.maximum(corners.min(axis=0) - margin, 0).astype(int)
    x1, y1 = np.minimum(corners.max(axis=0) + margin, (image_width, image_height)).astype(int)
    table_original_image = original_image[y0:y1, x0:x1].copy()
    table_binarized_image = cv2.cvtColor(binarized_image[y0:y1, x0:x1], cv2.COLOR_GRAY2RGB)

    # One contour per cell, in the coordinates of the clipped table
    cell_grid = []
    for row in range(first_row, last_row + 1):
        cell_grid.append([np.array(grid.cell_polygon((row, column)), dtype=np.int32).reshape(-1, 1, 2) - (x0, y0)
                          for column in range(first_column, last_column + 1)])
    contours = [contour for row in cell_grid for contour in row]

    # Save the detected grid for inspection (and for reuse: SegmentedTable.from_saved)
    save_dir = os.path.join(transient_transcription_output_dir, station)
    os.makedirs(save_dir, exist_ok=True)
    full_detected_table_with_labels = grid.highlight_all_cells(original_image, thickness=3)
    cv2.imwrite(os.path.join(save_dir, f'taulu_grid_{month_filename}.jpg'), full_detected_table_with_labels)
    grid.save(os.path.join(save_dir, f'taulu_grid_{month_filename}.json'))

    return [contours, table_original_image, table_binarized_image, table_original_image, full_detected_table_with_labels, cell_grid]
