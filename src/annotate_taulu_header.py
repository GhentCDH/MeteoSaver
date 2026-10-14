#!/usr/bin/env python
# Annotates the header of the table, for the taulu cell detection engine (cell_detection_engine = taulu in configuration.ini).
# This only has to be done once, on one image of a sheet: the header is then found on all the other sheets of the same format.
# The dashboard (src/tui.py) runs this script with its ANNOTATE HEADER button.
#
#   uv run src/annotate_taulu_header.py <image of a sheet> [header image to write, default: taulu/header.png]
#
# A window opens:
#  1. Click the four corners of the header of the table, to crop it.
#  2. Click the ruled lines of the header, two clicks (the two ends) per line, in any order: the line above the header,
#     the line below it (the line above the first row of data) and every vertical line that continues into the table.
#     Right-click removes the last line; 'n' saves, 'q' quits without saving.
# The header image and its annotation (.json, with the same name) are saved; set template_path in taulu/taulu.toml to the header image.
#
# Exit codes: 0 saved, 1 error, 2 cancelled or not enough lines annotated (the existing header is then kept as it was).
import os
import shutil
import sys
import tempfile

from taulu import Taulu, TableTemplate

if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        sys.exit(f'Usage: {sys.argv[0]} <image of a sheet> [header image to write, default: taulu/header.png]')
    image_path = sys.argv[1]
    header_path = sys.argv[2] if len(sys.argv) == 3 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'taulu', 'header.png')
    header_path = os.path.splitext(os.path.abspath(header_path))[0] + '.png'  # taulu reads the header as .png, with the annotation next to it as .json
    if not os.path.isfile(image_path):
        print(f'Image not found: {image_path}', flush=True)
        sys.exit(1)
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    # Annotate in a temporary directory: taulu writes the cropped header before the lines are annotated, and quits the
    # program when 'q' is pressed. The existing header and its annotation are only replaced by a complete new pair.
    with tempfile.TemporaryDirectory() as temporary_dir:
        new_header = os.path.join(temporary_dir, 'header.png')
        new_annotation = os.path.join(temporary_dir, 'header.json')
        print(f'Annotate the header of {image_path} in the window that opened.', flush=True)
        try:
            Taulu.annotate(image_path, new_header, backend='gui')
        except SystemExit:
            pass  # 'q' pressed in the window
        except AssertionError as e:
            print(f'Not saved: {e}', flush=True)  # e.g. not exactly four corners clicked to crop the header
            sys.exit(2)
        if not os.path.isfile(new_annotation):
            print('Cancelled: the header was not saved.', flush=True)
            sys.exit(2)

        template = TableTemplate.from_saved(new_annotation)
        horizontal_lines = len(template._h_rules)
        vertical_lines = len(template._v_rules)
        if horizontal_lines < 2 or vertical_lines < 2:
            print(f'Not saved: {horizontal_lines} horizontal and {vertical_lines} vertical lines were annotated. At least the lines above '
                  f'and below the header (2 horizontal) and the column borders (2 or more vertical) are needed.', flush=True)
            sys.exit(2)

        shutil.move(new_header, header_path)
        shutil.move(new_annotation, os.path.splitext(header_path)[0] + '.json')
    print(f'Header saved to {header_path} ({template.cols} columns, annotation: {os.path.splitext(header_path)[0]}.json)', flush=True)
