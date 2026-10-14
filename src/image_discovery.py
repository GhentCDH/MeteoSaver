#!/usr/bin/env python
import os
import re

# File types that are read as images of data sheets
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

# Expected name of an image: "STN_YYYYMM_SF" (standard format) or "STN_YYYYMM_HD" (hand drawn), where STN is the station number
SHEET_NAME = re.compile(r'^([^_]+)_(\d{6})_(SF|HD)', re.IGNORECASE)


def find_sheets(data_directory):
    '''
    Searches a directory and all the folders within it for images of data sheets.

    The folders do not need a particular name or structure. The station of a sheet is taken from the name of the image
    (STN_YYYYMM_SF). An image with another name is still transcribed, under the name of the folder it is in.

    Parameters
    --------------
    data_directory : str
        Directory to search through.

    Returns
    --------------
    sheets_by_station : dict
        For each station, the paths of its images of standard format sheets, in alphabetical order.
    messages : list
        What the user should know about the search: folders without images, and images that are left out or that have an unexpected name.
    '''
    sheets_by_station = {}
    messages = []

    if not os.path.isdir(data_directory):
        return sheets_by_station, [f"Image directory not found: {os.path.abspath(data_directory)}"]

    folders_with_images = set()  # Folders with an image in them or in one of the folders within them
    all_folders = []
    names_per_station = set()
    for folder, subfolders, files in os.walk(data_directory):
        subfolders.sort()
        all_folders.append(folder)
        for file in sorted(files):
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            parent = folder
            while parent not in folders_with_images:
                folders_with_images.add(parent)
                if os.path.samefile(parent, data_directory):
                    break
                parent = os.path.dirname(parent)

            path = os.path.join(folder, file)
            match = SHEET_NAME.match(file)
            if match and match.group(3).upper() == 'HD':
                messages.append(f"Left out, hand drawn (HD) sheets are not transcribed: {path}")
                continue
            if match:
                station = match.group(1)
            else:
                station = os.path.basename(os.path.abspath(folder))
                messages.append(f"Name does not follow STN_YYYYMM_SF, so the month is unknown (transcribed as a month of 31 days under '{station}', and left out of the data formatting): {path}")
            if (station, file) in names_per_station:
                messages.append(f"Left out, another image of station {station} has the same name: {path}")
                continue
            names_per_station.add((station, file))
            sheets_by_station.setdefault(station, []).append(path)

    # Mention the folders without images. Of a folder without images, the folders within it are not mentioned separately.
    for folder in all_folders:
        if folder not in folders_with_images and (os.path.samefile(folder, data_directory) or os.path.dirname(folder) in folders_with_images):
            messages.append(f"No images found in: {folder}")

    return dict(sorted(sheets_by_station.items())), messages
