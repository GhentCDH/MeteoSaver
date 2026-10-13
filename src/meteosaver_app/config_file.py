""" Reads the settings shown in the dashboard from the configuration file and writes changed ones back.

Values are written line by line rather than through configparser, which would drop every comment in the file. """

import configparser
import os
import re
from pathlib import Path

SECTION = re.compile(r"^\s*\[(.+)\]\s*$")

# The directories MeteoSaver writes to. In the dashboard they are set together, as one results directory.
OUTPUT_DIRECTORY_KEYS = [
    "pre_QA_QC_transcribed_hydroclimate_data_dir",
    "post_QA_QC_transcribed_hydroclimate_data_dir",
    "final_refined_daily_hydroclimate_data_dir",
    "transient_transcription_output_dir",
    "validation_dir",
    "result_maps_dir",
    "glm_output_dir",
]

# Folder name of an output directory that is not in the configuration file yet
DEFAULT_FOLDERS = {"glm_output_dir": "08_GLM_OCR_output/"}


def read_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.optionxform = str  # keep the capitals in the keys
    config.read(path, encoding="utf-8")
    return config


def results_directory(config: configparser.ConfigParser) -> str:
    """ The directory that holds all the output directories. """
    return os.path.commonpath([
        os.path.normpath(config["Directories"][key].strip()) for key in OUTPUT_DIRECTORY_KEYS if config.has_option("Directories", key)
    ])


def output_directories(config: configparser.ConfigParser, results: str) -> dict[tuple[str, str], str]:
    """ The output directories moved into another results directory, each keeping its own folder name. """
    moved = {}
    for key in OUTPUT_DIRECTORY_KEYS:
        current = config.get("Directories", key, fallback=DEFAULT_FOLDERS.get(key, "")).strip()
        folder = os.path.basename(os.path.normpath(current))
        moved[("Directories", key)] = os.path.join(results, folder) + ("/" if current.endswith("/") else "")
    return moved


def write_values(path: Path, values: dict[tuple[str, str], str]) -> list[tuple[str, str, str]]:
    """ Sets the (section, key) values that differ from the file, leaving every other line as it is.
        Returns (key, old value, new value) for each changed setting. """
    config = read_config(path)
    pending = {
        (section, key): value for (section, key), value in values.items()
        if not config.has_option(section, key) or config[section][key].strip() != value
    }
    if not pending:
        return []

    changes = [(key, config.get(section, key, fallback="").strip(), value) for (section, key), value in pending.items()]
    lines = path.read_text(encoding="utf-8").split("\n")
    section = None
    last_line_of_section: dict[str, int] = {}
    for number, line in enumerate(lines):
        header = SECTION.match(line)
        if header:
            section = header.group(1)
        elif section is not None and line.strip():
            last_line_of_section[section] = number
            match = re.match(r"^(\s*([^#;=\s][^=]*?)\s*=)", line)
            if match and (section, match.group(2)) in pending:
                lines[number] = f"{match.group(2)} = {pending.pop((section, match.group(2)))}"
        if header:
            last_line_of_section.setdefault(section, number)

    # Settings that are not in the file yet go at the end of their section, last ones first so the line numbers stay valid
    for (section, key), value in sorted(pending.items(), key=lambda item: -last_line_of_section.get(item[0][0], -1)):
        if section not in last_line_of_section:
            raise KeyError(f"Section [{section}] is missing from {path}")
        lines.insert(last_line_of_section[section] + 1, f"{key} = {value}")

    path.write_text("\n".join(lines), encoding="utf-8")
    return changes
