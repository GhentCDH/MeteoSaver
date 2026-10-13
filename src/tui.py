#!/usr/bin/env python
"""
MeteoSaver dashboard: set the main options of the configuration file and run the modules from the terminal.

Usage (in the main project directory):
    uv run python src/tui.py
    uv run python src/tui.py --config another_configuration.ini
"""
import argparse
from pathlib import Path

from meteosaver_app.app import MeteoSaverApp
from meteosaver_app.pipeline_runner import DEFAULT_CONFIG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MeteoSaver dashboard")
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help="Configuration file to show, edit and run with")
    args = parser.parse_args()
    MeteoSaverApp(args.config.resolve()).run()
