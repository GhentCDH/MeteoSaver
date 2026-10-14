""" Builds the pipeline subprocess command and checks what the dashboard needs before a run. """

import os
import re
import shutil
import signal
import subprocess
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPT = REPO_ROOT / "src" / "main.py"
ANNOTATE_HEADER_SCRIPT = REPO_ROOT / "src" / "annotate_taulu_header.py"
DEFAULT_CONFIG = REPO_ROOT / "configuration.ini"

# Printed by main.py around each module (announce_stage) and for each sheet. Matched here by format
# rather than imported, since importing main.py would load the models into the dashboard.
STAGE_MARKER = re.compile(r"^\[stage:(\w+)\] (start|done)$")
PROGRESS_MARKER = re.compile(r"^\[progress\] station=(\S+) \((\d+)/(\d+)\) sheet=(\d+)/(\d+) file=(.+)$")
REVIEW_MARKER = re.compile(r"^\[review\] page=(.+)$")  # printed by main.py for each review page it wrote

ERROR_PREFIX = "[ ERROR ]"
ERROR_CONTINUATION = "[ ERROR ] |"  # a further line of the same error (its traceback)
WARNING_PREFIX = "[ WARN ]"


def build_pipeline_command(stages: list[str]) -> list[str]:
    """ One command for all given stages, so they share a process and the OCR model is loaded once. """
    return [sys.executable, "-u", str(MAIN_SCRIPT), "--stages", *stages]


def build_annotate_header_command(image: Path, header: Path) -> list[str]:
    """ Annotation of the header for the taulu cell detection engine. It opens its own (OpenCV) window, in a separate
        process: taulu quits the program when the annotation is cancelled. """
    return [sys.executable, "-u", str(ANNOTATE_HEADER_SCRIPT), str(image), str(header)]


def display_available() -> bool:
    """ Whether a window can be opened (for the annotation of the header). On Linux this needs a desktop session. """
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def parse_stage_marker(line: str) -> tuple[str, str] | None:
    """ (stage key, "start" | "done") for a stage marker line, None for any other line. """
    match = STAGE_MARKER.match(line.strip())
    return (match.group(1), match.group(2)) if match else None


def parse_review_marker(line: str) -> str | None:
    """ The path of the review page for a review marker line, None for any other line. """
    match = REVIEW_MARKER.match(line.strip())
    return match.group(1) if match else None


def open_in_browser(page: str) -> bool:
    """ Opens a local HTML file in the default browser. False if no browser could be started. """
    try:
        return webbrowser.open(Path(page).resolve().as_uri())
    except webbrowser.Error:
        return False


def parse_progress(line: str) -> str | None:
    """ The text for the progress line of the dashboard, None for any line that is not a progress marker. """
    match = PROGRESS_MARKER.match(line.strip())
    if not match:
        return None
    station, station_number, stations, sheet, sheets, _file = match.groups()
    return f"STATION {station} ({station_number}/{stations})  ·  SHEET {sheet}/{sheets}"


def classify_line(line: str) -> str | None:
    """ Picks a log color for one line of pipeline output. "error" is a line of an error the pipeline
        reported itself (main.py), "fail" only looks like trouble. """
    if line.startswith(ERROR_PREFIX):
        return "error"
    if line.startswith(WARNING_PREFIX):
        return "warn"
    lowered = line.lower()
    if "| error |" in lowered or "traceback" in lowered or "failed" in lowered:
        return "fail"
    if "| warning |" in lowered or "warning:" in lowered:
        return "warn"
    if "finished processing" in lowered or "validation complete" in lowered:
        return "ok"
    return None


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def check_gpu() -> tuple[bool, str]:
    """ Whether torch sees a GPU. Asked in a separate process, so torch is not loaded into the dashboard. """
    code = "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as e:
        return False, f"{type(e).__name__}: {e}"
    name = result.stdout.strip()
    return bool(name), name or "no GPU found, GLM-OCR will be slow"


def describe_exit(returncode: int) -> str:
    """ Why the pipeline process ended, for a non-zero exit. A killed process cannot log a reason itself. """
    if returncode >= 0:
        return f"The pipeline process stopped with exit code {returncode} before finishing."

    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        name = f"signal {-returncode}"

    reason = f"The pipeline process was killed by {name} before finishing, so it could not log why."
    if -returncode == signal.SIGKILL:
        reason += (" This is almost always the system running out of memory:"
                   " check with `journalctl -k | grep -i 'out of memory'`.")
    return reason


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"
