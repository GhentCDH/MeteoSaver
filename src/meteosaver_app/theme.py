""" Palette of the dashboard: a green base with a paler accent, the darkest derived tint as the dominant surface. """

from textual.theme import Theme


def scale(hex_color: str, factor: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (min(255, max(0, round(c * factor))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


BASE = "#41DC8E"
ACCENT = "#A6F0CB"
DARK = scale(BASE, 0.08)
PANEL_BG = scale(BASE, 0.15)
BORDER = scale(BASE, 0.60)
RUNNING = "#E8C97A"
DONE = "#E6FFF2"  # not the usual green, which would not stand out against the green base
SKIPPED = scale(BASE, 0.40)
FAILED = "#E88A8A"
ABORTED = "#E0A96D"
WARNING = "#FFFD01"
DIM = scale(BASE, 0.45)

LEFT_COLUMN_WIDTH = 80  # wide enough for the two-line MeteoSaver logo next to the weather art inside its frame

LOG_STYLES = {
    "header": f"bold {ACCENT}",
    "ok": f"bold {DONE}",
    "fail": f"bold {FAILED}",
    "error": f"bold {FAILED}",
    "warn": f"bold {WARNING}",
    "dim": DIM,
}

METEO_THEME = Theme(
    name="meteo",
    primary=ACCENT,
    secondary=BASE,
    accent=ACCENT,
    foreground=BASE,
    background=DARK,
    surface=DARK,
    panel=PANEL_BG,
    success=DONE,
    warning=WARNING,
    error=FAILED,
    boost=BORDER,
    dark=True,
)
