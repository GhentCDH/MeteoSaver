""" Animated banner and boot splash. """

import asyncio

from rich.text import Text
from terminaltexteffects.utils.graphics import Color as EffectColor
from textual import on
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textualeffects.widgets import EffectLabel, effects

from meteosaver_app.constants import AppConstants
from meteosaver_app.theme import ACCENT, BASE, BORDER, DARK, LEFT_COLUMN_WIDTH


def rows_of(text: str) -> list[str]:
    """ The lines of a block of ASCII art, padded to the same width. """
    rows = text.strip("\n").split("\n")
    width = max(len(row) for row in rows)
    return [row.ljust(width) for row in rows]


def side_by_side(left: list[str], right: list[str], gap: int = 2) -> list[str]:
    """ Two blocks next to each other, the shorter one centred vertically. """
    height = max(len(left), len(right))

    def padded(rows: list[str]) -> list[str]:
        blank = " " * len(rows[0])
        above = (height - len(rows)) // 2
        return [blank] * above + rows + [blank] * (height - len(rows) - above)

    return [l + " " * gap + r for l, r in zip(padded(left), padded(right))]


def stacked(top: list[str], bottom: list[str]) -> list[str]:
    width = max(len(top[0]), len(bottom[0]))
    return [row.center(width) for row in top + bottom]


WEATHER_ROWS = rows_of(AppConstants.WEATHER_ART)
BANNER_ROWS = side_by_side(stacked(rows_of(AppConstants.LOGO_TOP), rows_of(AppConstants.LOGO_BOTTOM)), WEATHER_ROWS)
SPLASH_TEXT = "\n".join(stacked(WEATHER_ROWS + [""], rows_of(AppConstants.LOGO)) + ["", AppConstants.SUBTITLE.center(len(rows_of(AppConstants.LOGO)[0]))])


def build_banner(total_width: int) -> str:
    """ Plain text banner: the logo and the weather art inside a double-line frame. EffectLabel pads
        its width by 2, so this is built 2 columns narrower than the panel column. """
    width = total_width - 2

    def framed(text: str) -> str:
        return f"║{text.center(width)}║"

    lines = [f"╔{'═' * width}╗"]
    lines += [framed(row) for row in BANNER_ROWS]
    lines.append(framed(AppConstants.SUBTITLE))
    lines.append(f"╚{'═' * width}╝")
    return "\n".join(lines)


BANNER = build_banner(LEFT_COLUMN_WIDTH - 2)

FINAL_COLOR = (EffectColor(BASE.lstrip("#")),)
HIGHLIGHT_CONFIG = {"final_gradient_stops": FINAL_COLOR}
RAIN_CONFIG = {
    "rain_colors": tuple(EffectColor(color.lstrip("#")) for color in (ACCENT, BASE, BORDER)),
    "rain_symbols": ("╱", "│", "╎", "'", "."),
    "final_gradient_stops": FINAL_COLOR,
}


class FixedEffectLabel(EffectLabel):
    """ EffectLabel with two fixes: no per-frame delay (its default is visibly slow), and a canvas
        height that counts every line of a multi-line string, not one fewer. """

    FRAME_DELAY = 0.02

    def __init__(self, text: str, effect: str = "Beams", config: dict | None = None, frame_delay: float | None = None) -> None:
        super().__init__(text, effect, config or {})
        self._source_text = text
        if frame_delay is not None:
            self.FRAME_DELAY = frame_delay
        line_count = len(text.split("\n"))
        self.height = line_count
        self.styles.height = line_count

    async def run_effect(self) -> None:
        effect = effects[self.effect](self._source_text)
        for key in self.config:
            if hasattr(effect.effect_config, key):
                setattr(effect.effect_config, key, self.config[key])

        effect.terminal_config.canvas_width = self.width
        effect.terminal_config.canvas_height = self.height
        effect.terminal_config.ignore_terminal_dimensions = True
        for frame in effect:
            self.text = frame
            self.update(Text.from_ansi(self.text))
            await asyncio.sleep(self.FRAME_DELAY)
        self.post_message(self.EffectFinished(self.effect))


class LoopingEffectLabel(FixedEffectLabel):
    """ Repeats its effect with a pause between sweeps. Looping is driven from EffectFinished,
        since overriding on_mount would race EffectLabel's own on_mount. """

    REPEAT_EVERY = 7.0

    @on(EffectLabel.EffectFinished)
    def _schedule_next_sweep(self, message: EffectLabel.EffectFinished) -> None:
        message.stop()
        self.run_worker(self._restart_after_pause, exclusive=True)

    async def _restart_after_pause(self) -> None:
        await asyncio.sleep(self.REPEAT_EVERY)
        await self.run_effect()


class BootSplashScreen(ModalScreen):
    """ Rains the logo into place once at startup, then holds it briefly. Skippable with any key or click. """

    CSS = f"""
    BootSplashScreen {{
        background: {DARK};
        align: center middle;
    }}
    """

    BINDINGS = [("escape,enter,space", "skip", "Skip")]

    HOLD_AFTER_FINISHED = 1.0
    SAFETY_TIMEOUT = 10.0
    FRAME_DELAY = 0.006

    def compose(self) -> ComposeResult:
        yield FixedEffectLabel(SPLASH_TEXT, effect="Expand", config=RAIN_CONFIG, frame_delay=self.FRAME_DELAY)

    def on_mount(self) -> None:
        self.set_timer(self.SAFETY_TIMEOUT, self.action_skip)

    @on(EffectLabel.EffectFinished)
    def _on_finished(self, message: EffectLabel.EffectFinished) -> None:
        self.set_timer(self.HOLD_AFTER_FINISHED, self.action_skip)

    def action_skip(self) -> None:
        if self.is_current:
            self.dismiss()

    def on_click(self) -> None:
        self.action_skip()
