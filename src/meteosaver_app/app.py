""" MeteoSaver dashboard: edits the main settings of the configuration file and runs the enabled modules
in one subprocess, streaming its log. """

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import Button, Checkbox, Footer, Input, Label, RichLog, Select, Static

from image_discovery import find_sheets
from meteosaver_app.clipboard import ClipboardInput
from meteosaver_app.config_file import output_directories, read_config, results_directory, write_values
from meteosaver_app.pipeline_runner import (
    ERROR_CONTINUATION,
    REPO_ROOT,
    build_pipeline_command,
    check_gpu,
    classify_line,
    describe_exit,
    format_duration,
    open_in_browser,
    parse_progress,
    parse_review_marker,
    parse_stage_marker,
    tesseract_available,
)
from meteosaver_app.splash import BANNER, BootSplashScreen, HIGHLIGHT_CONFIG, LoopingEffectLabel
from meteosaver_app.theme import (
    ABORTED, ACCENT, BASE, BORDER, DARK, DONE, FAILED, LEFT_COLUMN_WIDTH, LOG_STYLES, PANEL_BG,
    RUNNING, SKIPPED, METEO_THEME,
)

OCR_MODELS = ["GLM-OCR", "Tesseract-OCR", "EasyOCR"]
ROW_GROUPINGS = ["columnwise", "kmeans"]


@dataclass
class Stage:
    number: int
    key: str
    label: str
    default_on: bool = True  # whether the stage is ticked when the dashboard starts
    checkbox: Checkbox = field(init=False, default=None)
    status: Label = field(init=False, default=None)
    started: float = field(init=False, default=0.0)
    elapsed: float = field(init=False, default=0.0)  # over the sheets and stations done so far
    errors: int = field(init=False, default=0)


class MeteoSaverApp(App):
    CSS = f"""
    Screen {{
        background: {DARK};
        color: {BASE};
    }}

    #body {{
        height: 1fr;
    }}

    #left-column {{
        width: {LEFT_COLUMN_WIDTH};
        height: 100%;
    }}

    #banner {{
        background: {DARK};
        margin-bottom: 1;
    }}

    #log-column {{
        width: 1fr;
        height: 100%;
    }}

    #controls-scroll {{
        height: 1fr;
        scrollbar-color: {BORDER};
        scrollbar-color-hover: {ACCENT};
        scrollbar-color-active: {ACCENT};
        scrollbar-background: {DARK};
    }}

    .panel {{
        height: auto;
        border: double {BASE};
        background: {DARK};
        color: {BASE};
        padding: 1 2;
        margin-bottom: 1;
    }}

    .panel-title {{
        text-style: bold underline;
        color: {ACCENT};
        margin-bottom: 1;
    }}

    .stage-row {{
        height: 1;
        align: left middle;
    }}

    .stage-label {{
        width: 42;
    }}

    .stage-status {{
        width: 1fr;
        text-align: right;
    }}

    #progress {{
        margin-top: 1;
        color: {BORDER};
    }}

    .status-pending {{ color: {BASE}; }}
    .status-running {{ color: {RUNNING}; text-style: bold; }}
    .status-done {{ color: {DONE}; text-style: bold; }}
    .status-skipped {{ color: {SKIPPED}; }}
    .status-failed {{ color: {FAILED}; text-style: bold; }}
    .status-aborted {{ color: {ABORTED}; text-style: bold; }}

    .button-row {{
        height: auto;
        margin-top: 1;
        align: center middle;
    }}

    Input {{
        background: {PANEL_BG};
        color: {BASE};
        border: solid {BASE};
    }}

    Input:focus {{
        border: solid {ACCENT};
    }}

    Input:disabled {{
        color: {SKIPPED};
        border: solid {SKIPPED};
    }}

    Select {{
        height: auto;
    }}

    Select > SelectCurrent {{
        background: {PANEL_BG};
        color: {BASE};
        border: solid {BASE};
    }}

    Select:focus > SelectCurrent {{
        border: solid {ACCENT};
    }}

    Select > SelectOverlay {{
        background: {PANEL_BG};
        color: {BASE};
        border: solid {ACCENT};
    }}

    Checkbox {{
        height: 1;
        border: none;
        padding: 0 1;
        background: {DARK};
        color: {BASE};
    }}

    Checkbox:focus {{
        background-tint: {PANEL_BG} 40%;
    }}

    Button {{
        background: {PANEL_BG};
        color: {BASE};
        border: solid {BASE};
    }}

    Button:hover {{
        background: {BASE};
        color: {DARK};
    }}

    Button:focus {{
        text-style: bold;
        border: solid {ACCENT};
    }}

    Button:disabled {{
        color: {SKIPPED};
        border: solid {SKIPPED};
    }}

    #run-btn {{
        border: solid {DONE};
        color: {DONE};
    }}

    #abort-btn {{
        border: solid {FAILED};
        color: {FAILED};
    }}

    #log {{
        height: 1fr;
        border: double {BASE};
        background: {DARK};
        color: {BASE};
        scrollbar-color: {BORDER};
        scrollbar-color-hover: {ACCENT};
        scrollbar-color-active: {ACCENT};
        scrollbar-background: {DARK};
    }}

    Label {{
        color: {BASE};
    }}

    #alarm {{
        display: none;
        height: 3;
        content-align: center middle;
        text-style: bold;
        background: {FAILED};
        color: {DARK};
    }}

    /* An error in the run turns the frames red until the next run */
    .has-errors #alarm {{
        display: block;
    }}

    .has-errors #log, .has-errors .panel {{
        border: double {FAILED};
    }}

    .has-errors .panel-title {{
        color: {FAILED};
    }}
    """

    BINDINGS = [("q", "quit", "Quit")]
    TITLE = "MeteoSaver"

    def __init__(self, config_path: Path) -> None:
        super().__init__()
        self.register_theme(METEO_THEME)
        self.theme = "meteo"
        self.config_path = config_path
        self.config = read_config(config_path)
        self.stages: list[Stage] = [
            Stage(1, "transcription", "1/5  TRANSCRIPTION"),
            # The QA/QC module in two parts: the clean-up recalculates nothing, the quality control (which includes it) does
            Stage(2, "cleanup", "2/5  CLEAN-UP (NO RECALC)"),
            Stage(3, "qaqc", "3/5  QUALITY CONTROL (RECALC)"),
            Stage(4, "formatting", "4/5  DATA FORMATTING"),
            Stage(5, "validation", "5/5  VALIDATION"),
            # Not a module of the pipeline itself: a page per sheet to check the readings, opened in the browser (needs GLM-OCR)
            Stage(6, "review", "+    REVIEW PAGE (BROWSER)", default_on=False),
        ]
        self._review_pages: list[str] = []  # the review pages written by this run
        self._process: subprocess.Popen | None = None
        self._aborted = False
        self._elapsed_timer: Timer | None = None
        self._active_stages: list[Stage] = []  # the module that is running, if any
        self._errors: list[tuple[str, list[str]]] = []  # (where it happened, lines of the error) per error of this run
        self._in_error = False  # whether the previous output line belonged to an error
        self._progress_text = ""

    # -- layout --

    def _setting(self, section: str, key: str, fallback: str = "") -> str:
        return self.config.get(section, key, fallback=fallback).strip()

    def compose(self) -> ComposeResult:
        ocr_model = self._setting("Transcription", "ocr_model", OCR_MODELS[0])
        row_grouping = self._setting("Transcription", "row_grouping", "kmeans")

        with Horizontal(id="body"):
            with Vertical(id="left-column"):
                banner = LoopingEffectLabel(BANNER, effect="Highlight", config=HIGHLIGHT_CONFIG)
                banner.id = "banner"
                yield banner
                with VerticalScroll(id="controls-scroll"):
                    with Vertical(classes="panel", id="paths-panel"):
                        yield Label("PATHS", classes="panel-title")
                        yield Label("IMAGES (input dir, searched with all folders in it)")
                        yield ClipboardInput(value=self._setting("Directories", "full_datadir"), id="images-input")
                        yield Label("RESULTS (output dir)")
                        yield ClipboardInput(value=results_directory(self.config), id="results-input")

                    with Vertical(classes="panel"):
                        yield Label("STAGES", classes="panel-title")
                        for stage in self.stages:
                            with Horizontal(classes="stage-row"):
                                stage.checkbox = Checkbox(value=stage.default_on, id=f"stage-{stage.key}-check")
                                yield stage.checkbox
                                yield Label(stage.label, classes="stage-label")
                                stage.status = Label("PENDING", classes="stage-status status-pending")
                                yield stage.status
                        yield Label("", id="progress")

                    with Vertical(classes="panel"):
                        yield Label("OCR", classes="panel-title")
                        yield Label("OCR/HTR model")
                        yield Select([(model, model) for model in dict.fromkeys(OCR_MODELS + [ocr_model])],
                                     value=ocr_model, allow_blank=False, id="ocr-model-select")
                        yield Label("Row grouping")
                        yield Select([(grouping, grouping) for grouping in dict.fromkeys(ROW_GROUPINGS + [row_grouping])],
                                     value=row_grouping, allow_blank=False, id="row-grouping-select")
                        yield Label("GLM-OCR model dir")
                        yield ClipboardInput(value=self._setting("Transcription", "glm_model_dir"), id="glm-model-input")
                        yield Label("GLM-OCR batch size")
                        yield ClipboardInput(value=self._setting("Transcription", "glm_batch_size", "64"), id="glm-batch-input")

                    with Vertical(classes="panel"):
                        yield Label("QA/QC", classes="panel-title")
                        yield Label("Maximum temperature (°C)")
                        yield ClipboardInput(value=self._setting("QAQC", "max_temperature_threshold"), id="max-temperature-input")
                        yield Label("Minimum temperature (°C)")
                        yield ClipboardInput(value=self._setting("QAQC", "min_temperature_threshold"), id="min-temperature-input")
                        yield Label("Uncertainty margin (°C)")
                        yield ClipboardInput(value=self._setting("QAQC", "uncertainty_margin"), id="uncertainty-margin-input")

            with Vertical(id="log-column"):
                yield Static("", id="alarm")
                yield RichLog(id="log", wrap=True, highlight=False, markup=False)
                with Horizontal(classes="button-row"):
                    yield Button("RUN PIPELINE", id="run-btn", variant="success")
                    yield Button("ABORT", id="abort-btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._update_glm_inputs()
        self.push_screen(BootSplashScreen(), self._after_splash)

    def _value(self, widget_id: str) -> str:
        return self.query_one(f"#{widget_id}", Input).value.strip()

    def _path(self, text: str) -> Path:
        """ Paths in the configuration file are relative to the main project directory. """
        return REPO_ROOT / os.path.expanduser(text)

    def _after_splash(self, _result: object = None) -> None:
        self._log("=== SYSTEM CHECK ===", "header")
        self._log(f"Configuration file: {self.config_path}", "dim")
        self._check_images(self._path(self._value("images-input")))

        ocr_model = self.query_one("#ocr-model-select", Select).value
        if ocr_model == "GLM-OCR":
            self._log_check("GLM-OCR model", self._path(self._value("glm-model-input")).is_dir())
            self.run_worker(self._startup_gpu_check, thread=True, group="gpu-check")
        elif ocr_model == "Tesseract-OCR":
            self._log_check("Tesseract", tesseract_available())
        self._log("")
        self._log("Ready. Check the settings, toggle stages, press RUN PIPELINE.")
        self._log("The settings are saved to the configuration file when a run starts.", "dim")

    def _check_images(self, images: Path) -> bool:
        """ Logs what the search for images finds, the same search main.py does. False if there is nothing to run on. """
        sheets_by_station, messages = find_sheets(str(images))
        for message in messages:
            self._log(f"[ WARN ]  {message}", "warn")
        sheets = sum(len(paths) for paths in sheets_by_station.values())
        if not sheets:
            self._log(f"[ FAIL ]  No images of data sheets found in {images} or in the folders within it", "fail")
            return False
        self._log(f"[ OK ]  Images: {sheets} sheets in {len(sheets_by_station)} stations ({', '.join(sheets_by_station)})", "ok")
        return True

    def _startup_gpu_check(self) -> None:
        ok, detail = check_gpu()
        self.call_from_thread(self._log, f"[ {'OK' if ok else 'WARN'} ]  GPU: {detail}", "ok" if ok else "warn")

    def _log(self, text: str, style: str | None = None) -> None:
        log = self.query_one("#log", RichLog)
        log.write(Text(text, style=LOG_STYLES.get(style, style)) if style else text)

    def _log_check(self, name: str, ok: bool) -> None:
        self._log(f"[ {'OK' if ok else 'FAIL'} ]  {name}", "ok" if ok else "fail")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "ocr-model-select":
            self._update_glm_inputs()

    def _update_glm_inputs(self) -> None:
        # The GLM-OCR settings only apply while GLM-OCR is the selected model
        uses_glm = self.query_one("#ocr-model-select", Select).value == "GLM-OCR"
        for widget_id in ("glm-model-input", "glm-batch-input"):
            self.query_one(f"#{widget_id}", Input).disabled = not uses_glm

    # -- settings --

    def _collect_settings(self) -> dict[tuple[str, str], str] | None:
        """ The settings as they are on screen, keyed by (section, key) of the configuration file.
            Logs what is wrong and returns None if they cannot be used for a run. """
        images = self._value("images-input")
        results = self._value("results-input")
        ocr_model = self.query_one("#ocr-model-select", Select).value
        transcribing = self.stages[0].checkbox.value
        reviewing = any(stage.key == "review" and stage.checkbox.value for stage in self.stages)

        if reviewing and ocr_model != "GLM-OCR":
            self._log("[ FAIL ] The review page shows the text as read by GLM-OCR: select the GLM-OCR model, or switch the review page off.", "fail")
            return None
        if not images or not results:
            self._log("[ FAIL ] The IMAGES and RESULTS directories are required.", "fail")
            return None
        if not self._check_images(self._path(images)):
            return None

        settings = {
            ("Directories", "full_datadir"): images,
            ("Transcription", "ocr_model"): ocr_model,
            ("Transcription", "row_grouping"): self.query_one("#row-grouping-select", Select).value,
        }
        if os.path.normpath(results) != results_directory(self.config):
            settings.update(output_directories(self.config, results))

        if ocr_model == "GLM-OCR":
            model_dir = self._value("glm-model-input")
            batch_size = self._value("glm-batch-input")
            if transcribing and not self._path(model_dir).is_dir():
                self._log(f"[ FAIL ] GLM-OCR model dir not found: {self._path(model_dir)}", "fail")
                return None
            if not (batch_size.isdigit() and int(batch_size) > 0):
                self._log(f"[ FAIL ] GLM-OCR batch size must be a positive number: {batch_size}", "fail")
                return None
            settings[("Transcription", "glm_model_dir")] = model_dir
            settings[("Transcription", "glm_batch_size")] = batch_size
        elif ocr_model == "Tesseract-OCR" and transcribing and not tesseract_available():
            self._log("[ FAIL ] Tesseract-OCR is selected, but the tesseract program was not found.", "fail")
            return None

        for name, widget_id, key in (
            ("Maximum temperature", "max-temperature-input", "max_temperature_threshold"),
            ("Minimum temperature", "min-temperature-input", "min_temperature_threshold"),
            ("Uncertainty margin", "uncertainty-margin-input", "uncertainty_margin"),
        ):
            value = self._value(widget_id)
            try:
                float(value)
            except ValueError:
                self._log(f"[ FAIL ] {name} must be a number: {value}", "fail")
                return None
            settings[("QAQC", key)] = value
        if float(settings[("QAQC", "min_temperature_threshold")]) >= float(settings[("QAQC", "max_temperature_threshold")]):
            self._log("[ FAIL ] The minimum temperature must be below the maximum temperature.", "fail")
            return None
        return settings

    def _save_settings(self, settings: dict[tuple[str, str], str]) -> None:
        changes = write_values(self.config_path, settings)
        self.config = read_config(self.config_path)
        for key, old, new in changes:
            self._log(f"Saved {key} = {new}  (was {old or 'not set'})", "dim")

    # -- run / abort --

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            self._start_pipeline()
        elif event.button.id == "abort-btn":
            self._abort_pipeline()

    def _set_status(self, stage: Stage, text: str, css_class: str) -> None:
        stage.status.set_classes(f"stage-status {css_class}")
        stage.status.update(text)

    def _set_progress(self, text: str) -> None:
        self.query_one("#progress", Label).update(text)

    def _start_pipeline(self) -> None:
        if not any(stage.checkbox.value for stage in self.stages):
            self._log("[ FAIL ] No stage is enabled.", "fail")
            return
        settings = self._collect_settings()
        if settings is None:
            return

        self._log("\n=== PIPELINE START ===", "header")
        try:
            self._save_settings(settings)
        except (OSError, KeyError) as e:
            self._log(f"[ FAIL ] Could not save the settings to {self.config_path}: {e}", "fail")
            return

        for stage in self.stages:
            stage.errors = 0
            stage.elapsed = 0.0
            if stage.checkbox.value:
                self._set_status(stage, "PENDING", "status-pending")
            else:
                self._set_status(stage, "SKIPPED", "status-skipped")

        self._aborted = False
        self._errors = []
        self._review_pages = []
        self._in_error = False
        self._set_progress("")
        self.screen.remove_class("has-errors")
        self.query_one("#run-btn", Button).disabled = True
        self.query_one("#abort-btn", Button).disabled = False
        self.run_worker(self._run_pipeline_worker, thread=True, exclusive=True, group="pipeline")

    def _abort_pipeline(self) -> None:
        self._aborted = True
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        for stage in self._active_stages:
            self._set_status(stage, "ABORTED", "status-aborted")

    def _run_pipeline_worker(self) -> None:
        pipeline_start = time.monotonic()
        failed = False
        selected = [stage for stage in self.stages if stage.checkbox.value]
        stage_by_key = {stage.key: stage for stage in selected}

        try:
            self._elapsed_timer = self.call_from_thread(self.set_interval, 1.0, self._tick)
            try:
                command = build_pipeline_command([stage.key for stage in selected])
                self.call_from_thread(self._log, f"$ {' '.join(command)}", "dim")
                # Child stdout is a pipe, so Python would block-buffer it without this flag.
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "METEOSAVER_CONFIG": str(self.config_path)}
                self._process = subprocess.Popen(
                    command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                for line in self._process.stdout:
                    progress = parse_progress(line)
                    if progress is not None:
                        self._progress_text = progress
                        self._finish_active_stages()  # a sheet that failed never reported its module as done
                        self.call_from_thread(self._set_progress, progress)
                        self.call_from_thread(self._log, f"\n--- {progress} ---", "header")
                        continue

                    page = parse_review_marker(line)
                    if page is not None:
                        self._show_review_page(page)
                        continue

                    marker = parse_stage_marker(line)
                    if marker is None or marker[0] not in stage_by_key:
                        style = classify_line(line)
                        self._track_error(line.rstrip(), style == "error")
                        self.call_from_thread(self._log, line.rstrip(), style)
                        continue

                    stage, event = stage_by_key[marker[0]], marker[1]
                    if event == "start":
                        self._finish_active_stages()
                        self._begin(stage)
                    else:
                        self._finish_active_stages()
                returncode = self._process.wait()
            except Exception as e:
                self.call_from_thread(self._log, f"[ FAIL ] {type(e).__name__}: {e}", "fail")
                returncode = None
                failed = True

            self._stop_timer()

            if not self._aborted and (failed or returncode != 0):
                failed = True
                if returncode is not None:
                    reason = describe_exit(returncode)
                    self._track_error(reason, True)
                    self.call_from_thread(self._log, f"[ FAIL ] {reason}", "fail")
                for stage in self._active_stages:
                    self.call_from_thread(self._set_status, stage, "FAILED", "status-failed")

            if not failed and not self._aborted:
                self._finish_active_stages()
                total = time.monotonic() - pipeline_start
                self.call_from_thread(self._log, "\n=== TIMELINE ===", "header")
                for stage in selected:
                    self.call_from_thread(self._log, f"{stage.label:<32} {format_duration(stage.elapsed):>10}")
                    if stage.errors:
                        self.call_from_thread(
                            self._set_status, stage, f"⚠ {stage.errors} ERR  DONE {format_duration(stage.elapsed)}", "status-failed"
                        )
                    else:
                        self.call_from_thread(self._set_status, stage, f"DONE {format_duration(stage.elapsed)}", "status-done")
                self.call_from_thread(self._log, f"{'TOTAL':<32} {format_duration(total):>10}", "header")
                if not self._errors:
                    self.call_from_thread(self._log, "\nPipeline complete.", "ok")

            if self._errors and not self._aborted:
                self.call_from_thread(self._log_error_summary, failed)
            elif self._aborted:
                self.call_from_thread(self._log, "\n=== PIPELINE ABORTED ===", "warn")
        finally:
            self._active_stages = []
            self._process = None
            self.call_from_thread(self._reset_buttons)

    def _show_review_page(self, page: str) -> None:
        """ The first review page of a run is opened in the browser as soon as it is written; the others are only
            logged, so a run over many sheets does not open a tab for each. """
        self._review_pages.append(page)
        if len(self._review_pages) == 1:
            if open_in_browser(page):
                self.call_from_thread(self._log, f"Review page opened in the browser: {page}", "ok")
            else:
                self.call_from_thread(self._log, f"Review page (no browser could be started, open this file): {page}", "warn")
        else:
            self.call_from_thread(self._log, f"Review page: {page}", "dim")

    def _begin(self, stage: Stage) -> None:
        """ Called from the pipeline worker thread when a module starts on a sheet or station. """
        stage.started = time.monotonic()
        self._active_stages.append(stage)
        self.call_from_thread(self._set_status, stage, f"RUNNING {format_duration(stage.elapsed)}", "status-running")

    def _finish_active_stages(self) -> None:
        """ Called from the pipeline worker thread. A module runs once per sheet or station, so between
            those it waits with the time it took so far. """
        for stage in self._active_stages:
            stage.elapsed += time.monotonic() - stage.started
            if not self._aborted:
                self.call_from_thread(self._set_status, stage, f"WAITING {format_duration(stage.elapsed)}", "status-pending")
        self._active_stages = []

    def _track_error(self, line: str, is_error: bool) -> None:
        """ Called from the pipeline worker thread for every output line. An error is its first line plus
            the continuation lines after it (its traceback), every new error updates the alarm. """
        if not is_error:
            self._in_error = False
            return

        if self._in_error and line.startswith(ERROR_CONTINUATION):
            self._errors[-1][1].append(line)
            return

        self._in_error = True
        where = " / ".join([stage.label.strip() for stage in self._active_stages] + [self._progress_text]).strip(" /") or "PIPELINE"
        self._errors.append((where, [line]))
        for stage in self._active_stages:
            stage.errors += 1
        self.call_from_thread(self._raise_alarm, len(self._errors))

    def _raise_alarm(self, count: int) -> None:
        self.query_one("#alarm", Static).update(
            f"⚠  {count} ERROR{'S' if count != 1 else ''} IN THIS RUN  ⚠\nfull details in the log and in the summary at the end"
        )
        if not self.screen.has_class("has-errors"):
            self.screen.add_class("has-errors")
            self.notify("The pipeline reported an error, see the log.", title="ERROR", severity="error", timeout=10)

    def _log_error_summary(self, failed: bool) -> None:
        """ Repeats every error of the run in full at the end of the log, so none is lost in the output above. """
        count = len(self._errors)
        self._log(f"\n=== ⚠ {count} ERROR{'S' if count != 1 else ''} ===", "error")
        for number, (where, lines) in enumerate(self._errors, start=1):
            self._log(f"\n--- error {number}/{count} during {where} ---", "error")
            for line in lines:
                self._log(line, "error")
        outcome = "Pipeline FAILED" if failed else "Pipeline finished, but not everything went well"
        self._log(f"\n{outcome}: {count} error{'s' if count != 1 else ''}, listed above.", "error")

    def _tick(self) -> None:
        if self._aborted:
            return
        for stage in list(self._active_stages):
            running = stage.elapsed + time.monotonic() - stage.started
            self._set_status(stage, f"RUNNING {format_duration(running)}", "status-running")

    def _stop_timer(self) -> None:
        if self._elapsed_timer is not None:
            self.call_from_thread(self._elapsed_timer.stop)
            self._elapsed_timer = None

    def _reset_buttons(self) -> None:
        self.query_one("#run-btn", Button).disabled = False
        self.query_one("#abort-btn", Button).disabled = True

    def action_quit(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        self.exit()
