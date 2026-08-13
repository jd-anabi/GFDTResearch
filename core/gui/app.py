"""Build the QApplication + MainWindow and install a last-resort excepthook that surfaces unhandled
errors in a dialog instead of silently killing the app."""
import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from core import config, registry

from . import app_icon, fonts, mpl_theme, settings, theming
from .main_window import MainWindow


def _install_excepthook(parent_getter):
    """Route any otherwise-unhandled exception to a dialog with a collapsible traceback.

    Installed BEFORE MainWindow() is built, so a failure during panel construction (e.g. a bad cell
    file breaking a panel's prefill) shows a dialog instead of a bare console traceback -- app.py used
    to install this only after MainWindow(), leaving that whole window unguarded.
    """
    def _excepthook(exc_type, exc, tb):
        tb_text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            box = QMessageBox(parent_getter())
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Unhandled error")
            box.setText(f"{exc_type.__name__}: {exc}")
            box.setDetailedText(tb_text)
            box.exec()
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook


def build_app(argv=None):
    # Quiet the per-time-segment bar: it wraps segs in {1,2,3} and nests a whole level under the
    # training-data bar for nothing. The solver's per-step bar stays ON -- it feeds the Solver
    # Performance meter. The GUI is the only caller that flips this; the CLI keeps both bars.
    config.QUIET_SEGMENT_BAR = True

    # Register saved user-defined models BEFORE MainWindow: the model combos read VALID_MODELS at
    # construction. Per-file failures land in registry.load_errors (shown on the Settings screen).
    registry.load_user_models()

    # Before the QApplication, and so before any window: this is the Windows shell's grouping key, and
    # setting it late leaves the taskbar button showing the interpreter's icon instead of ours.
    app_icon.set_windows_app_user_model_id()

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setOrganizationName(settings.ORG)   # both are needed for a stable QSettings store on Windows
    app.setApplicationName(settings.APP)
    # On the APPLICATION, not on the window: every dialog, message box and popped-out figure window
    # inherits it, so there is no per-window wiring to forget. PRISM had no icon at all before this.
    app.setWindowIcon(app_icon.app_icon())
    # Fusion repaints every control from the palette (the native Windows style doesn't), so a light/dark
    # colour-scheme switch recolours consistently on Win/Mac/Linux. Theming lives here (never in
    # MainWindow) so it stays out of the test path.
    app.setStyle("Fusion")
    qs = settings.settings()
    fonts.load_app_font(app, prefer_inter=settings.get_force_inter(qs))  # base UI font before any widget
    appearance = theming.Appearance(app)
    appearance.set_system_accent(settings.get_system_accent(qs))         # OS accent (opt-in) ...
    appearance.set_mode(settings.get_appearance(qs))                     # ... then restore + apply Fluent
    mpl_theme.install(appearance)   # matplotlib figures follow the theme (built after this point)

    window_ref = {}
    _install_excepthook(lambda: window_ref.get("w"))
    try:
        window = MainWindow(appearance=appearance)
    except Exception as e:                   # noqa: BLE001 -- a construction failure must still show
        box = QMessageBox(None)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("PRISM could not start")
        box.setText(f"The application failed to start:\n{e}")
        box.setDetailedText(traceback.format_exc())
        box.exec()
        # We already showed the dialog and logged the traceback. Exit via SystemExit so the installed
        # sys.excepthook does NOT fire and pop a SECOND "Unhandled error" dialog for the same failure.
        traceback.print_exc()
        raise SystemExit(1) from e
    window_ref["w"] = window
    return app, window
