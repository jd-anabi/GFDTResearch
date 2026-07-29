"""The PRISM main window: a NavShell (persistent "PRISM" title + back arrow) over a home/splash screen,
four section screens, and two Settings-reached screens (the Settings/Help screen and the user-defined
model builder). Replaces the old flat four-tab layout; the section panels are reused unchanged in
behaviour -- only where they are mounted changes. Cross-validation now lives inside the FDT Analysis
section, and the SBI panel is split into the Parameter Inference section's gated tabs."""
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QMainWindow, QMenu, QMessageBox

from core import registry
from core.config import VALID_MODELS
from core.Helpers import model_store

from . import settings, theming
from .panels.base_panel import BasePanel
from .panels.crossval_panel import CrossValPanel
from .panels.fdt_panel import FdtPanel
from .panels.reduction_panel import ReductionPanel
from .panels.simulate_panel import SimulatePanel
from .screens.home_screen import HomeScreen
from .screens.inference_screen import InferenceScreen
from .screens.model_builder_screen import ModelBuilderScreen
from .screens.nav_shell import NavShell
from .screens.section_screen import SectionScreen
from .screens.settings_screen import SettingsScreen


# ── screen-geometry guards ────────────────────────────────────────────────────
# All three degrade to the requested value if Qt cannot tell us about the screen (offscreen tests),
# so they are safe to call unconditionally.
def _available(widget=None):
    """The usable desktop rect (excludes the taskbar), or None when there is no screen."""
    from PySide6.QtGui import QGuiApplication
    screen = (widget.screen() if widget is not None and widget.screen() else
              QGuiApplication.primaryScreen())
    return screen.availableGeometry() if screen is not None else None


def _fits_on_screen(w: int, h: int) -> tuple:
    """Shrink a preferred size to what the screen can actually show, leaving a small margin."""
    rect = _available()
    if rect is None:
        return w, h
    return min(w, rect.width() - 40), min(h, rect.height() - 60)


def _on_a_visible_screen(widget) -> bool:
    """True when the window's frame actually intersects some attached screen.

    Geometry saved on an external monitor restores off-screen once that monitor is gone -- the window
    exists, is 'open', and cannot be reached.
    """
    from PySide6.QtGui import QGuiApplication
    frame = widget.frameGeometry()
    screens = QGuiApplication.screens()
    if not screens:
        return True                      # headless: nothing to validate against
    return any(s.availableGeometry().intersects(frame) for s in screens)


def _centre_on_primary(widget) -> None:
    rect = _available(widget)
    if rect is None:
        return
    frame = widget.frameGeometry()
    frame.moveCenter(rect.center())
    widget.move(frame.topLeft())


class MainWindow(QMainWindow):
    """The application shell: a NavShell over Home plus the five section screens.

    Always opens on Home -- the last screen is deliberately not restored. Owns the only QSettings
    WRITE site (``_save_state``, reached from closeEvent), so panel selections and layouts persist
    from here; appearance settings are written eagerly instead, as they are applied immediately.
    """
    def __init__(self, appearance=None):
        super().__init__()
        self.appearance = appearance         # theming.Appearance or None (tests build without one)
        self.setWindowTitle("PRISM")
        # 820px exceeded the usable height of a 1366x768 laptop and of 1920x1080 at 150% scaling, so
        # the action buttons started below the fold. Open at the smaller of the preference and what
        # the screen can actually show.
        self.resize(*_fits_on_screen(1300, 820))
        # A window can always be made this small; without it Qt lets the user (or a restored
        # geometry) shrink the shell until the controls column is unusable.
        self.setMinimumSize(900, 600)

        self.nav = NavShell()
        self.setCentralWidget(self.nav)

        # Section screens (built once; their panels are the existing ones, reused verbatim).
        self.reduction_screen = SectionScreen(
            "Reduction Map", [("NWK → Hopf reduction map", ReductionPanel())])
        self.fdt_screen = SectionScreen(
            "FDT Analysis",
            [("FDT analysis", FdtPanel()), ("Sweep study cross-validation", CrossValPanel())])
        self.inference_screen = InferenceScreen("Parameter Inference")
        self.simulate_screen = SectionScreen(
            "Simulate", [("Live simulation", SimulatePanel())])

        home = HomeScreen(
            live_sections={"Reduction Map", "FDT Analysis", "Parameter Inference", "Simulate"})
        self.nav.add_screen(home)                                    # index 0 -- Home
        idx_red = self.nav.add_screen(self.reduction_screen)
        idx_fdt = self.nav.add_screen(self.fdt_screen)
        idx_inf = self.nav.add_screen(self.inference_screen)
        idx_sim = self.nav.add_screen(self.simulate_screen)

        # The Settings/Help screen: reached only from the gear popover's "Full settings…" (never a Home
        # button), so the back arrow still returns Home.
        current_mode = (appearance.mode() if appearance is not None
                        else settings.get_appearance(settings.settings()))
        qs0 = settings.settings()
        self.settings_screen = SettingsScreen(self.set_appearance_mode, current_mode,
                                              on_open_builder=self._open_model_builder,
                                              on_edit_model=self._edit_user_model,
                                              on_delete_model=self._delete_user_model,
                                              on_system_accent=self._set_system_accent,
                                              system_accent=settings.get_system_accent(qs0),
                                              on_force_inter=self._set_force_inter,
                                              force_inter=settings.get_force_inter(qs0))
        idx_settings = self.nav.add_screen(self.settings_screen)

        # The model builder: reached only from the Settings "User-defined models" group (never a Home
        # tile); its own Back button returns to Settings, the nav back arrow still returns Home.
        self.model_builder_screen = ModelBuilderScreen(
            on_saved=self._on_user_models_changed,
            on_back=lambda: self.nav.go_to(self._section_index["Settings"]))
        idx_builder = self.nav.add_screen(self.model_builder_screen)

        self._section_index = {"Reduction Map": idx_red, "FDT Analysis": idx_fdt,
                               "Parameter Inference": idx_inf, "Simulate": idx_sim,
                               "Settings": idx_settings, "Model builder": idx_builder}
        home.navigate.connect(lambda name: self.nav.go_to(self._section_index[name]))
        self._build_settings_menu(current_mode)
        self.nav.go_home()                                          # ALWAYS open on Home

        # Restore geometry only. Each panel restored its own selections in its __init__; we deliberately
        # do NOT restore the last screen -- the app always opens on Home.
        qs = settings.settings()
        geom = qs.value("window/geometry")
        if geom is not None:
            # CHECK the return value, and then check the RESULT. restoreGeometry returns False for an
            # unusable blob, and even on success it will happily restore a window onto a monitor that
            # is no longer attached -- geometry saved on an external display came back off-screen with
            # no way to reach it.
            if not self.restoreGeometry(geom) or not _on_a_visible_screen(self):
                self.resize(*_fits_on_screen(1300, 820))
                _centre_on_primary(self)

    def _build_settings_menu(self, current_mode):
        """Attach the gear's appearance popover: exclusive Light/Dark/Auto/System + 'Full settings…'."""
        menu = QMenu(self)
        group = QActionGroup(self)
        group.setExclusive(True)
        self._appearance_actions = {}
        for mode, label in theming.MODE_LABELS:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(mode == current_mode)         # setChecked emits toggled, not triggered
            group.addAction(act)
            act.triggered.connect(lambda _checked=False, m=mode: self.set_appearance_mode(m))
            self._appearance_actions[mode] = act
        menu.addSeparator()
        full = menu.addAction("Full settings…")
        full.triggered.connect(lambda: self.nav.go_to(self._section_index["Settings"]))
        self.nav.btn_settings.setMenu(menu)

    def set_appearance_mode(self, mode):
        """Apply + persist an appearance mode, and keep the gear menu and Settings screen in sync.

        Safe to call from either entry point (gear action or Settings radio): the reflected updates use
        setChecked (which emits `toggled`, not `triggered`) and the Settings radio blocks its own signal,
        so there is no feedback loop.
        """
        qs = settings.settings()
        settings.set_appearance(qs, mode)
        qs.sync()
        if self.appearance is not None:
            self.appearance.set_mode(mode)
        act = self._appearance_actions.get(mode)
        if act is not None and not act.isChecked():
            act.setChecked(True)
        self.settings_screen.set_mode(mode)

    # ── appearance extras (Settings checkboxes) ───────────────────────────────
    def _set_system_accent(self, enabled: bool):
        """Persist + live-apply the follow-OS-accent toggle (no-op appearance under tests)."""
        qs = settings.settings()
        settings.set_system_accent(qs, enabled)
        qs.sync()
        if self.appearance is not None:
            self.appearance.set_system_accent(enabled)

    def _set_force_inter(self, enabled: bool):
        """Persist + live-apply the Inter-everywhere toggle (custom-painted caches settle on restart)."""
        qs = settings.settings()
        settings.set_force_inter(qs, enabled)
        qs.sync()
        from PySide6.QtWidgets import QApplication
        from . import fonts
        app = QApplication.instance()
        if app is not None:
            fonts.load_app_font(app, prefer_inter=enabled)

    # ── user-defined models (Settings group + builder screen) ─────────────────
    def _open_model_builder(self):
        self.model_builder_screen.reset()
        self.nav.go_to(self._section_index["Model builder"])

    def _edit_user_model(self, name: str):
        try:
            self.model_builder_screen.load_existing(name)
        except Exception as e:                       # noqa: BLE001 -- a corrupt file must not crash
            QMessageBox.warning(self, "Cannot edit model", f"Could not load '{name}':\n{e}")
            return
        self.nav.go_to(self._section_index["Model builder"])

    def _delete_user_model(self, name: str):
        if BasePanel._running:
            QMessageBox.information(self, "A task is running",
                                    "Wait for the running task to finish before deleting a model.")
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Delete model")
        box.setText(f"Delete the user-defined model '{name}'?")
        box.setInformativeText("This removes its definition and its generated Bounds/Cells/Units files.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec() != QMessageBox.Yes:
            return
        try:
            model_store.delete_user_model(name)
        except Exception as e:                       # noqa: BLE001
            QMessageBox.warning(self, "Delete failed", str(e))
            return
        registry.unregister(name)
        self._on_user_models_changed()

    def _on_user_models_changed(self, _name: str = ""):
        """After a save/delete: re-sync every model combo + the Settings list."""
        self._refresh_model_combos()
        self.settings_screen.refresh_models()

    def _refresh_model_combos(self):
        """Reload every panel's model combo from the live VALID_MODELS list, keeping the selection
        (falling back to NADROWSKI if the previous selection was deleted). The panel's model-changed
        hook is re-fired ONLY when the selection actually changed: the hooks refresh cell/bounds
        pickers (which resets them to their first entry), so firing on a mere item-list update would
        silently discard the user's picker selections app-wide on every model save/delete."""
        for panel in self._all_panels():
            combo = getattr(panel, "model_combo", None)
            if combo is None:
                continue
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(VALID_MODELS)
            combo.setCurrentText(current if current in VALID_MODELS else "NADROWSKI")
            combo.blockSignals(False)
            if combo.currentText() != current:
                handler = getattr(panel, "_on_model_changed", None)
                if handler is not None:
                    handler(combo.currentText())

    def _all_panels(self):
        return (self.reduction_screen.panels() + self.fdt_screen.panels()
                + self.inference_screen.panels() + self.simulate_screen.panels())

    def panel(self, cls):
        """The first panel of type ``cls`` across all screens (convenience for callers + tests)."""
        return next((p for p in self._all_panels() if isinstance(p, cls)), None)

    def closeEvent(self, event):
        """On close, offer to cancel a running task first; otherwise close normally.

        A cancel is cooperative (it lands at the run's next checkpoint), so even "Cancel & quit" does
        not stop the process instantly -- but it does stop it, instead of leaving it holding the CPU/GPU
        invisibly after the window is gone, which is what "Quit anyway" still does.
        """
        if BasePanel._running:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("A task is still running")
            box.setText("A task is still running.")
            box.setInformativeText(
                "Cancelling stops it at its next checkpoint (up to ~1 min during training). "
                "Quitting anyway closes the window now, but the task keeps running in the background "
                "until it finishes on its own.")
            cancel_quit = box.addButton("Cancel task && quit", QMessageBox.AcceptRole)
            quit_anyway = box.addButton("Quit anyway", QMessageBox.DestructiveRole)
            box.addButton("Don't quit", QMessageBox.RejectRole)
            box.setDefaultButton(cancel_quit)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel_quit:
                BasePanel.request_cancel_all()
            elif clicked is not quit_anyway:
                event.ignore()
                return

        self._save_state()
        super().closeEvent(event)

    def _save_state(self):
        """Persist window geometry + each panel's selections. Called only when a close is accepted."""
        qs = settings.settings()
        qs.setValue("window/geometry", self.saveGeometry())
        for panel in self._all_panels():
            panel.save_settings(qs)
        qs.sync()
