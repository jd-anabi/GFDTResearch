"""Fluent design tokens + palette/stylesheet builders (bespoke, no third-party widget library).

Single source of truth for the app's Fluent look. Pure + import-safe (only needs QtGui for QPalette/QColor),
so it can be unit-tested without a display. ``theming.Appearance`` applies ``build_palette`` + ``build_qss``
for the resolved light/dark scheme; everything else in the app stays palette-driven, so these two functions
recolour the WHOLE chrome.

Two things are LOAD-BEARING and must stay correct in ``build_palette`` regardless of the stylesheet:
  * ``QPalette.Mid``       -- read by widgets/progress_pane.py ``_Sparkline`` (baseline) + help_badge border.
  * ``QPalette.Highlight`` -- read by ``_Sparkline`` (trace) + help_badge hover + selection colour.
QSS does not feed those custom-painted reads; the palette does.
"""
from string import Template

from PySide6.QtGui import QColor, QPalette

# ── tokens ────────────────────────────────────────────────────────────────────
RADIUS_SM = 4
RADIUS_MD = 8
CTL_H = 32              # standard Fluent control height
CTL_H_SM = 28
SPACE = (4, 8, 12, 16, 20, 24)

# ── Layout sizing ─────────────────────────────────────────────────────────────────────────────────
# The controls column had a hard 460px MAXIMUM, which no amount of window-widening could escape, so
# any form wider than that got a permanent horizontal scrollbar. The cap is gone; these set the floor
# and the starting split instead, and the user's own drag is remembered from there.
CONTROLS_MIN_W = 360      # narrower than this and the two-field composite rows stop being readable
DEFAULT_SPLIT = (420, 880)  # initial (controls, results) split; overridden by the saved layout
FIELD_MIN_W = 96          # a numeric field must show ~8 characters; "0.033333" used to scroll inside
                          # a 3-6 character box and could not be read back without clicking into it
PATH_FIELD_MIN_W = 220    # path fields hold absolute paths, so they need materially more
DEFAULT_RESULTS_SPLIT = (520, 200)  # initial (figures, log) split in the results column
HELP_VIEWER_MIN_H = 260   # Settings' help viewer: tall enough to be usable, it scrolls itself
LABEL_MAX_W = 190         # ceiling on a form's label column: one long option name (e.g. "F0 (ND
                          # forcing amplitude)") used to narrow every field in the same form

# Type ramp: (pixel size, weight). Weight 600 = SemiBold (Fluent's "Strong"/heading weight).
TYPE = {
    "caption":     (12, 400),
    "body":        (14, 400),
    "body_strong": (14, 600),
    "heading":     (16, 600),
    "subtitle":    (20, 600),
    "title":       (28, 600),
}
BODY_PX = TYPE["body"][0]

# Neutral + accent palettes. Tuned to Fluent (Win11) light/dark. Retune here in ONE place.
LIGHT = {
    "window":       "#F3F3F3",
    "base":         "#FFFFFF",   # cards, inputs, popups
    "alt_base":     "#F7F7F7",
    "mid":          "#D1D1D1",   # borders / dividers  (-> QPalette.Mid)
    "mid_strong":   "#C4C4C4",
    "text":         "#1B1B1B",
    "text_2nd":     "#616161",   # secondary / placeholder / captions
    "text_disabled":"#A0A0A0",
    "button":       "#FBFBFB",
    "button_hover": "#F0F0F0",
    "button_press": "#E5E5E5",
    "accent":       "#0F6CBD",   # -> QPalette.Highlight
    "accent_hover": "#115EA3",
    "accent_press": "#0C3B5E",
    "on_accent":    "#FFFFFF",
    "tooltip_bg":   "#FFFFFF",
    "scrollbar":    "#B8B8B8",
    "scrollbar_hover":"#9A9A9A",
}
DARK = {
    "window":       "#202020",
    "base":         "#2B2B2B",
    "alt_base":     "#333333",
    "mid":          "#3D3D3D",
    "mid_strong":   "#4A4A4A",
    "text":         "#F5F5F5",
    "text_2nd":     "#C7C7C7",
    "text_disabled":"#6E6E6E",
    "button":       "#313131",
    "button_hover": "#3A3A3A",
    "button_press": "#454545",
    "accent":       "#4CA0E0",
    "accent_hover": "#62B0EE",
    "accent_press": "#3A88CC",
    "on_accent":    "#1B1B1B",
    "tooltip_bg":   "#2B2B2B",
    "scrollbar":    "#4A4A4A",
    "scrollbar_hover":"#5E5E5E",
}


def _c(hex_str: str) -> QColor:
    return QColor(hex_str)


def system_accent() -> "str | None":
    """The Windows OS accent colour as '#RRGGBB', or None (non-Windows, or any read failure).

    Primary source is the Explorer accent (an ABGR DWORD -- the colour the user picked in Windows
    Settings); fallback is the DWM ColorizationColor (ARGB). Both reads are wrapped: this must never
    raise, it degrades to the fixed Fluent-blue token."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Accent") as key:
            value, _ = winreg.QueryValueEx(key, "AccentColorMenu")
        r, g, b = value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF          # ABGR
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:                                       # noqa: BLE001
        pass
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\DWM") as key:
            value, _ = winreg.QueryValueEx(key, "ColorizationColor")
        r, g, b = (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF          # ARGB
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:                                       # noqa: BLE001
        return None


def tokens(dark: bool, accent: "str | None" = None) -> dict:
    """The token dict for a scheme, optionally re-based on an accent override ('#RRGGBB').

    The override recomputes accent/accent_hover/accent_press (hover/press follow the fixed tokens'
    direction: light mode darkens on interaction, dark mode lightens then darkens) and picks on_accent
    black/white by luminance so CTA text stays readable on any OS accent."""
    base = DARK if dark else LIGHT
    if not accent:
        return base
    c = _c(accent)
    if not c.isValid():
        return base
    t = dict(base)
    hover = c.lighter(112) if dark else c.darker(112)
    press = c.darker(120) if dark else c.darker(145)
    luminance = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    t["accent"] = c.name().upper()
    t["accent_hover"] = hover.name().upper()
    t["accent_press"] = press.name().upper()
    t["on_accent"] = "#1B1B1B" if luminance > 160 else "#FFFFFF"
    return t


def build_palette(dark: bool, accent: "str | None" = None) -> QPalette:
    """A fully-populated QPalette from the tokens. Sets every role that matters (including the
    Disabled group -- the whole controls column is disabled per run), because partially-QSS'd complex
    controls fall back to the palette for their un-styled sub-parts."""
    t = tokens(dark, accent)
    p = QPalette()

    def setall(role, color, groups=(QPalette.Active, QPalette.Inactive)):
        for g in groups:
            p.setColor(g, role, _c(color))

    setall(QPalette.Window, t["window"])
    setall(QPalette.WindowText, t["text"])
    setall(QPalette.Base, t["base"])
    setall(QPalette.AlternateBase, t["alt_base"])
    setall(QPalette.Text, t["text"])
    setall(QPalette.Button, t["button"])
    setall(QPalette.ButtonText, t["text"])
    setall(QPalette.BrightText, "#FFFFFF")
    setall(QPalette.ToolTipBase, t["tooltip_bg"])
    setall(QPalette.ToolTipText, t["text"])
    setall(QPalette.PlaceholderText, t["text_2nd"])
    setall(QPalette.Highlight, t["accent"])                 # LOAD-BEARING (custom paint + selection)
    setall(QPalette.HighlightedText, t["on_accent"])
    setall(QPalette.Link, t["accent"])
    setall(QPalette.LinkVisited, t["accent_press"])
    # Bevel/edge roles kept coherent so any Fusion-drawn sub-part we don't fully QSS matches.
    setall(QPalette.Mid, t["mid"])                          # LOAD-BEARING (_Sparkline baseline, badge)
    setall(QPalette.Midlight, t["alt_base"])
    setall(QPalette.Light, t["base"])
    setall(QPalette.Dark, t["mid_strong"])
    setall(QPalette.Shadow, "#000000" if not dark else "#000000")

    # Disabled group: BasePanel disables the entire controls column during a run.
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, _c(t["text_disabled"]))
    p.setColor(QPalette.Disabled, QPalette.Base, _c(t["alt_base"]))
    p.setColor(QPalette.Disabled, QPalette.Button, _c(t["button"]))
    p.setColor(QPalette.Disabled, QPalette.Highlight, _c(t["mid"]))
    p.setColor(QPalette.Disabled, QPalette.HighlightedText, _c(t["text_disabled"]))
    return p


# ── stylesheet ──────────────────────────────────────────────────────────────
# Named-class selectors ONLY -- never `*` or bare `QWidget` (would paint backgrounds behind the
# custom-painted _Sparkline, the pyqtgraph view, matplotlib QLabels, and the anim.py snapshot overlays).
# $-substitution (string.Template) so CSS braces stay literal.
_QSS = Template("""
/* ---- type ramp (replaces the old inline font-size stylesheets) ---- */
QLabel[type="title"]       { font-size: ${fs_title}px; font-weight: 600; }
QLabel[type="subtitle"]    { font-size: ${fs_subtitle}px; font-weight: 600; }
QLabel[type="heading"]     { font-size: ${fs_heading}px; font-weight: 600; }
QLabel[type="caption"]     { font-size: ${fs_caption}px; color: $text_2nd; }

/* ---- push buttons: neutral + accent variant ---- */
QPushButton {
    min-height: ${ctl_h}px; padding: 4px 14px;
    border: 1px solid $mid; border-radius: ${radius_sm}px;
    background: $button; color: $text;
}
QPushButton:hover   { background: $button_hover; }
QPushButton:pressed { background: $button_press; }
QPushButton:focus   { border: 1px solid $accent; }
QPushButton:disabled{ color: $text_disabled; border-color: $mid; background: $button; }
QPushButton[accent="true"] {
    background: $accent; color: $on_accent; border: 1px solid $accent; font-weight: 600;
}
QPushButton[accent="true"]:hover   { background: $accent_hover; border-color: $accent_hover; }
QPushButton[accent="true"]:pressed { background: $accent_press; border-color: $accent_press; }
QPushButton[accent="true"]:focus   { border: 1px solid $text; }
QPushButton[accent="true"]:disabled{ background: $mid; border-color: $mid; color: $text_disabled; }
/* compact square icon buttons (e.g. the picker refresh) -- override the wide default padding so a
   fixed-width glyph button isn't squeezed to a sliver, and match the adjacent input height. */
QPushButton#iconButton { padding: 0px; min-height: ${ctl_h_sm}px; font-size: ${fs_icon_btn}px; }

/* ---- text inputs (covers FloatField/IntField/PathField.edit subclasses of QLineEdit) ---- */
QLineEdit, QPlainTextEdit {
    min-height: ${ctl_h_sm}px; padding: 2px 8px;
    border: 1px solid $mid; border-radius: ${radius_sm}px;
    background: $base; color: $text; selection-background-color: $accent;
    selection-color: $on_accent;
}
QLineEdit:focus, QPlainTextEdit:focus { border: 1px solid $accent; }
QLineEdit:disabled { color: $text_disabled; background: $alt_base; }

/* ---- combo box + its (separate top-level) popup. Do NOT null ::down-arrow (no arrow asset). ---- */
QComboBox {
    min-height: ${ctl_h_sm}px; padding: 2px 8px;
    border: 1px solid $mid; border-radius: ${radius_sm}px; background: $base; color: $text;
}
QComboBox:hover { border-color: $mid_strong; }
QComboBox:focus { border: 1px solid $accent; }
QComboBox::drop-down { width: 22px; border: none; background: transparent; }
QComboBox QAbstractItemView {
    border: 1px solid $mid; border-radius: ${radius_sm}px;
    background: $base; color: $text;
    selection-background-color: $accent; selection-color: $on_accent; outline: 0;
}

/* ---- group box as a Fluent card ---- */
QGroupBox {
    background: $base; border: 1px solid $mid; border-radius: ${radius_md}px;
    margin-top: 14px; padding: 10px 12px 12px 12px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 4px; background: $base; color: $text_2nd;
}

/* ---- tabs as a Fluent pivot (accent underline on the selected tab) ---- */
QTabWidget::pane { border: none; }
QTabBar { qproperty-drawBase: 0; }
QTabBar::tab {
    background: transparent; color: $text_2nd; border: none;
    padding: 6px 14px; margin-right: 4px; border-bottom: 2px solid transparent;
}
QTabBar::tab:hover     { color: $text; }
QTabBar::tab:selected  { color: $text; border-bottom: 2px solid $accent; }
QTabBar::tab:disabled  { color: $text_disabled; }

/* ---- progress bars ---- */
QProgressBar {
    border: none; border-radius: 3px; background: $alt_base; text-align: center; color: $text;
}
QProgressBar::chunk { border-radius: 3px; background: $accent; }

/* ---- tool buttons (nav glyphs + help badge) ---- */
QToolButton { border: none; background: transparent; border-radius: ${radius_sm}px; color: $text; }
QToolButton:hover   { background: $button_hover; }
QToolButton:pressed { background: $button_press; }
QToolButton#navBack, QToolButton#navSettings { font-size: ${fs_nav_btn}px; padding: 2px 6px; }
QToolButton#helpBadge {
    border: 1px solid $mid; border-radius: 8px; color: $text_2nd;
    font-size: ${fs_badge}px; padding: 0px; background: transparent;
}
QToolButton#helpBadge:hover { border-color: $accent; color: $accent; }

/* ---- scroll bars: thin, no arrow buttons ---- */
QScrollBar:vertical   { background: transparent; width: 12px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle:vertical   { background: $scrollbar; min-height: 28px; border-radius: 6px; margin: 2px; }
QScrollBar::handle:horizontal { background: $scrollbar; min-width: 28px; border-radius: 6px; margin: 2px; }
QScrollBar::handle:hover { background: $scrollbar_hover; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; background: none; border: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* ---- menus (gear popover) ---- */
QMenu { background: $base; border: 1px solid $mid; border-radius: ${radius_md}px; padding: 4px; }
QMenu::item { padding: 6px 24px 6px 12px; border-radius: ${radius_sm}px; color: $text; }
QMenu::item:selected { background: $accent; color: $on_accent; }
QMenu::separator { height: 1px; background: $mid; margin: 4px 8px; }

/* ---- tooltips (help badge UX) ---- */
QToolTip {
    background: $tooltip_bg; color: $text; border: 1px solid $mid;
    border-radius: ${radius_sm}px; padding: 4px 8px;
}
""")


def ui_scale() -> float:
    """How much larger the ACTUAL application font is than this file's design baseline (BODY_PX).

    Every control height and font size here is a literal pixel value tuned at a 14px body font. That
    ignores the OS "Make text bigger" setting entirely: at a larger system font the text grows but the
    32px control height does not, so labels clip, the 16px "?" badge crops its glyph, and forms
    overflow the viewport. Scaling the tokens by the real font size keeps the proportions.

    Clamped: below 1.0 there is nothing to gain and rounding starts to bite; above 2.0 the layout
    stops being a layout. Returns 1.0 with no QApplication (import-time use, headless tests), so this
    module stays pure and testable without a display.
    """
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return 1.0
        font = app.font()
        px = font.pixelSize()
        if px <= 0:                                   # point-sized font: convert via logical DPI
            from PySide6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()
            dpi = screen.logicalDotsPerInch() if screen is not None else 96.0
            px = font.pointSizeF() * dpi / 72.0
        if px <= 0:
            return 1.0
        return min(2.0, max(1.0, px / BODY_PX))
    except Exception:                                 # noqa: BLE001 -- styling must never break startup
        return 1.0


def scaled(px: int) -> int:
    """A design pixel value in the user's actual text scale."""
    return int(round(px * ui_scale()))


def _qss_vars(dark: bool, accent: "str | None" = None) -> dict:
    t = tokens(dark, accent)
    return {
        **t,
        "radius_sm": RADIUS_SM,
        "radius_md": RADIUS_MD,
        "ctl_h": scaled(CTL_H),
        "ctl_h_sm": scaled(CTL_H_SM),
        # Type ramp, so the QSS sizes track the OS text setting instead of being frozen literals.
        "fs_title": scaled(TYPE["title"][0]),
        "fs_subtitle": scaled(TYPE["subtitle"][0]),
        "fs_heading": scaled(TYPE["heading"][0]),
        "fs_caption": scaled(TYPE["caption"][0]),
        "fs_icon_btn": scaled(18),
        "fs_nav_btn": scaled(20),
        "fs_badge": scaled(13),
    }


def build_qss(dark: bool, accent: "str | None" = None) -> str:
    return _QSS.substitute(_qss_vars(dark, accent))
