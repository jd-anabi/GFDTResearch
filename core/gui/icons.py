"""A small bundled icon FONT (assets/icons/prism-icons.ttf), registered via QFontDatabase exactly like
the Inter text font in fonts.py. Buttons render an icon as TEXT (a private-use codepoint in the icon
family), so they recolour for FREE from the QSS ``color:`` on every theme flip -- no QIcon re-tint
machinery -- and a dedicated icon family sidesteps the per-font glyph-coverage/sizing quirks of using
stray unicode glyphs across Segoe UI / Inter / system fonts.

Registration is LAZY + memoized (a QApplication always exists by the time a widget is built, incl. the
offscreen test path, so no build_app call-order dependency). If the .ttf can't be loaded, ``apply_icon``
falls back to the unicode glyph text, so buttons never render blank and the app still starts.
"""
from pathlib import Path

from PySide6.QtGui import QFontDatabase

_ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"

# Semantic name -> (icon-font codepoint, unicode text fallback). The private-use codepoints match
# assets/icons/build_prism_icons.py CODEPOINTS; the fallbacks are the glyphs these buttons used pre-B-e.
NAMES = {
    "back":     (chr(0xE000), "←"),   # left arrow
    "settings": (chr(0xE001), "⚙"),   # gear
    "refresh":  (chr(0xE002), "⟳"),   # circular arrow
    "help":     (chr(0xE003), "?"),
    "close":    (chr(0xE004), "✕"),   # remove-a-row (the chi probe table)
}

_family = None            # resolved icon-font family name, or None if the .ttf was unavailable
_registered = False


def register():
    """Register the bundled icon font (idempotent); return its real family name, or None if unavailable.
    The family is discovered at runtime so swapping the .ttf never needs a code change here."""
    global _family, _registered
    if _registered:
        return _family
    _registered = True
    for ttf in sorted(_ICON_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(ttf))
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                _family = families[0]
                break
    return _family


def available() -> bool:
    return register() is not None


def family():
    return register()


def glyph(name: str) -> str:
    """The icon codepoint when the font is available, else the unicode text fallback."""
    icon, fallback = NAMES[name]
    return icon if available() else fallback


def apply_icon(widget, name: str) -> None:
    """Render icon ``name`` on ``widget`` as text. Sets ONLY the font family (QSS still owns size/colour);
    falls back to the unicode glyph in the widget's current font when the icon font is missing. Never
    leaves the widget blank."""
    icon, fallback = NAMES[name]
    if available():
        f = widget.font()
        f.setFamily(_family)
        widget.setFont(f)
        widget.setText(icon)
    else:
        widget.setText(fallback)


def icon_button(name: str, *, object_name: str | None = None, tooltip: str | None = None,
                tool: bool = True, parent=None):
    """Build a fresh QToolButton (or QPushButton) carrying icon ``name``. Convenience for the two nav
    buttons; the objectName-keyed QSS in design.py styles size/colour as before."""
    from PySide6.QtWidgets import QPushButton, QToolButton
    btn = QToolButton(parent) if tool else QPushButton(parent)
    if object_name:
        btn.setObjectName(object_name)
    if tool:
        btn.setAutoRaise(True)
    if tooltip:
        btn.setToolTip(tooltip)
    apply_icon(btn, name)
    return btn
