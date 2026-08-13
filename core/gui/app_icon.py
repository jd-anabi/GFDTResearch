"""The application / window icon, loaded from the PNG set in assets/app.

Why not the SVG directly: Qt's ``svg`` IMAGE-FORMAT plugin is absent in this environment, so
``QIcon("prism.svg")`` returns a NULL icon and the app silently keeps Qt's default mark with nothing
logged anywhere. assets/app/build_app_icon.py rasterises the SVG source to PNGs offline (via the QtSvg
*module*, which is a different thing and is available); this only ever loads those PNGs. Same
source-plus-committed-binary split as the icon font in icons.py.

Missing or unreadable PNGs are not an error: an app with no icon still runs, and refusing to start over
decoration would be absurd. The icon is simply empty, exactly as it was before this module existed.
"""
import sys
from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap

_APP_DIR = Path(__file__).resolve().parent / "assets" / "app"
_APP_USER_MODEL_ID = "PRISM.PRISM.DesktopApp.1"


def app_icon() -> QIcon:
    """A multi-resolution QIcon, or an empty one if the assets are unavailable. Every size found is
    added so Qt picks the nearest rather than downscaling 256 to 16, which turns a stroked mark to
    mush -- see the note in assets/app/prism.svg."""
    icon = QIcon()
    for png in sorted(_APP_DIR.glob("prism-*.png")):
        pm = QPixmap(str(png))
        if not pm.isNull():
            icon.addPixmap(pm)
    return icon


def set_windows_app_user_model_id() -> None:
    """Tell the Windows shell this process is its own application.

    Without it the taskbar groups PRISM under the HOST INTERPRETER: the button shows python.exe's
    icon, not ours, however faithfully setWindowIcon is called. It is a shell grouping key, so it has
    to be set BEFORE the first window is created. No-op off Windows, and best-effort everywhere: a
    failure here costs a taskbar icon, which is never worth stopping a launch for.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_USER_MODEL_ID)
    except Exception:                                   # noqa: BLE001 -- decoration, never fatal
        pass
