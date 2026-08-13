"""Render ``prism.svg`` to the committed PNG set used as the application/window icon.
Run:  python core/gui/assets/app/build_app_icon.py

WHY A PNG SET AND NOT THE SVG. Qt's `svg` IMAGE-FORMAT PLUGIN is absent in this environment --
``QImageReader.supportedImageFormats()`` lists no "svg" -- so ``QIcon("prism.svg")`` returns a null
icon and the window silently keeps Qt's default mark, with nothing logged. The QtSvg *module*
(QSvgRenderer) is a different thing and is available, which is what lets this script rasterise
offline. So: SVG is the editable source, PNGs are what ships, exactly the split the icon FONT uses
(assets/icons/build_prism_icons.py authors glyphs, the .ttf is committed).

Several sizes rather than one: Qt picks the nearest and scales, and a 256 -> 16 downscale of a
stroked mark turns to mush. Regenerate after editing the SVG; commit the PNGs.
"""
import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

SIZES = (16, 24, 32, 48, 64, 128, 256)
HERE = Path(__file__).resolve().parent


def build(svg_path: Path = None, out_dir: Path = None) -> list:
    svg_path = svg_path or (HERE / "prism.svg")
    out_dir = out_dir or HERE
    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise SystemExit(f"QSvgRenderer could not parse {svg_path}")
    written = []
    for n in SIZES:
        img = QImage(n, n, QImage.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))                      # transparent: the tile's rx corners show
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        renderer.render(p, QRectF(0, 0, n, n))
        p.end()
        out = out_dir / f"prism-{n}.png"
        if not img.save(str(out), "PNG"):
            raise SystemExit(f"failed to write {out}")
        written.append(out)
    return written


if __name__ == "__main__":
    app = QApplication(sys.argv)                          # QImage/QPainter need a QGuiApplication
    for p in build():
        print(f"wrote {p}")
