"""Generate ``prism-icons.ttf`` -- a tiny icon font of 5 original glyphs (back, settings, refresh, help,
close) used by core/gui/icons.py. Run:  python core/gui/assets/icons/build_prism_icons.py

The glyphs are authored procedurally in font units (em=1000, y-up, centred on (500, 500)) and compiled
with fontTools, so the font is fully reproducible offline (no third-party font needed). Icons are mapped
to the private-use codepoints U+E000..U+E004. They are original work for PRISM, released under MIT
(see LICENSE.txt). Regenerate after editing a glyph; commit the resulting .ttf.

CODEPOINTS here and ``icons.NAMES`` must stay in step -- the codepoint is the only thing that ties a
semantic name to a glyph, and a name present in one but not the other renders as .notdef (an empty box)
rather than failing. ``tests/test_user_models.py`` asserts every NAMES entry resolves.
"""
import math
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

EM = 1000
CX = CY = 500
CODEPOINTS = {"back": 0xE000, "settings": 0xE001, "refresh": 0xE002, "help": 0xE003,
              "close": 0xE004}


def _polar(a_deg, r, cx=CX, cy=CY):
    a = math.radians(a_deg)
    return (cx + r * math.cos(a), cy + r * math.sin(a))


def _arc(cx, cy, r, a0, a1, n):
    return [_polar(a0 + (a1 - a0) * i / n, r, cx, cy) for i in range(n + 1)]


def _draw(pen, *contours):
    for pts in contours:
        pts = [(round(x), round(y)) for x, y in pts]
        pen.moveTo(pts[0])
        for p in pts[1:]:
            pen.lineTo(p)
        pen.closePath()
    return pen.glyph()


def _back():
    # A left-pointing arrow (single filled contour): shaft to the right, triangular head on the left.
    pts = [(180, 500), (470, 700), (470, 580), (820, 580),
           (820, 420), (470, 420), (470, 300)]
    return _draw(TTGlyphPen(None), pts)


def _settings():
    # A gear: n rectangular teeth around a central round hole (the hole is a reversed contour so the
    # non-zero winding knocks it out).
    n, r_root, r_tip, r_hole = 8, 285, 405, 200
    pitch = 360.0 / n
    tw = pitch * 0.40                                   # angular half-width of a tooth top
    outer = []
    for k in range(n):
        th = k * pitch
        outer.append(_polar(th - tw, r_tip))
        outer.append(_polar(th + tw, r_tip))
        outer.append(_polar(th + pitch / 2, r_root))   # valley into the next tooth
    hole = _arc(CX, CY, r_hole, 360, 0, 40)            # reversed direction -> a hole
    return _draw(TTGlyphPen(None), outer, hole)


def _refresh():
    # A circular arrow: a thick ring open at the top, plus a triangular arrowhead at one end.
    r_out, r_in = 380, 265
    a0, a1 = 128, 128 + 284                             # sweep ~284 deg CCW, gap at the top
    ring = _arc(CX, CY, r_out, a0, a1, 48) + _arc(CX, CY, r_in, a1, a0, 48)
    r_mid = (r_out + r_in) / 2
    base_in, base_out = _polar(a1, r_in - 35), _polar(a1, r_out + 35)
    tip = _polar(a1, r_mid)
    tan = a1 + 90                                       # tangent (CCW sweep direction)
    apex = (tip[0] + 150 * math.cos(math.radians(tan)),
            tip[1] + 150 * math.sin(math.radians(tan)))
    head = [base_in, base_out, apex]
    return _draw(TTGlyphPen(None), ring, head)


def _help():
    # A question mark: a top hook that ends pointing straight DOWN at the centre, a centred stem
    # continuing it, and a dot below. Ending the hook at -90 deg keeps the stem visually connected.
    hcx, hcy = 500, 655
    r_out, r_in = 190, 108
    a0, a1 = 200, -90                                   # CW: lower-left, over the top, to bottom-centre
    hook = _arc(hcx, hcy, r_out, a0, a1, 44) + _arc(hcx, hcy, r_in, a1, a0, 44)
    stem = [(459, 330), (541, 330), (541, 485), (459, 485)]   # centred vertical bar under the hook end
    dot = _arc(500, 250, 60, 0, 360, 28)               # the dot
    return _draw(TTGlyphPen(None), hook, stem, dot)


def _bar(cx, cy, half_len, half_w, angle_deg):
    """One rotated rectangle, as a single contour. Point order is fixed so every bar winds the same
    way -- two bars in OPPOSITE directions would knock their overlap out under the non-zero fill rule
    and leave a hole where the strokes cross."""
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)                   # along the bar
    vx, vy = -uy, ux                                    # across it
    return [(cx + ux * half_len + vx * half_w, cy + uy * half_len + vy * half_w),
            (cx + ux * half_len - vx * half_w, cy + uy * half_len - vy * half_w),
            (cx - ux * half_len - vx * half_w, cy - uy * half_len - vy * half_w),
            (cx - ux * half_len + vx * half_w, cy - uy * half_len + vy * half_w)]


def _close():
    # An X: two crossed bars at +-45 deg. Deliberately a touch shorter and thicker than a text "x" --
    # it sits on a 32 px-wide button next to a file path, where a thin glyph reads as dirt.
    return _draw(TTGlyphPen(None), _bar(CX, CY, 300, 58, 45), _bar(CX, CY, 300, 58, -45))


def build(out_path: Path) -> Path:
    glyphs = {".notdef": TTGlyphPen(None).glyph(), "back": _back(), "settings": _settings(),
              "refresh": _refresh(), "help": _help(), "close": _close()}
    order = [".notdef", "back", "settings", "refresh", "help", "close"]

    fb = FontBuilder(EM, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({cp: name for name, cp in CODEPOINTS.items()})
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics({g: (EM, 0) for g in order})
    fb.setupHorizontalHeader(ascent=EM, descent=0)
    fb.setupNameTable({"familyName": "PRISM Icons", "styleName": "Regular",
                       "psName": "PRISMIcons-Regular"})
    fb.setupOS2(sTypoAscender=EM, sTypoDescender=0, usWinAscent=EM, usWinDescent=0)
    fb.setupPost()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(out_path))
    return out_path


if __name__ == "__main__":
    p = build(Path(__file__).resolve().parent / "prism-icons.ttf")
    print(f"wrote {p}")
