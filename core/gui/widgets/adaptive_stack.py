"""A QStackedWidget that sizes to the page you are LOOKING AT, not to the tallest page.

Qt's default QStackedWidget size hint is the maximum over all pages, because every page keeps its
size policy while hidden. That is right when the pages are similar and wrong when they are not, and
in this app they are not:

  * the Infer tab's stack has a two-row "simulated" page and a chi page with K+3 rows -- so the short
    page reserved the tall page's height and left a large dead gap;
  * the model builder creates one forcing stack PER STATE VARIABLE, whose "None" page is empty and
    whose "exponential" page is three fields plus a radio row;
  * the bounds/cell SourceToggle switches between a one-row file picker and a 320px-capped grid.

Setting every hidden page to Ignored means only the current page contributes a size hint, so the
container collapses to fit what is actually on screen.
"""
from PySide6.QtWidgets import QSizePolicy, QStackedWidget


class AdaptiveStack(QStackedWidget):
    """QStackedWidget whose height follows the CURRENT page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(self._resize_to_current)

    def addWidget(self, w):                       # noqa: N802 -- Qt naming
        index = super().addWidget(w)
        self._resize_to_current(self.currentIndex())
        return index

    def _resize_to_current(self, index: int) -> None:
        for i in range(self.count()):
            page = self.widget(i)
            if page is None:
                continue
            policy = page.sizePolicy()
            # The visible page keeps its natural vertical policy; the rest stop contributing.
            policy.setVerticalPolicy(QSizePolicy.Preferred if i == index else QSizePolicy.Ignored)
            page.setSizePolicy(policy)
        current = self.widget(index)
        if current is not None:
            current.adjustSize()
        self.adjustSize()
        self.updateGeometry()
