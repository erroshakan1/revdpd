"""Small GUI helpers: collapsible sections and wheel-proof input widgets."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QFrame, QHBoxLayout, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)


class NoWheelFilter(QObject):
    """Blocks mouse-wheel events so scrolling a panel cannot change values by accident."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and isinstance(obj, (QAbstractSpinBox, QComboBox)):
            event.ignore()
            return True
        return False


def block_wheel(root: QWidget, filt: NoWheelFilter) -> None:
    """Install ``filt`` on every spin box and combo box inside ``root``."""
    for w in root.findChildren(QAbstractSpinBox) + root.findChildren(QComboBox):
        w.installEventFilter(filt)
        w.setFocusPolicy(Qt.StrongFocus)      # no focus (and no value change) on wheel


class CollapsibleGroup(QWidget):
    """A titled section that can be folded away, optionally with an enable check box.

    Widgets are added to ``self.content`` (give it a layout), so it can replace a
    ``QGroupBox``; :meth:`isChecked` / :meth:`setChecked` work like the checkable box.
    """

    toggled = Signal(bool)

    def __init__(self, title: str, checkable: bool = False, collapsed: bool = False, parent=None):
        super().__init__(parent)
        self.title = title
        self.arrow = QToolButton()
        self.arrow.setObjectName("sectionHeader")
        self.arrow.setAutoRaise(True)
        self.arrow.setCheckable(True)
        self.arrow.setChecked(not collapsed)
        self.arrow.setToolTip("Show or hide this section")
        self.arrow.clicked.connect(self.set_expanded)
        self.check: QCheckBox | None = None
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(2)
        head.addWidget(self.arrow)
        if checkable:
            self.check = QCheckBox(title)
            self.check.toggled.connect(self._on_check)
            head.addWidget(self.check)
        else:
            self.arrow.setText(title)
            self.arrow.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        head.addStretch(1)
        self.content = QFrame()
        self.content.setObjectName("sectionContent")
        self.content.setFrameShape(QFrame.StyledPanel)
        self.content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(head)
        lay.addWidget(self.content)
        self.set_expanded(not collapsed)

    # ---------------------------------------------------------------- state
    def set_expanded(self, on: bool) -> None:
        self.arrow.setChecked(on)
        self.arrow.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.content.setVisible(on)

    def is_expanded(self) -> bool:
        return self.arrow.isChecked()

    def _on_check(self, on: bool) -> None:
        self.content.setEnabled(on)
        if on and not self.is_expanded():
            self.set_expanded(True)
        self.toggled.emit(on)

    def isChecked(self) -> bool:  # noqa: N802 - mimics QGroupBox
        return True if self.check is None else self.check.isChecked()

    def setChecked(self, on: bool) -> None:  # noqa: N802
        if self.check is not None:
            self.check.setChecked(on)
