"""A lightweight QPainter-based 3D ball-and-stick viewer with atom picking."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget


def _rot(axis: int, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class MoleculeView(QWidget):
    """Displays one molecule. Emits indices of clicked / box-selected atoms."""

    atomClicked = Signal(int)
    atomsBoxSelected = Signal(list)
    atomHovered = Signal(int)          # -1 when nothing is under the cursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.pos = np.zeros((0, 3))
        self.radii = np.zeros(0)
        self.colors: list[QColor] = []
        self.bonds: list[tuple[int, int]] = []
        self.labels: list[str] = []
        self.visible = np.zeros(0, bool)
        self.pickable = np.zeros(0, bool)
        self.rings: dict[int, QColor] = {}
        self.show_labels = False
        self.overlay_pos = np.zeros((0, 3))
        self.overlay_colors: list[QColor] = []
        self.overlay_bonds: list[tuple[int, int]] = []
        self.overlay_radius = 0.3
        self.placeholder = "Nothing loaded"
        self.info_text = ""
        self.rot = np.eye(3)
        self.center = np.zeros(3)
        self.zoom = 20.0
        self.pan = np.zeros(2)
        self._press = None
        self._last = None
        self._band: QRectF | None = None
        self._hover = -1
        self._screen = np.zeros((0, 2))
        self._depth = np.zeros(0)

    # ------------------------------------------------------------------ API
    def set_molecule(self, pos, radii, colors, bonds, labels, pickable=None, reset_view=True):
        self.pos = np.asarray(pos, float)
        self.radii = np.asarray(radii, float)
        self.colors = list(colors)
        self.bonds = list(bonds)
        self.labels = list(labels)
        n = len(self.pos)
        self.visible = np.ones(n, bool)
        self.pickable = np.ones(n, bool) if pickable is None else np.asarray(pickable, bool)
        self.rings = {}
        self._hover = -1
        if reset_view:
            self.reset_view()
        self.update()

    def clear(self, placeholder: str = "Nothing loaded"):
        self.placeholder = placeholder
        self.set_molecule(np.zeros((0, 3)), [], [], [], [])
        self.set_overlay(np.zeros((0, 3)), [], [])

    def set_colors(self, colors):
        self.colors = list(colors)
        self.update()

    def set_visible(self, mask):
        self.visible = np.asarray(mask, bool)
        self.update()

    def set_rings(self, rings: dict[int, QColor]):
        self.rings = dict(rings)
        self.update()

    def set_overlay(self, pos, colors, bonds, radius=0.3):
        self.overlay_pos = np.asarray(pos, float).reshape(-1, 3)
        self.overlay_colors = list(colors)
        self.overlay_bonds = list(bonds)
        self.overlay_radius = radius
        self.update()

    def set_info(self, text: str):
        self.info_text = text
        self.update()

    def set_show_labels(self, on: bool):
        self.show_labels = on
        self.update()

    def reset_view(self):
        pts = self.pos[self.visible] if len(self.pos) else self.pos
        if len(pts) == 0:
            self.rot, self.center, self.zoom, self.pan = np.eye(3), np.zeros(3), 20.0, np.zeros(2)
            return
        self.center = pts.mean(0)
        if len(pts) >= 2:
            _, _, vt = np.linalg.svd(pts - self.center, full_matrices=False)
            R = vt
            if np.linalg.det(R) < 0:
                R[2] *= -1
            self.rot = R
        else:
            self.rot = np.eye(3)
        p = (pts - self.center) @ self.rot.T
        ext = np.abs(p[:, :2]).max(0) + self.radii.max(initial=0.5) + 1e-6
        w, h = max(self.width(), 50), max(self.height(), 50)
        self.zoom = 0.45 * min(w / ext[0], h / ext[1])
        self.pan = np.zeros(2)
        self.update()

    # ------------------------------------------------------------ projection
    def _project(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = (x - self.center) @ self.rot.T
        sx = self.width() / 2 + self.pan[0] + p[:, 0] * self.zoom
        sy = self.height() / 2 + self.pan[1] - p[:, 1] * self.zoom
        return np.stack([sx, sy], 1), p[:, 2]

    # ---------------------------------------------------------------- paint
    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        pal = self.palette()
        qp.fillRect(self.rect(), pal.base())
        if len(self.pos) == 0:
            qp.setPen(pal.placeholderText().color())
            qp.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        scr, dep = self._project(self.pos)
        self._screen, self._depth = scr, dep
        zmin, zmax = dep.min(), dep.max()
        span = max(zmax - zmin, 1e-6)
        items = []
        for i in np.flatnonzero(self.visible):
            items.append((dep[i], 1, int(i)))
        for (a, b) in self.bonds:
            if self.visible[a] and self.visible[b]:
                items.append(((dep[a] + dep[b]) / 2 - 1e-3, 0, (a, b)))
        items.sort(key=lambda t: t[0])
        bond_w = max(1.5, 0.16 * self.zoom)
        for z, kind, d in items:
            fog = 0.55 + 0.45 * (z - zmin) / span
            if kind == 0:
                a, b = d
                pa, pb = QPointF(*scr[a]), QPointF(*scr[b])
                mid = (pa + pb) / 2
                for p0, c in ((pa, self.colors[a]), (pb, self.colors[b])):
                    pen = QPen(c.darker(int(100 / fog)), bond_w, Qt.SolidLine, Qt.RoundCap)
                    qp.setPen(pen)
                    qp.drawLine(p0, mid)
            else:
                i = d
                r = max(2.0, self.radii[i] * self.zoom)
                c = self.colors[i]
                cx, cy = scr[i]
                g = QRadialGradient(QPointF(cx - r / 3, cy - r / 3), r * 1.3)
                g.setColorAt(0, c.lighter(160))
                g.setColorAt(0.5, c.darker(int(100 / fog)))
                g.setColorAt(1, c.darker(int(190 / fog)))
                qp.setBrush(g)
                ring = self.rings.get(i)
                if ring is not None:
                    qp.setPen(QPen(ring, max(2.5, min(5.0, 0.15 * r))))
                elif i == self._hover and self.pickable[i]:
                    qp.setPen(QPen(pal.highlight().color(), 2.0))
                else:
                    qp.setPen(QPen(c.darker(250), 0.8))
                qp.drawEllipse(QPointF(cx, cy), r, r)
        # overlay (e.g. fitted all-atom structure drawn over CG beads)
        if len(self.overlay_pos):
            osc, _ = self._project(self.overlay_pos)
            for a, b in self.overlay_bonds:
                qp.setPen(QPen(QColor(40, 40, 40, 170), max(1.0, 0.6 * self.overlay_radius * self.zoom)))
                qp.drawLine(QPointF(*osc[a]), QPointF(*osc[b]))
            qp.setPen(Qt.NoPen)
            r = max(1.5, self.overlay_radius * self.zoom)
            for k, (x, y) in enumerate(osc):
                c = QColor(self.overlay_colors[k])
                c.setAlpha(210)
                qp.setBrush(c)
                qp.drawEllipse(QPointF(x, y), r, r)
        # labels
        if self.show_labels or self._hover >= 0:
            f = QFont(self.font())
            f.setPointSizeF(max(7.0, min(11.0, 0.35 * self.zoom)))
            qp.setFont(f)
            idx = np.flatnonzero(self.visible) if self.show_labels else [self._hover]
            for i in idx:
                if not self.show_labels and not self.visible[i]:
                    continue
                r = self.radii[i] * self.zoom
                x, y = scr[i]
                qp.setPen(pal.text().color())
                qp.drawText(QPointF(x + r * 0.7, y - r * 0.7), self.labels[i])
        if self.info_text:
            qp.setFont(self.font())
            qp.setPen(pal.placeholderText().color())
            qp.drawText(self.rect().adjusted(8, 0, -8, -6), Qt.AlignLeft | Qt.AlignBottom, self.info_text)
        if self._band is not None:
            qp.setPen(QPen(pal.highlight().color(), 1, Qt.DashLine))
            hc = QColor(pal.highlight().color())
            hc.setAlpha(40)
            qp.setBrush(hc)
            qp.drawRect(self._band)

    # ---------------------------------------------------------------- mouse
    def _pick(self, x: float, y: float) -> int:
        if len(self._screen) == 0:
            return -1
        d = np.hypot(self._screen[:, 0] - x, self._screen[:, 1] - y)
        r = np.maximum(self.radii * self.zoom, 4.0)
        ok = (d <= r) & self.visible & self.pickable
        if not ok.any():
            return -1
        cand = np.flatnonzero(ok)
        return int(cand[np.argmax(self._depth[cand])])

    def mousePressEvent(self, ev):
        p = ev.position()
        self._press = (p.x(), p.y(), ev.button(), ev.modifiers())
        self._last = (p.x(), p.y())
        if ev.button() == Qt.LeftButton and ev.modifiers() & Qt.ShiftModifier:
            self._band = QRectF(p, p)

    def mouseMoveEvent(self, ev):
        p = ev.position()
        if self._press is None:
            h = self._pick(p.x(), p.y())
            if h != self._hover:
                self._hover = h
                self.atomHovered.emit(h)
                self.update()
            return
        dx, dy = p.x() - self._last[0], p.y() - self._last[1]
        self._last = (p.x(), p.y())
        btn = self._press[2]
        if self._band is not None:
            self._band = QRectF(QPointF(self._press[0], self._press[1]), p).normalized()
        elif btn == Qt.LeftButton:
            if self._press[3] & Qt.ControlModifier:
                self.rot = _rot(2, -dx * 0.01) @ self.rot
            else:
                self.rot = _rot(0, dy * 0.01) @ _rot(1, dx * 0.01) @ self.rot
        elif btn in (Qt.RightButton, Qt.MiddleButton):
            self.pan += (dx, dy)
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._press is None:
            return
        p = ev.position()
        moved = np.hypot(p.x() - self._press[0], p.y() - self._press[1]) > 4
        if self._band is not None:
            if moved and len(self._screen):
                r = self._band
                inside = [int(i) for i in np.flatnonzero(self.visible & self.pickable)
                          if r.contains(QPointF(*self._screen[i]))]
                if inside:
                    self.atomsBoxSelected.emit(inside)
            self._band = None
        elif not moved and self._press[2] == Qt.LeftButton:
            i = self._pick(p.x(), p.y())
            if i >= 0:
                self.atomClicked.emit(i)
        self._press = None
        self.update()

    def mouseDoubleClickEvent(self, ev):
        # fast consecutive clicks on atoms must still count as clicks
        p = ev.position()
        if ev.button() == Qt.LeftButton and self._pick(p.x(), p.y()) >= 0:
            self.mousePressEvent(ev)
        else:
            self.reset_view()

    def wheelEvent(self, ev):
        f = 1.0015 ** ev.angleDelta().y()
        self.zoom = float(np.clip(self.zoom * f, 0.5, 2000))
        self.update()

    def leaveEvent(self, _ev):
        if self._hover != -1:
            self._hover = -1
            self.atomHovered.emit(-1)
            self.update()
