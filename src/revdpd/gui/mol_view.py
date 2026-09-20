"""A lightweight QPainter-based 3D ball-and-stick viewer with atom picking."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen,
                           QPolygonF, QRadialGradient)
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
        self.setMinimumSize(150, 140)
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
        self.alpha = 255          # opacity of the atoms/beads (translucent CG beads: ~150)
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

    def set_alpha(self, alpha: int):
        """Opacity of the spheres and bonds (0-255); translucency reveals an overlay."""
        self.alpha = int(alpha)
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
    def _sphere(self, qp: QPainter, x: float, y: float, r: float, c: QColor, fog: float,
                alpha: int = 255) -> None:
        """A shaded ball: bright specular highlight, body colour, darker rim."""
        base = QColor(c)
        light = QColor(base.lighter(175))
        body = QColor(base.darker(int(105 / fog)))
        dark = QColor(base.darker(int(175 / fog)))
        if alpha >= 250:
            light.setAlpha(255)
            body.setAlpha(255)
            dark.setAlpha(255)
        else:
            # glass-like: clear in the middle, denser towards the rim
            light.setAlpha(min(255, int(alpha * 0.9)))
            body.setAlpha(int(alpha * 0.55))
            dark.setAlpha(min(255, int(alpha * 1.5)))
        g = QRadialGradient(QPointF(x - 0.35 * r, y - 0.4 * r), 1.45 * r)
        g.setColorAt(0.0, light)
        g.setColorAt(0.45, body)
        g.setColorAt(1.0, dark)
        qp.setBrush(QBrush(g))
        qp.drawEllipse(QPointF(x, y), r, r)

    def _half_bond(self, qp: QPainter, pa: QPointF, pb: QPointF, c: QColor, w: float,
                   fog: float, alpha: int = 255, ra: float = 0.0) -> None:
        """The half of a bond that belongs to the atom at ``pa``, shaded like a cylinder.

        It starts at the surface of that atom's sphere (radius ``ra`` in pixels), so a bond
        never paints over the ball it grows out of - which would show through a translucent
        one. Halves are drawn separately so each is sorted by the depth of its own atom.
        """
        dx, dy = pb.x() - pa.x(), pb.y() - pa.y()
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        mid = QPointF((pa.x() + pb.x()) / 2, (pa.y() + pb.y()) / 2)
        t = min(ra * 0.92, 0.49 * length)
        p0 = QPointF(pa.x() + ux * t, pa.y() + uy * t)
        if (mid.x() - p0.x()) * ux + (mid.y() - p0.y()) * uy <= 0:
            return                      # the sphere already reaches the middle
        nx, ny = -uy * w, ux * w
        body = QColor(c.darker(int(112 / fog)))
        edge = QColor(c.darker(int(200 / fog)))
        hi = QColor(c.lighter(150))
        for col in (body, edge, hi):
            col.setAlpha(alpha)
        g = QLinearGradient(p0.x() - nx, p0.y() - ny, p0.x() + nx, p0.y() + ny)
        g.setColorAt(0.0, edge)
        g.setColorAt(0.28, hi)
        g.setColorAt(0.55, body)
        g.setColorAt(1.0, edge)
        qp.setPen(Qt.NoPen)
        qp.setBrush(QBrush(g))
        qp.drawPolygon(QPolygonF([QPointF(p0.x() - nx, p0.y() - ny), QPointF(mid.x() - nx, mid.y() - ny),
                                  QPointF(mid.x() + nx, mid.y() + ny), QPointF(p0.x() + nx, p0.y() + ny)]))

    def _label(self, qp: QPainter, x: float, y: float, text: str, colour: QColor,
               halo: QColor) -> None:
        """Text with a halo so it stays readable on top of atoms."""
        path = QPainterPath()
        path.addText(QPointF(x, y), qp.font(), text)
        qp.setPen(QPen(halo, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        qp.setBrush(Qt.NoBrush)
        qp.drawPath(path)
        qp.setPen(Qt.NoPen)
        qp.setBrush(colour)
        qp.drawPath(path)

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        pal = self.palette()
        base = pal.base().color()
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0.0, base.lighter(103) if base.lightness() > 127 else base.lighter(130))
        bg.setColorAt(1.0, base.darker(107) if base.lightness() > 127 else base.darker(105))
        qp.fillRect(self.rect(), QBrush(bg))
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
                items.append((0.75 * dep[a] + 0.25 * dep[b] - 1e-3, 0, (a, b)))
                items.append((0.75 * dep[b] + 0.25 * dep[a] - 1e-3, 0, (b, a)))
        if len(self.overlay_pos):
            osc, odep = self._project(self.overlay_pos)
            for a, b in self.overlay_bonds:
                items.append((0.75 * odep[a] + 0.25 * odep[b] - 1e-3, 2, (a, b)))
                items.append((0.75 * odep[b] + 0.25 * odep[a] - 1e-3, 2, (b, a)))
            for k in range(len(self.overlay_pos)):
                items.append((odep[k], 3, k))
        items.sort(key=lambda t: t[0])
        # bond radius follows the spheres, so it looks the same at any zoom level
        bond_w = max(1.0, 0.30 * float(np.median(self.radii)) * self.zoom)
        ov_r = max(1.5, self.overlay_radius * self.zoom)
        for z, kind, d in items:
            fog = 0.6 + 0.4 * (z - zmin) / span
            if kind == 0:
                a, b = d
                ra = max(2.0, self.radii[a] * self.zoom)
                self._half_bond(qp, QPointF(*scr[a]), QPointF(*scr[b]), self.colors[a],
                                bond_w, fog, min(255, self.alpha + 70), ra)
            elif kind == 1:
                i = d
                r = max(2.0, self.radii[i] * self.zoom)
                x, y = scr[i]
                ring = self.rings.get(i)
                if ring is not None:
                    qp.setPen(QPen(ring, max(2.5, min(5.0, 0.15 * r))))
                elif i == self._hover and self.pickable[i]:
                    qp.setPen(QPen(pal.highlight().color(), 2.0))
                else:
                    rim = QColor(self.colors[i].darker(230))
                    rim.setAlpha(255 if self.alpha >= 250 else min(255, self.alpha + 80))
                    qp.setPen(QPen(rim, 0.8 if self.alpha >= 250 else 1.2))
                self._sphere(qp, x, y, r, self.colors[i], fog, self.alpha)
            elif kind == 2:
                a, b = d
                self._half_bond(qp, QPointF(*osc[a]), QPointF(*osc[b]), self.overlay_colors[a],
                                0.55 * ov_r, fog, 255, ov_r)
            else:
                k = d
                qp.setPen(Qt.NoPen)
                self._sphere(qp, osc[k, 0], osc[k, 1], ov_r, self.overlay_colors[k], fog)
        # labels
        if self.show_labels or self._hover >= 0:
            f = QFont(self.font())
            f.setPointSizeF(max(7.0, min(10.0, 0.28 * self.zoom)))
            qp.setFont(f)
            halo = QColor(pal.base().color())
            halo.setAlpha(220)
            idx = np.flatnonzero(self.visible) if self.show_labels else [self._hover]
            for i in idx:
                if not self.show_labels and not self.visible[i]:
                    continue
                r = self.radii[i] * self.zoom
                x, y = scr[i]
                self._label(qp, x + r * 0.7, y - r * 0.7, self.labels[i], pal.text().color(), halo)
        if self.info_text:
            qp.setFont(self.font())
            qp.setPen(pal.placeholderText().color())
            qp.setBrush(Qt.NoBrush)
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
