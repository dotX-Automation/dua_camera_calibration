"""
Qt widgets of the calibration GUI: live view, indicators, samples, results and setup form.

dotX Automation s.r.l. <info@dotxautomation.com>

September 28, 2026
"""

# Copyright 2026 dotX Automation s.r.l.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time

import cv2
from dua_camera_calibration import texts
from dua_camera_calibration.settings import CHOICES
import numpy as np
from python_qt_binding.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from python_qt_binding.QtGui import (QBrush, QColor, QFont, QFontDatabase, QImage,
                                     QKeySequence, QPainter, QPen, QPolygonF)
from python_qt_binding.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                                         QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
                                         QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                                         QLineEdit, QListWidget, QListWidgetItem,
                                         QPlainTextEdit, QProgressBar, QPushButton,
                                         QShortcut, QSizePolicy, QSlider, QSpinBox,
                                         QTableWidget, QTableWidgetItem, QToolTip,
                                         QVBoxLayout, QWidget)

GREEN = QColor(40, 200, 80)
AMBER = QColor(255, 176, 0)
RED = QColor(220, 40, 40)
GREY = QColor(150, 150, 150)
_FLASH_S = 0.3


def _bar_color(p: float) -> str:
    """Return the CSS color of a progress value: red < 0.5, amber < 1, green."""
    return '#dc2828' if p < 0.5 else '#ffb000' if p < 1.0 else '#28c850'


def _monospace() -> QFont:
    """Return the system fixed-width font."""
    return QFontDatabase.systemFont(QFontDatabase.FixedFont)


class LiveView(QWidget):
    """
    Camera panes (1 mono, 2 stereo) drawn with QPainter: image, overlay, heatmap, hint arrow.

    Coordinates: a full-res pixel u maps to display px (u + 0.5) * scale - 0.5, and to the
    widget point x0 + (u + 0.5) * scale * k, k being the widget/display zoom of the pane.
    """

    def __init__(self, parent=None):
        """Create an empty view."""
        super().__init__(parent)
        self.n_panes = 1
        self.heatmap = False
        self.lossy = False
        self.placeholder = ''
        self.show_verdict = True   # False in REVIEW/VERIFY: neutral outline, no tags
        self._disp = None     # DisplayFrame: also keeps the arrays wrapped by QImage alive
        self._ov = None
        self._guidance = None
        self._hints = ()
        self._flash_t = 0.0
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, disp) -> None:
        """Show a pipeline.DisplayFrame (None clears)."""
        self._disp = disp
        self.update()

    def set_overlay(self, ov) -> None:
        """Set the latest pipeline.Overlay (None clears)."""
        self._ov = ov
        self.update()

    def set_guidance(self, state, hints=()) -> None:
        """Set the guidance.GuidanceState (heatmap) and the hints to draw as arrows."""
        self._guidance = state
        self._hints = tuple(hints)
        self.update()

    def flash(self) -> None:
        """Start the capture flash."""
        self._flash_t = time.monotonic()
        self.update()

    def pane_boxes(self) -> list:
        """Return the widget rectangle of each pane."""
        n, gap = max(1, self.n_panes), 4
        w = (self.width() - gap * (n - 1)) / n
        return [QRectF(i * (w + gap), 0, w, self.height()) for i in range(n)]

    def fit_sizes(self, src_sizes) -> tuple:
        """Return the display (w, h) of each camera: pane size in device pixels, never upscaled."""
        dpr = self.devicePixelRatioF()
        out = []
        for box, (w, h) in zip(self.pane_boxes(), src_sizes):
            s = min(box.width() * dpr / w, box.height() * dpr / h, 1.0)
            out.append((max(1, int(round(w * s))), max(1, int(round(h * s)))))
        return tuple(out)

    def paintEvent(self, event):
        """Paint every pane."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), Qt.black)
        disp = self._disp
        for i, box in enumerate(self.pane_boxes()):
            if disp is None or i >= len(disp.images):
                p.setPen(QColor(220, 220, 220))
                p.drawText(box.adjusted(20, 20, -20, -20), Qt.AlignCenter | Qt.TextWordWrap,
                           self.placeholder)
                continue
            self._paint_pane(p, i, box, disp)
        age = time.monotonic() - self._flash_t
        if age < _FLASH_S:
            p.setPen(QPen(QColor(255, 255, 255, int(255 * (1 - age / _FLASH_S))), 8))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(self.rect()).adjusted(4, 4, -4, -4))
            QTimer.singleShot(30, self.update)
        p.end()

    def _paint_pane(self, p, i, box, disp) -> None:
        """Paint pane i: image, heatmap, overlay, hint arrow, rectified lines, warnings."""
        a = disp.images[i]
        h, w = a.shape[:2]
        k = min(box.width() / w, box.height() / h)
        r = QRectF(box.x() + (box.width() - w * k) / 2, box.y() + (box.height() - h * k) / 2,
                   w * k, h * k)
        img = QImage(a.data, w, h, a.strides[0], QImage.Format_Grayscale8)
        p.drawImage(r, img)
        s = disp.scales[i] * k

        def to_q(u, v):
            return QPointF(r.x() + (u + 0.5) * s, r.y() + (v + 0.5) * s)

        state = self._guidance
        if self.heatmap and state is not None and i < len(state.coverage):
            self._paint_heatmap(p, r, np.asarray(state.coverage[i]))
        ov = self._ov
        centroid = None
        if (ov is not None and ov.gen == disp.gen and ov.rectified == disp.rectified
                and i < len(ov.points) and ov.points[i] is not None and len(ov.points[i])):
            age = disp.recv_time - ov.recv_time
            p.setOpacity(1.0 if age < 0.1 else max(0.3, 1.0 - age / 0.6))
            centroid = self._paint_overlay(p, i, ov, np.asarray(ov.points[i]), to_q,
                                           self.show_verdict)
            p.setOpacity(1.0)
        for hint in self._hints:
            if hint.camera == i and hint.target is not None:
                start = centroid or r.center()
                end = QPointF(r.x() + hint.target[0] * r.width(),
                              r.y() + hint.target[1] * r.height())
                self._paint_arrow(p, start, end)
                break
        if disp.rectified and self.n_panes > 1:
            p.setPen(QPen(QColor(0, 255, 255, 110), 1))
            y = r.y()
            while y < r.bottom():
                p.drawLine(QPointF(r.x(), y), QPointF(r.right(), y))
                y += 32
        if self.lossy:
            p.setPen(AMBER)
            p.drawText(r.adjusted(6, 4, -6, -4), Qt.AlignTop | Qt.AlignLeft,
                       'JPEG input: corners may be biased')

    @staticmethod
    def _paint_heatmap(p, r, grid) -> None:
        """Fill empty coverage cells red, fading with coverage; outline empty corner cells."""
        rows, cols = grid.shape
        cw, ch = r.width() / cols, r.height() / rows
        p.setPen(Qt.NoPen)
        for (y, x), v in np.ndenumerate(grid):
            if v < 1.0:
                p.fillRect(QRectF(r.x() + x * cw, r.y() + y * ch, cw, ch),
                           QColor(220, 30, 30, int(90 * (1.0 - v))))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(RED, 2))
        for y, x in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)):
            if grid[y, x] == 0:
                p.drawRect(QRectF(r.x() + x * cw + 1, r.y() + y * ch + 1, cw - 2, ch - 2))

    @staticmethod
    def _paint_overlay(p, i, ov, pts, to_q, verdict=True):
        """Draw corners, outline and reason tag of camera i; return the centroid point."""
        v = ov.verdict
        good = v.ok or v.reason == 'ACCEPTED'
        det = ov.detections[i] if i < len(ov.detections) else None
        if ov.rectified or det is None:
            hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        else:
            hull = np.asarray(det.outline)
        pen = QPen((GREEN if good else AMBER) if verdict else QColor(0, 200, 255), 3)
        pen.setStyle(Qt.SolidLine if good or not verdict else Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(QPolygonF([to_q(x, y) for x, y in hull]))
        n = len(pts)
        for j, (x, y) in enumerate(pts):
            c = QColor.fromHsvF(0.8 * j / max(1, n - 1), 1.0, 1.0)
            p.setPen(QPen(c, 1.5))
            p.drawEllipse(to_q(x, y), 3.0, 3.0)
        cx, cy = pts.mean(axis=0)
        centre = to_q(cx, cy)
        tag = texts.REASON_TAGS.get(v.reason)
        if verdict and tag and v.params.get('camera', i) == i:
            f = p.font()
            f.setBold(True)
            f.setPointSizeF(f.pointSizeF() * 1.4)
            p.setFont(f)
            p.setPen(AMBER)
            p.drawText(QRectF(centre.x() - 80, centre.y() - 14, 160, 28), Qt.AlignCenter, tag)
            f.setBold(False)
            f.setPointSizeF(f.pointSizeF() / 1.4)
            p.setFont(f)
        return centre

    @staticmethod
    def _paint_arrow(p, a, b) -> None:
        """Draw a hint arrow from a to b with a dashed target circle."""
        d = b - a
        L = math.hypot(d.x(), d.y())
        pen = QPen(QColor(0, 200, 255), 3)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if L > 30:
            ux, uy = d.x() / L, d.y() / L
            tip = b - QPointF(ux, uy) * 22
            p.drawLine(a, tip)
            left = tip - QPointF(ux * 14 - uy * 8, uy * 14 + ux * 8)
            right = tip - QPointF(ux * 14 + uy * 8, uy * 14 - ux * 8)
            p.setBrush(QBrush(QColor(0, 200, 255)))
            p.drawPolygon(QPolygonF([tip, left, right]))
            p.setBrush(Qt.NoBrush)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawEllipse(b, 20.0, 20.0)


class IndicatorPanel(QWidget):
    """Progress bars of the guidance indicators, per camera, each with its hint line."""

    IDS = tuple(texts.INDICATOR_LABELS)

    def __init__(self, parent=None):
        """Create the panel for one camera."""
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setColumnStretch(1, 1)
        self._rows = {}
        self.set_cameras(1)

    def set_cameras(self, n: int) -> None:
        """Rebuild the rows for n cameras."""
        if len(self._rows) == n * len(self.IDS):
            return
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._rows = {}
        row = 0
        for cam in range(n):
            if n > 1:
                title = QLabel(f'<b>{texts.CAMERA_NAMES[2][cam]} camera</b>')
                self._grid.addWidget(title, row, 0, 1, 2)
                row += 1
            for ind in self.IDS:
                label = QLabel(texts.INDICATOR_LABELS[ind])
                label.setToolTip(texts.INDICATOR_HELP[ind])
                bar = QProgressBar()
                bar.setRange(0, 1000)
                bar.setTextVisible(True)
                bar.setToolTip(texts.INDICATOR_HELP[ind])
                hint = QLabel('')
                hint.setWordWrap(True)
                hint.setStyleSheet('color: gray')
                self._grid.addWidget(label, row, 0)
                self._grid.addWidget(bar, row, 1)
                self._grid.addWidget(hint, row + 1, 0, 1, 2)
                row += 2
                self._rows[(cam, ind)] = (bar, hint, [None])
                self._style(bar, 0.0, self._rows[(cam, ind)][2])

    @staticmethod
    def _style(bar, p, cache) -> None:
        """Set the chunk color of a bar (only when it changes)."""
        color = _bar_color(p)
        if cache[0] != color:
            cache[0] = color
            bar.setStyleSheet('QProgressBar {border: 1px solid #888; border-radius: 3px; '
                              'text-align: center} QProgressBar::chunk {background: %s}'
                              % color)

    def set_state(self, state) -> None:
        """Show a guidance.GuidanceState (None resets)."""
        hints = {}
        for h in (state.hints if state is not None else ()):
            if h.indicator is not None:
                hints.setdefault((h.camera, h.indicator), texts.hint_text(h))
        inds = {(i.camera, i.id): i for i in (state.indicators if state is not None else ())}
        for key, (bar, hint, cache) in self._rows.items():
            ind = inds.get(key)
            progress = ind.progress if ind is not None else 0.0
            bar.setValue(int(round(1000 * progress)))
            bar.setFormat(texts.indicator_value(ind) if ind is not None else '')
            self._style(bar, progress, cache)
            hint.setText(hints.get(key, ''))
            hint.setVisible(bool(hint.text()))


class SampleList(QWidget):
    """Captured samples with Remove / Remove last / Clear buttons."""

    def __init__(self, parent=None):
        """Create the list."""
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel('Samples: 0')
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setFont(_monospace())
        self.list.setMinimumHeight(160)
        self.remove_btn = QPushButton('Remove')
        self.remove_last_btn = QPushButton('Remove last')
        self.clear_btn = QPushButton('Clear')
        row = QHBoxLayout()
        for b in (self.remove_btn, self.remove_last_btn, self.clear_btn):
            row.addWidget(b)
        lay.addWidget(self.title)
        lay.addWidget(self.list, 1)
        lay.addLayout(row)
        QShortcut(QKeySequence(Qt.Key_Delete), self.list, self.remove_btn.click,
                  context=Qt.WidgetShortcut)

    def set_samples(self, snapshot, result=None, min_n: int = 0, max_n: int = 0) -> None:
        """Show the samples of a DBSnapshot, annotated with a CalibrationResult if given."""
        samples = snapshot.samples if snapshot is not None else ()
        self.title.setText(f'Samples: {len(samples)} (min {min_n}, max {max_n})')
        selected = set(self.selected_ids())
        self.list.clear()
        for s in samples:
            f = next((f for f in s.features if f is not None), None) or {}
            text = texts.fmt(texts.SAMPLE_ROW, {'id': s.id, 'x': f.get('x'), 'y': f.get('y'),
                                                'size': f.get('size'),
                                                'tx': f.get('tilt_x_deg'),
                                                'ty': f.get('tilt_y_deg')})
            if len(s.views) > 1:
                text += ' · ' + '+'.join(n[0] for n, v in zip(('L', 'R'), s.views)
                                         if v is not None)
            if result is not None:
                text += self._annotation(s.id, result)
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, s.id)
            self.list.addItem(item)
            item.setSelected(s.id in selected)

    @staticmethod
    def _annotation(sid, result) -> str:
        """Return the result suffix of a sample row."""
        reason = result.rejected.get(sid)
        if reason is not None:
            return ' · ' + texts.REJECTS.get(reason, reason)
        rms = [pv[sid] for pv in result.per_view_rms if sid in pv]
        return texts.fmt(' · {rms:.2f} px', {'rms': max(rms)}) if rms else ' · not used'

    def selected_ids(self) -> list:
        """Return the ids of the selected samples."""
        return [it.data(Qt.UserRole) for it in self.list.selectedItems()]

    def select_id(self, sid: int) -> None:
        """Select and show the sample with the given id."""
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole) == sid:
                self.list.setCurrentItem(it)
                self.list.scrollToItem(it)
                return


class ErrorBars(QWidget):
    """Per-view RMS bars (rejected views greyed) with a median line; click picks a sample."""

    picked = Signal(int)

    def __init__(self, title: str = '', parent=None):
        """Create an empty chart."""
        super().__init__(parent)
        self.title = title
        self._data = []       # [(sample id, rms | None, reject reason | None)]
        self.setMinimumHeight(120)
        self.setMouseTracking(True)

    def set_data(self, data) -> None:
        """Set the bars: iterable of (sample id, rms or None, reject reason or None)."""
        self._data = sorted(data)
        self.update()

    def _plot_rect(self) -> QRectF:
        """Return the area of the bars."""
        return QRectF(self.rect()).adjusted(44, 26, -6, -8)

    def _index_at(self, x) -> int:
        """Return the bar index under widget x, or -1."""
        r = self._plot_rect()
        if not self._data or not r.left() <= x < r.right():
            return -1
        return min(len(self._data) - 1, int((x - r.left()) / r.width() * len(self._data)))

    def paintEvent(self, event):
        """Draw the bars, the median line and the axis labels."""
        p = QPainter(self)
        r = self._plot_rect()
        pal = self.palette()
        p.setPen(pal.windowText().color())
        p.drawText(QRectF(0, 0, self.width(), 18), Qt.AlignLeft, self.title)
        rms = [e for _, e, _ in self._data if e is not None]
        if not rms:
            p.end()
            return
        top = max(rms) * 1.1
        med = float(np.median(rms))
        bw = r.width() / len(self._data)
        for k, (_, e, why) in enumerate(self._data):
            hgt = r.height() * (e / top if e is not None else 1.0)
            color = QColor(90, 140, 220) if why is None else QColor(160, 160, 160, 110)
            p.fillRect(QRectF(r.x() + k * bw + 0.5, r.bottom() - hgt, max(1.0, bw - 1), hgt),
                       color)
        y = r.bottom() - r.height() * med / top
        p.setPen(QPen(RED, 1, Qt.DashLine))
        p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(pal.windowText().color())
        fh = p.fontMetrics().height()
        p.drawText(QRectF(0, r.top() - fh / 2, 40, fh), Qt.AlignRight, f'{top:.2f}')
        if abs(y - r.top()) > fh:
            p.drawText(QRectF(0, y - fh / 2, 40, fh), Qt.AlignRight, f'{med:.2f}')
        p.drawText(QRectF(0, r.bottom() - fh, 40, fh), Qt.AlignRight, 'px')
        p.end()

    def mouseMoveEvent(self, event):
        """Show the tooltip of the bar under the mouse."""
        k = self._index_at(event.pos().x())
        if k < 0:
            QToolTip.hideText()
            return
        sid, e, why = self._data[k]
        text = f'#{sid}: ' + (f'{e:.3f} px' if e is not None else '')
        if why is not None:
            text += ' ' + texts.REJECTS.get(why, why)
        QToolTip.showText(event.globalPos(), text, self)

    def mousePressEvent(self, event):
        """Emit picked with the sample id of the clicked bar."""
        k = self._index_at(event.pos().x())
        if k >= 0:
            self.picked.emit(int(self._data[k][0]))


_DIST_NAMES = {'plumb_bob': ('k1', 'k2', 'p1', 'p2', 'k3'),
               'rational_polynomial': ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'),
               'equidistant': ('k1', 'k2', 'k3', 'k4')}


def _table(headers, rows) -> QTableWidget:
    """Return a read-only table sized to its content."""
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setFont(_monospace())
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            it = QTableWidgetItem(v)
            it.setTextAlignment((Qt.AlignLeft if c == 0 else Qt.AlignRight) | Qt.AlignVCenter)
            t.setItem(r, c, it)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    t.resizeRowsToContents()
    height = t.horizontalHeader().height() + sum(t.rowHeight(r) for r in range(len(rows)))
    t.setFixedHeight(height + 2 * t.frameWidth())
    t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return t


def _param_rows(cam) -> list:
    """Return the (name, value, sigma, 95% CI %) rows of a CameraCalib."""
    rows = []
    K = cam.K
    values = [('fx', K[0, 0]), ('fy', K[1, 1]), ('cx', K[0, 2]), ('cy', K[1, 2])]
    names = _DIST_NAMES.get(cam.distortion_model, tuple(f'd{i}' for i in range(len(cam.D))))
    values += list(zip(names, np.ravel(cam.D)))
    for name, v in values:
        s = cam.sigmas.get(name)
        ci = 196.0 * s / abs(v) if s is not None and abs(v) > 1e-12 else None
        fmt_v = '{:.3f}' if name in ('fx', 'fy', 'cx', 'cy') else '{:.5g}'
        sig = ('fixed' if cam.sigmas else '') if s is None else f'{s:.2g}'
        rows.append((name, fmt_v.format(v), sig, f'{ci:.3g}' if ci is not None else ''))
    rows.append(('RMS [px]', texts.fmt('{r:.4f}', {'r': cam.rms}), '', ''))
    rows.append(('views', str(cam.n_views), '', ''))
    return rows


def _stereo_rows(st) -> list:
    """Return the rows of the stereo extrinsics table."""
    rvec = cv2.Rodrigues(np.asarray(st.R, np.float64))[0].ravel()
    rows = []
    for k, name in enumerate(('rx', 'ry', 'rz')):
        s = st.sigmas.get(name)
        rows.append((name + ' [deg]', f'{math.degrees(rvec[k]):.4f}',
                     texts.fmt('{s:.2g}', {'s': math.degrees(s) if s is not None else None})))
    for k, name in enumerate(('tx', 'ty', 'tz')):
        rows.append((name + ' [m]', f'{float(np.ravel(st.T)[k]):.6f}',
                     texts.fmt('{s:.2g}', {'s': st.sigmas.get(name)})))
    rows.append(('baseline [m]', f'{float(np.linalg.norm(st.T)):.6f}', ''))
    rows.append(('stereo RMS [px]', f'{st.rms:.4f}', ''))
    rows.append(('pairs', str(st.n_pairs), ''))
    return rows


class ResultsPanel(QWidget):
    """Calibration results, per-view errors, rectification controls, live check and actions."""

    def __init__(self, parent=None):
        """Create the panel."""
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self.header = QLabel('No calibration yet.')
        self.header.setWordWrap(True)
        lay.addWidget(self.header)
        self._tables = QVBoxLayout()
        lay.addLayout(self._tables)
        self.bars = []
        self.warnings = QLabel('')
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet('color: #c07000')
        lay.addWidget(self.warnings)

        box = QGroupBox('Rectification')
        form = QFormLayout(box)
        self.policy = QComboBox()
        self.stereo_label = QLabel(texts.POLICY_NAMES['stereo'])
        self.stereo_label.setToolTip(texts.POLICY_TIPS['stereo'])
        self.alpha = QSlider(Qt.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha_label = QLabel('0.00')
        arow = QHBoxLayout()
        arow.addWidget(self.alpha, 1)
        arow.addWidget(self.alpha_label)
        self.zero_disparity = QCheckBox('Zero disparity')
        self.rectified = QCheckBox('Rectified view (R)')
        self.P = QPlainTextEdit()
        self.P.setReadOnly(True)
        self.P.setFont(_monospace())
        self.P.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.P.setFixedHeight(8 * self.P.fontMetrics().height())
        self.rect_warnings = QLabel('')
        self.rect_warnings.setWordWrap(True)
        self.rect_warnings.setStyleSheet('color: #c07000')
        form.addRow('Policy', self.policy)
        form.addRow('', self.stereo_label)
        form.addRow('alpha', arow)
        form.addRow('', self.zero_disparity)
        form.addRow('', self.rectified)
        form.addRow(self.P)
        form.addRow(self.rect_warnings)
        self._rect_form = form
        lay.addWidget(box)

        live = QGroupBox('Live check (median of the last 30 frames)')
        self.live = QLabel('')
        QVBoxLayout(live).addWidget(self.live)
        self.live.setFont(_monospace())
        self.live.setToolTip('\n'.join(f'{n}: {tip}' for n, _, tip in texts.VALIDATION.values()))
        lay.addWidget(live)

        cbox = QGroupBox('Commit (SetCameraInfo)')
        self._commit_form = QFormLayout(cbox)
        self.services = []
        lay.addWidget(cbox)

        buttons = QGridLayout()
        self.save_btn = QPushButton('Save')
        self.dataset_btn = QPushButton('Save dataset')
        self.commit_btn = QPushButton('Commit')
        self.recalibrate_btn = QPushButton('Recalibrate')
        self.back_btn = QPushButton('Back')
        for k, b in enumerate((self.save_btn, self.dataset_btn, self.commit_btn,
                               self.recalibrate_btn, self.back_btn)):
            buttons.addWidget(b, k // 3, k % 3)
        lay.addLayout(buttons)
        lay.addStretch(1)

    def set_mode(self, n_cameras: int, policies: tuple) -> None:
        """Configure mono/stereo rows and the selectable policies (policy codes)."""
        stereo = n_cameras > 1
        for w in (self.policy, self._rect_form.labelForField(self.policy)):
            w.setVisible(not stereo)
        self.stereo_label.setVisible(stereo)
        self.zero_disparity.setVisible(stereo)
        current = self.policy.currentData()
        self.policy.blockSignals(True)
        self.policy.clear()
        for code in policies:
            self.policy.addItem(texts.POLICY_NAMES[code], code)
            self.policy.setItemData(self.policy.count() - 1, texts.POLICY_TIPS[code],
                                    Qt.ToolTipRole)
        k = self.policy.findData(current)
        self.policy.setCurrentIndex(max(0, k))
        self.policy.blockSignals(False)
        while len(self.services) < n_cameras:
            combo = QComboBox()
            combo.setEditable(True)
            combo.setMinimumContentsLength(20)
            self.services.append(combo)
            self._commit_form.addRow(combo)
        for k, combo in enumerate(self.services):
            combo.setVisible(k < n_cameras)
            label = texts.CAMERA_NAMES[2][k] if n_cameras > 1 else 'Service'
            combo.setToolTip(f'{label} SetCameraInfo service')

    def set_services(self, names, defaults) -> None:
        """Fill the service comboboxes with discovered names, keeping the edited text."""
        for combo, default in zip(self.services, defaults):
            text = combo.currentText() or default
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(list(names))
            combo.setEditText(text)
            combo.blockSignals(False)

    def service_names(self, n: int) -> tuple:
        """Return the selected service names of the first n cameras."""
        return tuple(c.currentText().strip() for c in self.services[:n])

    def _clear_tables(self) -> None:
        """Remove the parameter tables and error bars."""
        while self._tables.count():
            w = self._tables.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.bars = []

    def set_result(self, result, total: int, source: str = '') -> None:
        """Show a CalibrationResult (None clears); source names a loaded file instead."""
        self._clear_tables()
        if result is None:
            self.header.setText(source or 'No calibration yet.')
            self.warnings.setText('')
            return
        used = set().union(*[set(u) for u in result.used_ids])
        self.header.setText(texts.fmt(texts.RESULT_HEADER, {
            'used': len(used), 'total': total, 'rms': result.rms,
            'duration': result.duration_s}))
        self.show_cameras(result.cameras)
        if result.stereo is not None:
            self._tables.addWidget(QLabel('<b>Right camera w.r.t. left</b>'))
            self._tables.addWidget(_table(('parameter', 'value', '±σ'),
                                          _stereo_rows(result.stereo)))
        n = len(result.cameras)
        for k, pv in enumerate(result.per_view_rms):
            name = texts.camera_prefix(k, n) + 'per-view RMS (grey: not used)'
            bars = ErrorBars(name)
            ids = set(pv) | set(result.rejected)
            bars.set_data((s, pv.get(s), result.rejected.get(s) if s not in pv else None)
                          for s in ids)
            self.bars.append(bars)
            self._tables.addWidget(bars)
        self.warnings.setText('\n'.join(
            texts.warning_text(c, p) for c, p in result.warnings))

    def show_cameras(self, cameras) -> None:
        """Add one parameter table per CameraCalib."""
        n = len(cameras)
        for k, cam in enumerate(cameras):
            title = texts.camera_prefix(k, n) + f'{cam.model}, {cam.distortion_model}, ' \
                f'{cam.size[0]}×{cam.size[1]}'
            self._tables.addWidget(QLabel(f'<b>{title}</b>'))
            self._tables.addWidget(_table(('parameter', 'value', '±σ', '95% CI [%]'),
                                          _param_rows(cam)))

    def set_rect(self, rect) -> None:
        """Show the P matrices and warnings of a Rectification (None clears)."""
        if rect is None:
            self.P.setPlainText('')
            self.rect_warnings.setText('')
            return
        n = len(rect.P)
        lines = []
        for k, P in enumerate(rect.P):
            lines.append(texts.camera_prefix(k, n) + 'P')
            lines += ['[' + ' '.join(f'{v:10.3f}' for v in row) + ' ]' for row in P]
        self.P.setPlainText('\n'.join(lines))
        msgs = [texts.warning_text(c, p) for c, p in rect.warnings]
        if rect.policy != self.policy.currentData() and n == 1 and self.policy.isVisible():
            msgs.append('Applied policy: ' + texts.POLICY_NAMES.get(rect.policy, rect.policy))
        self.rect_warnings.setText('\n'.join(msgs))

    def set_validation(self, rows) -> None:
        """Show live check rows: iterable of (label, text)."""
        self.live.setText('\n'.join(f'{k:<20} {v}' for k, v in rows) or 'No board in view.')


# Setup form: (group, title, ((key, label), ...)); widgets come from the params spec.
FORM = (
    ('source', 'Source', (
        ('source.mode', 'Setup'), ('source.transport', 'Transport'),
        ('source.topic', 'Topic'), ('source.left_topic', 'Left topic'),
        ('source.right_topic', 'Right topic'), ('source.sync_slop', 'Sync slop [s]'))),
    ('board', 'Board', (
        ('board.type', 'Type'), ('board.cols', 'Columns'), ('board.rows', 'Rows'),
        ('board.square_size', 'Square size [m]'), ('board.marker_size', 'Marker size [m]'),
        ('board.dictionary', 'Dictionary'), ('board.legacy_pattern', 'Legacy pattern'),
        ('board.detector', 'Detector'), ('detection.max_pixels', 'Detection pixels'))),
    ('calib', 'Calibration', (
        ('camera.model', 'Camera model'), ('calib.distortion_model', 'Distortion'),
        ('calib.fix_k3', 'Fix k3'), ('calib.fix_aspect_ratio', 'Fix aspect ratio'),
        ('calib.fix_principal_point', 'Fix principal point'),
        ('calib.zero_tangent_dist', 'Zero tangential dist.'),
        ('calib.max_views', 'Max views'),
        ('rectify.fisheye_max_fov_deg', 'Fisheye max FOV [deg]'))),
    ('capture', 'Capture & gates', (
        ('capture.mode', 'Capture mode'), ('capture.min_samples', 'Min samples'),
        ('capture.max_samples', 'Max samples'), ('capture.keep_images', 'Keep images'),
        ('gates.max_blur_px', 'Max blur [px]'), ('gates.max_motion_px', 'Max motion [px]'),
        ('gates.max_tilt_deg', 'Max tilt [deg]'), ('gates.min_novelty', 'Min novelty'),
        ('guidance.target_rel_ci', 'Target rel. 95% CI'))),
    ('output', 'Output', (('camera.name', 'Camera name'), ('output.dir', 'Directory'))),
)

TOPIC_KEYS = ('source.topic', 'source.left_topic', 'source.right_topic')

# key -> predicate on the current values: the field is shown only when it holds
VISIBLE = {
    'source.topic': lambda v: v['source.mode'] == 'mono',
    'source.left_topic': lambda v: v['source.mode'] == 'stereo',
    'source.right_topic': lambda v: v['source.mode'] == 'stereo',
    'source.sync_slop': lambda v: v['source.mode'] == 'stereo',
    'board.marker_size': lambda v: v['board.type'] == 'charuco',
    'board.dictionary': lambda v: v['board.type'] == 'charuco',
    'board.legacy_pattern': lambda v: v['board.type'] == 'charuco',
    'board.detector': lambda v: v['board.type'] == 'chessboard',
    'detection.max_pixels': lambda v: v['board.type'] == 'chessboard',
    'gates.max_blur_px': lambda v: v['board.type'] == 'chessboard',
    'calib.distortion_model': lambda v: v['camera.model'] == 'pinhole',
    'calib.fix_aspect_ratio': lambda v: v['camera.model'] == 'pinhole',
    'calib.zero_tangent_dist': lambda v: v['camera.model'] == 'pinhole',
    'calib.fix_k3': lambda v: (v['camera.model'] == 'pinhole'
                               and v['calib.distortion_model'] == 'plumb_bob'),
    'rectify.fisheye_max_fov_deg': lambda v: v['camera.model'] == 'fisheye',
}


class SetupForm(QWidget):
    """Settings form generated from FORM and the params spec; emits changed(key, value)."""

    changed = Signal(str, object)
    refresh_requested = Signal()

    def __init__(self, spec: dict, parent=None):
        """Create the form from the params spec (settings.load_spec())."""
        super().__init__(parent)
        self._spec = spec
        self._values = {k: e['default_value'] for k, e in spec.items()}
        self._fields = {}      # key -> (editor, label, form, row widget)
        self._avail = {}
        self._discovered = {'raw': (), 'compressed': ()}
        self.groups = {}
        lay = QVBoxLayout(self)
        for group, title, entries in FORM:
            box = QGroupBox(title.replace('&', '&&'))
            form = QFormLayout(box)
            for key, label in entries:
                editor, row = self._editor(key)
                tip = spec[key]['description']
                if spec[key].get('constraints'):
                    tip += '\n' + spec[key]['constraints']
                editor.setToolTip(tip)
                lab = QLabel(label)
                lab.setToolTip(tip)
                form.addRow(lab, row)
                self._fields[key] = (editor, lab, row)
            if group == 'source':
                self.refresh_btn = QPushButton('Refresh topics')
                self.refresh_btn.clicked.connect(self.refresh_requested.emit)
                form.addRow('', self.refresh_btn)
            self.groups[group] = box
            lay.addWidget(box)
        lay.addStretch(1)

    def _editor(self, key) -> tuple:
        """Return (editor, row widget) for a spec key."""
        e = self._spec[key]
        if key in CHOICES or key == 'board.dictionary':
            items = CHOICES.get(key) or sorted(n for n in dir(cv2.aruco) if n.startswith('DICT_'))
            w = QComboBox()
            for it in items:
                w.addItem(str(it), it)
            w.activated.connect(lambda _, k=key, w=w: self._emit(k, w.currentData()))
            return w, w
        if key in TOPIC_KEYS:
            w = QComboBox()
            w.setEditable(True)
            w.setInsertPolicy(QComboBox.NoInsert)
            w.setMinimumContentsLength(18)
            avail = QLabel('')
            avail.setStyleSheet('color: gray')
            self._avail[key] = avail
            w.activated.connect(lambda _, k=key, w=w: self._emit(k, w.currentText().strip()))
            w.lineEdit().editingFinished.connect(
                lambda k=key, w=w: self._emit(k, w.currentText().strip()))
            w.currentTextChanged.connect(lambda _, k=key: self._update_avail(k))
            row = QWidget()
            v = QVBoxLayout(row)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(w)
            v.addWidget(avail)
            return w, row
        t = e['type']
        if t == 'bool':
            w = QCheckBox()
            w.toggled.connect(lambda b, k=key: self._emit(k, bool(b)))
            return w, w
        if t in ('integer', 'double'):
            w = QSpinBox() if t == 'integer' else QDoubleSpinBox()
            if t == 'double':
                w.setDecimals(6 if (e.get('min_value') or 1.0) < 1e-3 else 4)
                w.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
            cast = int if t == 'integer' else float
            w.setRange(cast(e.get('min_value', -10 ** 9)), cast(e.get('max_value', 10 ** 9)))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda x, k=key: self._emit(k, x))
            return w, w
        w = QLineEdit()
        w.editingFinished.connect(lambda k=key, w=w: self._emit(k, w.text().strip()))
        if key != 'output.dir':
            return w, w
        w.setPlaceholderText('<current directory>/calibrations')
        browse = QPushButton('Browse…')
        browse.clicked.connect(lambda _=False, w=w: self._browse(w))
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w, 1)
        h.addWidget(browse)
        return w, row

    def _browse(self, edit) -> None:
        """Pick the output directory."""
        d = QFileDialog.getExistingDirectory(self, 'Output directory', edit.text() or '.')
        if d:
            edit.setText(d)
            self._emit('output.dir', d)

    def _emit(self, key, value) -> None:
        """Store and emit a changed value (only if it differs)."""
        if self._values.get(key) == value:
            return
        self._values[key] = value
        self._update_visibility()
        self.changed.emit(key, value)

    def set_values(self, values: dict) -> None:
        """Show the given {key: value} settings without emitting changed."""
        self._values.update({k: v for k, v in values.items() if k in self._spec})
        for key, (w, _, _) in self._fields.items():
            v = self._values[key]
            w.blockSignals(True)
            if isinstance(w, QComboBox):
                k = w.findData(v)
                if k < 0 and w.isEditable():
                    w.setEditText(str(v))
                elif k < 0:
                    w.addItem(str(v), v)
                    w.setCurrentIndex(w.count() - 1)
                else:
                    w.setCurrentIndex(k)
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.setValue(v)
            else:
                w.setText(str(v))
            w.blockSignals(False)
        for key in TOPIC_KEYS:
            self._update_avail(key)
        self._update_visibility()

    def set_topics(self, discovered: dict) -> None:
        """Fill the topic comboboxes with discovered base topics {'raw': (...), ...}."""
        self._discovered = {k: tuple(v) for k, v in discovered.items()}
        names = sorted(set().union(*self._discovered.values())) if self._discovered else []
        for key in TOPIC_KEYS:
            w = self._fields[key][0]
            w.blockSignals(True)
            w.clear()
            w.addItems(names)
            w.setEditText(str(self._values[key]))
            w.blockSignals(False)
            self._update_avail(key)

    def _update_avail(self, key) -> None:
        """Show which transports publish the topic typed in a topic combobox."""
        text = self._fields[key][0].currentText().strip()
        have = [t for t in ('raw', 'compressed') if text in self._discovered.get(t, ())]
        self._avail[key].setText('available: ' + (', '.join(have) or 'none (not discovered)'))

    def _update_visibility(self) -> None:
        """Hide the fields that do not apply to the current choices."""
        for key, (_, lab, row) in self._fields.items():
            pred = VISIBLE.get(key)
            show = pred is None or pred(self._values)
            lab.setVisible(show)
            row.setVisible(show)

    def set_group_enabled(self, group: str, enabled: bool) -> None:
        """Enable or disable a FORM group."""
        self.groups[group].setEnabled(enabled)
