"""
Main window of the calibration GUI: state machine, 15 ms polling tick and user flows.

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

from collections import deque
from dataclasses import replace
import os
import sys
import time
import traceback

import cv2
from dua_camera_calibration import calibration, fileio, rectification, texts
from dua_camera_calibration.pipeline import JobRunner, Session, ViewConfig, write_run
from dua_camera_calibration.rectification import POLICIES, Rectification
from dua_camera_calibration.settings import board_error, load_spec
from dua_camera_calibration.widgets import (IndicatorPanel, LiveView, ResultsPanel, SampleList,
                                            SetupForm)
import numpy as np
from python_qt_binding.QtCore import PYQT_VERSION_STR, QSettings, Qt, QT_VERSION_STR, QTimer
from python_qt_binding.QtWidgets import (QAction, QApplication, QDockWidget, QFileDialog, QLabel,
                                         QMainWindow, QMessageBox, QPlainTextEdit,
                                         QProgressBar, QPushButton, QScrollArea, QTabWidget,
                                         QTextBrowser, QVBoxLayout, QWidget)

STATES = ('SETUP', 'CAPTURING', 'CALIBRATING', 'REVIEW', 'VERIFY')

# state -> controls that may be enabled (dynamic predicates in MainWindow._predicates)
ENABLED = {
    'SETUP': {'group.source', 'group.board', 'group.calib', 'group.capture', 'group.output',
              'start', 'clear', 'load', 'reset'},
    'CAPTURING': {'group.calib', 'group.capture', 'group.output', 'stop', 'pause', 'capture',
                  'remove', 'clear', 'calibrate', 'dataset', 'reset'},
    'CALIBRATING': {'group.output', 'cancel'},
    'REVIEW': {'group.calib', 'group.output', 'remove', 'calibrate', 'recalibrate', 'rect',
               'save', 'dataset', 'commit', 'load', 'back', 'reset'},
    'VERIFY': {'group.source', 'group.board', 'group.output', 'rect', 'save', 'commit', 'load',
               'back', 'reset'},
}

TICK_MS = 15
COMMIT_TIMEOUT_S = 5.0
NO_DATA_S = 2.0
STALL_S = 5.0


def pick_service(topic: str, services) -> str:
    """Return the service sharing the longest '/'-segment prefix with topic ('' if none)."""
    seg = [s for s in topic.split('/') if s]
    best, key = '', None
    for name in services:
        parts = [s for s in name.split('/') if s]
        n = 0
        while n < min(len(seg), len(parts)) and seg[n] == parts[n]:
            n += 1
        k = (n, -len(name))
        if n > 0 and (key is None or k > key):
            best, key = name, k
    return best


def _calibrate_job(token, snapshot, board, cfg, progress):
    """Run calibration.calibrate on the job thread; return (token, result)."""
    return token, calibration.calibrate(snapshot, board, cfg, progress=progress)


def _rectify_job(cams, stereo, cfg, file_rect, sizes):
    """Compute the rectification (or keep the file one) and the display maps if sizes."""
    rect = file_rect if cfg.policy == 'file' else rectification.rectify(cams, stereo, cfg)
    maps = None
    if sizes:
        maps = tuple(rectification.make_maps(c, R, P, s)
                     for c, R, P, s in zip(cams, rect.R, rect.P, sizes))
    return rect, maps, sizes


def _save_files(out_dir, names, cams, rect) -> str:
    """Write the camera YAML files of a loaded (VERIFY) calibration into a new run dir."""
    path = fileio.run_dir(out_dir, '_'.join(names))
    files = (names[0],) if len(cams) == 1 else ('left', 'right')
    for f, name, cam, R, P in zip(files, names, cams, rect.R, rect.P):
        fileio.write_camera_yaml(os.path.join(path, f + '.yaml'), name, cam, R, P,
                                 {'policy': rect.policy, 'alpha': rect.alpha})
    return path


def _is_tmp(path: str) -> bool:
    """Return True if path resolves under /tmp."""
    real = os.path.realpath(path)
    return real == '/tmp' or real.startswith('/tmp/')


class MainWindow(QMainWindow):
    """Calibration main window; every method runs on the Qt thread except report_error."""

    def __init__(self, node, settings, warnings=(), spec=None):
        """Create the window for a ros_io.CalibrationNode (or duck type) and Settings."""
        super().__init__()
        self.node = node
        self.spin_thread = None          # set by the app entry for the liveness check
        self._settings = settings
        self._spec = spec or load_spec()
        self._session = Session(settings)
        self._jobs = JobRunner()
        self._state = 'SETUP'
        self._paused = False
        self._gen = 0
        self._versions = {'display': 0, 'overlay': 0, 'guidance': 0, 'db': -1}
        self._frame_t = None             # monotonic time of the last frame of this generation
        self._overlay_t = 0.0
        self._subscribed_t = time.monotonic()
        self._src_sizes = None
        self._view_sizes = ()
        self._source_status = ('WAITING', {})
        self._banner_color = None
        self._overlay = None
        self._guidance = None
        self._result = None
        self._calib_total = 0
        self._calib_token = 0
        self._calib_submitted = 0
        self._calib_t0 = 0.0
        self._stage = ''
        self._cams = None
        self._stereo = None
        self._file_rect = None
        self._file_names = ()
        self._rect = None
        self._rect_pending = False
        self._saved = True
        self._services_ready = False
        self._commit_pending = None
        self._val_hist = deque(maxlen=30)
        self._errors = deque(maxlen=200)     # appended from any thread (report_error)
        self._seen_session_errors = set()
        self._n_errors = 0
        self._session_err_total = 0
        self._force_quit = False
        self._shut = False
        self._slow_t = {'stats': 0.0, 'checks': 0.0}
        self._build_ui()
        self.form.set_values(settings.as_dict())
        self._apply_mode()
        for key, value in warnings:
            self.log(f'Invalid parameter {key} = {value!r}: default used.')
        self._refresh_topics()
        QTimer.singleShot(1500, self._refresh_topics)   # DDS discovery is incomplete at first
        self._resubscribe()
        self._set_state('SETUP')
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(TICK_MS)
        qs = QSettings('dotxautomation', 'dua_camera_calibration')
        if qs.value('geometry') is not None:
            self.restoreGeometry(qs.value('geometry'))
            self.restoreState(qs.value('state'))

    # UI construction

    def _action(self, text, slot, shortcut=None, checkable=False) -> QAction:
        """Create a window-wide QAction."""
        a = QAction(text, self)
        if shortcut is not None:
            a.setShortcut(shortcut)
        a.setCheckable(checkable)
        (a.toggled if checkable else a.triggered).connect(slot)
        self.addAction(a)
        return a

    def _build_ui(self) -> None:
        """Create docks, central view, toolbar, menus and status bar."""
        self.setWindowTitle('dotX camera calibration')
        self.resize(1500, 900)
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(2, 2, 2, 2)
        self.view = LiveView()
        self.banner1 = QLabel('')
        self.banner2 = QLabel('')
        for b in (self.banner1, self.banner2):
            b.setWordWrap(True)
            f = b.font()
            f.setPointSizeF(f.pointSizeF() * 1.2)
            b.setFont(f)
        v.addWidget(self.view, 1)
        v.addWidget(self.banner1)
        v.addWidget(self.banner2)
        self.setCentralWidget(central)

        self.form = SetupForm(self._spec)
        self.form.changed.connect(self._on_setting)
        self.form.refresh_requested.connect(self._refresh_topics)
        self._dock('Setup', 'setup', self.form, Qt.LeftDockWidgetArea)

        guidance = QWidget()
        g = QVBoxLayout(guidance)
        self.indicators = IndicatorPanel()
        self.help_btn = QPushButton('What do these mean?')
        self.help_btn.setCheckable(True)
        self.help = QTextBrowser()
        self.help.setHtml(texts.HELP_HTML)
        self.help.setVisible(False)
        self.help_btn.toggled.connect(self.help.setVisible)
        self.samples = SampleList()
        g.addWidget(self.indicators)
        g.addWidget(self.help_btn)
        g.addWidget(self.help, 2)
        g.addWidget(self.samples, 1)
        gscroll = QScrollArea()
        gscroll.setWidgetResizable(True)
        gscroll.setWidget(guidance)
        self.results = ResultsPanel()
        self.tabs = QTabWidget()
        self.tabs.addTab(gscroll, 'Guidance')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.results)
        self.tabs.addTab(scroll, 'Results')
        right = self._dock('Guidance and results', 'right', self.tabs, Qt.RightDockWidgetArea,
                           scroll=False)
        self.resizeDocks([right], [470], Qt.Horizontal)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_dock = self._dock('Log', 'log', self.log_view, Qt.BottomDockWidgetArea,
                                   scroll=False)
        self.log_dock.hide()

        A = self._action
        self.act_start = A('Start capture', lambda: self._set_state('CAPTURING'))
        self.act_stop = A('Stop capture', lambda: self._set_state('SETUP'))
        self.act_pause = A('Pause', self._on_pause, 'P', checkable=True)
        self.act_capture = A('Capture', self._capture_now, Qt.Key_Space)
        self.act_calibrate = A('Calibrate', self._calibrate, 'C')
        self.act_cancel = A('Cancel', self._cancel)
        self.act_reset = A('Reset', self._reset, 'Ctrl+R')
        self.act_reset.setToolTip('Discard samples and calibration, unlock the setup and '
                                  'restart from SETUP (Ctrl+R)')
        self.act_save = A('Save', lambda: self._save(False), 'Ctrl+S')
        self.act_dataset = A('Save dataset', lambda: self._save(True))
        self.act_commit = A('Commit', self._commit)
        self.act_load = A('Load calibration…', self._load, 'Ctrl+O')
        self.act_remove_last = A('Remove last sample', self._remove_last, Qt.Key_Backspace)
        self.act_rectified = A('Rectified view', self.results.rectified.setChecked, 'R',
                               checkable=True)
        self.act_heatmap = A('Coverage heatmap', self._on_heatmap, 'H', checkable=True)
        self.act_log = self.log_dock.toggleViewAction()
        self.act_help = A('Help', self._show_help, 'F1')
        self.act_full = A('Full screen', self._toggle_full, 'F11')
        self.act_escape = A('Cancel / exit full screen', self._escape, Qt.Key_Escape)
        self.act_about = A('About', self._about)
        self.act_quit = A('Quit', self.close, 'Ctrl+Q')

        mb = self.menuBar()
        for title, acts in (
                ('File', (self.act_load, self.act_save, self.act_dataset, self.act_commit, None,
                          self.act_quit)),
                ('Capture', (self.act_start, self.act_stop, self.act_pause, self.act_capture,
                             self.act_remove_last, None, self.act_calibrate, self.act_cancel,
                             None, self.act_reset)),
                ('View', (self.act_rectified, self.act_heatmap, self.act_log, self.act_full)),
                ('Help', (self.act_help, self.act_about))):
            m = mb.addMenu(title)
            for a in acts:
                m.addSeparator() if a is None else m.addAction(a)
        tb = self.addToolBar('Main')
        tb.setObjectName('toolbar')
        for a in (self.act_start, self.act_stop, self.act_pause, self.act_capture,
                  self.act_calibrate, self.act_cancel, self.act_save, self.act_commit,
                  self.act_load, self.act_reset):
            tb.addAction(a)
        self.state_label = QLabel('')
        tb.addSeparator()
        tb.addWidget(self.state_label)
        self._calib_button = tb.widgetForAction(self.act_calibrate)

        self.stats_label = QLabel('')
        self.error_badge = QPushButton('')
        self.error_badge.setFlat(True)
        self.error_badge.setStyleSheet('color: #dc2828; font-weight: bold')
        self.error_badge.clicked.connect(lambda: self.log_dock.setVisible(True))
        self.error_badge.hide()
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumWidth(120)
        self.busy.hide()
        sb = self.statusBar()
        sb.addWidget(self.stats_label, 1)
        sb.addPermanentWidget(self.error_badge)
        sb.addPermanentWidget(self.busy)

        r = self.results
        r.save_btn.clicked.connect(lambda: self._save(False))
        r.dataset_btn.clicked.connect(lambda: self._save(True))
        r.commit_btn.clicked.connect(self._commit)
        r.recalibrate_btn.clicked.connect(self._calibrate)
        r.back_btn.clicked.connect(self._back)
        r.policy.activated.connect(lambda _: self._on_rect_control())
        r.alpha.valueChanged.connect(lambda _: self._on_rect_control())
        r.zero_disparity.toggled.connect(lambda _: self._on_rect_control())
        r.rectified.toggled.connect(self._on_rectified)
        s = self.samples
        s.remove_btn.clicked.connect(lambda: self._remove(s.selected_ids()))
        s.remove_last_btn.clicked.connect(self._remove_last)
        s.clear_btn.clicked.connect(self._clear)

        self._rect_timer = QTimer(self)
        self._rect_timer.setSingleShot(True)
        self._rect_timer.timeout.connect(self._submit_rectify)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._update_view_sizes)
        self.view.installEventFilter(self)

        self._controls = {
            'start': (self.act_start,), 'stop': (self.act_stop,), 'pause': (self.act_pause,),
            'capture': (self.act_capture,), 'calibrate': (self.act_calibrate,),
            'cancel': (self.act_cancel,), 'recalibrate': (r.recalibrate_btn,),
            'remove': (s.remove_btn, s.remove_last_btn, self.act_remove_last),
            'clear': (s.clear_btn,), 'save': (self.act_save, r.save_btn),
            'dataset': (self.act_dataset, r.dataset_btn),
            'commit': (self.act_commit, r.commit_btn), 'load': (self.act_load,),
            'back': (r.back_btn,), 'reset': (self.act_reset,),
            'rect': (r.policy, r.alpha, r.zero_disparity, r.rectified, self.act_rectified),
        }

    def _dock(self, title, name, widget, area, scroll=True) -> QDockWidget:
        """Add a dock holding widget (in a scroll area if scroll)."""
        d = QDockWidget(title, self)
        d.setObjectName(name)
        if scroll:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(widget)
            widget = sa
        d.setWidget(widget)
        self.addDockWidget(area, d)
        return d

    def eventFilter(self, obj, event):
        """Debounce live view resizes into a display size update."""
        if obj is self.view and event.type() == event.Resize:
            self._resize_timer.start(150)
        return False

    # logging and errors

    def log(self, msg: str, error: bool = False, count: bool = True) -> None:
        """Append a line to the log dock and the ROS logger (Qt thread)."""
        self.log_view.appendPlainText(time.strftime('%H:%M:%S ') + msg)
        logger = getattr(self.node, 'get_logger', None)
        if logger is not None:
            try:
                (logger().error if error else logger().info)(msg)
            except Exception:
                pass
        if error and count:
            self._n_errors += 1
        if error:
            self._update_badge()

    def _update_badge(self) -> None:
        """Show the number of errors (GUI errors plus every pipeline occurrence)."""
        n = self._n_errors + self._session_err_total
        self.error_badge.setText(f'! {n} errors')
        self.error_badge.setVisible(n > 0)

    def report_error(self, where: str, exc) -> None:
        """Queue an error for the log and status bar; safe from any thread."""
        if isinstance(exc, BaseException):
            msg = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        else:
            msg = str(exc)
        self._errors.append((where, msg.strip()))

    def _dialog(self, title: str, text: str) -> None:
        """Show a warning dialog outside the tick."""
        QTimer.singleShot(0, lambda: QMessageBox.warning(self, title, text))

    # state machine

    def _set_state(self, state: str) -> None:
        """Enter a state and apply its capture mode and enabled controls."""
        self._state = state
        if state == 'SETUP':
            self.act_pause.setChecked(False)   # a new capture never starts paused
        self._apply_capture()
        if state in ('REVIEW', 'VERIFY'):
            self.tabs.setCurrentIndex(1)
            self._refresh_services()
        elif state in ('SETUP', 'CAPTURING'):
            self.tabs.setCurrentIndex(0)
        self.state_label.setText(f'  state: {state}  ')
        self.view.show_verdict = state in ('SETUP', 'CAPTURING')
        self._refresh_enabled()
        self._update_banner()

    def _n_samples(self) -> int:
        """Return the number of samples in the session database."""
        db = getattr(self._session, 'db', None)
        return len(db.snapshot().samples) if db is not None else 0

    def _predicates(self) -> dict:
        """Return the dynamic enable conditions of the controls."""
        n = self._n_samples()
        min_views = self._settings.calib_config().min_views
        return {
            'group.source': n == 0, 'group.board': n == 0,
            'start': self._frame_t is not None, 'remove': n > 0, 'clear': n > 0,
            'calibrate': n >= min_views and not self._jobs.busy('calibrate'),
            'recalibrate': n >= min_views and not self._jobs.busy('calibrate'),
            'dataset': n > 0 and self._settings['capture.keep_images'],
            'commit': self._services_ready and self._rect is not None
            and self._commit_pending is None,
            'save': self._rect is not None, 'load': n == 0 or self._state != 'SETUP',
        }

    def _refresh_enabled(self) -> None:
        """Apply the ENABLED table and the predicates to every control."""
        allowed, pred = ENABLED[self._state], self._predicates()
        for name, widgets in self._controls.items():
            on = name in allowed and pred.get(name, True)
            for w in widgets:
                w.setEnabled(on)
        for group, box in self.form.groups.items():
            key = 'group.' + group
            on = key in allowed and pred.get(key, True)
            self.form.set_group_enabled(group, on)
            locked = key in allowed and not on     # allowed by the state, locked by samples
            box.setToolTip('Locked while samples exist: Clear them or Reset (Ctrl+R) to '
                           'change these settings.' if locked else '')
        r = self.results
        if 'rect' in allowed:
            fixed = self._state == 'VERIFY' and self._settings.n_cameras > 1
            r.policy.setEnabled(not fixed)
            r.zero_disparity.setEnabled(not fixed)
            r.alpha.setEnabled(not fixed and r.policy.currentData() not in ('k', 'file'))
        tip = ''
        if not pred['commit'] and self._state in ('REVIEW', 'VERIFY'):
            names = r.service_names(self._settings.n_cameras)
            tip = texts.fmt(texts.COMMIT_NO_SERVER, {'service': ', '.join(names) or '?'})
        r.commit_btn.setToolTip(tip)

    def _apply_capture(self) -> None:
        """Set the session capture mode from state, settings and pause."""
        mode = self._settings['capture.mode'] if self._state == 'CAPTURING' else 'off'
        self._session.set_capture(mode, self._paused)

    def _apply_mode(self) -> None:
        """Adapt panes, indicators and results to mono or stereo."""
        n = self._settings.n_cameras
        self.view.n_panes = n
        self.indicators.set_cameras(n)
        pol = POLICIES if n == 1 else ()
        if self._state == 'VERIFY':
            pol = ('file',) + pol
        self.results.set_mode(n, pol)
        self.results.zero_disparity.setChecked(self._settings['rectify.zero_disparity'])
        k = self.results.policy.findData(self._settings['rectify.policy'])
        if k >= 0 and self._state != 'VERIFY':
            self.results.policy.setCurrentIndex(k)
        self.results.alpha.blockSignals(True)
        self.results.alpha.setValue(int(round(100 * self._settings['rectify.alpha'])))
        self.results.alpha.blockSignals(False)
        self.results.alpha_label.setText(f"{self._settings['rectify.alpha']:.2f}")

    # settings and source

    def _on_setting(self, key: str, value) -> None:
        """Apply a setting changed in the form."""
        changes = {key: value}
        if key in ('source.topic', 'source.left_topic', 'source.right_topic'):
            if value.endswith('/compressed'):
                changes = {key: value[:-len('/compressed')], 'source.transport': 'compressed'}
            self.form.set_values(changes)
        new = self._settings.updated(changes)
        problem = board_error(new) if any(k.startswith('board.') for k in changes) else None
        if (key == 'source.mode' and self._state == 'VERIFY' and self._cams is not None
                and new.n_cameras != len(self._cams)):
            problem = (f'The loaded calibration has {len(self._cams)} camera(s): press Back '
                       f'before switching the setup to {value}.')
        if problem is not None:
            self.log('Setting rejected: ' + problem)
            self._dialog('Invalid setting', problem)
            self.form.set_values({k: self._settings[k] for k in changes})
            return
        self._settings = new
        self._session.configure(self._settings)
        if key.startswith('source.'):
            self._apply_mode()
            self._resubscribe()
        if key == 'capture.mode':
            self._apply_capture()
        self._refresh_enabled()

    def _refresh_topics(self) -> None:
        """Fill the topic comboboxes from discovery."""
        try:
            self.form.set_topics(self.node.discover_topics())
        except Exception as e:
            self.report_error('discover_topics', e)

    def _resubscribe(self) -> None:
        """Start a new generation and subscribe to the configured source."""
        self._gen = self._session.next_generation()
        self.node.set_source(self._settings, self._gen, self._session.push)
        self._frame_t = None
        self._subscribed_t = time.monotonic()
        self._src_sizes = None
        self._overlay = None
        self.view.set_frame(None)
        self.view.set_overlay(None)
        self._source_status = ('WAITING', {})
        self._update_banner()

    def _topics_text(self) -> str:
        """Return the subscribed topics as text."""
        return ', '.join(self._settings.topics())

    # tick

    def _on_tick(self) -> None:
        """Poll slots, versions, jobs and commits; never blocks."""
        try:
            self._tick()
        except Exception as e:
            self.report_error('tick', e)
        while self._errors:
            where, msg = self._errors.popleft()
            self.log(f'Error in {where}: {msg}', error=True)

    def _tick(self) -> None:
        """Do one polling step."""
        s, now, ver = self._session, time.monotonic(), self._versions
        v, disp = s.slot_display.get()
        if v != ver['display']:
            ver['display'] = v
            if disp is not None and disp.gen == self._gen:
                first = self._frame_t is None
                self._frame_t = now
                sizes = tuple(getattr(disp, 'src_sizes', ())) or tuple(
                    (int(round(im.shape[1] / sc)), int(round(im.shape[0] / sc)))
                    for im, sc in zip(disp.images, disp.scales))
                if sizes != self._src_sizes:
                    self._src_sizes = sizes
                    self._update_view_sizes()
                    self._check_size()
                self.view.set_frame(disp)
                if first:
                    self._refresh_enabled()
                    self._update_banner()
        v, ov = s.slot_overlay.get()
        if v != ver['overlay']:
            ver['overlay'] = v
            self._overlay_t = now
            if ov is not None and ov.gen == self._gen:
                self._overlay = ov
                self.view.set_overlay(ov)
                if ov.verdict.reason == 'ACCEPTED':
                    self.view.flash()
                if ov.validation is not None:
                    self._val_hist.append(ov.validation)
                self._update_banner()
        v, g = s.slot_guidance.get()
        if v != ver['guidance'] and g is not None:
            ver['guidance'] = v
            self._guidance = g
            self.indicators.set_state(g)
            self._update_hints()
            ready = g.ready
            self.act_calibrate.setText('Calibrate (ready)' if ready else 'Calibrate')
            if self._calib_button is not None:
                self._calib_button.setStyleSheet(
                    'background: #28c850; font-weight: bold' if ready else '')
        db = getattr(s, 'db', None)
        if db is not None and db.version != ver['db']:
            ver['db'] = db.version
            self._refresh_samples()
            self._refresh_enabled()
        for kind, fut in self._jobs.pop_done():
            self._job_done(kind, fut)
        self._poll_commit(now)
        self.busy.setVisible(self._jobs.busy())
        if self._state == 'CALIBRATING':
            self._update_banner()
        if now - self._slow_t['stats'] >= 0.5:
            self._slow_t['stats'] = now
            self._update_stats()
        if now - self._slow_t['checks'] >= 1.0:
            self._slow_t['checks'] = now
            self._slow_checks(now)

    def _update_stats(self) -> None:
        """Update the status bar statistics, the live check and the session errors."""
        s = self._session
        st = s.stats()
        lat = st.get('latency_s')
        self.stats_label.setText(texts.fmt(
            'cam {rx_hz:.1f} Hz · det {det_hz:.1f} Hz · dropped {dropped} · latency {lat:.0f} ms'
            ' · samples {samples} ({mem_mb:.0f} MB)',
            {**st, 'lat': 1000 * lat if lat is not None else None}))
        lossy = bool(st.get('lossy', False))
        if lossy != self.view.lossy:
            self.view.lossy = lossy
            if lossy:
                self.log(texts.STATUS['LOSSY_TRANSPORT'])
        # log each distinct (where, message) once; repeats only raise the badge count
        errors = list(getattr(s, 'errors', ()))
        total = sum(e[3] for e in errors)
        for _, where, msg, _ in errors:
            if (where, msg) not in self._seen_session_errors:
                self._seen_session_errors.add((where, msg))
                self.log(f'{where}: {msg}', error=True, count=False)
        if total != self._session_err_total:
            self._session_err_total = total
            self._update_badge()
        if self._state in ('REVIEW', 'VERIFY'):
            self.results.set_validation(self._live_rows())

    def _slow_checks(self, now) -> None:
        """Check the source, services and thread liveness (every second)."""
        if self._frame_t is None or now - self._frame_t > NO_DATA_S:
            try:
                counts = self.node.publisher_counts(self._settings)
            except Exception as e:
                counts = ()
                self.report_error('publisher_counts', e)
            topics = self._settings.topics()
            missing = [t for t, c in zip(topics, counts) if c == 0]
            img_err = getattr(self._session, 'image_error', None)
            if missing:
                self._source_status = ('NO_PUBLISHER', {'topic': ', '.join(missing)})
            elif img_err is not None:
                code = getattr(img_err, 'code', '')
                error = texts.fmt(texts.IMAGE_ERRORS[code], img_err.params) \
                    if code in texts.IMAGE_ERRORS else str(img_err)
                self._source_status = ('DECODE_ERROR', {'error': error})
            elif now - self._subscribed_t > NO_DATA_S:
                self._source_status = ('NO_DATA', {'n': min(counts, default=0),
                                                   'topic': self._topics_text()})
            else:
                self._source_status = ('WAITING', {'topic': self._topics_text()})
            self._update_banner()
        if self._state in ('REVIEW', 'VERIFY'):
            names = self.results.service_names(self._settings.n_cameras)
            try:
                self._services_ready = all(names) and all(self.node.service_ready(n)
                                                          for n in names)
            except Exception as e:
                self._services_ready = False
                self.report_error('service_ready', e)
        if self.spin_thread is not None and not self.spin_thread.is_alive() and not self._shut:
            self.banner1.setText(texts.fmt(texts.STATUS['EXECUTOR_STOPPED'],
                                           {'error': 'thread exited'}))
        if not self._session.alive() and not self._shut:
            self.banner1.setText(texts.fmt(texts.STATUS['DECODE_ERROR'],
                                           {'error': 'pipeline threads stopped'}))
        self._refresh_enabled()

    # banner, hints, samples, live check

    def _extra(self) -> dict:
        """Return the extra format fields of verdict texts."""
        return {'cols': self._settings['board.cols'], 'rows': self._settings['board.rows'],
                'topic': self._topics_text()}

    def _update_banner(self) -> None:
        """Set banner line 1 (verdict) and line 2 (next hint)."""
        color, l1, l2 = '#c07000', '', ''
        now = time.monotonic()
        state, ov = self._state, self._overlay
        n = self._settings.n_cameras
        if state == 'CALIBRATING':
            cfg = self._settings.calib_config()
            l1 = texts.fmt(texts.STATUS['CALIBRATING'], {
                'n': self._calib_total, 'model': f'{cfg.model}, {cfg.distortion_model}',
                'elapsed': now - self._calib_t0,
                'stage': texts.STAGES.get(self._stage, self._stage)})
        elif self._frame_t is None or now - self._frame_t > NO_DATA_S:
            code, params = self._source_status
            l1 = texts.fmt(texts.STATUS[code], {'topic': self._topics_text(), **params})
            self.view.placeholder = l1
            self.view.update()
        elif self._frame_t - self._overlay_t > STALL_S:
            l1 = texts.fmt(texts.STATUS['DETECTOR_STALLED'],
                           {'s': self._frame_t - self._overlay_t})
        elif state in ('REVIEW', 'VERIFY'):
            color = '#28a050'
            l1 = texts.fmt(texts.STATUS[state], {'file': ', '.join(self._file_names)})
        elif ov is not None:
            v = ov.verdict
            if state == 'SETUP' and v.hard_ok:
                l1, color = texts.STATUS['SETUP_OK'], '#28a050'
            elif state == 'CAPTURING' and self._paused:
                l1 = texts.STATUS['PAUSED']
            elif (state == 'CAPTURING' and self._settings['capture.mode'] == 'manual'
                  and v.hard_ok and v.reason != 'ACCEPTED'):
                l1, color = texts.STATUS['MANUAL_READY'], '#28a050'
            else:
                prefix = ''
                if 'camera' in v.params and v.reason not in ('OK', 'ACCEPTED'):
                    prefix = texts.camera_prefix(v.params['camera'], n)
                l1 = prefix + texts.reason_text(v, self._extra())
                if v.ok or v.reason == 'ACCEPTED':
                    color = '#28a050'
            hints = tuple(ov.hints or ())
            if not hints and self._guidance is not None and state == 'CAPTURING':
                hints = self._guidance.hints[:1]
            if hints and state in ('SETUP', 'CAPTURING'):
                h = hints[0]
                l2 = 'Next: ' + texts.camera_prefix(h.camera, n) + texts.hint_text(h)
        self.banner1.setText(l1)
        if self._banner_color != color:
            self._banner_color = color
            self.banner1.setStyleSheet(f'color: {color}; font-weight: bold')
        self.banner2.setText(l2)
        self._update_hints()

    def _update_hints(self) -> None:
        """Send the arrow hints to the live view (frame hints first, then guidance)."""
        hints = ()
        if self._state == 'CAPTURING':
            ov_hints = tuple(self._overlay.hints or ()) if self._overlay is not None else ()
            hints = ov_hints + (self._guidance.hints if self._guidance is not None else ())
            first = {}
            for h in hints:
                first.setdefault(h.camera, h)
            hints = tuple(first.values())
        self.view.set_guidance(self._guidance, hints)

    def _refresh_samples(self) -> None:
        """Rebuild the sample list."""
        db = getattr(self._session, 'db', None)
        result = self._result if self._state == 'REVIEW' else None
        self.samples.set_samples(db.snapshot() if db is not None else None, result,
                                 self._settings['capture.min_samples'],
                                 self._settings['capture.max_samples'])

    def _live_rows(self) -> list:
        """Return the live check rows: median of the last 30 validation metrics."""
        rows = []
        for key, (label, unit, _) in texts.VALIDATION.items():
            vals = [m.values[key] for m in self._val_hist if key in m.values]
            if not vals:
                continue
            if isinstance(vals[0], tuple):
                per = []
                for k in range(len(vals[0])):
                    xs = [x[k] for x in vals if len(x) > k and x[k] is not None]
                    per.append(f'{np.median(xs):.3f}' if xs else '-')
                rows.append((label, ' / '.join(per) + ' ' + unit))
            else:
                xs = [x for x in vals if x is not None]
                if xs:
                    med = float(np.median(xs)) * (100 if unit == '%' else 1)
                    rows.append((label, f'{med:.4g} {unit}'))
        return rows

    def _check_size(self) -> None:
        """Warn if the loaded calibration does not match the image size."""
        if self._state != 'VERIFY' or self._cams is None or self._src_sizes is None:
            return
        for cam, (w, h) in zip(self._cams, self._src_sizes):
            if tuple(cam.size) != (w, h):
                msg = texts.fmt(texts.SIZE_MISMATCH, {'cw': cam.size[0], 'ch': cam.size[1],
                                                      'w': w, 'h': h})
                self.results.warnings.setText(msg)
                self.log(msg, error=True)

    # display sizes and rectification

    def _update_view_sizes(self) -> None:
        """Send the pane sizes to the session; rebuild rectification maps if shown."""
        if self._src_sizes is None:
            return
        sizes = self.view.fit_sizes(self._src_sizes)
        if sizes == self._view_sizes:
            return
        self._view_sizes = sizes
        if self.results.rectified.isChecked():
            self._rect_timer.start(150)
        else:
            self._session.set_view(ViewConfig(sizes, None))

    def _on_rect_control(self) -> None:
        """Handle policy, alpha or zero-disparity changes (150 ms debounce)."""
        r = self.results
        alpha = r.alpha.value() / 100.0
        r.alpha_label.setText(f'{alpha:.2f}')
        policy = r.policy.currentData()
        changes = {'rectify.alpha': alpha, 'rectify.zero_disparity': r.zero_disparity.isChecked()}
        if policy in POLICIES:
            changes['rectify.policy'] = policy
        self._settings = self._settings.updated(changes)
        if self._state == 'VERIFY' and policy != 'file':
            self._saved = False
        self._refresh_enabled()
        self._rect_timer.start(150)

    def _on_rectified(self, on: bool) -> None:
        """Toggle the rectified live view."""
        self.act_rectified.setChecked(on)
        if on:
            self._submit_rectify()
        else:
            self._session.set_view(ViewConfig(self._view_sizes, None))

    def _submit_rectify(self) -> None:
        """Submit a rectify job (or mark it pending if one is running)."""
        if self._cams is None:
            return
        if self._jobs.busy('rectify'):
            self._rect_pending = True
            return
        r = self.results
        policy = r.policy.currentData() or 'square'
        cfg = replace(self._settings.rectify_config(), policy=policy,
                      alpha=r.alpha.value() / 100.0, zero_disparity=r.zero_disparity.isChecked())
        sizes = None
        if r.rectified.isChecked():
            sizes = self._view_sizes or tuple(tuple(c.size) for c in self._cams)
        self._jobs.submit('rectify', _rectify_job, self._cams, self._stereo, cfg,
                          self._file_rect, sizes)

    # jobs

    def _job_done(self, kind: str, fut) -> None:
        """Dispatch a finished job (already removed from the pending table)."""
        exc = fut.exception()
        if kind == 'calibrate':
            if self._calib_submitted != self._calib_token or self._state != 'CALIBRATING':
                return                      # cancelled: drop the result and any failure
            if exc is not None:
                self._discard_calibration('CAPTURING')
                if isinstance(exc, calibration.CalibrationError):
                    msg = texts.error_text(exc.code, exc.params)
                else:
                    msg = f'{type(exc).__name__}: {exc}'
                    self.report_error('calibrate', exc)
                self.log('Calibration failed: ' + msg, error=True)
                self._dialog('Calibration failed', msg)
                return
            self._on_calibrated(fut.result()[1])
        elif kind == 'rectify':
            if exc is not None:
                self.report_error('rectify', exc)
            else:
                self._on_rectified_done(*fut.result())
            if self._rect_pending:
                self._rect_pending = False
                self._submit_rectify()
        elif kind == 'save':
            if exc is not None:
                self.report_error('save', exc)
                self._dialog('Save failed', f'{type(exc).__name__}: {exc}')
                return
            path, full = fut.result()
            if full:
                self._saved = True
            msg = texts.fmt(texts.SAVED, {'path': path})
            self.log(msg)
            self.statusBar().showMessage(msg, 10000)

    def _on_calibrated(self, result) -> None:
        """Show a calibration result and enter REVIEW."""
        self._result = result
        self._saved = False
        self._cams, self._stereo, self._file_rect = result.cameras, result.stereo, None
        self._rect = None
        self._val_hist.clear()
        self.results.set_result(result, self._calib_total)
        for bars in self.results.bars:
            bars.picked.connect(self._pick_sample)
        self.log(texts.fmt('Calibrated: RMS {rms:.4f} px in {d:.2f} s',
                           {'rms': result.rms, 'd': result.duration_s}))
        for code, params in result.warnings:
            self.log(texts.warning_text(code, params))
        self._set_state('REVIEW')
        self._refresh_samples()
        self._submit_rectify()

    def _on_rectified_done(self, rect, maps, sizes) -> None:
        """Apply a rectification job result."""
        if self._cams is None or self._state not in ('REVIEW', 'VERIFY'):
            return
        self._rect = rect
        self.results.set_rect(rect)
        for code, params in rect.warnings:
            self.log(texts.warning_text(code, params))
        self._session.set_active_calibration(self._cams, rect)
        if maps is not None and self.results.rectified.isChecked():
            self._session.set_view(ViewConfig(sizes, maps))
        else:
            self._session.set_view(ViewConfig(self._view_sizes, None))
        self._val_hist.clear()
        self._refresh_enabled()

    def _pick_sample(self, sid: int) -> None:
        """Select a sample picked in the error bars."""
        self.tabs.setCurrentIndex(0)
        self.samples.select_id(sid)

    # user flows

    def _on_pause(self, on: bool) -> None:
        """Pause or resume automatic capture."""
        self._paused = on
        self._apply_capture()
        self._update_banner()

    def _on_heatmap(self, on: bool) -> None:
        """Toggle the coverage heatmap."""
        self.view.heatmap = on
        self.view.update()

    def _capture_now(self) -> None:
        """Capture the next frame passing every gate but novelty."""
        if self._state == 'CAPTURING':
            self._session.request_capture()

    def _remove(self, ids) -> None:
        """Remove samples by id."""
        if ids:
            self._session.remove_samples(ids)

    def _remove_last(self) -> None:
        """Remove the most recent sample."""
        db = getattr(self._session, 'db', None)
        samples = db.snapshot().samples if db is not None else ()
        if samples:
            self._session.remove_samples([max(s.id for s in samples)])

    def _clear(self) -> None:
        """Remove every sample after confirmation."""
        n = self._n_samples()
        if n and QMessageBox.question(self, 'Clear samples',
                                      texts.fmt(texts.CONFIRM_CLEAR, {'n': n})) \
                == QMessageBox.Yes:
            self._session.clear_samples()

    def _calibrate(self) -> None:
        """Start calibrating the current samples (confirm if guidance is not ready)."""
        if self._state not in ('CAPTURING', 'REVIEW') or self._jobs.busy('calibrate'):
            return
        n = self._n_samples()
        cfg = self._settings.calib_config()
        if n < cfg.min_views:
            return
        g = self._guidance
        if g is None or not g.ready:
            nc = self._settings.n_cameras
            missing = sorted({texts.camera_prefix(i.camera, nc) + texts.INDICATOR_LABELS[i.id]
                              for i in (g.indicators if g is not None else ())
                              if i.progress < 1.0})
            if n < self._settings['capture.min_samples']:
                missing.append(f"{n} of {self._settings['capture.min_samples']} samples")
            text = texts.fmt(texts.CONFIRM_FORCE, {'missing': ', '.join(missing) or 'unknown'})
            if QMessageBox.question(self, 'Calibrate', text) != QMessageBox.Yes:
                return
            if self._state not in ('CAPTURING', 'REVIEW'):   # the dialog kept ticking
                return
        self._calib_token += 1
        self._calib_submitted = self._calib_token
        snap = self._session.db.snapshot()
        fut = self._jobs.submit('calibrate', _calibrate_job, self._calib_token, snap,
                                self._settings.board(), cfg, self._set_stage)
        if fut is None:
            return
        self._calib_total = len(snap.samples)
        self._calib_t0 = time.monotonic()
        self._stage = ''
        self._session.set_active_calibration(None, None)
        self.results.rectified.setChecked(False)
        self._set_state('CALIBRATING')

    def _set_stage(self, name: str) -> None:
        """Record the calibration stage (called on the job thread; read by the tick)."""
        self._stage = name

    def _cancel(self) -> None:
        """Discard the running calibration (the thread finishes in the background)."""
        # ponytail: cancel only discards the result; a subprocess would allow a real abort
        if self._state == 'CALIBRATING':
            self._calib_token += 1
            self._discard_calibration('CAPTURING')

    def _reset(self) -> None:
        """Discard samples and calibration, resubscribe and return to SETUP."""
        if self._state == 'CALIBRATING':
            return
        n = self._n_samples()
        unsaved = self._result is not None and not self._saved
        if (n or unsaved) and QMessageBox.question(
                self, 'Reset', texts.fmt(texts.CONFIRM_RESET, {'n': n})) != QMessageBox.Yes:
            return
        if self._state == 'CALIBRATING':      # the dialog kept ticking
            return
        self._calib_token += 1
        self._session.clear_samples()
        self._val_hist.clear()
        self._discard_calibration('SETUP')
        self._resubscribe()
        self._refresh_enabled()
        self.log('Reset: samples and calibration discarded.')

    def _escape(self) -> None:
        """Cancel the calibration or leave full screen."""
        if self._state == 'CALIBRATING':
            self._cancel()
        elif self.isFullScreen():
            self.showNormal()

    def _back(self) -> None:
        """Leave REVIEW (to CAPTURING) or VERIFY (to SETUP), discarding the calibration."""
        target = {'REVIEW': 'CAPTURING', 'VERIFY': 'SETUP'}.get(self._state)
        if target is not None:
            self._discard_calibration(target)

    def _discard_calibration(self, target: str) -> None:
        """Drop the result or loaded calibration, clear the results panel, enter target."""
        self._result = self._cams = self._stereo = self._file_rect = self._rect = None
        self._file_names = ()
        self._saved = True
        self._session.set_active_calibration(None, None)
        self.results.rectified.setChecked(False)
        self._session.set_view(ViewConfig(self._view_sizes, None))
        self.results.set_result(None, 0)
        self.results.set_rect(None)
        self._set_state(target)
        self._apply_mode()
        self._refresh_samples()

    def _save(self, dataset_only: bool) -> None:
        """Save the calibration (and dataset) or only the dataset, on the job thread."""
        if self._jobs.busy('save'):
            return
        out = self._settings.output_dir()
        if _is_tmp(out) and QMessageBox.question(
                self, 'Save', texts.fmt(texts.CONFIRM_TMP, {'path': out})) != QMessageBox.Yes:
            return
        db = getattr(self._session, 'db', None)
        snap = db.snapshot() if db is not None else None
        board, st = self._settings.board(), self._settings
        if dataset_only:
            if snap is None or not snap.samples:
                return
            self._jobs.submit('save', lambda: (write_run(out, st, None, None, snap, board,
                                                         dataset=True), False))
        elif self._state == 'VERIFY' and self._rect is not None:
            names, cams, rect = self._file_names, self._cams, self._rect
            self._jobs.submit('save', lambda: (_save_files(out, names, cams, rect), True))
        elif self._result is not None and self._rect is not None:
            result, rect = self._result, self._rect
            dataset = bool(st['capture.keep_images'] and snap is not None and snap.samples)
            self._jobs.submit('save', lambda: (write_run(out, st, result, rect, snap, board,
                                                         dataset=dataset), True))

    def _refresh_services(self) -> None:
        """Fill the commit service comboboxes from discovery."""
        n = self._settings.n_cameras
        try:
            services = sorted(self.node.discover_services())
        except Exception as e:
            services = []
            self.report_error('discover_services', e)
        topics = self._settings.topics()
        defaults = [self._settings['commit.service'] if n == 1 and
                    self._settings['commit.service'] else pick_service(t, services)
                    for t in topics]
        self.results.set_services(services, defaults)

    def _commit(self) -> None:
        """Send the calibration to the SetCameraInfo services."""
        if self._rect is None or self._commit_pending is not None or self._cams is None:
            return
        from dua_camera_calibration import ros_io   # rclpy import only when committing
        names = self.results.service_names(len(self._cams))
        if not all(names) or len(set(names)) != len(names):
            msg = ('Choose one SetCameraInfo service per camera: the names must be non-empty '
                   'and different (' + ', '.join(n or '<empty>' for n in names) + ').')
            self.log('Commit refused: ' + msg, error=True)
            self._dialog('Commit', msg)
            return
        infos = tuple(ros_io.camera_info(c, R, P)
                      for c, R, P in zip(self._cams, self._rect.R, self._rect.P))
        futures = self.node.commit(names, infos)
        self._commit_pending = (tuple(futures), names, time.monotonic())
        self._refresh_enabled()

    def _poll_commit(self, now) -> None:
        """Report finished or timed-out commits."""
        if self._commit_pending is None:
            return
        futures, names, t0 = self._commit_pending
        if all(f.done() for f in futures):
            self._commit_pending = None
            failed = []
            for f, name in zip(futures, names):
                try:
                    res = f.result()
                    ok, msg = bool(res.success), res.status_message
                except Exception as e:
                    ok, msg = False, str(e)
                text = texts.fmt(texts.COMMIT_OK if ok else texts.COMMIT_FAILED,
                                 {'service': name, 'msg': msg})
                self.log(text, error=not ok)
                self.statusBar().showMessage(text, 10000)
                if not ok:
                    failed.append(text)
            if failed:
                self._dialog('Commit failed', '\n'.join(failed))
            self._refresh_enabled()
        elif now - t0 > COMMIT_TIMEOUT_S:
            self._commit_pending = None
            self.node.cancel(futures)
            text = texts.fmt(texts.COMMIT_TIMEOUT, {'service': ', '.join(names),
                                                    't': COMMIT_TIMEOUT_S})
            self.log(text, error=True)
            self._dialog('Commit timed out', text)
            self._refresh_enabled()

    def _load(self) -> None:
        """Load calibration YAML file(s) and enter VERIFY."""
        n = self._settings.n_cameras
        paths = []
        for k in range(n):
            title = 'Load calibration' if n == 1 else \
                f'Select the {texts.CAMERA_NAMES[2][k].upper()} calibration'
            p, _ = QFileDialog.getOpenFileName(self, title, self._settings.output_dir(),
                                               'YAML (*.yaml *.yml)')
            if not p:
                return
            paths.append(p)
        self.load_files(paths)

    def load_files(self, paths) -> None:
        """Load calibration YAML files (one per camera) and enter VERIFY."""
        if 'load' not in ENABLED[self._state] or not self._predicates()['load']:
            return
        if self._result is not None and not self._saved and QMessageBox.question(
                self, 'Load calibration', texts.CONFIRM_DISCARD) != QMessageBox.Yes:
            return
        try:
            loaded = [fileio.read_camera_yaml(p) for p in paths]
        except Exception as e:
            self.report_error('load', e)
            self._dialog('Load failed', f'{type(e).__name__}: {e}')
            return
        names, cams, Rs, Ps = zip(*loaded)
        self._result, self._stereo = None, None
        self._cams = tuple(cams)
        self._file_names = tuple(n or os.path.splitext(os.path.basename(p))[0]
                                 for n, p in zip(names, paths))
        self._file_rect = Rectification(tuple(Rs), tuple(Ps), 'file', 0.0)
        self._rect = self._file_rect
        self._saved = True
        self._val_hist.clear()
        self._state = 'VERIFY'
        self._apply_mode()
        self.results.policy.setCurrentIndex(self.results.policy.findData('file'))
        self.results.set_result(None, 0, 'Loaded: ' + ', '.join(paths))
        self.results.show_cameras(self._cams)
        self.results.set_rect(self._rect)
        self._session.set_active_calibration(self._cams, self._rect)
        self._set_state('VERIFY')
        self._check_size()
        if self.results.rectified.isChecked():
            self._submit_rectify()

    def _show_help(self) -> None:
        """Show the help panel."""
        self.tabs.setCurrentIndex(0)
        self.help_btn.setChecked(True)

    def _toggle_full(self) -> None:
        """Toggle full screen."""
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _about(self) -> None:
        """Show versions, useful in bug reports."""
        QMessageBox.about(self, 'About', (
            f'dua_camera_calibration 4.0.0\nOpenCV {cv2.__version__}\nQt {QT_VERSION_STR}\n'
            f'PyQt {PYQT_VERSION_STR}\nPython {sys.version.split()[0]}\n'
            'dotX Automation s.r.l.'))

    # shutdown

    def quit_now(self) -> None:
        """Quit without confirmation (signals)."""
        self._force_quit = True
        self.close()
        QApplication.quit()

    def closeEvent(self, event):
        """Ask before discarding an unsaved calibration, then shut down."""
        if not self._force_quit and self._result is not None and not self._saved:
            if QMessageBox.question(self, 'Quit', texts.CONFIRM_QUIT) != QMessageBox.Yes:
                event.ignore()
                return
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        """Stop tick, session threads, jobs and subscriptions (idempotent)."""
        if self._shut:
            return
        self._shut = True
        for step in (self._timer.stop, self._rect_timer.stop, self._session.stop,
                     self._jobs.shutdown, self.node.clear_source):
            try:
                step()
            except Exception as e:
                print(f'shutdown: {type(e).__name__}: {e}', file=sys.stderr)
        qs = QSettings('dotxautomation', 'dua_camera_calibration')
        qs.setValue('geometry', self.saveGeometry())
        qs.setValue('state', self.saveState())
