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

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')   # before any QApplication

import threading  # noqa: E402
import time  # noqa: E402

from conftest import CAM, make_settings  # noqa: E402
from dua_camera_calibration import fileio, gui  # noqa: E402
from dua_camera_calibration.pipeline import RawFrame  # noqa: E402
from dua_camera_calibration.rectification import Rectification  # noqa: E402
from dua_camera_calibration.synthetic_camera import make_image_msg  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from python_qt_binding.QtCore import QSettings, Qt  # noqa: E402
from python_qt_binding.QtTest import QTest  # noqa: E402
from python_qt_binding.QtWidgets import QApplication, QMessageBox  # noqa: E402
from rclpy.task import Future  # noqa: E402
from sensor_msgs.srv import SetCameraInfo  # noqa: E402

SHOTS = os.environ.get('DCC_SCREENSHOT_DIR')


class FakeNode:
    """Duck type of ros_io.CalibrationNode: frames are pushed by the test."""

    def __init__(self):
        """Start with no source."""
        self.gen, self.sink, self.commits, self.cleared = None, None, [], False

    def discover_topics(self):
        """Return one raw topic."""
        return {'raw': ('/fake/image_raw',), 'compressed': ()}

    def discover_services(self):
        """Return one SetCameraInfo service per side."""
        return ['/fake/set_camera_info', '/fake/left/set_camera_info',
                '/fake/right/set_camera_info']

    def set_source(self, settings, gen, sink):
        """Store the sink the test pushes into."""
        self.gen, self.sink = gen, sink

    def clear_source(self):
        """Drop the sink."""
        self.sink, self.cleared = None, True

    def publisher_counts(self, settings):
        """Pretend every topic has a publisher."""
        return (1,) * settings.n_cameras

    def service_ready(self, name):
        """Pretend every service is up."""
        return True

    def commit(self, names, infos):
        """Answer immediately with success."""
        self.commits.append((names, infos))
        out = []
        for _ in names:
            f = Future()
            f.set_result(SetCameraInfo.Response(success=True, status_message='stored'))
            out.append(f)
        return out

    def cancel(self, futures):
        """Do nothing."""


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Return the QApplication with QSettings redirected and dialogs auto-accepted."""
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path / 'qs'))
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: QMessageBox.Ok)
    return QApplication.instance() or QApplication([])


def wait_until(pred, timeout, feed=None):
    """Process events (and push frames through feed()) until pred() or timeout."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if feed is not None:
            feed()
        QTest.qWait(30)
        if pred():
            return True
    return pred()


def feeder(node, grays):
    """Return a callable pushing the given gray images (one per camera) as a RawFrame."""
    def push():
        stamp = time.time_ns()
        if node.sink is not None:
            node.sink(RawFrame(node.gen, time.monotonic(), stamp,
                               tuple(make_image_msg(g, stamp) for g in grays)))
    return push


def shot(win, name):
    """Save a screenshot if DCC_SCREENSHOT_DIR is set."""
    if SHOTS:
        os.makedirs(SHOTS, exist_ok=True)
        QTest.qWait(100)
        win.grab().save(os.path.join(SHOTS, name))


def enabled(win, name):
    """Return whether every widget of a control is enabled."""
    return all(w.isEnabled() for w in win._controls[name])


def close(win):
    """Close the window and check that the pipeline threads are gone."""
    win.close()
    QTest.qWait(50)
    assert not win._session.alive() or not win._session._threads
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        left = [t for t in threading.enumerate()
                if t is not threading.main_thread() and (not t.daemon or t.name in (
                    'converter', 'detector'))]
        if not left:
            break
        time.sleep(0.05)
    assert not left, left


def test_mono_flow(app, tmp_path, mono_views):
    """SETUP -> CAPTURING -> CALIBRATING -> REVIEW, policy switch, commit, save, shutdown."""
    node = FakeNode()
    settings = make_settings(source__topic='/fake/image_raw', output__dir=str(tmp_path / 'out'))
    win = gui.MainWindow(node, settings)
    win.show()
    win.activateWindow()
    QApplication.setActiveWindow(win)
    assert win._state == 'SETUP'
    assert not enabled(win, 'start') and not enabled(win, 'calibrate')
    assert win.form.groups['source'].isEnabled()

    push = feeder(node, (mono_views[0],))
    assert wait_until(lambda: win._overlay is not None, 10, push)
    assert enabled(win, 'start') and not enabled(win, 'capture')
    assert win.view._disp is not None and win.view._disp.images[0].flags['C_CONTIGUOUS']
    assert win._src_sizes == (tuple(CAM.size),)
    shot(win, 'setup.png')

    win.act_start.trigger()
    assert win._state == 'CAPTURING'
    assert enabled(win, 'capture') and enabled(win, 'pause') and not enabled(win, 'start')
    assert not win.form.groups['source'].isEnabled()

    QTest.keyClick(win, Qt.Key_P)
    assert win._paused and win.act_pause.isChecked()
    QTest.keyClick(win, Qt.Key_P)
    assert not win._paused
    calls = []
    real = win._session.request_capture
    win._session.request_capture = lambda: (calls.append(1), real())
    QTest.keyClick(win, Qt.Key_Space)
    assert calls

    for gray in mono_views:
        n = len(win._session.db)
        wait_until(lambda: len(win._session.db) > n, 2.0, feeder(node, (gray,)))
    win._session.set_capture('off', False)   # freeze the db, then let the tick catch up
    n = len(win._session.db)
    assert n >= 10, n
    assert wait_until(lambda: win.samples.list.count() == len(win._session.db), 2.0)
    win._session.set_capture('auto', False)
    win.act_heatmap.setChecked(True)
    push = feeder(node, (mono_views[-1],))
    wait_until(lambda: False, 0.3, push)
    shot(win, 'capturing.png')
    win.act_heatmap.setChecked(False)

    win.act_calibrate.trigger()
    assert win._state == 'CALIBRATING'
    assert enabled(win, 'cancel') and not enabled(win, 'capture')
    assert wait_until(lambda: win._state == 'REVIEW', 30)
    res = win._result
    assert res.rms < 0.3
    fx = res.cameras[0].K[0, 0]
    assert abs(fx / CAM.K[0, 0] - 1) < 0.01
    assert enabled(win, 'save') or wait_until(lambda: enabled(win, 'save'), 5)
    assert wait_until(lambda: win._rect is not None, 10)
    assert win._rect.policy == 'square'
    P_square = win._rect.P[0].copy()
    assert abs(P_square[0, 0] - P_square[1, 1]) < 1e-9

    r = win.results
    r.policy.setCurrentIndex(r.policy.findData('k'))
    r.policy.activated.emit(r.policy.currentIndex())
    assert wait_until(lambda: win._rect.policy == 'k', 5)
    assert not np.allclose(win._rect.P[0], P_square)
    assert np.allclose(win._rect.P[0][:, :3], res.cameras[0].K)
    r.policy.setCurrentIndex(r.policy.findData('square'))
    r.policy.activated.emit(r.policy.currentIndex())
    assert wait_until(lambda: win._rect.policy == 'square', 5)

    assert wait_until(lambda: enabled(win, 'commit'), 3)
    win.act_commit.trigger()
    assert wait_until(lambda: win._commit_pending is None, 3)
    names, infos = node.commits[-1]
    assert names == ('/fake/set_camera_info',)
    assert np.allclose(infos[0].p, win._rect.P[0].ravel())
    assert 'Committed to /fake/set_camera_info' in win.log_view.toPlainText()

    win.act_save.trigger()
    assert wait_until(lambda: win._saved, 10)
    runs = os.listdir(tmp_path / 'out')
    assert runs and os.path.exists(tmp_path / 'out' / runs[0] / 'camera.yaml')

    r.rectified.setChecked(True)
    push = feeder(node, (mono_views[3],))
    assert wait_until(lambda: win.view._disp is not None and win.view._disp.rectified, 10, push)
    wait_until(lambda: False, 0.6, push)
    shot(win, 'review.png')
    r.rectified.setChecked(False)

    r.back_btn.click()
    assert win._state == 'CAPTURING' and win._result is None

    # VERIFY: reload the saved file, re-frame it with another policy
    win.act_calibrate.trigger()
    assert wait_until(lambda: win._state == 'REVIEW', 30)
    saved = str(tmp_path / 'out' / runs[0] / 'camera.yaml')
    win.load_files([saved])
    assert win._state == 'VERIFY' and win._rect.policy == 'file'
    assert r.policy.currentData() == 'file'
    assert enabled(win, 'save') and not enabled(win, 'calibrate')
    P_file = win._rect.P[0].copy()
    r.policy.setCurrentIndex(r.policy.findData('k'))
    r.policy.activated.emit(r.policy.currentIndex())
    assert wait_until(lambda: win._rect.policy == 'k', 5)
    assert not np.allclose(win._rect.P[0], P_file) and not win._saved
    push = feeder(node, (mono_views[5],))
    assert wait_until(lambda: len(win._val_hist) > 0, 5, push)
    win._update_stats()
    assert 'Reprojection RMS' in r.live.text()
    r.back_btn.click()
    assert win._state == 'SETUP' and win._cams is None
    close(win)
    assert node.cleared


def test_stereo_smoke(app, tmp_path, stereo_views):
    """Stereo panes, pair capture and per-camera indicators."""
    node = FakeNode()
    settings = make_settings(source__mode='stereo', source__left_topic='/fake/left/image_raw',
                             source__right_topic='/fake/right/image_raw',
                             output__dir=str(tmp_path / 'out'))
    win = gui.MainWindow(node, settings)
    win.show()
    assert win.view.n_panes == 2 and len(win.indicators._rows) == 10
    push = feeder(node, stereo_views[0])
    assert wait_until(lambda: win._overlay is not None, 10, push)
    win.act_start.trigger()
    for pair in stereo_views[:4]:
        n = len(win._session.db)
        wait_until(lambda: len(win._session.db) > n, 2.0, feeder(node, pair))
    assert len(win._session.db) >= 3
    assert win._guidance is not None or wait_until(lambda: win._guidance is not None, 5)
    shot(win, 'stereo.png')
    close(win)


def make_win(tmp_path, **changes):
    """Return (window, node) showing a first frame of mono_views-like size."""
    node = FakeNode()
    settings = make_settings(source__topic='/fake/image_raw', output__dir=str(tmp_path / 'out'),
                             **changes)
    win = gui.MainWindow(node, settings)
    win.show()
    return win, node


def add_fake_samples(win, n):
    """Add n samples without detections (enough to exercise the flows, not to calibrate)."""
    for i in range(n):
        win._session.db.add((None,), (None,), None, i)
    QTest.qWait(50)


def test_stop_reconfigure_restart_and_reset(app, tmp_path, mono_views):
    """Start -> Stop -> reconfigure -> Start works; Reset unlocks the setup and restarts."""
    win, node = make_win(tmp_path)
    push = feeder(node, (mono_views[0],))
    assert wait_until(lambda: win._overlay is not None, 10, push)
    win.act_pause.setChecked(True)
    win.act_start.trigger()
    win.act_stop.trigger()
    assert win._state == 'SETUP' and not win.act_pause.isChecked()
    assert win.form.groups['board'].isEnabled()        # no samples: still configurable
    gen = node.gen
    win.form._emit('source.topic', '/fake/other')
    win.form._emit('board.cols', 9)
    assert node.gen > gen and win._settings['board.cols'] == 9
    assert win._settings['source.topic'] == '/fake/other'
    win.act_start.trigger()
    assert win._state == 'CAPTURING'
    assert wait_until(lambda: win._overlay is not None and win._overlay.gen == node.gen, 10,
                      feeder(node, (mono_views[0],)))

    add_fake_samples(win, 3)
    win.act_stop.trigger()
    win._refresh_enabled()
    box = win.form.groups['board']
    assert not box.isEnabled() and 'Reset' in box.toolTip()
    assert enabled(win, 'reset')
    gen = node.gen
    win.act_reset.trigger()
    assert win._state == 'SETUP' and len(win._session.db) == 0
    assert node.gen > gen and win._result is None and win._cams is None
    assert win.form.groups['board'].isEnabled() and win.form.groups['source'].isEnabled()
    assert wait_until(lambda: win._overlay is not None and win._overlay.gen == node.gen, 10,
                      feeder(node, (mono_views[0],)))
    close(win)


def test_setting_validation_and_error_log(app, tmp_path):
    """Invalid ChArUco settings are refused; pipeline errors are logged once, counted always."""
    win, node = make_win(tmp_path)
    win.form._emit('board.square_size', 0.01)
    win.form._emit('board.type', 'charuco')             # default marker 0.018 >= square 0.01
    assert win._settings['board.type'] == 'chessboard'
    assert win.form._values['board.type'] == 'chessboard'
    assert 'Setting rejected' in win.log_view.toPlainText()
    for _ in range(5):
        win._session.report('detector', ValueError('boom'))
    win._update_stats()
    win._update_stats()
    assert win.log_view.toPlainText().count('boom') == 1
    assert '5' in win.error_badge.text()
    close(win)


def test_verify_guards(app, tmp_path, mono_views, monkeypatch):
    """VERIFY: confirm before dropping a result, lock setup with samples, refuse mode/services."""
    win, node = make_win(tmp_path)
    assert wait_until(lambda: win._overlay is not None, 10, feeder(node, (mono_views[0],)))
    add_fake_samples(win, 2)
    path = str(tmp_path / 'c.yaml')
    fileio.write_camera_yaml(path, 'c', CAM, np.eye(3), np.hstack([CAM.K, np.zeros((3, 1))]))
    win._set_state('REVIEW')
    win._result, win._saved = object(), False
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.No)
    win.load_files([path])
    assert win._state == 'REVIEW'
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.Yes)
    win.load_files([path])
    assert win._state == 'VERIFY'
    assert not win.form.groups['board'].isEnabled() and not win.form.groups['source'].isEnabled()
    win._session.clear_samples()
    QTest.qWait(50)
    win._refresh_enabled()
    assert win.form.groups['source'].isEnabled()
    win.form._emit('source.mode', 'stereo')
    assert win._settings['source.mode'] == 'mono' and win.form._values['source.mode'] == 'mono'

    win.results.services[0].setEditText('')
    win._commit()
    assert not node.commits and 'Commit refused' in win.log_view.toPlainText()
    win._cams = (CAM, CAM)
    win._rect = Rectification((np.eye(3),) * 2, (win._rect.P[0],) * 2, 'file', 0.0)
    win.results.set_mode(2, ())
    for c in win.results.services[:2]:
        c.setEditText('/fake/set_camera_info')
    win._commit()
    assert not node.commits and win._commit_pending is None
    close(win)


def test_cancelled_failure_is_silent(app, tmp_path, mono_views, monkeypatch):
    """A calibration failing after Cancel shows nothing; a real failure resets the results."""
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: warnings.append(a))
    win, node = make_win(tmp_path)
    assert wait_until(lambda: win._overlay is not None, 10, feeder(node, (mono_views[0],)))
    add_fake_samples(win, 8)
    win._set_state('CAPTURING')
    win._calibrate()
    assert win._state == 'CALIBRATING'
    win._cancel()
    assert win._state == 'CAPTURING'
    assert wait_until(lambda: not win._jobs.busy() and not win._jobs._pending, 10)
    QTest.qWait(100)
    assert not warnings and 'Calibration failed' not in win.log_view.toPlainText()
    win._calibrate()
    assert wait_until(lambda: win._state == 'CAPTURING', 10)
    QTest.qWait(100)
    assert warnings and win._result is None and win._cams is None
    close(win)
