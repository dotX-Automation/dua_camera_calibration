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
import threading
import time

from conftest import BOARD, CAM, make_settings
import cv2
from dua_camera_calibration import calibration, fileio, rectification
from dua_camera_calibration.dua_camera_calibration_cli import flatten, load_params_file
from dua_camera_calibration.pipeline import (camera_files, convert_raw, Frame, JobRunner,
                                             LatestSlot, make_display, RawFrame, Session,
                                             ViewConfig, write_run)
from dua_camera_calibration.ros_io import camera_info, topic_bases
from dua_camera_calibration.synthetic_camera import make_compressed_msg, make_image_msg
import numpy as np
import pytest
from sensor_msgs.msg import Image
import yaml


def frame(grays, seq=0, stamp_ns=0, gen=0):
    grays = grays if isinstance(grays, tuple) else (grays,)
    return Frame(gen, seq, time.monotonic(), stamp_ns, grays, ('mono8',) * len(grays), False)


def wait_for(cond, timeout=10.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.01)
    return True


def test_settings():
    s = make_settings(board__type='hexagon', source__transport='compressed')
    assert s['board.type'] == 'chessboard'           # invalid choice replaced by the default
    assert s.topics() == ('/camera/image_raw/compressed',)
    s2 = s.updated({'source.mode': 'stereo'})
    assert s2.n_cameras == 2 and len(s2.topics()) == 2
    _, warnings = type(s).from_mapping({'board.cols': 'abc'})
    assert warnings == [('board.cols', 'abc')]


def test_topic_bases():
    nt = [('/a/image_raw', ['sensor_msgs/msg/Image']),
          ('/a/image_raw/compressed', ['sensor_msgs/msg/CompressedImage']),
          ('/a/image_raw/zstd', ['sensor_msgs/msg/CompressedImage']),
          ('/a/depth/compressedDepth', ['sensor_msgs/msg/CompressedImage']),
          ('/b/image', ['sensor_msgs/msg/Image']), ('/c', ['std_msgs/msg/String'])]
    assert topic_bases(nt) == {'raw': ('/a/image_raw', '/b/image'),
                               'compressed': ('/a/image_raw',)}


def test_latest_slot():
    slot = LatestSlot()
    assert slot.get() == (0, None)
    slot.put(1)
    slot.put(2)
    assert slot.get() == (2, 2)                      # overwrite, drop-oldest
    t0 = time.monotonic()
    assert slot.wait_newer(2, 0.05) == (2, 2)        # timeout: unchanged
    assert time.monotonic() - t0 >= 0.04
    threading.Timer(0.02, slot.put, (3,)).start()
    assert slot.wait_newer(2, 2.0) == (3, 3)


def test_job_runner():
    jobs = JobRunner()
    gate = threading.Event()
    f = jobs.submit('calibrate', gate.wait)
    assert f is not None and jobs.submit('calibrate', gate.wait) is None   # re-entrancy guard
    assert jobs.busy('calibrate') and not jobs.busy('save')
    gate.set()
    f.result(2.0)
    assert jobs.pop_done() == [('calibrate', f)] and not jobs.busy()
    jobs.shutdown()


def test_convert_raw(mono_views):
    g = mono_views[0]
    bgr = Image(encoding='bgr8', width=g.shape[1], height=g.shape[0], step=3 * g.shape[1],
                data=cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).tobytes())
    msgs = (make_image_msg(g, 5), bgr, make_compressed_msg(g, 5, 'png'),
            make_compressed_msg(g, 5, 'jpeg'))
    out = [convert_raw(RawFrame(3, 1.0, 5, (m,)), 7) for m in msgs]
    for f in out[:3]:
        assert np.array_equal(f.grays[0], g) and not f.lossy
    assert out[3].lossy and np.abs(out[3].grays[0].astype(int) - g).mean() < 3
    assert (out[0].gen, out[0].seq, out[0].stamp_ns, out[0].encodings) == (3, 7, 5, ('mono8',))
    pair = convert_raw(RawFrame(0, 0.0, 5, (msgs[0], msgs[2])), 0)
    assert len(pair.grays) == 2 and not pair.lossy


def test_make_display(mono_views):
    f = frame(mono_views[0])
    assert make_display(f, ViewConfig(())).images[0] is f.grays[0]      # native
    d = make_display(f, ViewConfig(((320, 1000),)))
    assert d.images[0].shape == (240, 320) and d.scales == (0.5,) and not d.rectified
    assert make_display(f, ViewConfig(((2000, 2000),))).scales == (1.0,)  # never upscale
    rect = rectification.rectify((CAM,), None, rectification.RectifyConfig())
    maps = (rectification.make_maps(CAM, rect.R[0], rect.P[0], (320, 240)),)
    d = make_display(f, ViewConfig((), maps))
    assert d.rectified and d.images[0].shape == (240, 320) and d.scales == (0.5,)
    assert d.images[0].flags['C_CONTIGUOUS'] and d.images[0].dtype == np.uint8


def test_process_frame_reasons(mono_views):
    s = Session(make_settings(), threaded=False)
    blank = np.full((480, 640), 128, np.uint8)
    ov = s.process_frame(frame(blank))
    assert ov.verdict.reason == 'NOT_DETECTED' and ov.hints and ov.detections == (None,)
    assert s.process_frame(frame(mono_views[0], 1)).verdict.reason == 'MOVING'  # no prev
    assert s.process_frame(frame(mono_views[0], 2)).verdict.reason == 'OK'      # capture off
    s.set_capture('auto', True)
    assert s.process_frame(frame(mono_views[0], 3)).verdict.reason == 'OK'      # paused
    s.set_capture('auto', False)
    ov = s.process_frame(frame(mono_views[0], 4))
    assert ov.verdict.reason == 'ACCEPTED' and ov.verdict.params == {'id': 0}
    assert len(s.db) == 1 and s.db.snapshot().samples[0].images_png is not None
    assert s.process_frame(frame(mono_views[0], 5)).verdict.reason == 'TOO_SIMILAR'
    s.set_capture('manual', False)
    assert s.process_frame(frame(mono_views[0], 6)).verdict.reason == 'TOO_SIMILAR'
    s.request_capture()
    assert s.process_frame(frame(mono_views[0], 7)).verdict.params == {'id': 1}  # hard_ok
    assert s.process_frame(frame(mono_views[0], 8)).verdict.reason == 'TOO_SIMILAR'
    s.configure(make_settings().updated({'capture.max_samples': 2}))
    s.set_capture('auto', False)
    s.process_frame(frame(mono_views[1], 9))
    ov = s.process_frame(frame(mono_views[1], 10))
    assert ov.verdict.reason == 'LIMIT' and len(s.db) == 2
    assert s.refresh_guidance() is not None and s.refresh_guidance() is None  # only if changed
    s.remove_samples([0])
    assert [x.id for x in s.db.snapshot().samples] == [1]
    s.configure(make_settings(board__cols=7))       # board change clears the samples
    assert len(s.db) == 0
    with pytest.raises(ValueError):
        s.set_capture('sometimes', False)


def test_generation_drop(mono_views):
    s = Session(make_settings(), threaded=False)
    raw = RawFrame(0, 0.0, 0, (make_image_msg(mono_views[0]),))
    s.push(raw)
    assert s.slot_in.get()[0] == 1
    assert s.next_generation() == 1
    s.push(raw)                                     # stale generation: dropped
    assert s.slot_in.get()[0] == 1
    s.push(RawFrame(1, 0.0, 0, raw.msgs))
    assert s.slot_in.get()[0] == 2


def test_threaded_pipeline(mono_views):
    s = Session(make_settings(), threaded=True)
    try:
        s.set_view(ViewConfig(((320, 240),)))
        s.set_capture('auto', False)
        gen = s.next_generation()
        msgs = [make_image_msg(v) for v in mono_views[:3]]
        for k in range(60):
            s.push(RawFrame(gen, time.monotonic(), k * 50000000, (msgs[k // 20],)))
            time.sleep(0.02)
        assert wait_for(lambda: len(s.db) >= 2)
        assert s.slot_display.get()[1].images[0].shape == (240, 320)
        ov = s.slot_overlay.get()[1]
        assert ov.gen == gen and ov.detections[0] is not None
        assert wait_for(lambda: s.slot_guidance.get()[1] is not None
                        and s.slot_guidance.get()[1].db_version == s.db.version)
        st = s.stats()
        assert set(st) == {'rx_hz', 'det_hz', 'dropped', 'latency_s', 'samples', 'mem_mb', 'lossy'}
        assert st['samples'] == len(s.db) and st['mem_mb'] > 0 and st['rx_hz'] > 0
        s.push(RawFrame(gen, 0.0, 0, (Image(encoding='weird', width=1, height=1, step=1,
                                            data=b'x'),)))
        assert wait_for(lambda: any(w == 'converter' for _, w, _, _ in s.errors))
        assert s.image_error is not None and s.alive()
    finally:
        s.stop()
        s.stop()                                    # idempotent
    assert not s.alive()


def test_errors_coalesced():
    s = Session(make_settings(), threaded=False)
    for _ in range(3):
        s.report('detector', RuntimeError('boom'))
    s.report('converter', RuntimeError('other'))
    errs = s.errors
    assert [(w, m, c) for _, w, m, c in errs] == [('detector', 'RuntimeError: boom', 3),
                                                  ('converter', 'RuntimeError: other', 1)]


def test_write_run_and_validation(mono_views, tmp_path):
    settings = make_settings(camera__name='cam_a')
    s = Session(settings, threaded=False, check_motion=False)
    s.set_capture('auto', False)
    for i, v in enumerate(mono_views):
        s.process_frame(frame(v, i))
        if i % 5 == 4:
            s.refresh_guidance()
    snap = s.db.snapshot()
    assert len(snap.samples) >= 10
    result = calibration.calibrate(snap, BOARD, settings.calib_config())
    rect = rectification.rectify(result.cameras, None, settings.rectify_config())
    assert abs(result.cameras[0].K[0, 0] / CAM.K[0, 0] - 1) < 0.01
    path = write_run(str(tmp_path), settings, result, rect, snap, BOARD, dataset=True)
    assert os.path.basename(path).startswith('cam_a_')
    name, cam, R, P = fileio.read_camera_yaml(os.path.join(path, 'cam_a.yaml'))
    assert name == 'cam_a' and np.array_equal(cam.K, result.cameras[0].K)
    assert np.array_equal(cam.D, result.cameras[0].D) and np.array_equal(P, rect.P[0])
    info = camera_info(cam, R, P, 'f')
    assert list(info.k) == list(cam.K.ravel()) and list(info.p) == list(P.ravel())
    assert info.width == 640 and info.distortion_model == 'plumb_bob'
    snap2, board2, meta = fileio.read_dataset(os.path.join(path, 'dataset'))
    assert board2 == BOARD and meta['settings']['camera.name'] == 'cam_a'
    assert [x.id for x in snap2.samples] == [x.id for x in snap.samples]
    png = snap2.samples[0].images_png[0]
    assert np.array_equal(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE),
                          mono_views[0])
    with open(os.path.join(path, 'report.yaml')) as f:
        assert flatten(yaml.safe_load(f)['settings']) == settings.as_dict()
    assert load_params_file(os.path.join(path, 'settings.yaml')) == settings.as_dict()
    assert camera_files(settings, 2) == (('left.yaml', 'cam_a_left'),
                                         ('right.yaml', 'cam_a_right'))
    only = write_run(str(tmp_path / 'ds'), settings, None, None, snap, BOARD, dataset=True)
    assert sorted(os.listdir(only)) == ['dataset', 'settings.yaml']

    # live validation and rectified overlay points with the active calibration
    s.set_active_calibration(result.cameras, rect)
    ov = s.process_frame(frame(mono_views[0], 99))
    assert not ov.rectified and ov.validation.values['reproj_rms_px'][0] < 0.5
    maps = (rectification.make_maps(result.cameras[0], rect.R[0], rect.P[0], (640, 480)),)
    s.set_view(ViewConfig((), maps))
    ov = s.process_frame(frame(mono_views[0], 100))
    assert ov.rectified
    expect = rectification.rectify_points(result.cameras[0], rect.R[0], rect.P[0],
                                          ov.detections[0].corners)
    assert np.allclose(ov.points[0], expect)
    s.set_active_calibration(None, None)
    assert s.process_frame(frame(mono_views[0], 101)).validation is None
