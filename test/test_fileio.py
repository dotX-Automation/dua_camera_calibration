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

from datetime import datetime
import os
import subprocess

import cv2
from dua_camera_calibration.calibration import (CalibrationResult, CameraCalib,
                                                StereoExtrinsics)
from dua_camera_calibration.detection import BoardSpec, Detection
from dua_camera_calibration.fileio import (read_camera_yaml, read_dataset, run_dir,
                                           write_camera_yaml, write_dataset, write_report)
from dua_camera_calibration.rectification import Rectification
from dua_camera_calibration.samples import SampleDB
import numpy as np
import pytest
import yaml

PARSER_KEYS = {'image_width', 'image_height', 'camera_name', 'camera_matrix', 'distortion_model',
               'distortion_coefficients', 'rectification_matrix', 'projection_matrix'}
RNG = np.random.default_rng(7)


def camera(model='pinhole'):
    K = np.array([[1000.0 / 3, 0.0, 639.5 + 1e-9], [0.0, 1001.123456789012, 359.25], [0, 0, 1]])
    n, dm = (4, 'equidistant') if model == 'fisheye' else (5, 'plumb_bob')
    D = RNG.normal(0.0, 0.1, n)
    D[0] = 1.2345678901234567e-17
    return CameraCalib(model, dm, (1280, 720), K, D, {'fx': 0.25}, 0.21, 30)


def test_camera_yaml_round_trip(tmp_path):
    R = cv2.Rodrigues(RNG.normal(0.0, 0.1, 3))[0]
    P = np.hstack([camera().K, [[-0.1 / 3], [0.0], [0.0]]])
    for model in ('pinhole', 'fisheye'):
        cam = camera(model)
        path = str(tmp_path / f'{model}.yaml')
        write_camera_yaml(path, 'my_cam', cam, R, P, {'rms': 0.21, 'note': 'two\nlines'})
        text = open(path).read()
        assert text.startswith('# rms: 0.21\n# note: two lines\nimage_width: 1280\n')
        doc = yaml.safe_load(text)
        assert set(doc) == PARSER_KEYS
        assert doc['distortion_coefficients']['rows'] == 1
        assert doc['distortion_coefficients']['cols'] == len(cam.D)
        assert (doc['projection_matrix']['rows'], doc['projection_matrix']['cols']) == (3, 4)
        name, back, R2, P2 = read_camera_yaml(path)
        assert name == 'my_cam' and back.model == model and back.size == (1280, 720)
        assert back.distortion_model == cam.distortion_model
        assert np.array_equal(back.K, cam.K) and np.array_equal(back.D, cam.D)   # exact floats
        assert np.array_equal(R2, R) and np.array_equal(P2, P)


def test_camera_yaml_real_parser(tmp_path):
    packages = pytest.importorskip('ament_index_python.packages')
    try:
        prefix = packages.get_package_prefix('camera_calibration_parsers')
    except packages.PackageNotFoundError:
        pytest.skip('camera_calibration_parsers not installed')
    convert = os.path.join(prefix, 'lib', 'camera_calibration_parsers', 'convert')
    if not os.path.exists(convert):
        pytest.skip('camera_calibration_parsers convert not found')
    cam = camera()
    src, dst = str(tmp_path / 'a.yaml'), str(tmp_path / 'b.yaml')
    write_camera_yaml(src, 'my_cam', cam, np.eye(3), np.hstack([cam.K, np.zeros((3, 1))]),
                      {'rms': 0.21})
    subprocess.run([convert, src, dst], check=True, capture_output=True, timeout=10)
    name, back, _, _ = read_camera_yaml(dst)
    assert name == 'my_cam' and back.distortion_model == 'plumb_bob'
    np.testing.assert_allclose(back.K, cam.K, rtol=1e-6)
    np.testing.assert_allclose(back.D, cam.D, rtol=1e-6, atol=1e-20)


def test_report(tmp_path):
    cam = camera()
    st = StereoExtrinsics(np.eye(3), np.array([-0.12, 0.0, 0.0005]), np.eye(3), np.eye(3), 0.3,
                          {'tx': np.float64(1e-4)}, 20)
    res = CalibrationResult((cam, cam), st, ((0, 1, 2), (0, 2)), {3: 'OUTLIER'},
                            ({0: 0.1, 1: np.float64(0.2), 2: 0.3}, {0: 0.1, 2: 0.2}), 0.3,
                            (('SOME_WARNING', {'value': np.float32(1.5)}),), 1.25)
    rect = Rectification((np.eye(3), np.eye(3)), (np.zeros((3, 4)), np.ones((3, 4))), 'square',
                         0.0)
    path = str(tmp_path / 'report.yaml')
    write_report(path, res, rect, {'board': BoardSpec(), 'topics': ('/l', '/r')})
    doc = yaml.safe_load(open(path))
    assert doc['rms'] == 0.3 and doc['rejected'] == {3: 'OUTLIER'}
    assert doc['cameras'][0]['K'] == cam.K.tolist() and doc['cameras'][0]['sigmas'] == {'fx': 0.25}
    assert doc['cameras'][1]['per_view_rms'] == {0: 0.1, 2: 0.2}
    assert doc['stereo']['baseline'] == pytest.approx(np.hypot(0.12, 0.0005))
    assert doc['rectification']['policy'] == 'square'
    assert doc['rectification']['P'][1][0] == [1.0] * 4
    assert doc['warnings'] == [{'code': 'SOME_WARNING', 'params': {'value': 1.5}}]
    assert doc['settings']['board']['cols'] == 8 and doc['settings']['topics'] == ['/l', '/r']


def test_dataset_round_trip(tmp_path):
    board = BoardSpec(type='charuco', cols=5, rows=4, marker_size=0.02)
    pts = board.object_points()
    corners = RNG.uniform(0, 600, (len(pts), 2)).astype(np.float32)
    d = Detection(corners, np.arange(len(pts), dtype=np.int32), pts, True,
                  corners[[0, 3, -1, -4]], (640, 480), 1.75, 7)
    part = Detection(corners[:8], np.arange(8, dtype=np.int32), pts[:8], False, corners[:4],
                     (640, 480), None, 4)
    png = cv2.imencode('.png', RNG.integers(0, 255, (48, 64), dtype=np.uint8))[1].tobytes()
    feats = {'x': 0.5, 'y': 0.25, 'size': 1 / 3, 'tilt_x_deg': -12.5, 'tilt_y_deg': 0.1}
    db = SampleDB(((640, 480), (640, 480)))
    db.add((d, part), (feats, feats), (png, None), 123456789012345)
    db.add((None, d), (None, feats), None, 5)
    path = str(tmp_path / 'ds')
    write_dataset(path, db.snapshot(), board, {'model': 'fisheye', 'mode': 'stereo'})
    assert sorted(os.listdir(path)) == ['cam0', 'session.yaml']
    snap, board2, meta = read_dataset(path)
    assert board2 == board and meta == {'model': 'fisheye', 'mode': 'stereo'}
    assert snap.image_sizes == ((640, 480), (640, 480))
    a, b = snap.samples
    assert (a.id, a.stamp_ns, b.id, b.stamp_ns) == (0, 123456789012345, 1, 5)
    assert a.images_png == (png, None) and b.images_png is None
    assert a.features == (feats, feats) and b.features == (None, feats) and b.views[0] is None
    for got, exp in ((a.views[0], d), (a.views[1], part), (b.views[1], d)):
        for key in ('corners', 'ids', 'object_points', 'outline'):
            x, y = getattr(got, key), getattr(exp, key)
            assert x.dtype == y.dtype and np.array_equal(x, y)
        assert (got.complete, got.image_size, got.sharpness, got.window) == \
            (exp.complete, exp.image_size, exp.sharpness, exp.window)


def test_run_dir(tmp_path):
    path = run_dir(str(tmp_path), 'left/cam', datetime(2026, 9, 28, 7, 5, 3))
    assert path == os.path.join(str(tmp_path), 'left_cam_20260928-070503') and os.path.isdir(path)
