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

import cv2
from dua_camera_calibration import calibration
from dua_camera_calibration.calibration import CalibConfig, CameraCalib
from dua_camera_calibration.detection import BoardSpec, Detection
from dua_camera_calibration.guidance import (evaluate_frame, frame_hints, FrameVerdict,
                                             GateConfig, GuidanceConfig, GuidanceEngine)
from dua_camera_calibration.samples import SampleDB
import numpy as np
import pytest

W, H = 1280, 720
F = 1280.0          # equals the bootstrap focal length: tilts are exact without an estimate
K = np.array([[F, 0.0, (W - 1) / 2], [0.0, F, (H - 1) / 2], [0.0, 0.0, 1.0]])
BOARD = BoardSpec()  # 8 x 6 inner corners, 25 mm
MONO = ((W, H),)
STEREO = ((W, H), (W, H))
# 3 x 3 fronto-parallel views covering the whole image (board 448 x 320 px at 0.5 m)
GRID9 = [(u, v) for v in (0.25, 0.5, 0.75) for u in (0.2, 0.5, 0.8)]


def rot(rx, ry):
    """Rotation Rx(rx) Ry(ry), degrees."""
    a, b = np.radians(rx), np.radians(ry)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return Rx @ Ry


def det(u=0.5, v=0.5, z=0.5, rx=0.0, ry=0.0, sharpness=1.0, shift=(0.0, 0.0), size=(W, H),
        noise=0.0, rng=None):
    """Board centered at the normalized image point (u, v), depth z [m], rotated by rx, ry."""
    k = K.copy()
    k[0, 2], k[1, 2] = (size[0] - 1) / 2, (size[1] - 1) / 2
    obj = BOARD.object_points()
    R = rot(rx, ry)
    t = z * np.linalg.solve(k, [u * size[0], v * size[1], 1.0]) - R @ obj.mean(axis=0)
    img, _ = cv2.projectPoints(obj.astype(np.float64), cv2.Rodrigues(R)[0], t, k, None)
    c = img.reshape(-1, 2) + shift
    if noise:
        c = c + rng.normal(0.0, noise, c.shape)
    c = c.astype(np.float32)
    n = BOARD.grid[0]
    return Detection(corners=c, ids=np.arange(len(c), dtype=np.int32), object_points=obj,
                     complete=True, outline=c[[0, n - 1, -1, -n]], image_size=size,
                     sharpness=sharpness, window=5)


def gate(views, prev='same', snap=None, gates=GateConfig(), sizes=MONO) -> FrameVerdict:
    return evaluate_frame(views, sizes, views if prev == 'same' else prev, 0.033, snap, None,
                          gates, BOARD)


def fake_quick(rel_of_n):
    """quick_calibrate stand-in returning the true camera with rel_ci = rel_of_n(len(views))."""
    def quick(views, size, cfg, init=None, max_iter=30):
        r = rel_of_n(len(views)) / 1.96
        return CameraCalib('pinhole', 'plumb_bob', size, K.copy(), np.zeros(5),
                           {'fx': r * F, 'fy': r * F, 'cx': r * size[0], 'cy': r * size[1]},
                           0.1, len(views))
    return quick


def add(db, views):
    """Add a sample with features computed by evaluate_frame."""
    sizes = db.snapshot().image_sizes
    v = gate(views, snap=None, gates=GateConfig(check_motion=False), sizes=sizes)
    assert v.hard_ok, v.reason
    db.add(v.views, v.features, None, 0)


def ind(state, name, cam=0):
    return next(i for i in state.indicators if i.id == name and i.camera == cam)


def test_gates_in_order():
    assert gate((None,)).reason == 'NOT_DETECTED'
    d = det()
    few = Detection(d.corners[:5], d.ids[:5], d.object_points[:5], False, d.outline, d.image_size)
    assert gate((few,)).reason == 'TOO_FEW_CORNERS'
    edge = det(shift=(5.5 - float(det().corners[:, 0].min()), 0.0))   # closer than window + 1
    assert gate((edge,)).reason == 'NEAR_BORDER'
    inside = det(shift=(6.5 - float(det().corners[:, 0].min()), 0.0))
    assert gate((inside,)).ok
    assert gate((det(sharpness=3.5),)).reason == 'BLURRY'
    assert gate((det(sharpness=None),)).ok
    blurry_edge = det(sharpness=3.5, shift=(5.5 - float(det().corners[:, 0].min()), 0.0))
    assert gate((blurry_edge,)).reason == 'NEAR_BORDER'              # border before blur
    moved = (det(shift=(5.0, 0.0)),)
    assert gate((det(sharpness=3.5),), prev=moved).reason == 'BLURRY'   # blur before motion
    assert gate((det(ry=65.0),), prev=moved).reason == 'MOVING'        # motion before tilt
    assert gate((det(ry=65.0),)).reason == 'TOO_TILTED'
    v = gate((det(),))
    assert v.ok and v.hard_ok and v.reason == 'OK' and v.views[0] is not None


def test_motion_gate():
    d = det()
    assert gate((d,), prev=(det(shift=(3.0, 0.0)),)).reason == 'MOVING'
    assert gate((d,), prev=(det(shift=(1.0, 0.0)),)).ok
    v = gate((d,), prev=None)
    assert v.reason == 'MOVING' and not v.hard_ok        # no previous detection counts as moving
    assert gate((d,), prev=None, gates=GateConfig(check_motion=False)).ok
    # threshold scales with the image diagonal: 2.25 px at 1080p
    big = (1920, 1080)
    d2 = det(size=big)
    assert gate((d2,), prev=(det(size=big, shift=(2.0, 0.0)),), sizes=(big,)).ok
    v = gate((d2,), prev=(det(size=big, shift=(2.5, 0.0)),), sizes=(big,))
    assert v.reason == 'MOVING' and v.params['limit'] == pytest.approx(2.25, rel=1e-3)
    assert v.metrics[0]['motion_px'] == pytest.approx(2.5, abs=1e-3)


def test_features_and_tilt_signs():
    f = gate((det(rx=30.0),)).features[0]
    assert f['tilt_x_deg'] == pytest.approx(30.0, abs=0.5) and abs(f['tilt_y_deg']) < 0.5
    f = gate((det(ry=-30.0),)).features[0]
    assert f['tilt_y_deg'] == pytest.approx(-30.0, abs=0.5) and abs(f['tilt_x_deg']) < 0.5
    f = gate((det(u=0.25, v=0.75),)).features[0]
    assert f['x'] == pytest.approx(0.25, abs=1e-3) and f['y'] == pytest.approx(0.75, abs=1e-3)
    assert f['size'] == pytest.approx(np.sqrt(448 * 320 / (W * H)), rel=1e-3)
    assert gate((det(ry=50.0),)).metrics[0]['tilt'] == pytest.approx(50.0, abs=0.5)


def test_novelty_and_coverage():
    db = SampleDB(MONO)
    add(db, (det(),))
    snap = db.snapshot()
    v = gate((det(),), snap=snap)
    assert not v.ok and v.hard_ok and v.reason == 'TOO_SIMILAR'
    assert v.metrics[0]['novelty'] == pytest.approx(0.0, abs=1e-6)
    assert gate((det(ry=30.0),), snap=snap).ok                    # far in feature space
    small = gate((det(u=0.53),), snap=snap)                       # same cells, close features
    assert small.reason == 'TOO_SIMILAR'
    v = gate((det(u=0.6),), snap=snap)                            # close features, new cells
    assert v.metrics[0]['novelty'] < GateConfig().min_novelty and v.ok


def test_stereo_single_side_rule():
    d = det()
    v = gate((d, d), sizes=STEREO)
    assert v.ok and v.views == (d, d)
    v = gate((d, None), sizes=STEREO)                  # empty DB: fills empty cells
    assert v.ok and v.hard_ok and v.views == (d, None) and v.features[1] is None
    db = SampleDB(STEREO)
    add(db, (d, d))
    v = gate((d, None), snap=db.snapshot(), sizes=STEREO)
    assert not v.ok and not v.hard_ok and v.reason == 'NOT_DETECTED'
    assert v.params['camera'] == 1
    e = det(u=0.2, v=0.25)                             # left side hits new cells
    v = gate((e, det(sharpness=5.0)), snap=db.snapshot(), sizes=STEREO)
    assert v.ok and v.views[0] is e and v.views[1] is None
    v = gate((d, d), snap=db.snapshot(), sizes=STEREO)
    assert v.reason == 'TOO_SIMILAR' and v.hard_ok and not v.ok


def test_frame_hints():
    expect = {'MOVING': 'HOLD_STILL', 'BLURRY': 'IMPROVE_FOCUS_OR_LIGHT',
              'NEAR_BORDER': 'MOVE_AWAY_FROM_EDGE', 'NOT_DETECTED': 'SHOW_WHOLE_BOARD',
              'TOO_FEW_CORNERS': 'SHOW_WHOLE_BOARD', 'TOO_TILTED': 'TILT_LESS'}
    for reason, code in expect.items():
        (h,) = frame_hints(FrameVerdict(False, False, reason, {'camera': 1}))
        assert h.code == code and h.camera == 1
    assert frame_hints(FrameVerdict(True, True, 'OK')) == ()
    assert frame_hints(FrameVerdict(False, True, 'TOO_SIMILAR')) == ()


def test_engine_indicators_and_hints(monkeypatch):
    monkeypatch.setattr(calibration, 'quick_calibrate', fake_quick(lambda n: 0.001))
    db = SampleDB(MONO)
    eng = GuidanceEngine(BOARD, CalibConfig(), GuidanceConfig(), MONO)
    add(db, (det(),))
    s = eng.update(db.snapshot())
    assert s.hints[0].code == 'COVER_REGION'
    assert s.hints[0].params['region'] == 'top-left'
    assert s.hints[0].target == pytest.approx((1 / 16, 1 / 12))
    assert s.hints[-1].code == 'KEEP_GOING' and not s.ready
    assert s.coverage[0].shape == (6, 8)
    for u, v in GRID9:
        add(db, (det(u, v),))
    s = eng.update(db.snapshot())
    assert ind(s, 'coverage').progress == 1.0
    assert ind(s, 'tilt_x').progress == pytest.approx(1 / 3)
    assert ind(s, 'distance').progress == pytest.approx(1 / 3)
    assert ind(s, 'uncertainty').progress == 1.0 and s.estimate is not None
    assert s.hints[0].code == 'TILT_LEFT_RIGHT' and s.hints[0].params == {'direction': 'right'}
    assert [h.code for h in s.hints[:3]] == ['TILT_LEFT_RIGHT', 'TILT_UP_DOWN', 'MOVE_CLOSER']
    assert s.hints[1].params == {'direction': 'up'}
    for rx, ry in ((30, 0), (-30, 0), (0, 30), (0, -30)):
        add(db, (det(rx=rx, ry=ry),))
    s = eng.update(db.snapshot())
    assert ind(s, 'tilt_x').progress == 1.0 and ind(s, 'tilt_y').progress == 1.0
    assert s.hints[0].code == 'MOVE_CLOSER' and not s.ready
    add(db, (det(z=0.3),))
    add(db, (det(z=1.0),))
    s = eng.update(db.snapshot())
    assert all(i.progress == 1.0 for i in s.indicators)
    assert s.ready and s.hints[0].code == 'READY' and s.hints[0].params == {'samples': 16}
    assert eng.update(db.snapshot()) is s                     # same version: cached


def test_ready_rule(monkeypatch):
    target = GuidanceConfig().target_rel_ci
    cases = ((lambda n: 4 * target, True), (lambda n: 1000 * target / n ** 2, False))
    for rel_of_n, plateau in cases:
        monkeypatch.setattr(calibration, 'quick_calibrate', fake_quick(rel_of_n))
        db = SampleDB(MONO)
        eng = GuidanceEngine(BOARD, CalibConfig(), GuidanceConfig(), MONO)
        poses = GRID9 + [(0.35, 0.4), (0.65, 0.6), (0.4, 0.62)]    # fronto only: tilts not full
        for k, (u, v) in enumerate(poses):
            add(db, (det(u, v),))
            s = eng.update(db.snapshot())
            assert not s.ready or k == len(poses) - 1
        assert ind(s, 'uncertainty').progress < 1.0 and ind(s, 'tilt_y').progress < 1.0
        assert s.ready == plateau


def test_estimate_failure_keeps_previous(monkeypatch):
    monkeypatch.setattr(calibration, 'quick_calibrate', fake_quick(lambda n: 0.01))
    db = SampleDB(MONO)
    eng = GuidanceEngine(BOARD, CalibConfig(), GuidanceConfig(), MONO)
    for u, v in GRID9[:5]:
        add(db, (det(u, v),))
    eng.update(db.snapshot())
    est = eng.estimate
    assert est is not None and est.rel_ci == pytest.approx(0.01) and est.n_views == 5

    def boom(*args, **kwargs):
        raise RuntimeError('ill-conditioned')

    monkeypatch.setattr(calibration, 'quick_calibrate', boom)
    add(db, (det(*GRID9[5]),))
    s = eng.update(db.snapshot())
    assert eng.estimate is est and s.estimate is est


def test_stereo_engine(monkeypatch):
    monkeypatch.setattr(calibration, 'quick_calibrate', fake_quick(lambda n: 0.001))
    db = SampleDB(STEREO)
    eng = GuidanceEngine(BOARD, CalibConfig(), GuidanceConfig(), STEREO)
    add(db, (det(), det()))
    add(db, (det(0.2, 0.25), None))
    s = eng.update(db.snapshot())
    assert len(s.indicators) == 10 and len(s.coverage) == 2
    assert {i.camera for i in s.indicators} == {0, 1}
    assert s.hints[-1].code == 'KEEP_GOING' and s.hints[-1].params == {'samples': 2, 'pairs': 1}
    assert ind(s, 'coverage', 0).progress > ind(s, 'coverage', 1).progress


def test_live_estimate_with_quick_calibrate():
    rng = np.random.default_rng(1)
    db = SampleDB(MONO)
    poses = [{'u': u, 'v': v} for u, v in GRID9] + [
        {'rx': 30}, {'rx': -30}, {'ry': 30}, {'ry': -30}, {'z': 0.3}, {'z': 1.0}]
    for p in poses:
        add(db, (det(noise=0.1, rng=rng, **p),))
    eng = GuidanceEngine(BOARD, CalibConfig(), GuidanceConfig(), MONO)
    try:
        calibration.quick_calibrate([d for _, d in db.snapshot().views_of(0)], (W, H),
                                    CalibConfig())
    except NotImplementedError:
        pytest.skip('calibration.quick_calibrate not implemented yet')
    s = eng.update(db.snapshot())
    assert s.estimate is not None
    cam = s.estimate.cams[0]
    assert cam.K[0, 0] == pytest.approx(F, rel=0.01) and cam.K[1, 1] == pytest.approx(F, rel=0.01)
    assert 0.0 < s.estimate.rel_ci < 0.05
