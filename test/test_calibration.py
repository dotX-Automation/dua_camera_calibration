"""
Accuracy, robustness and speed tests of the calibration core on synthetic projections.

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

import time

import cv2
from dua_camera_calibration.calibration import (CalibConfig, calibrate, CalibrationError,
                                                quick_calibrate)
from dua_camera_calibration.detection import BoardSpec, Detection
from dua_camera_calibration.samples import DBSnapshot, Sample
import numpy as np
import pytest

W, H = 1280, 720
SIZE = (W, H)
BOARD = BoardSpec(cols=12, rows=8, square_size=0.03)
K_PIN = np.array([[900.0, 0.0, 645.0], [0.0, 905.0, 355.0], [0.0, 0.0, 1.0]])
D_PLUMB = np.array([-0.28, 0.09, 8e-4, -5e-4, 0.0])
D_RATIONAL = np.array([0.8, 0.1, 8e-4, -5e-4, 0.001, 1.1, 0.3, 0.01])
K_PIN2 = np.array([[910.0, 0.0, 630.0], [0.0, 912.0, 365.0], [0.0, 0.0, 1.0]])
D_PLUMB2 = np.array([-0.27, 0.085, -4e-4, 6e-4, 0.0])
K_FISH = np.array([[500.0, 0.0, 642.0], [0.0, 501.0, 358.0], [0.0, 0.0, 1.0]])
D_FISH = np.array([0.02, -0.01, 0.003, -5e-4])
K_FISH2 = np.array([[505.0, 0.0, 635.0], [0.0, 503.0, 362.0], [0.0, 0.0, 1.0]])
D_FISH2 = np.array([0.015, -0.008, 0.002, -3e-4])
R_LR = cv2.Rodrigues(np.radians([0.5, -1.0, 0.3]))[0]
T_LR = np.array([-0.12, 0.002, 0.001])
I3, ZERO3 = np.eye(3), np.zeros(3)


def project(model, K, D, obj, rvec, tvec):
    """Project (N, 3) points with the pinhole or fisheye model."""
    fn = cv2.fisheye.projectPoints if model == 'fisheye' else cv2.projectPoints
    return fn(np.asarray(obj, np.float64).reshape(-1, 1, 3), np.asarray(rvec, float).reshape(3, 1),
              np.asarray(tvec, float).reshape(3, 1), K, D)[0].reshape(-1, 2)


def make_views(rng, rig, n, noise=0.15, board=BOARD, margin=5.0):
    """
    Return n random board poses as lists of per-camera (N, 2) corners (+ Gaussian noise).

    rig: one (model, K, D, R, T) per camera, X_cam = R X_cam0 + T. Every corner lies inside
    every image, in front of every camera and within the non-folding part of pinhole models.
    """
    obj = board.object_points().astype(np.float64)
    out = []
    while len(out) < n:
        R = cv2.Rodrigues(np.radians([rng.uniform(-40, 40), rng.uniform(-40, 40), 0.0]))[0] @ \
            cv2.Rodrigues(np.radians([0.0, 0.0, rng.uniform(-20, 20)]))[0]
        ray = np.linalg.solve(rig[0][1], [rng.uniform(0, W), rng.uniform(0, H), 1.0])
        t = rng.uniform(0.4, 1.3) * ray - R @ obj.mean(axis=0)
        views = []
        for model, K, D, Rc, Tc in rig:
            pc = (Rc @ R @ obj.T).T + Rc @ t + Tc
            if pc[:, 2].min() < 0.1 or (model == 'pinhole' and np.max(
                    np.linalg.norm(pc[:, :2] / pc[:, 2:], axis=1)) > 1.2):
                break
            p = project(model, K, D, obj, cv2.Rodrigues(Rc @ R)[0], Rc @ t + Tc)
            if p.min() < margin or p[:, 0].max() > W - 1 - margin or \
                    p[:, 1].max() > H - 1 - margin:
                break
            views.append(p + rng.normal(0, noise, p.shape))
        else:
            out.append(views)
    return out


def detection(corners, ids=None, board=BOARD):
    """Build a Detection from full-board corners, optionally keeping only ids."""
    ids = np.arange(board.n_corners) if ids is None else np.asarray(ids)
    c = np.asarray(corners, np.float32)[ids]
    n = board.grid[0]
    outline = c[[0, n - 1, -1, -n]] if len(ids) == board.n_corners else c[[0, 0, -1, -1]]
    return Detection(c, ids.astype(np.int32), board.object_points()[ids],
                     len(ids) == board.n_corners, outline, SIZE)


def snapshot(poses, n_cams=1):
    """Build a DBSnapshot; poses[i][c] are the corners of camera c (None: not seen)."""
    samples = tuple(
        Sample(i, i, tuple(None if c is None else detection(c) for c in p), (None,) * n_cams)
        for i, p in enumerate(poses))
    return DBSnapshot(1, samples, (SIZE,) * n_cams)


def _check_intrinsics(cam, K, rel_f=0.003, c_px=1.0):
    assert abs(cam.K[0, 0] / K[0, 0] - 1) < rel_f
    assert abs(cam.K[1, 1] / K[1, 1] - 1) < rel_f
    assert np.hypot(cam.K[0, 2] - K[0, 2], cam.K[1, 2] - K[1, 2]) < c_px


@pytest.mark.parametrize('dist, D', [('plumb_bob', D_PLUMB), ('rational_polynomial', D_RATIONAL)])
def test_pinhole_accuracy(dist, D):
    rng = np.random.default_rng(1)
    poses = make_views(rng, [('pinhole', K_PIN, D, I3, ZERO3)], 60, noise=0.1)
    stages = []
    res = calibrate(snapshot(poses), BOARD, CalibConfig(distortion_model=dist), stages.append)
    cam = res.cameras[0]
    _check_intrinsics(cam, K_PIN)
    assert cam.distortion_model == dist and cam.D.shape == (len(D),)
    assert 0.1 < res.rms < 0.2 and res.rms == cam.rms
    assert 'solving' in stages and 'sigmas' in stages
    assert len(res.per_view_rms[0]) == len(res.used_ids[0]) == cam.n_views
    expected = {'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'}
    if dist == 'rational_polynomial':
        expected |= {'k3', 'k4', 'k5', 'k6'}
    assert set(cam.sigmas) == expected


def test_fisheye_accuracy():
    rng = np.random.default_rng(2)
    poses = make_views(rng, [('fisheye', K_FISH, D_FISH, I3, ZERO3)], 60, noise=0.1)
    res = calibrate(snapshot(poses), BOARD, CalibConfig(model='fisheye'))
    cam = res.cameras[0]
    _check_intrinsics(cam, K_FISH, rel_f=0.005)
    assert cam.distortion_model == 'equidistant' and cam.D.shape == (4,)
    assert set(cam.sigmas) == {'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'}
    assert not res.warnings


def test_fisheye_ill_conditioned_view_dropped(monkeypatch):
    rng = np.random.default_rng(3)
    poses = make_views(rng, [('fisheye', K_FISH, D_FISH, I3, ZERO3)], 12)
    real, calls = cv2.fisheye.calibrate, []

    def fake(obj, img, *args, **kwargs):
        calls.append(kwargs['flags'])
        if len(calls) == 1:
            raise cv2.error('CALIB_CHECK_COND - Ill-conditioned matrix for input array 3')
        if len(calls) == 2:
            raise cv2.error('(-215:Assertion failed) svd.w.at<double>(0) / ... < thresh_cond')
        return real(obj, img, *args, **kwargs)

    monkeypatch.setattr(cv2.fisheye, 'calibrate', fake)
    res = calibrate(snapshot(poses), BOARD, CalibConfig(model='fisheye', outlier_rejection=False))
    assert res.rejected == {3: 'ILL_CONDITIONED'}
    assert res.warnings == (('FISHEYE_CHECK_COND_DISABLED', {'camera': 0}),)
    assert calls[0] & cv2.fisheye.CALIB_CHECK_COND and not calls[2] & cv2.fisheye.CALIB_CHECK_COND
    _check_intrinsics(res.cameras[0], K_FISH, rel_f=0.01, c_px=3.0)


def test_flags():
    rng = np.random.default_rng(4)
    poses = make_views(rng, [('pinhole', K_PIN, D_PLUMB, I3, ZERO3)], 30)
    cfg = CalibConfig(fix_aspect_ratio=True, fix_principal_point=True, zero_tangent_dist=True,
                      fix_k3=False)
    cam = calibrate(snapshot(poses), BOARD, cfg).cameras[0]
    assert cam.K[0, 0] == cam.K[1, 1]
    assert (cam.K[0, 2], cam.K[1, 2]) == ((W - 1) / 2, (H - 1) / 2)
    assert cam.D[2] == cam.D[3] == 0.0
    assert set(cam.sigmas) == {'fx', 'fy', 'k1', 'k2', 'k3'}
    assert cam.sigmas['fx'] == cam.sigmas['fy']


def test_outlier_rejected():
    rng = np.random.default_rng(5)
    poses = make_views(rng, [('pinhole', K_PIN, D_PLUMB, I3, ZERO3)], 40, noise=0.1)
    poses[7] = [poses[7][0] + rng.normal(0, 1.0, poses[7][0].shape)]
    res = calibrate(snapshot(poses), BOARD, CalibConfig())
    assert res.rejected.get(7) == 'OUTLIER'
    assert 7 not in res.used_ids[0] and len(res.used_ids[0]) >= 35
    _check_intrinsics(res.cameras[0], K_PIN)


def test_cold_start_pose_minimum_recovered():
    # this far, tilted view near the left edge lands in a wrong pose minimum when calibrateCamera
    # starts cold (RMS 0.69 px instead of 0.14 px); the warm re-solve fixes it
    rng = np.random.default_rng(5)
    poses = make_views(rng, [('pinhole', K_PIN, D_PLUMB, I3, ZERO3)], 20, noise=0.1)
    bad = project('pinhole', K_PIN, D_PLUMB, BOARD.object_points(),
                  np.radians([-34.9, -24.0, 12.3]), [-0.725, 0.04, 1.188])
    poses.append([bad + rng.normal(0, 0.1, bad.shape)])
    res = calibrate(snapshot(poses), BOARD, CalibConfig(outlier_rejection=False))
    assert res.rms < 0.16 and res.per_view_rms[0][20] < 0.2


def test_view_selection():
    rng = np.random.default_rng(6)
    poses = make_views(rng, [('pinhole', K_PIN, D_PLUMB, I3, ZERO3)], 100)
    stages = []
    res = calibrate(snapshot(poses), BOARD, CalibConfig(max_views=40, outlier_rejection=False),
                    stages.append)
    assert len(res.used_ids[0]) == 40
    assert sorted(res.rejected) == sorted(set(range(100)) - set(res.used_ids[0]))
    assert set(res.rejected.values()) == {'NOT_SELECTED'}
    assert stages[0] == 'selecting'
    _check_intrinsics(res.cameras[0], K_PIN)


def test_sigmas_match_empirical_spread():
    rng = np.random.default_rng(7)
    exact = make_views(rng, [('pinhole', K_PIN, D_PLUMB, I3, ZERO3)], 15, noise=0.0)
    cfg = CalibConfig(outlier_rejection=False)
    est, sig = [], []
    for _ in range(40):
        noisy = [[v + rng.normal(0, 0.2, v.shape) for v in p] for p in exact]
        cam = calibrate(snapshot(noisy), BOARD, cfg).cameras[0]
        est.append([cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2], cam.D[0]])
        sig.append([cam.sigmas[k] for k in ('fx', 'fy', 'cx', 'cy', 'k1')])
    ratio = np.std(est, axis=0, ddof=1) / np.mean(sig, axis=0)
    assert np.all((ratio > 1 / 1.5) & (ratio < 1.5)), ratio


def test_timing_150_samples():
    rng = np.random.default_rng(8)
    snap = snapshot(make_views(rng, [('pinhole', K_PIN, D_PLUMB, I3, ZERO3)], 150))
    t0 = time.monotonic()
    res = calibrate(snap, BOARD, CalibConfig())
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0 and res.duration_s <= elapsed
    assert len(res.used_ids[0]) <= 60
    _check_intrinsics(res.cameras[0], K_PIN)


def test_quick_calibrate():
    rng = np.random.default_rng(9)
    views = [detection(p[0]) for p in
             make_views(rng, [('pinhole', K_PIN, D_PLUMB, I3, ZERO3)], 40)]
    cfg = CalibConfig()
    first = quick_calibrate(views[:20], SIZE, cfg)
    t0 = time.monotonic()
    cam = quick_calibrate(views, SIZE, cfg, init=first)
    assert time.monotonic() - t0 < 0.2
    _check_intrinsics(cam, K_PIN, rel_f=0.005, c_px=2.0)
    assert cam.n_views == 40 and {'fx', 'fy', 'cx', 'cy'} <= set(cam.sigmas)


def _stereo_rig(model):
    if model == 'fisheye':
        return [('fisheye', K_FISH, D_FISH, I3, ZERO3), ('fisheye', K_FISH2, D_FISH2, R_LR, T_LR)]
    return [('pinhole', K_PIN, D_PLUMB, I3, ZERO3), ('pinhole', K_PIN2, D_PLUMB2, R_LR, T_LR)]


@pytest.mark.parametrize('model', ['pinhole', 'fisheye'])
def test_stereo_accuracy(model):
    rng = np.random.default_rng(10)
    rig = _stereo_rig(model)
    poses = make_views(rng, rig, 60, noise=0.1)
    poses += [[p[0], None] for p in make_views(rng, rig[:1], 3, noise=0.1)]   # left-only
    res = calibrate(snapshot(poses, 2), BOARD, CalibConfig(model=model))
    st = res.stereo
    singles = {60, 61, 62}                   # camera 0 has 63 views: selection is active
    assert abs(np.linalg.norm(st.T) / np.linalg.norm(T_LR) - 1) < 0.01
    rot_err = np.degrees(np.linalg.norm(cv2.Rodrigues(st.R @ R_LR.T)[0]))
    assert rot_err < 0.1
    # fisheye RMS is measured in normalized space scaled by f: the periphery inflates it
    outliers = {i for i, why in res.rejected.items() if why == 'OUTLIER'}
    assert st.n_pairs == 60 - len(outliers - singles) and res.rms == st.rms and 0.1 < st.rms < 0.4
    assert set(st.sigmas) == {'rx', 'ry', 'rz', 'tx', 'ty', 'tz'}
    # conditional on the fixed intrinsics, hence much smaller than the errors above
    assert all(0 < v < 1e-3 for v in st.sigmas.values())
    assert all(i in res.used_ids[0] or i in res.rejected for i in singles)
    assert len(res.used_ids[0]) <= 60 and not singles & set(res.used_ids[1])
    _check_intrinsics(res.cameras[0], rig[0][1], rel_f=0.005, c_px=2.0)
    _check_intrinsics(res.cameras[1], rig[1][1], rel_f=0.005, c_px=2.0)


def test_errors():
    rng = np.random.default_rng(11)
    rig = _stereo_rig('pinhole')
    poses = make_views(rng, rig, 4)
    with pytest.raises(CalibrationError) as e:
        calibrate(snapshot([[p[0]] for p in poses]), BOARD, CalibConfig())
    assert e.value.code == 'TOO_FEW_VIEWS' and e.value.params['n'] == 4
    poses += [[p[0], None] for p in make_views(rng, rig[:1], 6)]
    poses += [[None, p[1]] for p in make_views(rng, rig, 6)]
    with pytest.raises(CalibrationError) as e:
        calibrate(snapshot(poses, 2), BOARD, CalibConfig())
    assert e.value.code == 'TOO_FEW_PAIRS' and e.value.params == {'n': 4, 'min': 6}
