"""
Invariant tests of the P policies, stereo rectification, maps and validation metrics.

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

import cv2
from dua_camera_calibration.calibration import CameraCalib, StereoExtrinsics
from dua_camera_calibration.rectification import (make_maps, rectify, rectify_points,
                                                  RectifyConfig, validate_views)
import numpy as np
import pytest
from test_calibration import (BOARD, D_FISH, D_FISH2, D_PLUMB, D_PLUMB2, detection, H, I3,
                              K_FISH, K_FISH2, K_PIN, K_PIN2, make_views, R_LR, SIZE, T_LR, W,
                              ZERO3)

PIN = CameraCalib('pinhole', 'plumb_bob', SIZE, K_PIN, D_PLUMB)
PIN2 = CameraCalib('pinhole', 'plumb_bob', SIZE, K_PIN2, D_PLUMB2)
FISH = CameraCalib('fisheye', 'equidistant', SIZE, K_FISH, D_FISH)
FISH2 = CameraCalib('fisheye', 'equidistant', SIZE, K_FISH2, D_FISH2)
EXTR = StereoExtrinsics(R_LR, T_LR, np.eye(3), np.eye(3), 0.0)
MAP_TOL = 1 / 32 + 1e-3          # CV_16SC2 maps are quantized to 1/32 px


def _float_maps(cam, R, P, size):
    return cv2.convertMaps(*make_maps(cam, R, P, size), cv2.CV_32FC1)


def _border_pixels():
    xs, ys = np.arange(W, dtype=float), np.arange(H, dtype=float)
    return np.vstack([np.c_[xs, 0 * xs], np.c_[xs, 0 * xs + H - 1],
                      np.c_[0 * ys, ys], np.c_[0 * ys + W - 1, ys]])


def _inside(x, y, tol):
    return x.min() >= -tol and y.min() >= -tol and x.max() <= W - 1 + tol and \
        y.max() <= H - 1 + tol


@pytest.mark.parametrize('cam', [PIN, FISH], ids=['pinhole', 'fisheye'])
@pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
def test_square_and_aspect(cam, alpha):
    sq = rectify((cam,), None, RectifyConfig(policy='square', alpha=alpha))
    assert sq.policy == 'square' and not sq.warnings and sq.alpha == alpha
    assert abs(sq.P[0][0, 0] - sq.P[0][1, 1]) < 1e-9
    assert np.array_equal(sq.R[0], I3) and np.all(sq.P[0][:, 3] == 0)
    asp = rectify((cam,), None, RectifyConfig(policy='aspect', alpha=alpha)).P[0]
    assert abs(asp[0, 0] / asp[1, 1] - cam.K[0, 0] / cam.K[1, 1]) < 1e-12


@pytest.mark.parametrize('cam', [PIN, FISH], ids=['pinhole', 'fisheye'])
@pytest.mark.parametrize('policy', ['square', 'aspect'])
def test_alpha0_output_border_maps_inside_source(cam, policy):
    rect = rectify((cam,), None, RectifyConfig(policy=policy, alpha=0.0))
    mx, my = _float_maps(cam, rect.R[0], rect.P[0], SIZE)
    bx = np.r_[mx[0], mx[-1], mx[:, 0], mx[:, -1]]
    by = np.r_[my[0], my[-1], my[:, 0], my[:, -1]]
    assert _inside(bx, by, MAP_TOL)
    if cam.model == 'pinhole':      # tight (fisheye: the FOV clip is usually the limit)
        assert min(bx.min(), by.min(), W - 1 - bx.max(), H - 1 - by.max()) < 0.5


@pytest.mark.parametrize('policy', ['square', 'aspect'])
def test_alpha1_source_border_lands_inside_output(policy):
    rect = rectify((PIN,), None, RectifyConfig(policy=policy, alpha=1.0))
    p = rectify_points(PIN, rect.R[0], rect.P[0], _border_pixels())
    assert _inside(p[:, 0], p[:, 1], 0.05)
    assert min(p[:, 0].min(), p[:, 1].min(), W - 1 - p[:, 0].max(), H - 1 - p[:, 1].max()) < 0.5


@pytest.mark.parametrize('alpha', [0.0, 1.0])
def test_opencv_and_k_policies(alpha):
    P = rectify((PIN,), None, RectifyConfig(policy='opencv', alpha=alpha)).P[0]
    assert np.allclose(P[:, :3], cv2.getOptimalNewCameraMatrix(K_PIN, D_PLUMB, SIZE, alpha)[0])
    P = rectify((FISH,), None, RectifyConfig(policy='opencv', alpha=alpha)).P[0]
    assert np.allclose(P[:, :3], cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_FISH, D_FISH, SIZE, I3, balance=alpha))
    rect = rectify((PIN,), None, RectifyConfig(policy='k', alpha=alpha))
    assert rect.policy == 'k' and np.array_equal(rect.P[0][:, :3], K_PIN)


@pytest.mark.parametrize('cam', [
    CameraCalib('pinhole', 'plumb_bob', SIZE,
                np.array([[500.0, 0, 640], [0, 500.0, 360], [0, 0, 1]]),
                np.array([-0.9, 0.0, 0.0, 0.0, 0.0])),
    CameraCalib('fisheye', 'equidistant', SIZE, K_FISH, np.array([-0.5, 0.05, 0.0, 0.0]))],
    ids=['pinhole', 'fisheye'])
def test_non_invertible_model_falls_back_to_k(cam):
    rect = rectify((cam,), None, RectifyConfig(policy='square'))
    assert rect.policy == 'k' and np.array_equal(rect.P[0][:, :3], cam.K)
    assert [w[0] for w in rect.warnings] == ['MODEL_NOT_INVERTIBLE']
    assert rect.warnings[0][1]['camera'] == 0


@pytest.mark.parametrize('model', ['pinhole', 'fisheye'])
@pytest.mark.parametrize('alpha', [0.0, 1.0])
def test_stereo_rectification(model, alpha):
    cams = (PIN, PIN2) if model == 'pinhole' else (FISH, FISH2)
    rect = rectify(cams, EXTR, RectifyConfig(alpha=alpha))
    assert rect.policy == 'stereo' and len(rect.P) == 2
    for P in rect.P:
        assert abs(P[0, 0] - P[1, 1]) < 1e-9
    assert rect.P[0][0, 0] == rect.P[1][0, 0]
    rig = [(c.model, c.K, c.D, R, T) for c, R, T in zip(cams, (I3, R_LR), (ZERO3, T_LR))]
    for left, right in make_views(np.random.default_rng(20), rig, 10, noise=0.0):
        pl = rectify_points(cams[0], rect.R[0], rect.P[0], left)
        pr = rectify_points(cams[1], rect.R[1], rect.P[1], right)
        assert np.abs(pl[:, 1] - pr[:, 1]).max() < 0.1


@pytest.mark.parametrize('cam', [PIN, FISH], ids=['pinhole', 'fisheye'])
def test_half_size_maps_match_downsampled_full_maps(cam):
    rect = rectify((cam,), None, RectifyConfig())
    fx, fy = _float_maps(cam, rect.R[0], rect.P[0], SIZE)
    hx, hy = _float_maps(cam, rect.R[0], rect.P[0], (W // 2, H // 2))
    down = [m.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3)) for m in (fx, fy)]
    assert max(np.abs(hx - down[0]).max(), np.abs(hy - down[1]).max()) < 0.1


@pytest.mark.parametrize('cam', [PIN, FISH], ids=['pinhole', 'fisheye'])
def test_validate_mono(cam):
    corners = make_views(np.random.default_rng(21), [(cam.model, cam.K, cam.D, I3, ZERO3)], 1,
                         noise=0.0)[0][0]
    det = detection(corners)
    v = validate_views((det,), (cam,), None, BOARD).values
    assert v['reproj_rms_px'][0] < 1e-3 and v['straightness_rms_px'][0] < 1e-3
    D = cam.D.copy()
    D[0] += 0.1
    wrong = CameraCalib(cam.model, cam.distortion_model, SIZE, cam.K, D)
    assert validate_views((det,), (wrong,), None, BOARD).values['straightness_rms_px'][0] > 0.05
    assert validate_views((None,), (cam,), None, BOARD) is None


@pytest.mark.parametrize('model', ['pinhole', 'fisheye'])
def test_validate_stereo(model):
    cams = (PIN, PIN2) if model == 'pinhole' else (FISH, FISH2)
    rect = rectify(cams, EXTR, RectifyConfig())
    rig = [(c.model, c.K, c.D, R, T) for c, R, T in zip(cams, (I3, R_LR), (ZERO3, T_LR))]
    left, right = make_views(np.random.default_rng(22), rig, 1, noise=0.0)[0]
    ids = np.arange(10, BOARD.n_corners)                # partial right view
    v = validate_views((detection(left), detection(right, ids)), cams, rect, BOARD).values
    assert v['epipolar_rms_px'] < 0.01 and abs(v['square_size_err']) < 1e-3
    assert abs(v['square_size_m'] - BOARD.square_size) < 1e-4
    assert len(v['reproj_rms_px']) == 2 and max(v['straightness_rms_px']) < 1e-3
    only_left = validate_views((detection(left), None), cams, rect, BOARD).values
    assert only_left['reproj_rms_px'][1] is None and 'epipolar_rms_px' not in only_left
