"""
Rectified projection (P) policies, rectification maps and live validation metrics.

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

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# CONTRACT (do not change field names/types without updating every user):

POLICIES = ('square', 'aspect', 'opencv', 'k')


@dataclass(frozen=True)
class RectifyConfig:
    """Rectification options."""

    policy: str = 'square'            # mono only, see POLICIES
    alpha: float = 0.0                # 0: all output pixels valid, 1: all source pixels kept
    zero_disparity: bool = True       # stereo only
    fisheye_max_fov_deg: float = 120.0


@dataclass(frozen=True)
class Rectification:
    """
    Per-camera rectification (R) and projection (P) matrices.

    policy is the one actually applied: 'k' after a MODEL_NOT_INVERTIBLE fallback, 'stereo'
    for stereo pairs.
    """

    R: tuple                          # tuple[(3, 3) float64, ...]
    P: tuple                          # tuple[(3, 4) float64, ...]
    policy: str
    alpha: float
    warnings: tuple = ()              # tuple[(code, params dict), ...]


@dataclass(frozen=True)
class ValidationMetrics:
    """
    Live validation of a calibration on one frame.

    Keys: 'reproj_rms_px', 'straightness_rms_px' (tuples with one entry per camera, None when
    that camera has nothing to measure); stereo adds 'epipolar_rms_px', 'square_size_m',
    'square_size_err' (relative).
    """

    values: dict = field(default_factory=dict)


WARNING_CODES = ('MODEL_NOT_INVERTIBLE',)

_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
_BORDER_STEP_PX = 8
_ROUNDTRIP_TOL_PX = 0.1
_FAR = 1e6                            # stands for "beyond the fisheye model range"


def rectify(cameras: tuple, stereo, cfg: RectifyConfig) -> Rectification:
    """Compute R and P for each camera (mono: cameras has 1 element and stereo is None)."""
    if stereo is None:
        K, policy, warnings = _mono(cameras[0], cfg)
        return Rectification((np.eye(3),), (np.hstack([K, np.zeros((3, 1))]),), policy,
                             cfg.alpha, tuple(warnings))
    c0, c1 = cameras
    T = np.asarray(stereo.T, np.float64).reshape(3, 1)
    flags = cv2.CALIB_ZERO_DISPARITY if cfg.zero_disparity else 0
    # ponytail: both cameras are assumed to share c0.size
    if c0.model == 'fisheye':
        # fx' = fy' on both P verified on OpenCV 4.11 (test_rectification checks it)
        R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
            c0.K, c0.D, c1.K, c1.D, tuple(c0.size), stereo.R, T, flags,
            balance=cfg.alpha, fov_scale=1.0)
    else:
        R1, R2, P1, P2, *_ = cv2.stereoRectify(c0.K, c0.D, c1.K, c1.D, tuple(c0.size), stereo.R,
                                               T, flags=flags, alpha=cfg.alpha)
    return Rectification((R1, R2), (P1, P2), 'stereo', cfg.alpha)


def make_maps(cam, R: np.ndarray, P: np.ndarray, out_size: tuple):
    """
    Build remap tables from the full-resolution source image to a rectified image of out_size.

    out_size is (w, h). If out_size differs from cam.size, P is rescaled with
    u_d = (u + 0.5) * s - 0.5. Returns (map1, map2) for cv2.remap (CV_16SC2).
    """
    sx, sy = out_size[0] / cam.size[0], out_size[1] / cam.size[1]
    S = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1]])
    Kd = S @ np.asarray(P, np.float64)[:, :3]
    init = cv2.fisheye.initUndistortRectifyMap if cam.model == 'fisheye' \
        else cv2.initUndistortRectifyMap
    return init(cam.K, cam.D, R, Kd, tuple(out_size), cv2.CV_16SC2)


def rectify_points(cam, R: np.ndarray, P: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Rectify (N, 2) distorted full-res pixels into (N, 2) rectified full-res pixels."""
    src = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if cam.model == 'fisheye':
        return cv2.fisheye.undistortPoints(src, cam.K, cam.D, R=R, P=P,
                                           criteria=_CRITERIA).reshape(-1, 2)
    return cv2.undistortPointsIter(src, cam.K, cam.D, R, P, _CRITERIA).reshape(-1, 2)


def undistort_normalized(cam, pts: np.ndarray):
    """
    Undistort (N, 2) pixels of cam into normalized image coordinates.

    Return (xy (N, 2) float64, valid (N,) bool). Fisheye points that OpenCV cannot undistort
    onto the image plane (theta beyond the model range or past 90 deg) are invalid.
    """
    src = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if cam.model != 'fisheye':
        xy = cv2.undistortPointsIter(src, cam.K, cam.D, None, None, _CRITERIA).reshape(-1, 2)
        return xy, np.isfinite(xy).all(axis=1)
    xy = cv2.fisheye.undistortPoints(src, cam.K, cam.D, criteria=_CRITERIA).reshape(-1, 2)
    d = (src.reshape(-1, 2) - cam.K[:2, 2]) / np.diag(cam.K)[:2]
    # OpenCV returns (-1e6, -1e6) when it fails and a mirrored point when theta > 90 deg
    return xy, (np.abs(xy).max(axis=1) < 1e5) & (np.sum(xy * d, axis=1) >= 0)


def validate_views(views: tuple, cameras: tuple, rect: Optional[Rectification],
                   board) -> Optional[ValidationMetrics]:
    """Compute live metrics for the detections of one frame (None if nothing to measure)."""
    if all(v is None for v in views):
        return None
    reproj, straight = [], []
    for det, cam in zip(views, cameras):
        ok = det is not None and len(det.ids) >= 4
        reproj.append(_reproj_rms(cam, det) if ok else None)
        straight.append(_straightness(cam, det, board) if ok else None)
    values = {'reproj_rms_px': tuple(reproj), 'straightness_rms_px': tuple(straight)}
    if len(views) == 2 and rect is not None and None not in views:
        values.update(_stereo_metrics(views, cameras, rect, board))
    return ValidationMetrics(values)


def _border(size):
    """Return the top, bottom, left, right image border samples (<= 8 px apart, exact corners)."""
    w, h = size
    xs = np.linspace(0, w - 1, int(np.ceil((w - 1) / _BORDER_STEP_PX)) + 1)
    ys = np.linspace(0, h - 1, int(np.ceil((h - 1) / _BORDER_STEP_PX)) + 1)
    return (np.c_[xs, np.zeros_like(xs)], np.c_[xs, np.full_like(xs, h - 1)],
            np.c_[np.zeros_like(ys), ys], np.c_[np.full_like(ys, w - 1), ys])


def _fov_limit(cfg):
    """Return the max normalized radius kept for fisheye cameras: tan(fov / 2)."""
    return np.tan(np.radians(cfg.fisheye_max_fov_deg) / 2)


def _undistorted_border(cam, cfg):
    """Return the undistorted border samples; fisheye ones are clipped at radius _fov_limit."""
    out = []
    for b in _border(cam.size):
        xy, valid = undistort_normalized(cam, b)
        if cam.model == 'fisheye':
            xy[~valid] = ((b[~valid] - cam.K[:2, 2]) / np.diag(cam.K)[:2]) * _FAR
            r = np.linalg.norm(xy, axis=1, keepdims=True)
            xy = xy * np.minimum(1.0, _fov_limit(cfg) / np.maximum(r, 1e-12))
        out.append(xy)
    return out


def _radial(cam, r):
    """Return the distorted radius (pinhole) or angle (fisheye) of undistorted r (theta)."""
    k = np.zeros(8)
    k[:len(cam.D)] = cam.D
    r2 = r * r
    if cam.model == 'fisheye':
        return r * (1 + r2 * (k[0] + r2 * (k[1] + r2 * (k[2] + r2 * k[3]))))
    num = 1 + r2 * (k[0] + r2 * (k[1] + r2 * k[4]))
    return r * num / (1 + r2 * (k[5] + r2 * (k[6] + r2 * k[7])))


def _invertibility(cam, cfg):
    """Return None if the model is invertible over the image, else the warning params."""
    pts = np.vstack(_border(cam.size))
    xy, valid = undistort_normalized(cam, pts)
    err = np.inf
    if cam.model == 'fisheye':
        # only the rendered part (theta <= fov / 2) has to be invertible
        theta_lim = np.radians(cfg.fisheye_max_fov_deg) / 2
        theta = np.arctan(np.linalg.norm(xy, axis=1))
        inside = valid & (theta <= theta_lim)
        r_max = theta_lim if not inside.all() else theta.max()
        if inside.any():
            back = cv2.fisheye.distortPoints(xy[inside].reshape(-1, 1, 2), cam.K, cam.D)
            err = np.linalg.norm(back.reshape(-1, 2) - pts[inside], axis=1).max()
    elif valid.all():
        back = cv2.projectPoints(np.c_[xy, np.ones(len(xy))], np.zeros(3), np.zeros(3),
                                 cam.K, cam.D)[0]
        err = np.linalg.norm(back.reshape(-1, 2) - pts, axis=1).max()
        r_max = np.linalg.norm(xy, axis=1).max()
    else:
        r_max = 0.0
    # ponytail: radial monotonicity only; tangential terms are covered by the round trip
    monotonic = bool(np.all(np.diff(_radial(cam, np.linspace(0, r_max, 2001))) > 0))
    if err < _ROUNDTRIP_TOL_PX and monotonic:
        return None
    return {'roundtrip_px': float(err), 'monotonic': monotonic}


def _mono(cam, cfg):
    """Return (new K, applied policy, warnings) for a single camera."""
    policy, warnings = cfg.policy, []
    if policy not in POLICIES:
        raise ValueError(f'unknown rectification policy {policy!r}')
    if policy != 'k':
        problem = _invertibility(cam, cfg)
        if problem is not None:
            warnings.append(('MODEL_NOT_INVERTIBLE', {'camera': 0, **problem}))
            policy = 'k'
    if policy == 'k':
        return np.array(cam.K, np.float64), policy, warnings
    if policy == 'opencv':
        if cam.model == 'fisheye':
            K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                cam.K, cam.D, tuple(cam.size), np.eye(3), balance=cfg.alpha)
        else:
            K = cv2.getOptimalNewCameraMatrix(cam.K, cam.D, tuple(cam.size), cfg.alpha)[0]
        return K, policy, warnings
    rho = 1.0 if policy == 'square' else cam.K[0, 0] / cam.K[1, 1]
    top, bottom, left, right = _undistorted_border(cam, cfg)
    both = np.vstack([top, bottom, left, right])
    inner = np.array([left[:, 0].max(), top[:, 1].max(), right[:, 0].min(), bottom[:, 1].min()])
    outer = np.r_[both.min(axis=0), both.max(axis=0)]
    w1, h1 = cam.size[0] - 1, cam.size[1] - 1

    def scale(rect, pick):
        return pick(w1 / (rho * (rect[2] - rect[0])), h1 / (rect[3] - rect[1]))

    a = cfg.alpha
    s = (1 - a) * scale(inner, max) + a * scale(outer, min)
    cn = ((1 - a) * (inner[:2] + inner[2:]) + a * (outer[:2] + outer[2:])) / 2
    fx, fy = rho * s, s
    return (np.array([[fx, 0, w1 / 2 - fx * cn[0]], [0, fy, h1 / 2 - fy * cn[1]], [0, 0, 1]]),
            policy, warnings)


def _reproj_rms(cam, det):
    obj = det.object_points.astype(np.float64).reshape(-1, 1, 3)
    img = det.corners.astype(np.float64).reshape(-1, 1, 2)
    fisheye = cam.model == 'fisheye'
    _, rvec, tvec = (cv2.fisheye.solvePnP if fisheye else cv2.solvePnP)(obj, img, cam.K, cam.D)
    proj = (cv2.fisheye.projectPoints if fisheye else cv2.projectPoints)(obj, rvec, tvec,
                                                                         cam.K, cam.D)[0]
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - det.corners) ** 2, axis=1))))


def _straightness(cam, det, board):
    """RMS distance [px] of undistorted corners from total-least-squares grid lines."""
    xy, valid = undistort_normalized(cam, det.corners)
    xy = xy * (cam.K[0, 0] + cam.K[1, 1]) / 2
    n_cols = board.grid[0]
    sse, n = 0.0, 0
    for key in (det.ids // n_cols, det.ids % n_cols):
        for k in np.unique(key[valid]):
            p = xy[valid & (key == k)]
            if len(p) >= 3:
                sse += np.linalg.svd(p - p.mean(axis=0), compute_uv=False)[-1] ** 2
                n += len(p)
    return float(np.sqrt(sse / n)) if n else None


def _stereo_metrics(views, cameras, rect, board):
    a, b = views
    common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
    pl = rectify_points(cameras[0], rect.R[0], rect.P[0], a.corners[ia])
    pr = rectify_points(cameras[1], rect.R[1], rect.P[1], b.corners[ib])
    ok = (np.abs(pl) < 1e5).all(axis=1) & (np.abs(pr) < 1e5).all(axis=1)
    common, pl, pr = common[ok], pl[ok], pr[ok]
    if len(common) < 2:
        return {}
    axis = 0 if abs(rect.P[1][1, 3]) > abs(rect.P[1][0, 3]) else 1   # vertical rig: x
    out = {'epipolar_rms_px': float(np.sqrt(np.mean((pl[:, axis] - pr[:, axis]) ** 2)))}
    X = cv2.triangulatePoints(rect.P[0], rect.P[1], pl.T, pr.T)
    X = (X[:3] / X[3]).T
    idx = {int(i): k for k, i in enumerate(common)}
    n_cols = board.grid[0]
    d = [np.linalg.norm(X[k] - X[idx[j]]) for i, k in idx.items()
         for j in ((i + 1) if i % n_cols < n_cols - 1 else -1, i + n_cols) if j in idx]
    if d:
        out['square_size_m'] = float(np.mean(d))
        out['square_size_err'] = float(np.mean(d) / board.square_size - 1)
    return out
