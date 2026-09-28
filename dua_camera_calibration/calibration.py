"""
Monocular and stereo camera calibration (pinhole and fisheye).

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
import re
import time
from typing import Callable, Optional

import cv2
from dua_camera_calibration.rectification import undistort_normalized
import numpy as np

# CONTRACT (do not change field names/types without updating every user):


class CalibrationError(Exception):
    """Calibration failure with a machine-readable code (see texts.WARNINGS/ERRORS)."""

    def __init__(self, code: str, **params):
        """Store the code and its format parameters."""
        super().__init__(code, params)
        self.code = code
        self.params = params


@dataclass(frozen=True)
class CalibConfig:
    """Calibration options."""

    model: str = 'pinhole'                 # 'pinhole' | 'fisheye'
    distortion_model: str = 'plumb_bob'    # 'plumb_bob' | 'rational_polynomial' (pinhole only)
    fix_k3: bool = True
    fix_aspect_ratio: bool = False
    fix_principal_point: bool = False
    zero_tangent_dist: bool = False
    max_views: int = 60
    min_views: int = 6
    outlier_rejection: bool = True


@dataclass(frozen=True)
class CameraCalib:
    """
    Intrinsic calibration of one camera.

    distortion_model is the ROS name: 'plumb_bob' (5 coeffs), 'rational_polynomial' (8),
    'equidistant' (4).
    sigmas: 1-sigma standard deviations by parameter name ('fx', 'fy', 'cx', 'cy', 'k1', ...);
    fixed parameters are absent. Empty when unknown (e.g. loaded from file).
    """

    model: str                    # 'pinhole' | 'fisheye'
    distortion_model: str
    size: tuple                   # (width, height)
    K: np.ndarray                 # (3, 3) float64
    D: np.ndarray                 # (n,) float64
    sigmas: dict = field(default_factory=dict)
    rms: float = float('nan')     # reprojection RMS [px] over used views
    n_views: int = 0


@dataclass(frozen=True)
class StereoExtrinsics:
    """Pose of the right camera w.r.t. the left one: X_right = R X_left + T."""

    R: np.ndarray                 # (3, 3) float64
    T: np.ndarray                 # (3,) float64 [m]
    E: np.ndarray                 # (3, 3)
    F: np.ndarray                 # (3, 3)
    rms: float                    # stereo reprojection RMS [px]
    sigmas: dict = field(default_factory=dict)   # 'rx','ry','rz' [rad], 'tx','ty','tz' [m]
    n_pairs: int = 0


@dataclass(frozen=True)
class CalibrationResult:
    """
    Output of calibrate().

    Stereo: rejected merges both cameras (a failure reason wins over NOT_SELECTED); check
    used_ids[cam] to know whether a camera used a sample.
    """

    cameras: tuple                # tuple[CameraCalib, ...] (1 mono, 2 stereo)
    stereo: Optional[StereoExtrinsics]
    used_ids: tuple               # per camera: tuple of sample ids used in the final solve
    rejected: dict                # sample id -> reason code (REJECT_CODES)
    per_view_rms: tuple           # per camera: dict sample id -> RMS [px] (used views)
    rms: float                    # overall RMS [px] (mono: camera RMS; stereo: stereo RMS)
    warnings: tuple = ()          # tuple[(code, params dict), ...]
    duration_s: float = 0.0


WARNING_CODES = ('FISHEYE_CHECK_COND_DISABLED',)
ERROR_CODES = ('TOO_FEW_VIEWS', 'TOO_FEW_PAIRS', 'SOLVE_FAILED')
REJECT_CODES = ('OUTLIER', 'ILL_CONDITIONED', 'NOT_SELECTED')
STAGES = ('selecting', 'solving', 'outliers', 'sigmas', 'stereo')

_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9)
_INIT_VIEWS = 30              # farthest-point subset solved before view selection
_MAX_ILL_COND_DROPS = 5
_MIN_PAIR_POINTS = 6
_ILL_COND = re.compile(r'Ill-conditioned matrix for input array (\d+)')
# Intrinsic columns of the cv2.projectPoints / cv2.fisheye.projectPoints Jacobians
_JCOL = {
    'pinhole': {'fx': 6, 'fy': 7, 'cx': 8, 'cy': 9, 'k1': 10, 'k2': 11, 'p1': 12, 'p2': 13,
                'k3': 14, 'k4': 15, 'k5': 16, 'k6': 17},
    'fisheye': {'fx': 0, 'fy': 1, 'cx': 2, 'cy': 3, 'k1': 4, 'k2': 5, 'k3': 6, 'k4': 7},
}
_POSE_COLS = {'pinhole': slice(0, 6), 'fisheye': slice(8, 14)}   # [r3, t3]


def calibrate(snapshot, board, cfg: CalibConfig,
              progress: Optional[Callable[[str], None]] = None) -> CalibrationResult:
    """
    Calibrate the cameras of a DBSnapshot (1 camera: mono, 2: stereo).

    progress, if given, receives the STAGES names. Raises CalibrationError (ERROR_CODES).
    """
    t0 = time.monotonic()
    rejected, warnings = {}, []
    cams, used, per_view = [], [], []
    for c in range(snapshot.n_cameras):
        items = [(s.id, s.views[c], s.features[c] if len(s.features) > c else None)
                 for s in snapshot.samples if s.views[c] is not None]
        rej = {}
        cam, ids, pv = _calibrate_camera(c, items, tuple(snapshot.image_sizes[c]), cfg,
                                         progress, rej, warnings)
        for sid, why in rej.items():
            if rejected.get(sid) in (None, 'NOT_SELECTED'):
                rejected[sid] = why
        cams.append(cam)
        used.append(ids)
        per_view.append(pv)
    stereo = None
    if len(cams) == 2:
        bad = {sid for sid, why in rejected.items() if why != 'NOT_SELECTED'}
        stereo = _stereo(snapshot, cams, cfg, bad, progress)
    return CalibrationResult(
        tuple(cams), stereo, tuple(used), rejected, tuple(per_view),
        stereo.rms if stereo is not None else cams[0].rms, tuple(warnings),
        time.monotonic() - t0)


def quick_calibrate(views: list, size: tuple, cfg: CalibConfig,
                    init: Optional[CameraCalib] = None, max_iter: int = 30) -> CameraCalib:
    """
    Fast single-camera solve on a list of Detection (caller bounds its length, e.g. <= 40).

    Warm-started from init when given; sigmas are filled. Used by the live guidance estimate.
    """
    n_dist = _n_dist(cfg)
    guess = None
    if init is not None and init.model == cfg.model and len(init.D) == n_dist:
        guess = (np.asarray(init.K, np.float64), np.asarray(init.D, np.float64))
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, max_iter, 1e-6)
    K, D, rvecs, tvecs, items = _solve(cfg, list(enumerate(views)), tuple(size), guess, 0, {}, [],
                                       criteria, check_cond=False)
    terms = [_view_terms(cfg, K, D, d, r, t) for (_, d), r, t in zip(items, rvecs, tvecs)]
    return _camera(cfg, size, K, D, items, terms)


def _stage(progress, name):
    if progress is not None:
        progress(name)


def _n_dist(cfg):
    if cfg.model == 'fisheye':
        return 4
    return 8 if cfg.distortion_model == 'rational_polynomial' else 5


def _free_params(cfg):
    """Return the names of the intrinsics estimated under cfg."""
    names = ['fx', 'fy'] + ([] if cfg.fix_principal_point else ['cx', 'cy'])
    if cfg.model == 'fisheye':
        return names + ['k1', 'k2', 'k3', 'k4']
    names += ['k1', 'k2'] + ([] if cfg.zero_tangent_dist else ['p1', 'p2'])
    if cfg.distortion_model == 'rational_polynomial':
        return names + ['k3', 'k4', 'k5', 'k6']
    return names + ([] if cfg.fix_k3 else ['k3'])


def _merged_f(cfg):
    """Return True when fx and fy are one parameter (fixed aspect ratio)."""
    return cfg.model == 'pinhole' and cfg.fix_aspect_ratio


def _flags(cfg, guess):
    if cfg.model == 'fisheye':
        # fisheye.calibrate has no aspect-ratio or tangential options: those settings do not apply
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        if cfg.fix_principal_point:
            flags |= cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
        return flags | (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS if guess else 0)
    flags = cv2.CALIB_USE_LU
    if cfg.distortion_model == 'rational_polynomial':
        flags |= cv2.CALIB_RATIONAL_MODEL
    elif cfg.fix_k3:
        flags |= cv2.CALIB_FIX_K3
    if cfg.fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if cfg.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if cfg.zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    return flags | (cv2.CALIB_USE_INTRINSIC_GUESS if guess else 0)


def _solve(cfg, items, size, init, cam, rejected, warnings, criteria=_CRITERIA,
           check_cond=True):
    """
    Run one OpenCV calibration on items [(id, Detection)], warm-started from init=(K, D).

    Return (K, D, rvecs, tvecs, items used). Fisheye views that OpenCV reports as
    ill-conditioned are dropped (recorded in rejected); after _MAX_ILL_COND_DROPS drops, or on
    another CHECK_COND failure, the solve is retried without CHECK_COND (warning recorded).
    """
    items = list(items)
    K0 = np.array(init[0], np.float64) if init is not None else None
    D0 = np.array(init[1], np.float64) if init is not None else None
    if cfg.model == 'pinhole':
        obj = [d.object_points.astype(np.float32) for _, d in items]
        img = [d.corners.astype(np.float32) for _, d in items]
        K, D = (np.eye(3), None) if init is None else (K0, D0)   # eye: fx/fy = 1 if fixed
        # A cold start initializes the poses by homography ignoring distortion, which can leave
        # strongly distorted views in a wrong pose minimum; a warm re-solve re-initializes every
        # pose from the estimated K, D (fisheye recomputes extrinsics at each iteration anyway).
        for warm in (False, True) if init is None else (True,):
            if warm and cfg.fix_aspect_ratio:
                K[0, 0] = K[1, 1]
            try:
                _, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, K, D,
                                                            flags=_flags(cfg, warm),
                                                            criteria=criteria)
            except cv2.error as e:
                raise CalibrationError('SOLVE_FAILED', msg=str(e)) from e
        return K, D.ravel()[:_n_dist(cfg)].astype(np.float64), rvecs, tvecs, items
    flags = _flags(cfg, init is not None) | (cv2.fisheye.CALIB_CHECK_COND if check_cond else 0)
    drops = 0
    while True:
        try:
            _, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                [d.object_points.astype(np.float64).reshape(1, -1, 3) for _, d in items],
                [d.corners.astype(np.float64).reshape(1, -1, 2) for _, d in items],
                size, K0, D0, flags=flags, criteria=criteria)
            return K, D.ravel().astype(np.float64), rvecs, tvecs, items
        except cv2.error as e:
            if not flags & cv2.fisheye.CALIB_CHECK_COND:
                raise CalibrationError('SOLVE_FAILED', msg=str(e)) from e
            m = _ILL_COND.search(str(e))
            if m is not None and drops < _MAX_ILL_COND_DROPS and int(m.group(1)) < len(items):
                rejected[items.pop(int(m.group(1)))[0]] = 'ILL_CONDITIONED'
                drops += 1
                if len(items) < cfg.min_views:
                    raise CalibrationError('TOO_FEW_VIEWS', n=len(items), min=cfg.min_views,
                                           camera=cam) from e
            else:
                flags &= ~cv2.fisheye.CALIB_CHECK_COND
                w = ('FISHEYE_CHECK_COND_DISABLED', {'camera': cam})
                if w not in warnings:
                    warnings.append(w)


def _project(model, K, D, obj, rvec, tvec):
    """Return the (N, 2) projections of obj and the OpenCV Jacobian."""
    rvec = np.asarray(rvec, np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, np.float64).reshape(3, 1)
    obj = np.asarray(obj, np.float64).reshape(-1, 1, 3)
    if model == 'fisheye':
        img, jac = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
    else:
        img, jac = cv2.projectPoints(obj, rvec, tvec, K, D)
    return img.reshape(-1, 2), jac


def _view_terms(cfg, K, D, det, rvec, tvec):
    """
    Return (Schur information block, sum of squared residuals) of one view.

    Lambda = A'A - A'B (B'B)^-1 B'A, A: free intrinsic columns, B: pose columns.
    """
    proj, jac = _project(cfg.model, K, D, det.object_points, rvec, tvec)
    names = _free_params(cfg)
    A = jac[:, [_JCOL[cfg.model][n] for n in names]]
    if _merged_f(cfg):
        A = np.column_stack([K[0, 0] / K[1, 1] * A[:, 0] + A[:, 1], A[:, 2:]])
    B = jac[:, _POSE_COLS[cfg.model]]
    AtB = A.T @ B
    info = A.T @ A - AtB @ np.linalg.solve(B.T @ B, AtB.T)
    return info, float(np.sum((proj - det.corners) ** 2))


def _pose(cfg, K, D, det):
    solve = cv2.fisheye.solvePnP if cfg.model == 'fisheye' else cv2.solvePnP
    _, rvec, tvec = solve(det.object_points.astype(np.float64).reshape(-1, 1, 3),
                          det.corners.astype(np.float64).reshape(-1, 1, 2), K, D)
    return rvec, tvec


def _camera(cfg, size, K, D, items, terms):
    """Build the CameraCalib, with sigma^2 = sum e^2 / (2M - p - 6N) and O(N) Schur covariance."""
    names = _free_params(cfg)
    n_pts = sum(len(d.corners) for _, d in items)
    sse = sum(t[1] for t in terms)
    dof = 2 * n_pts - (len(names) - _merged_f(cfg)) - 6 * len(items)
    sigmas = {}
    if dof > 0:
        try:
            sd = np.sqrt(np.abs(np.diag(np.linalg.inv(sum(t[0] for t in terms)) * (sse / dof))))
        except np.linalg.LinAlgError:
            sd = None
        if sd is not None and _merged_f(cfg):
            sigmas = {'fx': K[0, 0] / K[1, 1] * sd[0], 'fy': sd[0], **dict(zip(names[2:], sd[1:]))}
        elif sd is not None:
            sigmas = dict(zip(names, sd))
    dist = 'equidistant' if cfg.model == 'fisheye' else cfg.distortion_model
    return CameraCalib(cfg.model, dist, tuple(size), K, D,
                       {k: float(v) for k, v in sigmas.items()},
                       float(np.sqrt(sse / n_pts)), len(items))


def _features(det, feat, size):
    """Return the guidance feature vector (x, y, size, tilt_x/90, tilt_y/90) of a view."""
    keys = ('x', 'y', 'size', 'tilt_x_deg', 'tilt_y_deg')
    if feat and all(k in feat for k in keys):
        return [feat['x'], feat['y'], feat['size'], feat['tilt_x_deg'] / 90,
                feat['tilt_y_deg'] / 90]
    w, h = size
    c = det.corners.astype(np.float32)
    area = cv2.contourArea(cv2.convexHull(c))
    return [c[:, 0].mean() / w, c[:, 1].mean() / h, np.sqrt(area / (w * h)), 0.0, 0.0]


def _farthest_points(X, k):
    """Return k indices of X picked by farthest-point sampling."""
    chosen = [int(np.argmax(np.linalg.norm(X - X.mean(axis=0), axis=1)))]
    d = np.linalg.norm(X - X[chosen[0]], axis=1)
    d[chosen[0]] = -np.inf
    while len(chosen) < min(k, len(X)):
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
        d[chosen] = -np.inf
    return chosen


def _d_optimal(infos, k):
    """Return k indices greedily maximising log det of the summed information (D-optimal)."""
    acc = 1e-6 * np.diag(np.diag(infos.sum(axis=0)))   # scale-aware prior for the first picks
    free = list(range(len(infos)))
    chosen = []
    for _ in range(min(k, len(infos))):
        j = free.pop(int(np.argmax(np.linalg.slogdet(acc + infos[free])[1])))
        chosen.append(j)
        acc = acc + infos[j]
    return chosen


def _calibrate_camera(cam, items, size, cfg, progress, rejected, warnings):
    """Calibrate one camera from items [(id, Detection, features)]: (CameraCalib, ids, rms)."""
    if len(items) < cfg.min_views:
        raise CalibrationError('TOO_FEW_VIEWS', n=len(items), min=cfg.min_views, camera=cam)
    views = [(sid, d) for sid, d, _ in items]
    init = None
    if len(views) > cfg.max_views:
        _stage(progress, 'selecting')
        X = np.array([_features(d, f, size) for _, d, f in items])
        sub = _farthest_points(X, max(cfg.min_views, min(cfg.max_views, _INIT_VIEWS)))
        init = _solve(cfg, [views[i] for i in sub], size, None, cam, rejected, warnings)[:2]
        views = [v for v in views if v[0] not in rejected]
        infos = np.array([_view_terms(cfg, *init, d, *_pose(cfg, *init, d))[0] for _, d in views])
        keep = set(_d_optimal(infos, cfg.max_views))
        for i, (sid, _) in enumerate(views):
            if i not in keep:
                rejected[sid] = 'NOT_SELECTED'
        views = [v for i, v in enumerate(views) if i in keep]
    _stage(progress, 'solving')
    K, D, rvecs, tvecs, views = _solve(cfg, views, size, init, cam, rejected, warnings)
    terms = [_view_terms(cfg, K, D, d, r, t) for (_, d), r, t in zip(views, rvecs, tvecs)]
    for _ in range(2 if cfg.outlier_rejection else 0):
        rms = np.array([np.sqrt(t[1] / len(d.corners)) for (_, d), t in zip(views, terms)])
        med = np.median(rms)
        # floor: never reject views within 1.5x the median or below 0.1 px (clean data)
        thr = max(med + 3 * 1.4826 * np.median(np.abs(rms - med)), 1.5 * med, 0.1)
        # ponytail: dropped views are not replaced by unselected ones
        drop = {int(i) for i in np.argsort(-rms)[:max(0, len(views) - cfg.min_views)]
                if rms[i] > thr}
        if not drop:
            break
        _stage(progress, 'outliers')
        for i in drop:
            rejected[views[i][0]] = 'OUTLIER'
        views = [v for i, v in enumerate(views) if i not in drop]
        K, D, rvecs, tvecs, views = _solve(cfg, views, size, (K, D), cam, rejected, warnings)
        terms = [_view_terms(cfg, K, D, d, r, t) for (_, d), r, t in zip(views, rvecs, tvecs)]
    _stage(progress, 'sigmas')
    per_view = {sid: float(np.sqrt(t[1] / len(d.corners))) for (sid, d), t in zip(views, terms)}
    return _camera(cfg, size, K, D, views, terms), tuple(sid for sid, _ in views), per_view


def _stereo(snapshot, cams, cfg, bad, progress):
    """Estimate the extrinsics from the pairs (intrinsics fixed)."""
    _stage(progress, 'stereo')
    obj, left, right = [], [], []
    for sid, a, b in snapshot.pairs():
        if sid in bad:
            continue
        _, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
        o, pa, pb = (a.object_points[ia].astype(np.float64), a.corners[ia].astype(np.float64),
                     b.corners[ib].astype(np.float64))
        if cfg.model == 'fisheye':
            # ponytail: pinhole stereoCalibrate on normalized points (fisheye.stereoCalibrate
            # asserts easily); residuals are not reweighted, so the image periphery weighs more
            pa, va = undistort_normalized(cams[0], pa)
            pb, vb = undistort_normalized(cams[1], pb)
            o, pa, pb = o[va & vb], pa[va & vb], pb[va & vb]
        if len(o) >= _MIN_PAIR_POINTS:            # stereoCalibrate wants float32
            obj.append(o.astype(np.float32))
            left.append(pa.astype(np.float32))
            right.append(pb.astype(np.float32))
    if len(obj) < cfg.min_views:
        raise CalibrationError('TOO_FEW_PAIRS', n=len(obj), min=cfg.min_views)
    if len(obj) > cfg.max_views:
        # ponytail: evenly spaced pairs bound the dense (6N + 6)^2 LM; D-optimal if ever needed
        keep = np.linspace(0, len(obj) - 1, cfg.max_views).round().astype(int)
        obj, left, right = ([x[i] for i in keep] for x in (obj, left, right))
    if cfg.model == 'fisheye':
        K1 = K2 = np.eye(3)
        D1 = D2 = np.zeros(5)
        unit = float(np.mean([cams[0].K[0, 0], cams[0].K[1, 1], cams[1].K[0, 0], cams[1].K[1, 1]]))
    else:
        K1, D1, K2, D2, unit = cams[0].K, cams[0].D, cams[1].K, cams[1].D, 1.0
    try:
        rms, _, _, _, _, R, T, E, F, rvecs, tvecs, _ = cv2.stereoCalibrateExtended(
            obj, left, right, K1, D1, K2, D2, tuple(cams[0].size), None, None,
            flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_LU, criteria=_CRITERIA)
    except cv2.error as e:
        raise CalibrationError('SOLVE_FAILED', msg=str(e)) from e
    sigmas = _stereo_sigmas(obj, left, right, K1, D1, K2, D2, R, T, rvecs, tvecs)
    if cfg.model == 'fisheye':
        F = np.linalg.inv(cams[1].K).T @ E @ np.linalg.inv(cams[0].K)
        F = F / F[2, 2]
    return StereoExtrinsics(R, T.ravel(), E, F, float(rms) * unit, sigmas, len(obj))


def _stereo_sigmas(obj, left, right, K1, D1, K2, D2, R, T, rvecs, tvecs):
    """
    1-sigma of (rvec(R), T) by the Schur complement over the per-pair left poses.

    The right pose is compose(left pose, (R, T)); chain rule through cv2.composeRT.
    """
    # ponytail: conditional on the fixed intrinsics; principal point errors (~0.5 px) tilt R by
    # ~c/f and dominate the true rotation error; propagate the intrinsic covariance if needed
    r = cv2.Rodrigues(R)[0]
    info = np.zeros((6, 6))
    sse, n_pts = 0.0, 0
    for o, a, b, rv, tv in zip(obj, left, right, rvecs, tvecs):
        pl, jl = _project('pinhole', K1, D1, o, rv, tv)
        rv2, tv2, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2 = \
            cv2.composeRT(np.asarray(rv, np.float64).reshape(3, 1),
                          np.asarray(tv, np.float64).reshape(3, 1), r, T.reshape(3, 1))
        pr, jr = _project('pinhole', K2, D2, o, rv2, tv2)
        d_ext = np.block([[dr3dr2, dr3dt2], [dt3dr2, dt3dt2]])
        d_left = np.block([[dr3dr1, dr3dt1], [dt3dr1, dt3dt1]])
        A = np.vstack([np.zeros((len(pl) * 2, 6)), jr[:, :6] @ d_ext])
        B = np.vstack([jl[:, :6], jr[:, :6] @ d_left])
        AtB = A.T @ B
        info += A.T @ A - AtB @ np.linalg.solve(B.T @ B, AtB.T)
        sse += float(np.sum((pl - a) ** 2) + np.sum((pr - b) ** 2))
        n_pts += len(o)
    dof = 4 * n_pts - 6 - 6 * len(obj)
    try:
        sd = np.sqrt(np.abs(np.diag(np.linalg.inv(info) * (sse / dof))))
    except np.linalg.LinAlgError:
        return {}
    return {k: float(v) for k, v in zip(('rx', 'ry', 'rz', 'tx', 'ty', 'tz'), sd)}
