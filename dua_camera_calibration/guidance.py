"""
Per-frame gates, coverage indicators, user hints and live calibration estimate.

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
import math
from typing import Optional

import cv2
from dua_camera_calibration import calibration
import numpy as np

# CONTRACT (do not change field names/types without updating every user):

REASON_CODES = ('OK', 'ACCEPTED', 'NOT_DETECTED', 'TOO_FEW_CORNERS', 'NEAR_BORDER', 'BLURRY',
                'MOVING', 'TOO_TILTED', 'TOO_SIMILAR', 'LIMIT')
INDICATOR_IDS = ('coverage', 'tilt_x', 'tilt_y', 'distance', 'uncertainty')
HINT_CODES = ('HOLD_STILL', 'IMPROVE_FOCUS_OR_LIGHT', 'MOVE_AWAY_FROM_EDGE', 'SHOW_WHOLE_BOARD',
              'TILT_LESS', 'COVER_REGION', 'TILT_LEFT_RIGHT', 'TILT_UP_DOWN', 'MOVE_CLOSER',
              'MOVE_FARTHER', 'KEEP_GOING', 'READY')


@dataclass(frozen=True)
class GateConfig:
    """Per-frame acceptance thresholds."""

    max_blur_px: float = 3.0
    max_motion_px: float = 1.5        # at 720p, scaled by image diagonal / 1468.6
    max_tilt_deg: float = 60.0
    min_novelty: float = 0.15
    check_motion: bool = True         # False for unordered image folders
    model: str = 'pinhole'            # camera model of the tilt bootstrap before an estimate


@dataclass(frozen=True)
class GuidanceConfig:
    """Indicator and readiness options."""

    min_samples: int = 12
    max_samples: int = 150
    target_rel_ci: float = 0.005
    grid: tuple = (8, 6)              # coverage (cols, rows) if landscape, swapped for portrait


@dataclass(frozen=True)
class FrameVerdict:
    """
    Gate outcome for one frame.

    ok: frame should be captured in auto mode. hard_ok: all gates but novelty pass (manual
    capture). views: detections to store if captured (stereo: a side may be None when only one
    side is useful).
    features: per-view feature dicts (x, y, size, tilt_x_deg, tilt_y_deg) or None.
    """

    ok: bool
    hard_ok: bool
    reason: str                        # REASON_CODES
    params: dict = field(default_factory=dict)
    views: tuple = ()
    features: tuple = ()
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Indicator:
    """One progress indicator of one camera."""

    id: str                            # INDICATOR_IDS  # noqa: A003
    progress: float                    # [0, 1]
    params: dict = field(default_factory=dict)
    camera: int = 0


@dataclass(frozen=True)
class Hint:
    """One suggestion to the user; target is a normalized (x, y) image point for an arrow."""

    code: str                          # HINT_CODES
    params: dict = field(default_factory=dict)
    indicator: Optional[str] = None
    target: Optional[tuple] = None
    camera: int = 0


@dataclass(frozen=True)
class Estimate:
    """Quick live calibration used for tilt estimation and the uncertainty indicator."""

    cams: tuple                        # tuple[CameraCalib, ...]
    rel_ci: Optional[float]            # max relative 95% CI of fx, fy, cx, cy (None if unknown)
    n_views: int


@dataclass(frozen=True)
class GuidanceState:
    """Output of GuidanceEngine.update()."""

    db_version: int
    indicators: tuple                  # tuple[Indicator, ...]
    hints: tuple                       # tuple[Hint, ...] most important first
    ready: bool
    coverage: tuple                    # per camera (rows, cols) float array in [0, 1]
    estimate: Optional[Estimate] = None


# Implementation constants.
_MIN_CORNERS = 6
_REF_DIAG = 1468.6                     # 1280x720 diagonal, reference of max_motion_px
_GRID = GuidanceConfig.grid            # evaluate_frame has no GuidanceConfig: default grid
_COVER_FULL = 0.9                      # fraction of hit cells for full coverage
_HEAT_VIEWS = 3                        # views per cell for a saturated heatmap cell
_TILT_EDGE = 15.0                      # [deg] tilt bins: < -15, [-15, 15], > 15
_SIZE_EDGES = (0.25, 0.5)              # board size bins: far < 0.25 <= mid <= 0.5 < close
_EST_MIN_VIEWS = 5
_EST_MAX_VIEWS = 40
_EST_MAX_ITER = 30
_PLATEAU_SAMPLES = 5
_PLATEAU_GAIN = 0.10
_PLATEAU_COVERAGE = 0.8
# Tilts are right-handed rotations of the board normal about the camera X (tilt_x) and Y
# (tilt_y) axes. TILT_* hint 'direction' is where the board front face must turn, in image
# terms, to fill the (negative, fronto, positive) bin: 'up' = top edge away from the camera,
# 'left' = left edge away from the camera, 'none' = face the camera.
_TILT_DIRS = {'tilt_x': ('up', 'none', 'down'), 'tilt_y': ('right', 'none', 'left')}
_FRAME_HINTS = {'MOVING': 'HOLD_STILL', 'BLURRY': 'IMPROVE_FOCUS_OR_LIGHT',
                'NEAR_BORDER': 'MOVE_AWAY_FROM_EDGE', 'NOT_DETECTED': 'SHOW_WHOLE_BOARD',
                'TOO_FEW_CORNERS': 'SHOW_WHOLE_BOARD', 'TOO_TILTED': 'TILT_LESS'}


def _grid_shape(size, grid) -> tuple:
    """Return the (cols, rows) coverage grid matched to the image orientation."""
    return tuple(grid) if size[0] >= size[1] else tuple(grid)[::-1]


def _cells(det, size, grid) -> np.ndarray:
    """Return the flat indices of the coverage cells containing corners of det."""
    cols, rows = grid
    c = np.clip((det.corners[:, 0] * (cols / size[0])).astype(int), 0, cols - 1)
    r = np.clip((det.corners[:, 1] * (rows / size[1])).astype(int), 0, rows - 1)
    return np.unique(r * cols + c)


def _hits(dets, size, grid) -> np.ndarray:
    """Return the (rows, cols) number of views with corners in each coverage cell."""
    cols, rows = grid
    hits = np.zeros(rows * cols, np.int64)
    for d in dets:
        hits[_cells(d, size, grid)] += 1
    return hits.reshape(rows, cols)


def _intrinsics(cam, size, model) -> tuple:
    """Return (model, K, D) of cam, or of the bootstrap camera (f = w, fisheye f = w / pi)."""
    if cam is not None:
        return cam.model, cam.K, cam.D
    w, h = size
    f = w / math.pi if model == 'fisheye' else float(w)
    K = np.array([[f, 0.0, (w - 1) / 2], [0.0, f, (h - 1) / 2], [0.0, 0.0, 1.0]])
    return model, K, np.zeros(4 if model == 'fisheye' else 5)


def _tilts(det, size, cam, model) -> tuple:
    """Return the signed (tilt_x, tilt_y) [deg] of the board normal about the camera X, Y axes."""
    model, K, D = _intrinsics(cam, size, model)
    img = det.corners.reshape(-1, 1, 2).astype(np.float64)
    if model == 'fisheye':
        img = cv2.fisheye.undistortPoints(img, K, D)
        K, D = np.eye(3), None
    try:
        ok, rvec, _ = cv2.solvePnP(det.object_points.astype(np.float64), img, K, D,
                                   flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        ok = False
    if not ok:
        return 0.0, 0.0    # ponytail: degenerate (e.g. collinear) views count as fronto
    n = cv2.Rodrigues(rvec)[0][:, 2]
    n = -n if n[2] < 0 else n    # normal pointing away from the camera
    return math.degrees(math.atan2(-n[1], n[2])), math.degrees(math.atan2(n[0], n[2]))


def _features(det, size, cam=None, model='pinhole') -> dict:
    """Return the guidance features of a view: normalized centroid, size and tilts."""
    w, h = size
    cx, cy = det.corners.mean(axis=0)
    area = cv2.contourArea(cv2.convexHull(det.corners.astype(np.float32)))
    tx, ty = _tilts(det, size, cam, model)
    return {'x': float(cx) / w, 'y': float(cy) / h, 'size': math.sqrt(area / (w * h)),
            'tilt_x_deg': tx, 'tilt_y_deg': ty}


def _vec(f) -> np.ndarray:
    """Return the novelty feature vector of a feature dict."""
    return np.array([f['x'], f['y'], f['size'], f['tilt_x_deg'] / 90, f['tilt_y_deg'] / 90])


def _tilt(f) -> float:
    """Return the angle [deg] between the board normal and the optical axis."""
    tx, ty = math.radians(f['tilt_x_deg']), math.radians(f['tilt_y_deg'])
    return math.degrees(math.atan(math.hypot(math.tan(tx), math.tan(ty))))


def _motion(det, prev) -> Optional[float]:
    """Return the mean displacement [px] of the corners common to det and prev (None if none)."""
    if prev is None:
        return None
    _, a, b = np.intersect1d(det.ids, prev.ids, return_indices=True)
    if len(a) == 0:
        return None
    return float(np.linalg.norm(det.corners[a] - prev.corners[b], axis=1).mean())


def _check(i, det, size, prev, stored, cam, gates) -> tuple:
    """
    Gate view i against the previous frame and the stored (Detection, features) of its camera.

    Return (reason, params, metrics, features, fills): reason is None if every gate passes and
    'TOO_SIMILAR' if only novelty fails; fills is True if the view hits an empty coverage cell.
    """
    m = {}
    if det is None:
        return 'NOT_DETECTED', {'camera': i}, m, None, False
    m['n_corners'] = n = len(det.ids)
    if n < _MIN_CORNERS:
        return 'TOO_FEW_CORNERS', {'camera': i, 'value': n, 'limit': _MIN_CORNERS}, m, None, False
    w, h = size
    x, y = det.corners[:, 0], det.corners[:, 1]
    m['margin_px'] = margin = float(min(x.min(), y.min(), w - 1 - x.max(), h - 1 - y.max()))
    if margin < det.window + 1:
        params = {'camera': i, 'value': margin, 'limit': det.window + 1}
        return 'NEAR_BORDER', params, m, None, False
    m['sharpness'] = det.sharpness
    if det.sharpness is not None and det.sharpness > gates.max_blur_px:
        params = {'camera': i, 'value': det.sharpness, 'limit': gates.max_blur_px}
        return 'BLURRY', params, m, None, False
    if gates.check_motion:
        # ponytail: displacement per frame, dt_s unused; normalize by dt_s if rates vary a lot
        m['motion_px'] = motion = _motion(det, prev)
        limit = gates.max_motion_px * math.hypot(w, h) / _REF_DIAG
        if motion is None or motion > limit:
            return 'MOVING', {'camera': i, 'value': motion, 'limit': limit}, m, None, False
    feats = _features(det, size, cam, gates.model)
    m['tilt'] = tilt = _tilt(feats)
    if tilt > gates.max_tilt_deg:
        params = {'camera': i, 'value': tilt, 'limit': gates.max_tilt_deg}
        return 'TOO_TILTED', params, m, feats, False
    # ponytail: coverage recomputed per frame (~1 ms at 150 samples), cache by snapshot.version
    grid = _grid_shape(size, _GRID)
    hits = _hits([d for d, _ in stored], size, grid).ravel()
    fills = bool(np.any(hits[_cells(det, size, grid)] == 0))
    known = [_vec(f) for _, f in stored if f is not None]
    novelty = float(np.linalg.norm(np.array(known) - _vec(feats), axis=1).min()) if known else None
    m['novelty'] = novelty
    if fills or novelty is None or novelty > gates.min_novelty:
        return None, {'camera': i}, m, feats, fills
    params = {'camera': i, 'value': novelty, 'limit': gates.min_novelty}
    return 'TOO_SIMILAR', params, m, feats, fills


def evaluate_frame(views: tuple, image_sizes: tuple, prev_views: Optional[tuple],
                   dt_s: Optional[float], snapshot, estimate: Optional[Estimate],
                   gates: GateConfig, board) -> FrameVerdict:
    """
    Apply the gates to the detections of one frame (one entry per camera, None if not detected).

    Gates in order: NOT_DETECTED, TOO_FEW_CORNERS, NEAR_BORDER, BLURRY, MOVING, TOO_TILTED,
    TOO_SIMILAR. Stereo: both views are returned if both pass the hard gates and one is novel;
    a single view (other side None) is accepted only if it hits an empty coverage cell of its
    camera; otherwise the reason is the earliest failing gate. params carry 'camera' and, for
    measured gates, 'value' and 'limit'. metrics maps camera index -> measured values.
    Tilts use the estimate camera, or a pinhole bootstrap (f = w) before the first estimate.
    LIMIT and ACCEPTED are set by the caller.
    """
    samples = snapshot.samples if snapshot is not None else ()
    checks = []
    for i, det in enumerate(views):
        prev = prev_views[i] if prev_views is not None else None
        cam = estimate.cams[i] if estimate is not None else None
        stored = [(s.views[i], s.features[i]) for s in samples if s.views[i] is not None]
        checks.append(_check(i, det, tuple(image_sizes[i]), prev, stored, cam, gates))
    reasons = [c[0] for c in checks]
    feats = tuple(c[3] for c in checks)
    metrics = {i: c[2] for i, c in enumerate(checks)}
    hard = [r in (None, 'TOO_SIMILAR') for r in reasons]
    if all(hard):
        if None in reasons:
            return FrameVerdict(True, True, 'OK', {}, tuple(views), feats, metrics)
        return FrameVerdict(False, True, 'TOO_SIMILAR', checks[0][1], tuple(views), feats, metrics)
    if len(views) > 1:
        for i, c in enumerate(checks):
            if hard[i] and c[4]:
                keep = tuple(v if j == i else None for j, v in enumerate(views))
                kf = tuple(f if j == i else None for j, f in enumerate(feats))
                return FrameVerdict(True, True, 'OK', {'camera': i}, keep, kf, metrics)
    worst = min((c for c, h in zip(checks, hard) if not h), key=lambda c: REASON_CODES.index(c[0]))
    return FrameVerdict(False, False, worst[0], worst[1], tuple(views), feats, metrics)


def frame_hints(verdict: FrameVerdict) -> tuple:
    """Return the hints addressing a rejected frame (empty if accepted/OK or TOO_SIMILAR)."""
    code = _FRAME_HINTS.get(verdict.reason)
    if code is None:
        return ()
    target = (0.5, 0.5) if code == 'MOVE_AWAY_FROM_EDGE' else None
    return (Hint(code, dict(verdict.params), None, target, verdict.params.get('camera', 0)),)


def _region(x, y) -> str:
    """Return the region name ('top-left', 'top', ..., 'center') of a normalized point."""
    v = 'top' if y < 1 / 3 else 'bottom' if y > 2 / 3 else ''
    h = 'left' if x < 1 / 3 else 'right' if x > 2 / 3 else ''
    return '-'.join(p for p in (v, h) if p) or 'center'


def _cover_hint(hits, cam) -> Hint:
    """Return the COVER_REGION hint for the emptiest empty cell, corner cells first."""
    rows, cols = hits.shape
    pad = np.pad(hits, 1)
    near = sum(pad[r:r + rows, c:c + cols] for r in range(3) for c in range(3))
    corner = np.zeros(hits.shape, bool)
    corner[[0, 0, -1, -1], [0, -1, 0, -1]] = True
    r, c = min((tuple(rc) for rc in np.argwhere(hits == 0)),
               key=lambda rc: (not corner[rc], near[rc], rc))
    x, y = (c + 0.5) / cols, (r + 0.5) / rows
    return Hint('COVER_REGION', {'region': _region(x, y)}, 'coverage', (x, y), cam)


def _rel_ci(cam) -> Optional[float]:
    """Return max(1.96 sigma / value) of fx, fy and 1.96 sigma / size of cx, cy, or None."""
    w, h = cam.size
    s = cam.sigmas
    ref = (('fx', cam.K[0, 0]), ('fy', cam.K[1, 1]), ('cx', w), ('cy', h))
    terms = [abs(s[k] / d) for k, d in ref if k in s and np.isfinite(s[k])]
    return 1.96 * max(terms) if terms else None


def _spread(feats, k) -> list:
    """Return the indices of up to k views spread out in feature space (farthest points)."""
    if len(feats) <= k:
        return list(range(len(feats)))
    x = np.array([_vec(f) for f in feats])
    sel = [len(x) - 1]
    d = np.linalg.norm(x - x[-1], axis=1)
    d[-1] = -np.inf
    while len(sel) < k:
        j = int(np.argmax(d))
        sel.append(j)
        d = np.minimum(d, np.linalg.norm(x - x[j], axis=1))
        d[j] = -np.inf
    return sel


class GuidanceEngine:
    """
    Computes indicators, hints and the live estimate from DB snapshots (thread-confined).

    Indicators per camera: coverage = 0.75 min(1, hit / 0.9) + 0.25 corner cells hit / 4 (full
    iff >= 90 % cells and the 4 corner cells hit); tilt_x/tilt_y and distance = filled bins / 3,
    where distance uses the 'size' feature sqrt(hull area / image area), close to the board width
    fraction for boards shaped like the image; uncertainty = min(1, target / rel_ci).
    ready: N >= min_samples (stereo: also pairs >= min_samples) and either every indicator full
    or rel_ci improved < 10 % over the last 5 samples with every coverage >= 0.8.
    Hints: READY first when ready, then actionable hints by decreasing deficit, then KEEP_GOING.
    """

    def __init__(self, board, calib_cfg, guidance_cfg: GuidanceConfig, image_sizes: tuple):
        """Create the engine for a board, calibration options and per-camera image sizes."""
        self._calib_cfg = calib_cfg
        self._cfg = guidance_cfg
        self._sizes = tuple(tuple(s) for s in image_sizes)
        self._cams = [None] * len(self._sizes)   # latest successful quick solve per camera
        self._estimate = None
        self._history = {}                       # number of samples -> Estimate.rel_ci
        self._state = None

    @property
    def estimate(self) -> Optional[Estimate]:
        """Return the latest live estimate, None before the first successful quick solve."""
        return self._estimate

    def update(self, snapshot) -> GuidanceState:
        """Recompute the state for a snapshot (may take up to ~0.3 s: quick solve + sigmas)."""
        if self._state is not None and self._state.db_version == snapshot.version:
            return self._state
        n = len(snapshot.samples)
        dets = [[d for _, d in snapshot.views_of(i)] for i in range(len(self._sizes))]
        feats = [[_features(d, size, cam, self._calib_cfg.model) for d in ds]
                 for ds, size, cam in zip(dets, self._sizes, self._cams)]
        solved = self._solve(dets, feats, n)
        indicators, cands, grids, covs = [], [], [], []
        for i in range(len(self._sizes)):
            inds, hs, grid = self._camera(i, dets[i], feats[i])
            indicators += inds
            cands += hs
            grids.append(grid)
            covs.append(inds[0].progress)
        rel = self._estimate.rel_ci if solved else None
        self._history = {k: v for k, v in self._history.items() if k < n}
        if rel is not None:
            self._history[n] = rel
        old = self._history.get(n - _PLATEAU_SAMPLES)
        plateau = (rel is not None and old is not None and old - rel < _PLATEAU_GAIN * old
                   and min(covs) >= _PLATEAU_COVERAGE)
        params = {'samples': n}
        if len(self._sizes) > 1:
            params['pairs'] = len(snapshot.pairs())
        enough = n >= self._cfg.min_samples and params.get('pairs', n) >= self._cfg.min_samples
        ready = enough and (plateau or all(ind.progress >= 1.0 for ind in indicators))
        cands.sort(key=lambda c: (-c[0], HINT_CODES.index(c[1].code), c[1].camera))
        hints = [h for _, h in cands]
        if ready:
            hints.insert(0, Hint('READY', params))
        else:
            hints.append(Hint('KEEP_GOING', params))
        self._state = GuidanceState(snapshot.version, tuple(indicators), tuple(hints), ready,
                                    tuple(grids), self._estimate)
        return self._state

    def _solve(self, dets, feats, n) -> bool:
        """Refresh the quick solve of each camera; return True if every camera was solved now."""
        solved = []
        for i, (ds, fs) in enumerate(zip(dets, feats)):
            if len(ds) < _EST_MIN_VIEWS:
                solved.append(False)
                continue
            views = [ds[j] for j in _spread(fs, _EST_MAX_VIEWS)]
            try:
                self._cams[i] = calibration.quick_calibrate(views, self._sizes[i], self._calib_cfg,
                                                            self._cams[i], _EST_MAX_ITER)
                solved.append(True)
            except Exception:     # keep the previous solve: early views are often degenerate
                solved.append(False)
        if any(solved) and all(c is not None for c in self._cams):
            rels = [_rel_ci(c) for c in self._cams]
            self._estimate = Estimate(tuple(self._cams), None if None in rels else max(rels), n)
        return all(solved)

    def _camera(self, i, dets, feats) -> tuple:
        """Return (indicators, [(deficit, Hint)], heat grid) of camera i."""
        size = self._sizes[i]
        hits = _hits(dets, size, _grid_shape(size, self._cfg.grid))
        hit = hits > 0
        corners = hit[[0, 0, -1, -1], [0, -1, 0, -1]]
        cov = 0.75 * min(1.0, hit.mean() / _COVER_FULL) + 0.25 * corners.mean()
        inds = [Indicator('coverage', cov, {'cells_hit': int(hit.sum()), 'cells': hit.size,
                                            'corners_hit': int(corners.sum())}, i)]
        cands = [(1.0 - cov, _cover_hint(hits, i))] if cov < 1.0 else []
        for ind, key, code in (('tilt_x', 'tilt_x_deg', 'TILT_UP_DOWN'),
                               ('tilt_y', 'tilt_y_deg', 'TILT_LEFT_RIGHT')):
            t = np.array([f[key] for f in feats])
            bins = (bool(np.any(t < -_TILT_EDGE)), bool(np.any(np.abs(t) <= _TILT_EDGE)),
                    bool(np.any(t > _TILT_EDGE)))
            p = sum(bins) / 3
            inds.append(Indicator(ind, p, {'bins': bins}, i))
            if p < 1.0:
                b = next(b for b in (0, 2, 1) if not bins[b])
                hint = Hint(code, {'direction': _TILT_DIRS[ind][b]}, ind, None, i)
                cands.append((1.0 - p, hint))
        s = np.array([f['size'] for f in feats])
        lo, hi = _SIZE_EDGES
        bins = (bool(np.any(s < lo)), bool(np.any((s >= lo) & (s <= hi))), bool(np.any(s > hi)))
        p = sum(bins) / 3
        inds.append(Indicator('distance', p, {'bins': bins}, i))
        if p < 1.0:
            if not bins[2]:
                code, name = 'MOVE_CLOSER', 'close'
            elif not bins[0]:
                code, name = 'MOVE_FARTHER', 'far'
            else:
                name = 'mid'
                code = 'MOVE_FARTHER' if np.median(s) > (lo + hi) / 2 else 'MOVE_CLOSER'
            cands.append((1.0 - p, Hint(code, {'bin': name}, 'distance', None, i)))
        cam = self._cams[i]
        rel = _rel_ci(cam) if cam is not None and len(dets) >= _EST_MIN_VIEWS else None
        target = self._cfg.target_rel_ci
        up = 0.0 if rel is None else 1.0 if rel <= target else target / rel
        inds.append(Indicator('uncertainty', up, {'rel_ci': rel, 'target': target}, i))
        return inds, cands, np.minimum(hits / _HEAT_VIEWS, 1.0)
