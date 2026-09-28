"""
Calibration target specification and detection.

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

from dataclasses import dataclass
import re
from typing import Optional

import cv2
import numpy as np


def require_opencv(min_version=(4, 8)) -> None:
    """Raise RuntimeError if the loaded OpenCV is older than min_version."""
    m = re.match(r'(\d+)\.(\d+)', cv2.__version__)
    if m is None or (int(m.group(1)), int(m.group(2))) < tuple(min_version):
        raise RuntimeError(
            f'OpenCV >= {min_version[0]}.{min_version[1]} required, found {cv2.__version__}')


@dataclass(frozen=True)
class BoardSpec:
    """
    Calibration target.

    Chessboard: cols x rows INNER corners. ChArUco: cols x rows SQUARES.
    Corner ids are row-major over the inner-corner grid (id = r * n_cols + c).
    """

    type: str = 'chessboard'          # noqa: A003 'chessboard' | 'charuco'
    cols: int = 8
    rows: int = 6
    square_size: float = 0.025        # [m]
    marker_size: float = 0.018        # [m], ChArUco only
    dictionary: str = 'DICT_5X5_100'  # ChArUco only
    legacy_pattern: bool = False      # ChArUco only
    detector: str = 'classic'         # 'classic' | 'sb', chessboard only
    max_pixels: int = 1280 * 720      # detection downscale target (chessboard)

    @property
    def grid(self) -> tuple:
        """Return the inner-corner grid size (n_cols, n_rows)."""
        if self.type == 'charuco':
            return (self.cols - 1, self.rows - 1)
        return (self.cols, self.rows)

    @property
    def n_corners(self) -> int:
        """Return the number of inner corners of the full board."""
        c, r = self.grid
        return c * r

    def object_points(self) -> np.ndarray:
        """Return (n_corners, 3) float32 board-frame coordinates [m] of all corners, by id."""
        c, r = self.grid
        ys, xs = np.mgrid[0:r, 0:c]
        pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(c * r)], axis=1) * self.square_size
        return pts.astype(np.float32)


@dataclass(frozen=True)
class Detection:
    """
    Board detected in one image (full-resolution pixel coordinates).

    Arrays are never mutated after construction.
    """

    corners: np.ndarray          # (N, 2) float32 [px]
    ids: np.ndarray              # (N,) int32 corner ids (see BoardSpec)
    object_points: np.ndarray    # (N, 3) float32 [m], matching ids
    complete: bool               # all board corners detected
    outline: np.ndarray          # (4, 2) float32 [px] extreme detected grid corners TL, TR, BR, BL
    image_size: tuple            # (width, height)
    sharpness: Optional[float] = None   # edge transition width [px], None if unavailable
    window: int = 5              # cornerSubPix half-window used [px]


_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-3)


def _refine(gray: np.ndarray, corners: np.ndarray, rc: np.ndarray, factor: float = 0.3) -> tuple:
    """
    Refine corners at full resolution with half-window = clamp(round(factor * spacing), 2, 25).

    rc: (N, 2) grid (row, col) of each corner. The spacing is the minimum over all pairs of
    image distance / grid distance, so undetected neighbours are accounted for.
    Return (refined (N, 2) float32, half-window).
    """
    i, j = np.triu_indices(len(corners), 1)
    spacing = np.min(np.linalg.norm(corners[i] - corners[j], axis=1)
                     / np.linalg.norm((rc[i] - rc[j]).astype(np.float64), axis=1))
    win = int(np.clip(round(factor * float(spacing)), 2, 25))
    c = cv2.cornerSubPix(gray, np.ascontiguousarray(corners, dtype=np.float32).reshape(-1, 1, 2),
                         (win, win), (-1, -1), _SUBPIX_CRITERIA)
    return c.reshape(-1, 2), win


def _frozen(a: np.ndarray, dtype) -> np.ndarray:
    """Return a read-only C-contiguous copy of a."""
    a = np.array(a, dtype=dtype, order='C')
    a.flags.writeable = False
    return a


def _detection(board: BoardSpec, gray: np.ndarray, corners: np.ndarray, ids: np.ndarray,
               win: int, sharpness=None) -> Detection:
    """Build a Detection; the outline is the detected corners extreme along the grid diagonals."""
    n_cols = board.grid[0]
    r, c = ids // n_cols, ids % n_cols
    quad = [np.argmin(r + c), np.argmax(c - r), np.argmax(r + c), np.argmax(r - c)]
    return Detection(corners=_frozen(corners, np.float32), ids=_frozen(ids, np.int32),
                     object_points=_frozen(board.object_points()[ids], np.float32),
                     complete=ids.size == board.n_corners,
                     outline=_frozen(corners[quad], np.float32),
                     image_size=(gray.shape[1], gray.shape[0]), sharpness=sharpness, window=win)


def _canonical(g: np.ndarray) -> np.ndarray:
    """
    Re-order a (rows, cols, 2) corner grid into the canonical image-based orientation.

    First the handedness is made that of the board frame seen from the front (row direction x
    column direction > 0 in image coordinates, i.e. board Z away from the camera). Then, among
    the grid rotations that keep the (rows, cols) shape, pick the one whose rows run most
    horizontally left-to-right: id 0 is then the corner nearest the image top-left.
    """
    def cross(a):
        u, v = a[0, -1] - a[0, 0], a[-1, 0] - a[0, 0]
        return u[0] * v[1] - u[1] * v[0]

    def rightness(a):
        u = a[0, -1] - a[0, 0]
        return u[0] / (np.linalg.norm(u) + 1e-12)

    if cross(g) < 0:
        g = g[:, ::-1]
    cands = [g, g[::-1, ::-1]]
    if g.shape[0] == g.shape[1]:
        cands += [np.rot90(g, 1), np.rot90(g, -1)]
    # ponytail: an image-based rule is discontinuous where two candidates tie (non-square board
    # with vertical rows, square board rotated 45 deg); stereo pairs may disagree there
    return max(cands, key=rightness)


class ChessboardDetector:
    """Classic or SB chessboard detector (thread-confined)."""

    def __init__(self, board: BoardSpec):
        """Create the detector for a chessboard BoardSpec."""
        self.board = board

    def detect(self, gray: np.ndarray, cam=None) -> Optional[Detection]:
        """Detect the complete chessboard in a uint8 gray image (cam unused); None if not found."""
        board = self.board
        h, w = gray.shape
        s = min(1.0, np.sqrt(board.max_pixels / float(w * h)))
        small = gray
        if s < 1.0:
            small = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))),
                               interpolation=cv2.INTER_AREA)
        if board.detector == 'sb':
            ok, c = cv2.findChessboardCornersSB(
                small, board.grid,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY)
        else:
            ok, c = cv2.findChessboardCorners(
                small, board.grid,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK)
        if not ok:
            return None
        # Pixel centres: full = (small + 0.5) / scale - 0.5, per axis.
        scale = np.array([small.shape[1] / w, small.shape[0] / h], np.float32)
        c = (c.reshape(-1, 2) + 0.5) / scale - 0.5
        n_cols, n_rows = board.grid
        c = _canonical(c.reshape(n_rows, n_cols, 2)).reshape(-1, 2)
        ids = np.arange(board.n_corners, dtype=np.int32)
        c, win = _refine(gray, c, np.stack([ids // n_cols, ids % n_cols], axis=1))
        sharp = cv2.estimateChessboardSharpness(gray, board.grid, c.reshape(-1, 1, 2))[0][0]
        return _detection(board, gray, c, ids, win, float(sharp))


class CharucoDetector:
    """ChArUco detector (thread-confined); partial views with >= 6 non-collinear corners."""

    def __init__(self, board: BoardSpec):
        """Create the detector for a ChArUco BoardSpec."""
        self.board = board
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board.dictionary))
        self._board = cv2.aruco.CharucoBoard((board.cols, board.rows), board.square_size,
                                             board.marker_size, dictionary)
        self._board.setLegacyPattern(board.legacy_pattern)
        self._params = cv2.aruco.DetectorParameters()
        self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # The window must stay inside the gap between the chessboard corner and the markers:
        # measured on renders, 0.3 * spacing reaches the marker edges (0.3 px error), 0.9 * gap
        # gives 0.03 px. OpenCV's own ChArUco corners are off by ~(0.5, 0.5) px in 4.11.
        self._factor = min(0.3, 0.9 * (board.square_size - board.marker_size) / 2
                           / board.square_size)
        self._cam = None
        self._detector = cv2.aruco.CharucoDetector(self._board, cv2.aruco.CharucoParameters(),
                                                   self._params)

    def _set_camera(self, cam) -> None:
        """Rebuild the OpenCV detector when the camera estimate changes."""
        if cam is self._cam:
            return
        self._cam = cam
        cp = cv2.aruco.CharucoParameters()
        # ponytail: fisheye estimates are not fed (OpenCV expects pinhole D here); local
        # homography interpolation is used instead
        if cam is not None and cam.model == 'pinhole':
            cp.cameraMatrix = np.asarray(cam.K, np.float64)
            cp.distCoeffs = np.asarray(cam.D, np.float64).reshape(1, -1)
        self._detector = cv2.aruco.CharucoDetector(self._board, cp, self._params)

    def detect(self, gray: np.ndarray, cam=None) -> Optional[Detection]:
        """Detect ChArUco corners in a uint8 gray image; cam (CameraCalib) aids interpolation."""
        self._set_camera(cam)
        c, ids, _, _ = self._detector.detectBoard(gray)
        if ids is None or len(ids) < 6:
            return None
        ids = ids.reshape(-1).astype(np.int32)
        order = np.argsort(ids)
        ids, c = ids[order], c.reshape(-1, 2)[order]
        n_cols = self.board.grid[0]
        rc = np.stack([ids // n_cols, ids % n_cols], axis=1)
        if np.linalg.matrix_rank(rc - rc.mean(axis=0)) < 2:
            return None
        c, win = _refine(gray, c, rc, self._factor)
        return _detection(self.board, gray, c, ids, win)


def make_detector(board: BoardSpec):
    """Create the thread-confined detector for a board (use one per thread)."""
    if board.type == 'charuco':
        return CharucoDetector(board)
    return ChessboardDetector(board)
