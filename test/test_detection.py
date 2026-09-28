"""
Tests of chessboard and ChArUco detection on synthetic renders with exact ground truth.

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

import dataclasses
import time

import cv2
from dua_camera_calibration.calibration import CameraCalib
from dua_camera_calibration.detection import BoardSpec, make_detector
from dua_camera_calibration.synthetic import project_corners, random_poses, render_view
import numpy as np
import pytest

CAM = CameraCalib('pinhole', 'plumb_bob', (1280, 720),
                  np.array([[900.0, 0.0, 639.5], [0.0, 900.0, 359.5], [0.0, 0.0, 1.0]]),
                  np.array([-0.2, 0.08, 0.001, -0.0005, 0.0]))
FISHEYE = CameraCalib('fisheye', 'equidistant', (1280, 720),
                      np.array([[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]]),
                      np.array([0.05, -0.01, 0.002, 0.0]))
CHESS = BoardSpec(cols=9, rows=6, square_size=0.03)
CHARUCO = BoardSpec(type='charuco', cols=10, rows=7, square_size=0.03, marker_size=0.022)
# ChArUco refinement windows must stay inside the corner-marker gap: ~0.035 px mean, 0.056 worst.
CHARUCO_TOL = 0.06


def _errors(board, cam, det, rvec, tvec) -> np.ndarray:
    """Return per-corner errors [px] of a detection against the ground-truth projection."""
    assert det is not None
    return np.linalg.norm(det.corners - project_corners(board, cam, rvec, tvec)[det.ids], axis=1)


def _check_contract(board, det) -> None:
    """Assert dtypes, shapes, immutability and object-point consistency of a Detection."""
    n = len(det.ids)
    assert det.corners.dtype == np.float32 and det.corners.shape == (n, 2)
    assert det.ids.dtype == np.int32 and np.all(np.diff(det.ids) > 0)
    assert det.object_points.dtype == np.float32 and det.object_points.shape == (n, 3)
    assert np.array_equal(det.object_points, board.object_points()[det.ids])
    assert det.outline.dtype == np.float32 and det.outline.shape == (4, 2)
    assert det.image_size == CAM.size and 2 <= det.window <= 25
    for a in (det.corners, det.ids, det.object_points, det.outline):
        assert not a.flags.writeable


def _rotated_in_plane(board, rvec, tvec, angle: float) -> tuple:
    """Return the pose of the board rotated in its plane by angle about its centre."""
    R = cv2.Rodrigues(rvec)[0]
    R2 = R @ cv2.Rodrigues(np.array([0.0, 0.0, angle]))[0]
    c = board.object_points().astype(np.float64).mean(axis=0)
    return cv2.Rodrigues(R2)[0].ravel(), tvec + R @ c - R2 @ c


def test_chessboard_accuracy():
    """Classic detector: < 0.05 px mean corner error with mild blur and noise."""
    det = make_detector(CHESS)
    for i, (r, t) in enumerate(random_poses(CHESS, CAM, 6, seed=1)):
        d = det.detect(render_view(CHESS, CAM, r, t, blur_sigma=0.8, noise_sigma=1.0, seed=i))
        e = _errors(CHESS, CAM, d, r, t)
        assert d.complete and len(d.ids) == CHESS.n_corners
        assert np.array_equal(d.ids, np.arange(CHESS.n_corners))
        assert e.mean() < 0.05, (i, e.mean())
        _check_contract(CHESS, d)
        n_cols = CHESS.grid[0]
        assert np.array_equal(d.outline, d.corners[[0, n_cols - 1, -1, -n_cols]])
        assert 1.5 < d.sharpness < 3.5


@pytest.mark.parametrize('board', [
    dataclasses.replace(CHESS, max_pixels=640 * 360),      # downscaled detection path
    dataclasses.replace(CHESS, detector='sb')])
def test_chessboard_variants(board):
    """Downscaled classic detection and SB detection keep the accuracy."""
    det = make_detector(board)
    for i, (r, t) in enumerate(random_poses(board, CAM, 2, seed=2)):
        d = det.detect(render_view(board, CAM, r, t, blur_sigma=0.8, noise_sigma=1.0, seed=i))
        assert d.complete and _errors(board, CAM, d, r, t).mean() < 0.05


def test_sharpness_tracks_blur():
    """The transition width grows with the blur."""
    det = make_detector(CHESS)
    r, t = random_poses(CHESS, CAM, 1, seed=3)[0]
    sharp = [det.detect(render_view(CHESS, CAM, r, t, blur_sigma=b)).sharpness for b in (0.5, 2.0)]
    assert sharp[1] > sharp[0] + 1.0


@pytest.mark.parametrize('board,angle', [
    (CHESS, np.pi), (BoardSpec(cols=7, rows=7, square_size=0.03), np.pi / 2)])
def test_canonical_orientation_is_image_based(board, angle):
    """Board symmetries do not change the image-based corner labelling."""
    det = make_detector(board)
    r, t = random_poses(board, CAM, 1, seed=4, tilt_max_deg=25)[0]
    r2, t2 = _rotated_in_plane(board, r, t, angle)
    d1 = det.detect(render_view(board, CAM, r, t, blur_sigma=0.8))
    d2 = det.detect(render_view(board, CAM, r2, t2, blur_sigma=0.8))
    assert np.array_equal(d1.ids, d2.ids)
    assert np.abs(d1.corners - d2.corners).max() < 0.1
    g = d1.corners.reshape(board.rows, board.cols, 2)
    assert g[0, -1, 0] > g[0, 0, 0] and g[-1, 0, 1] > g[0, 0, 1]     # rows run right, cols down


def test_no_board_in_noise_is_fast():
    """Detectors return None on noise, quickly on textured clutter."""
    noise = np.random.default_rng(5).integers(0, 256, (720, 1280)).astype(np.uint8)
    clutter = cv2.GaussianBlur(noise, (0, 0), 2.0)
    for board, budget in [(CHESS, 0.05), (CHARUCO, 0.2)]:
        det = make_detector(board)
        assert det.detect(noise) is None      # white noise is the worst case: ~0.4 s classic
        t0 = time.monotonic()
        assert det.detect(clutter) is None
        assert time.monotonic() - t0 < budget


def test_charuco_complete_accuracy():
    """ChArUco: complete views, with and without a camera estimate."""
    det = make_detector(CHARUCO)
    for i, (r, t) in enumerate(random_poses(CHARUCO, CAM, 4, seed=6)):
        img = render_view(CHARUCO, CAM, r, t, blur_sigma=0.8, noise_sigma=1.0, seed=i)
        for cam in (None, CAM, CAM):      # plain, with a camera estimate, cached estimate
            d = det.detect(img, cam)
            assert d.complete and _errors(CHARUCO, CAM, d, r, t).mean() < CHARUCO_TOL
            _check_contract(CHARUCO, d)
        assert d.sharpness is None


def test_charuco_partial_view():
    """ChArUco: a board half out of the image yields an accurate partial detection."""
    det = make_detector(CHARUCO)
    r = np.array([0.2, -0.1, 0.05])
    c = CHARUCO.object_points().astype(np.float64).mean(axis=0)
    t = np.array([(0.0 - 639.5) / 900.0 * 0.5, 0.0, 0.5]) - cv2.Rodrigues(r)[0] @ c
    d = det.detect(render_view(CHARUCO, CAM, r, t, blur_sigma=0.8, noise_sigma=1.0))
    assert not d.complete and 6 <= len(d.ids) < CHARUCO.n_corners
    assert _errors(CHARUCO, CAM, d, r, t).mean() < CHARUCO_TOL
    _check_contract(CHARUCO, d)
    assert all(np.any(np.all(d.corners == p, axis=1)) for p in d.outline)


def test_charuco_ids_match_opencv_object_points():
    """Check that OpenCV ChArUco ids and object points match BoardSpec (legacy pattern too)."""
    board = dataclasses.replace(CHARUCO, rows=8, legacy_pattern=True)   # even rows: legacy matters
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board.dictionary))
    cb = cv2.aruco.CharucoBoard((board.cols, board.rows), board.square_size, board.marker_size,
                                dictionary)
    cb.setLegacyPattern(True)
    offset = np.array([board.square_size, board.square_size, 0.0], np.float32)
    assert np.allclose(cb.getChessboardCorners() - offset, board.object_points(), atol=1e-7)
    r, t = random_poses(board, CAM, 1, seed=7)[0]
    d = make_detector(board).detect(render_view(board, CAM, r, t, blur_sigma=0.8))
    assert d.complete and _errors(board, CAM, d, r, t).mean() < CHARUCO_TOL
    obj, _ = cb.matchImagePoints(d.corners.reshape(-1, 1, 2), d.ids.reshape(-1, 1))
    assert np.allclose(obj.reshape(-1, 3) - offset, d.object_points, atol=1e-7)


def test_charuco_rejects_few_or_collinear_corners():
    """Fewer than 6 or collinear ChArUco corners are rejected."""
    det = make_detector(CHARUCO)

    class Fake:
        """OpenCV detector stub returning fixed corners."""

        def __init__(self, ids):
            """Store the ids to return."""
            self.ids = np.array(ids, np.int32).reshape(-1, 1)

        def detectBoard(self, gray):
            """Return corners on a 50 px grid for the stored ids."""
            rc = np.stack([self.ids // 9, self.ids % 9], axis=-1).astype(np.float32)
            return 100 + 50 * rc[..., ::-1], self.ids, None, None

    gray = np.zeros((720, 1280), np.uint8)
    for ids in ([0, 1, 2, 3, 4, 5, 6, 7], [0, 10, 20, 30, 40, 50], [0, 1, 9, 10, 11]):
        det._detector = Fake(ids)
        assert det.detect(gray) is None


def test_fisheye_render_matches_projection():
    """Fisheye renders agree with fisheye projections."""
    det = make_detector(CHESS)
    for i, (r, t) in enumerate(random_poses(CHESS, FISHEYE, 2, seed=8, tilt_max_deg=30)):
        d = det.detect(render_view(CHESS, FISHEYE, r, t, blur_sigma=0.8, seed=i))
        assert _errors(CHESS, FISHEYE, d, r, t).mean() < 0.05
