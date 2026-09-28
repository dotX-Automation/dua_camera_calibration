"""
Synthetic calibration images with exact ground truth, for tests and the synthetic camera.

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

from functools import lru_cache

import cv2
import numpy as np

SUPERSAMPLING = 2   # samples per pixel per axis, box-filtered down


def board_texture(board, px_per_square: int = 64, margin_squares: int = 1) -> tuple:
    """
    Draw the board with a white margin.

    Return (uint8 image, scale [px/m], origin (x, y) [px]): the board-frame point (X, Y, 0) is at
    texture pixel origin + scale * (X, Y), pixel centres at integer coordinates.
    """
    p, m = int(px_per_square), int(margin_squares)
    if board.type == 'charuco':
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board.dictionary))
        cb = cv2.aruco.CharucoBoard((board.cols, board.rows), board.square_size,
                                    board.marker_size, dictionary)
        cb.setLegacyPattern(board.legacy_pattern)
        img = cb.generateImage(((board.cols + 2 * m) * p, (board.rows + 2 * m) * p),
                               marginSize=m * p, borderBits=1)
    else:
        rows, cols = np.indices((board.rows + 1, board.cols + 1))
        squares = np.where((rows + cols) % 2 == 0, 0, 255).astype(np.uint8)
        img = np.pad(np.kron(squares, np.ones((p, p), np.uint8)), m * p, constant_values=255)
    o = (m + 1) * p - 0.5
    return img, p / board.square_size, (o, o)


@lru_cache(maxsize=2)
def _rays(model: str, size: tuple, k: bytes, d: bytes) -> tuple:
    """Return the normalized undistorted (x, y) float32 maps of the supersampled pixel grid."""
    w, h = size
    s = SUPERSAMPLING
    xs = (np.arange(w * s) + 0.5) / s - 0.5
    ys = (np.arange(h * s) + 0.5) / s - 0.5
    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 1, 2)
    K = np.frombuffer(k).reshape(3, 3)
    D = np.frombuffer(d)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
    if model == 'fisheye':
        n = cv2.fisheye.undistortPoints(pts, K, D, criteria=crit).reshape(-1, 2)
        # cv2 clamps the distorted angle to pi/2: rays beyond it do not exist (background)
        r = np.hypot((pts[:, 0, 0] - K[0, 2]) / K[0, 0], (pts[:, 0, 1] - K[1, 2]) / K[1, 1])
        n[r >= np.pi / 2 - 1e-6] = np.nan
    else:
        n = cv2.undistortPointsIter(pts, K, D, None, None, crit)
    n = n.reshape(h * s, w * s, 2).astype(np.float32)
    return n[..., 0], n[..., 1]


def render_view(board, cam, rvec, tvec, blur_sigma: float = 0.0, noise_sigma: float = 0.0,
                background: int = 160, seed: int = 0) -> np.ndarray:
    """
    Render the board seen by cam (calibration.CameraCalib) at pose (rvec, tvec) board->camera.

    Every pixel ray is undistorted and intersected with the board plane (board frame as in
    BoardSpec.object_points: z = 0, X right, Y down, origin at corner id 0), with 2x2
    supersampling; the texture resolution follows the projected square size to avoid aliasing.
    Return uint8 gray (h, w).
    """
    w, h = cam.size
    K = np.ascontiguousarray(cam.K, np.float64)
    D = np.ascontiguousarray(cam.D, np.float64).ravel()
    n_cols, n_rows = board.grid
    g = project_corners(board, cam, rvec, tvec).reshape(n_rows, n_cols, 2)
    spacing = min(np.linalg.norm(np.diff(g, axis=0), axis=2).min(),
                  np.linalg.norm(np.diff(g, axis=1), axis=2).min())
    tex, scale, origin = board_texture(board, int(np.clip(round(SUPERSAMPLING * spacing), 8, 256)))

    x, y = _rays(cam.model, tuple(cam.size), K.tobytes(), D.tobytes())
    R = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]
    t = np.asarray(tvec, np.float64).ravel()
    n = R[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (n @ t) / (n[0] * x + n[1] * y + n[2])     # ray length along (x, y, 1)
    s[~(s > 0)] = np.nan
    bx = R[0, 0] * (s * x - t[0]) + R[1, 0] * (s * y - t[1]) + R[2, 0] * (s - t[2])
    by = R[0, 1] * (s * x - t[0]) + R[1, 1] * (s * y - t[1]) + R[2, 1] * (s - t[2])
    mx = np.nan_to_num((origin[0] + scale * bx).astype(np.float32), nan=-1e6)
    my = np.nan_to_num((origin[1] + scale * by).astype(np.float32), nan=-1e6)
    img = cv2.remap(tex.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=float(background))
    img = img.reshape(h, SUPERSAMPLING, w, SUPERSAMPLING).mean(axis=(1, 3))
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if noise_sigma > 0:
        img = img + np.random.default_rng(seed).normal(0.0, noise_sigma, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def project_corners(board, cam, rvec, tvec) -> np.ndarray:
    """Return the ground-truth (n_corners, 2) float32 image points of all corners, by id."""
    return _project(cam, board.object_points(), rvec, tvec).astype(np.float32)


def random_poses(board, cam, n: int, seed: int = 0, tilt_max_deg: float = 40.0,
                 fill: tuple = (0.3, 0.9)) -> list:
    """
    Draw n random board poses (rvec, tvec) that keep the whole board inside the image.

    fill: range of the board width over the image width (fronto-parallel equivalent). The tilt
    (angle between board normal and optical axis) is uniform in [0, tilt_max_deg], the in-plane
    rotation within +-20 deg. Every corner and the board outer edge stay 3% of the image
    height/width away from the border.
    """
    rng = np.random.default_rng(seed)
    w, h = cam.size
    K = np.asarray(cam.K, np.float64)
    n_cols, n_rows = board.grid
    sq = board.square_size
    obj = board.object_points().astype(np.float64)
    edge = np.array([[-sq, -sq, 0], [n_cols * sq, -sq, 0], [n_cols * sq, n_rows * sq, 0],
                     [-sq, n_rows * sq, 0]])
    pts = np.vstack([obj, edge])
    centre = obj.mean(axis=0)
    margin = 0.03 * min(w, h)
    poses = []
    for _ in range(2000 * n):
        if len(poses) == n:
            break
        z = K[0, 0] * (n_cols + 1) * sq / (rng.uniform(*fill) * w)
        phi, tilt = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(0, tilt_max_deg))
        R = cv2.Rodrigues(np.array([np.cos(phi), np.sin(phi), 0.0]) * tilt)[0] @ cv2.Rodrigues(
            np.array([0.0, 0.0, np.radians(rng.uniform(-20, 20))]))[0]
        u, v = rng.uniform(0, w), rng.uniform(0, h)
        p = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        rvec, tvec = cv2.Rodrigues(R)[0].ravel(), p - R @ centre
        if np.any((pts @ R.T + tvec)[:, 2] <= 0):
            continue
        q = _project(cam, pts, rvec, tvec)
        if np.all((q >= margin) & (q <= np.array([w - 1, h - 1]) - margin)):
            poses.append((rvec, tvec))
    if len(poses) < n:
        raise RuntimeError(f'random_poses: only {len(poses)} of {n} poses fit the image')
    return poses


def _project(cam, pts: np.ndarray, rvec, tvec) -> np.ndarray:
    """Project (M, 3) board points with the camera model to (M, 2) float64 pixels."""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 3)
    K = np.asarray(cam.K, np.float64)
    D = np.asarray(cam.D, np.float64).ravel()
    rvec = np.asarray(rvec, np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, np.float64).reshape(3, 1)
    if cam.model == 'fisheye':
        return cv2.fisheye.projectPoints(pts, rvec, tvec, K, D)[0].reshape(-1, 2)
    return cv2.projectPoints(pts, rvec, tvec, K, D)[0].reshape(-1, 2)
