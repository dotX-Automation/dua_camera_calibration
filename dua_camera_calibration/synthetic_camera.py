"""
Synthetic mono/stereo camera publishing rendered calibration boards, with SetCameraInfo servers.

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

import argparse
import sys

import cv2
from dua_camera_calibration.calibration import CameraCalib
from dua_camera_calibration.detection import BoardSpec
from dua_camera_calibration.synthetic import project_corners, random_poses, render_view
from dua_node_py.dua_node import NodeBase
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.srv import SetCameraInfo


def _set_header(msg, stamp_ns: int, frame_id: str) -> None:
    """Fill the header of a message."""
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(int(stamp_ns), 1000000000)
    msg.header.frame_id = frame_id


def make_image_msg(gray: np.ndarray, stamp_ns: int = 0, frame_id: str = 'camera') -> Image:
    """Return a mono8 sensor_msgs/Image of a uint8 gray image."""
    msg = Image()
    _set_header(msg, stamp_ns, frame_id)
    msg.height, msg.width = gray.shape
    msg.encoding, msg.is_bigendian, msg.step = 'mono8', 0, gray.shape[1]
    msg.data = np.ascontiguousarray(gray).tobytes()
    return msg


def make_compressed_msg(gray: np.ndarray, stamp_ns: int = 0, fmt: str = 'png',
                        frame_id: str = 'camera') -> CompressedImage:
    """Return a sensor_msgs/CompressedImage (fmt 'png' or 'jpeg') of a uint8 gray image."""
    msg = CompressedImage()
    _set_header(msg, stamp_ns, frame_id)
    ok, buf = cv2.imencode('.jpg' if fmt == 'jpeg' else '.png', gray)
    if not ok:
        raise RuntimeError(f'cannot encode {fmt}')
    msg.format = f'mono8; {fmt} compressed mono8'
    msg.data = buf.tobytes()
    return msg


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the synthetic camera options (ROS arguments must be removed already)."""
    p = argparse.ArgumentParser(prog='synthetic_camera',
                                description=__doc__.strip().split('\n')[0])
    p.add_argument('--mode', choices=('mono', 'stereo'), default='mono')
    p.add_argument('--transport', choices=('raw', 'compressed', 'both'), default='both')
    p.add_argument('--jpeg', action='store_true', help='compressed as JPEG instead of PNG')
    p.add_argument('--rate', type=float, default=10.0, help='frame rate [Hz]')
    p.add_argument('--ns', default='/synthetic', help='topic and service namespace')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fx', type=float, default=500.0, help='focal length [px] (fx = fy)')
    p.add_argument('--model', choices=('pinhole', 'fisheye'), default='pinhole')
    p.add_argument('--baseline', type=float, default=0.1, help='stereo baseline [m]')
    p.add_argument('--board-type', choices=('chessboard', 'charuco'), default='chessboard')
    p.add_argument('--cols', type=int, default=8)
    p.add_argument('--rows', type=int, default=6)
    p.add_argument('--square-size', type=float, default=0.025)
    p.add_argument('--marker-size', type=float, default=0.018)
    p.add_argument('--dictionary', default='DICT_5X5_100')
    p.add_argument('--poses', type=int, default=20, help='distinct board poses')
    p.add_argument('--transition', type=int, default=3, help='rendered frames between poses')
    p.add_argument('--dwell', type=float, default=1.0, help='time still at each pose [s]')
    p.add_argument('--noise', type=float, default=1.0, help='image noise sigma [gray levels]')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args(argv)


def board_of(args) -> BoardSpec:
    """Return the board of the options."""
    return BoardSpec(type=args.board_type, cols=args.cols, rows=args.rows,
                     square_size=args.square_size, marker_size=args.marker_size,
                     dictionary=args.dictionary)


def camera_of(args) -> CameraCalib:
    """Return the ground-truth camera of the options (both stereo cameras share it)."""
    w, h = args.width, args.height
    K = np.array([[args.fx, 0.0, (w - 1) / 2 + 3.0], [0.0, args.fx, (h - 1) / 2 - 2.0],
                  [0.0, 0.0, 1.0]])
    if args.model == 'fisheye':
        D = np.array([0.02, -0.005, 0.001, 0.0])
        return CameraCalib('fisheye', 'equidistant', (w, h), K, D)
    return CameraCalib('pinhole', 'plumb_bob', (w, h), K, np.array([-0.05, 0.02, 0.0, 0.0, 0.0]))


def key_poses(board, cam, n: int, seed: int = 0, baseline=None) -> list:
    """Return n board poses of the left camera; stereo: the right view must also fit."""
    if baseline is None:
        return random_poses(board, cam, n, seed)
    w, h = cam.size
    m = 0.03 * min(w, h)
    t_r = np.array([-baseline, 0.0, 0.0])
    keep = []
    for rvec, tvec in random_poses(board, cam, 6 * n, seed):
        q = project_corners(board, cam, rvec, tvec + t_r)
        if np.all((q >= m) & (q <= np.array([w - 1, h - 1]) - m)):
            keep.append((rvec, tvec))
            if len(keep) == n:
                return keep
    raise RuntimeError(f'only {len(keep)} of {n} stereo poses fit both images')


def render_pool(board, cam, poses: list, transition: int, baseline=None, noise: float = 0.0,
                seed: int = 0) -> list:
    """
    Render the cyclic trajectory through poses: per pose one still frame then transition frames.

    Return a list of per-camera gray tuples: entry k * (transition + 1) is pose k.
    """
    offsets = [np.zeros(3)] if baseline is None else [np.zeros(3), np.array([-baseline, 0, 0])]
    pool = []
    for k, (r0, t0) in enumerate(poses):
        r1, t1 = poses[(k + 1) % len(poses)]
        for j in range(transition + 1):
            a = j / (transition + 1)
            r, t = (1 - a) * r0 + a * r1, (1 - a) * t0 + a * t1
            pool.append(tuple(render_view(board, cam, r, t + o, noise_sigma=noise,
                                          seed=seed + len(pool)) for o in offsets))
    return pool


def schedule(n_poses: int, transition: int, dwell_frames: int) -> list:
    """Return the pool indices of one trajectory cycle (each still frame repeated)."""
    out = []
    for k in range(n_poses):
        base = k * (transition + 1)
        out += [base] * max(1, dwell_frames) + [base + j for j in range(1, transition + 1)]
    return out


class SyntheticCamera(NodeBase):
    """Publishes the rendered trajectory and stores CameraInfo received by SetCameraInfo."""

    def __init__(self, args: argparse.Namespace, node_name: str = 'synthetic_camera'):
        """Render the frame pool, then create publishers, services and the frame timer."""
        super().__init__(node_name, True)
        board, cam = board_of(args), camera_of(args)
        self.board, self.camera = board, cam
        self.baseline = args.baseline if args.mode == 'stereo' else None
        poses = key_poses(board, cam, args.poses, args.seed, self.baseline)
        self.get_logger().info(f'Rendering {len(poses) * (args.transition + 1)} frames...')
        pool = render_pool(board, cam, poses, args.transition, self.baseline, args.noise,
                           args.seed)
        self._schedule = schedule(len(poses), args.transition, round(args.dwell * args.rate))
        self._k = 0
        sides = [''] if self.baseline is None else ['/left', '/right']
        fmt = 'jpeg' if args.jpeg else 'png'
        self.received = {}          # service name -> last CameraInfo received
        self._outputs = []          # per side: [(publisher, [msg per pool entry])]
        for i, side in enumerate(sides):
            prefix = args.ns.rstrip('/') + side
            frame_id = 'synthetic' + side.replace('/', '_')
            outs = []
            if args.transport in ('raw', 'both'):
                outs.append((self.dua_create_publisher(Image, prefix + '/image_raw', 5),
                             [make_image_msg(g[i], 0, frame_id) for g in pool]))
            if args.transport in ('compressed', 'both'):
                outs.append((self.dua_create_publisher(CompressedImage,
                                                       prefix + '/image_raw/compressed', 5),
                             [make_compressed_msg(g[i], 0, fmt, frame_id) for g in pool]))
            self._outputs.append(outs)
            name = prefix + '/set_camera_info'
            self.dua_create_service_server(
                SetCameraInfo, name, lambda req, res, n=name: self._set_camera_info(n, req, res))
        self.dua_create_timer('frames', 1000.0 / args.rate, self._publish)
        self.get_logger().info(f'Publishing {args.mode} {args.transport} on {args.ns}')

    def _publish(self) -> None:
        """Publish the next frame of the trajectory (identical stamps on both sides)."""
        idx = self._schedule[self._k % len(self._schedule)]
        self._k += 1
        stamp = self.get_clock().now().to_msg()
        for outs in self._outputs:
            for pub, msgs in outs:
                msg = msgs[idx]
                msg.header.stamp = stamp
                pub.publish(msg)

    def _set_camera_info(self, name, request, response):
        """Store and log a received CameraInfo."""
        info = request.camera_info
        self.received[name] = info
        self.get_logger().info(f'{name}: {info.width}x{info.height} {info.distortion_model} '
                               f'K={list(info.k)} D={list(info.d)}')
        response.success, response.status_message = True, 'stored'
        return response


def main(argv=None) -> int:
    """Run the synthetic camera."""
    rclpy.init(args=argv)
    node = None
    try:
        args = parse_args(remove_ros_args(argv if argv is not None else sys.argv)[1:])
        node = SyntheticCamera(args)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
