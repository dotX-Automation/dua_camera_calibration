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

import os

# Isolate ROS tests from the host graph; read by rclpy.init, so set before any test runs.
os.environ['ROS_DOMAIN_ID'] = str(80 + os.getpid() % 20)
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

from dua_camera_calibration.settings import Settings  # noqa: E402
from dua_camera_calibration.synthetic import render_view  # noqa: E402
from dua_camera_calibration.synthetic_camera import (board_of, camera_of, key_poses,  # noqa: E402
                                                     parse_args)
import pytest  # noqa: E402

ARGS = parse_args([])
BOARD = board_of(ARGS)        # spec defaults: 8 x 6 inner corners, 25 mm
CAM = camera_of(ARGS)         # 640 x 480, fx = fy = 500, mild plumb_bob distortion
BASELINE = 0.1
N_VIEWS = 14


def make_settings(**changes) -> Settings:
    """Return spec-default settings with 'group__key' changes."""
    return Settings.from_mapping({k.replace('__', '.'): v for k, v in changes.items()})[0]


@pytest.fixture(scope='session')
def mono_views():
    """Render mono views (uint8 gray) of distinct poses."""
    return [render_view(BOARD, CAM, r, t, noise_sigma=1.0, seed=i)
            for i, (r, t) in enumerate(key_poses(BOARD, CAM, N_VIEWS, seed=1))]


@pytest.fixture(scope='session')
def stereo_views():
    """Render (left, right) views, the right camera BASELINE m to the right of the left one."""
    out = []
    for i, (r, t) in enumerate(key_poses(BOARD, CAM, N_VIEWS, seed=2, baseline=BASELINE)):
        tr = t + [-BASELINE, 0.0, 0.0]
        out.append((render_view(BOARD, CAM, r, t, noise_sigma=1.0, seed=i),
                    render_view(BOARD, CAM, r, tr, noise_sigma=1.0, seed=100 + i)))
    return out
