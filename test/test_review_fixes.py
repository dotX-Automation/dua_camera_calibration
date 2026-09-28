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

import subprocess
import sys
import time

from conftest import make_settings
from dua_camera_calibration import guidance
from dua_camera_calibration.gui import pick_service
from dua_camera_calibration.pipeline import Frame, make_display, Session, ViewConfig
from dua_camera_calibration.settings import board_error, Settings
import numpy as np
from test_guidance import BOARD, det


def test_board_error_and_fallback():
    """Invalid ChArUco combinations are reported and replaced when building settings."""
    ok = make_settings(board__type='charuco').as_dict()
    assert board_error(ok) is None
    assert 'smaller' in board_error({**ok, 'board.marker_size': 0.03})
    assert 'Unknown' in board_error({**ok, 'board.dictionary': 'DICT_NOPE'})
    assert 'needs' in board_error({**ok, 'board.dictionary': 'DICT_4X4_50',
                                   'board.cols': 20, 'board.rows': 20})
    assert board_error({**ok, 'board.type': 'chessboard', 'board.marker_size': 1.0}) is None
    s, warnings = Settings.from_mapping({'board.type': 'charuco', 'board.marker_size': 0.03})
    assert s['board.marker_size'] < s['board.square_size'] and board_error(s) is None
    assert [k for k, _ in warnings] == ['board.marker_size']
    s, warnings = Settings.from_mapping({'board.type': 'charuco', 'board.square_size': 0.01})
    assert s['board.type'] == 'chessboard' and ('board.type', 'charuco') in warnings


def test_worker_backs_off_on_persistent_error():
    """A detector that cannot be built must not spin the detector thread."""
    bad = make_settings().updated({'board.type': 'charuco', 'board.marker_size': 0.03})
    s = Session(bad)
    try:
        s.slot_detect.put(Frame(s.next_generation(), 1, 0.0, 0,
                                (np.zeros((48, 64), np.uint8),), ('mono8',), False))
        t0 = time.process_time()
        time.sleep(1.0)
        cpu = time.process_time() - t0
    finally:
        s.stop()
    assert s.errors_version < 50, s.errors_version
    assert cpu < 0.5, cpu


def test_jobrunner_does_not_block_exit():
    """A job still running at quit must not keep the interpreter alive."""
    code = ('import time\nfrom dua_camera_calibration.pipeline import JobRunner\n'
            'j = JobRunner()\nj.submit("calibrate", time.sleep, 30.0)\ntime.sleep(0.1)\n'
            'j.shutdown()\n')
    t0 = time.monotonic()
    subprocess.run([sys.executable, '-c', code], check=True, timeout=20)
    assert time.monotonic() - t0 < 10


def test_pick_service_needs_common_prefix():
    """No shared path segment means no default service."""
    services = ['/driver/left/set_camera_info', '/other/set_camera_info']
    assert pick_service('/cam/image_raw', services) == ''
    assert pick_service('/driver/left/image_raw', services) == '/driver/left/set_camera_info'


def test_display_frame_has_source_sizes():
    """make_display reports the exact source size even when downscaled."""
    g = np.zeros((481, 641), np.uint8)
    d = make_display(Frame(1, 1, 0.0, 0, (g,), ('mono8',), False), ViewConfig(((333, 250),)))
    assert d.src_sizes == ((641, 481),) and d.images[0].shape[1] < 641


def test_fisheye_tilt_bootstrap_uses_model():
    """Before an estimate the tilt gate uses the configured camera model."""
    view = det(u=0.8, v=0.3, rx=25.0, ry=-20.0)
    size = view.image_size
    tilts = {}
    for model in ('pinhole', 'fisheye'):
        v = guidance.evaluate_frame((view,), (size,), (view,), None, None, None,
                                    guidance.GateConfig(model=model), BOARD)
        tilts[model] = v.metrics[0]['tilt']
        want = guidance._tilt(guidance._features(view, size, None, model))
        assert abs(tilts[model] - want) < 1e-9
    assert abs(tilts['pinhole'] - tilts['fisheye']) > 1.0
    assert make_settings(camera__model='fisheye').gate_config().model == 'fisheye'
