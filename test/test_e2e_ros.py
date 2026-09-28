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

# conftest sets ROS_DOMAIN_ID and ROS_AUTOMATIC_DISCOVERY_RANGE before rclpy is initialized.

import os
import threading
import time

from conftest import CAM
from dua_camera_calibration import calibration, fileio, rectification
from dua_camera_calibration.pipeline import camera_files, Session, write_run
from dua_camera_calibration.ros_io import CalibrationNode, camera_info
from dua_camera_calibration.settings import Settings
from dua_camera_calibration.synthetic_camera import parse_args, SyntheticCamera
import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor


def wait_for(cond, timeout):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.05)
    return True


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


@pytest.mark.parametrize('mode,transport', [('mono', 'raw'), ('stereo', 'compressed')])
def test_capture_calibrate_commit(ros, tmp_path, mode, transport):
    synth = SyntheticCamera(parse_args(['--mode', mode, '--transport', 'both', '--rate', '20',
                                        '--dwell', '0.5', '--poses', '16', '--transition', '1']),
                            node_name=f'synthetic_{mode}')
    node = CalibrationNode()
    executor = SingleThreadedExecutor()
    executor.add_node(synth)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    sides = ('',) if mode == 'mono' else ('/left', '/right')
    bases = tuple(f'/synthetic{s}/image_raw' for s in sides)
    services = tuple(f'/synthetic{s}/set_camera_info' for s in sides)
    session = None
    try:
        assert wait_for(lambda: all(b in node.discover_topics()['raw']
                                    and b in node.discover_topics()['compressed']
                                    for b in bases), 10.0)
        assert wait_for(lambda: set(services) <= set(node.discover_services()), 10.0)
        mapping = dict(node.params(), **{'source.mode': mode, 'source.transport': transport})
        if mode == 'mono':
            mapping['source.topic'] = bases[0]
        else:
            mapping['source.left_topic'], mapping['source.right_topic'] = bases
        settings, warnings = Settings.from_mapping(mapping)
        assert warnings == []
        session = Session(settings)
        session.set_capture('auto', False)
        node.set_source(settings, session.next_generation(), session.push)
        assert wait_for(lambda: node.publisher_counts(settings) == (1,) * len(bases), 5.0)
        if mode == 'mono':
            assert wait_for(lambda: len(session.db) >= 8, 60.0), session.stats()
        else:
            assert wait_for(lambda: len(session.db.snapshot().pairs()) >= 8, 60.0), \
                session.stats()
        node.clear_source()
        session.stop()
        assert session.errors == [] and session.stats()['rx_hz'] > 0

        snap = session.db.snapshot()
        result = calibration.calibrate(snap, settings.board(), settings.calib_config())
        rect = rectification.rectify(result.cameras, result.stereo, settings.rectify_config())
        for cam in result.cameras:
            assert np.allclose(cam.K, CAM.K, rtol=0.02, atol=2.0), cam.K
        path = write_run(str(tmp_path), settings, result, rect, snap, settings.board())
        saved = [fileio.read_camera_yaml(os.path.join(path, f))
                 for f, _ in camera_files(settings, len(bases))]
        infos = tuple(camera_info(cam, R, P) for _, cam, R, P in saved)

        assert wait_for(lambda: all(node.service_ready(s) for s in services), 5.0)
        futures = node.commit(services, infos)
        assert wait_for(lambda: all(f.done() for f in futures), 5.0)
        assert all(f.result().success for f in futures)
        for name, (_, cam, R, P) in zip(services, saved):
            got = synth.received[name]
            assert (got.width, got.height) == cam.size
            assert got.distortion_model == cam.distortion_model
            assert list(got.k) == cam.K.ravel().tolist() and list(got.d) == cam.D.tolist()
            assert list(got.r) == R.ravel().tolist() and list(got.p) == P.ravel().tolist()
        node.cancel(futures)                          # harmless on completed futures
    finally:
        if session is not None:
            session.stop()
        executor.shutdown(timeout_sec=1.0)
        spin.join(2.0)
        node.destroy_node()
        synth.destroy_node()
    assert not spin.is_alive()
