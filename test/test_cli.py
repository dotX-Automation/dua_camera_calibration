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

import glob
import os

from conftest import BASELINE, CAM
import cv2
from dua_camera_calibration import fileio
from dua_camera_calibration.dua_camera_calibration_cli import main, pair_by_stamp
from dua_camera_calibration.synthetic_camera import make_compressed_msg, make_image_msg
import numpy as np
from rclpy.serialization import serialize_message
import rosbag2_py
import yaml


def write_images(dirpath, images):
    os.makedirs(dirpath)
    for i, g in enumerate(images):
        cv2.imwrite(os.path.join(dirpath, f'{i:03d}.png'), g)
    return str(dirpath)


def run_dir(out):
    (path,) = glob.glob(os.path.join(str(out), '*_*'))
    return path


def check_camera(path, fname, cname):
    name, cam, _, P = fileio.read_camera_yaml(os.path.join(path, fname))
    assert name == cname
    assert np.allclose(cam.K, CAM.K, rtol=0.01, atol=1.0), cam.K
    assert abs(P[0, 0] - P[1, 1]) < 1e-9          # default policy: square pixels
    return cam


def write_bag(path, streams, repeat=4):
    """Write {topic: [msgs of distinct poses]}: each pose repeated (still) at 10 Hz."""
    w = rosbag2_py.SequentialWriter()
    storage = rosbag2_py.get_default_storage_id()
    w.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=storage),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    for i, (topic, msgs) in enumerate(streams.items()):
        w.create_topic(rosbag2_py.TopicMetadata(i, topic, 'sensor_msgs/msg/' + type(
            msgs[0]).__name__, 'cdr'))
    k = 0
    for j in range(len(next(iter(streams.values())))):
        for _ in range(repeat):
            t = 1000000000 + k * 100000000
            for side, (topic, msgs) in enumerate(streams.items()):
                m = msgs[j]
                ts = t + side * 1000000                 # right side 1 ms late (within slop)
                m.header.stamp.sec, m.header.stamp.nanosec = divmod(ts, 1000000000)
                w.write(topic, serialize_message(m), ts)
            k += 1
    del w
    return str(path)


def test_images_mono_and_dataset(mono_views, tmp_path, capsys):
    d = write_images(tmp_path / 'img', mono_views)
    out = tmp_path / 'out'
    assert main(['--images', d, '--out', str(out), '--set', 'camera.name=mycam']) == 0
    path = run_dir(out)
    check_camera(path, 'mycam.yaml', 'mycam')
    assert 'RMS' in capsys.readouterr().out
    # the saved dataset (run directory given) calibrates again, with a CLI policy override
    out2 = tmp_path / 'out2'
    assert main(['--dataset', path, '--out', str(out2), '--policy', 'k']) == 0
    path2 = run_dir(out2)
    _, cam, _, P = fileio.read_camera_yaml(os.path.join(path2, 'mycam.yaml'))
    assert np.allclose(P[:, :3], cam.K) and not os.path.exists(os.path.join(path2, 'dataset'))


def test_images_stereo(stereo_views, tmp_path):
    left = write_images(tmp_path / 'l', [v[0] for v in stereo_views])
    right = write_images(tmp_path / 'r', [v[1] for v in stereo_views])
    out = tmp_path / 'out'
    assert main(['--images', left, '--right-images', right, '--out', str(out)]) == 0
    path = run_dir(out)
    check_camera(path, 'left.yaml', 'camera_left')
    check_camera(path, 'right.yaml', 'camera_right')
    with open(os.path.join(path, 'report.yaml')) as f:
        report = yaml.safe_load(f)
    assert abs(report['stereo']['baseline'] - BASELINE) < 0.002
    assert report['camera_files'] == {'left.yaml': 'camera_left', 'right.yaml': 'camera_right'}


def test_bag_mono_raw(mono_views, tmp_path):
    bag = write_bag(tmp_path / 'bag', {'/cam/image_raw': [make_image_msg(v) for v in mono_views]})
    out = tmp_path / 'out'
    params = tmp_path / 'p.yaml'
    params.write_text('/dua_camera_calibration:\n  ros__parameters:\n'
                      '    camera:\n      name: bagcam\n')
    assert main(['--bag', bag, '--topic', '/cam/image_raw', '--params-file', str(params),
                 '--out', str(out)]) == 0
    check_camera(run_dir(out), 'bagcam.yaml', 'bagcam')


def test_bag_stereo_compressed(stereo_views, tmp_path):
    streams = {'/s/left/image_raw/compressed': [make_compressed_msg(v[0]) for v in stereo_views],
               '/s/right/image_raw/compressed': [make_compressed_msg(v[1]) for v in stereo_views]}
    bag = write_bag(tmp_path / 'bag', streams)
    out = tmp_path / 'out'
    assert main(['--bag', bag, '--topic', '/s/left/image_raw', '--right-topic',
                 '/s/right/image_raw/compressed', '--slop', '0.01', '--out', str(out)]) == 0
    path = run_dir(out)
    check_camera(path, 'left.yaml', 'camera_left')
    check_camera(path, 'right.yaml', 'camera_right')


def test_pair_by_stamp():
    class M:
        def __init__(self, t):
            self.header = type('H', (), {})()
            self.header.stamp = type('S', (), {'sec': 0, 'nanosec': t})()

    L, R = [M(t) for t in (0, 100, 200, 300)], [M(t) for t in (5, 210, 290)]
    stream = [(0, L[0]), (1, R[0]), (0, L[1]), (0, L[2]), (1, R[1]), (0, L[3]), (1, R[2])]
    assert list(pair_by_stamp(stream, 20)) == [(L[0], R[0]), (L[2], R[1]), (L[3], R[2])]


def test_bad_arguments(tmp_path):
    assert main([]) == 2
    assert main(['--bag', str(tmp_path)]) == 2                       # --topic missing
    assert main(['--images', str(tmp_path / 'missing')]) == 2
    assert main(['--images', str(tmp_path), '--set', 'no.such=1']) == 2
    assert main(['--dataset', str(tmp_path)]) == 2
    assert main(['--images', str(tmp_path), '--right-topic', '/x']) == 2
