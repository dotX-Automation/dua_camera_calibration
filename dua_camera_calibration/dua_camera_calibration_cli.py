"""
Offline camera calibration from image folders, a saved dataset or a rosbag2.

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
from collections import Counter, deque
import glob
import os
import sys
import traceback

import cv2
from dua_camera_calibration import calibration, fileio, rectification
from dua_camera_calibration.detection import require_opencv
from dua_camera_calibration.pipeline import (convert_raw, Frame, RawFrame, Session, stamp_ns,
                                             write_run)
from dua_camera_calibration.rectification import POLICIES
from dua_camera_calibration.settings import load_spec, Settings
import numpy as np
import yaml

EXIT_OK, EXIT_ERROR, EXIT_ARGS = 0, 1, 2
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
IMAGE_TYPES = ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage')
GUIDANCE_EVERY = 5   # ponytail: live estimate refreshed every 5 samples; 1 if tilt gating lags


class ArgError(Exception):
    """Invalid command-line input (exit code 2)."""


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    p = argparse.ArgumentParser(
        prog='dua_camera_calibration_cli', description=__doc__.strip().split('\n')[0],
        epilog='Settings: spec defaults <- dataset settings <- --params-file <- --set <- '
               '--policy/--alpha. Example: --set board.cols=9 --set board.square_size=0.03')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--images', metavar='DIR', help='image folder (left folder for stereo)')
    src.add_argument('--dataset', metavar='DIR', help='dataset folder saved by a previous run')
    src.add_argument('--bag', metavar='PATH', help='rosbag2 folder or file')
    p.add_argument('--right-images', metavar='DIR', help='right image folder (stereo)')
    p.add_argument('--topic', help='bag image topic (left for stereo; base or /compressed)')
    p.add_argument('--right-topic', help='bag right image topic (stereo)')
    p.add_argument('--slop', type=float, default=None,
                   help='stereo stamp tolerance [s] (default: source.sync_slop)')
    p.add_argument('--params-file', metavar='YAML', help='ROS 2 parameters file of the node')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='override a setting (repeatable)')
    p.add_argument('--policy', choices=POLICIES, help='mono projection matrix policy')
    p.add_argument('--alpha', type=float, help='rectification alpha in [0, 1]')
    p.add_argument('--out', metavar='DIR', help='output base directory (default: output.dir)')
    return p


def flatten(d: dict, prefix: str = '') -> dict:
    """Flatten nested mappings into dotted keys."""
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def load_params_file(path: str) -> dict:
    """Return the flat parameters of a ROS 2 parameters file (first node with ros__parameters)."""
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    for node in doc.values():
        if isinstance(node, dict) and 'ros__parameters' in node:
            return flatten(node['ros__parameters'])
    raise ArgError(f'{path}: no <node>: ros__parameters: section')


def parse_sets(items, spec: dict) -> dict:
    """Return {key: value string} of the --set KEY=VALUE items."""
    out = {}
    for item in items:
        key, sep, value = item.partition('=')
        if not sep or key not in spec:
            raise ArgError(f'--set {item!r}: expected KEY=VALUE with KEY among the settings')
        out[key] = value
    return out


def image_files(dirpath: str) -> list:
    """Return the sorted image files of a folder."""
    if not os.path.isdir(dirpath):
        raise ArgError(f'{dirpath}: not a directory')
    files = sorted(f for f in glob.glob(os.path.join(dirpath, '*'))
                   if f.lower().endswith(IMAGE_EXTENSIONS))
    if not files:
        raise ArgError(f'{dirpath}: no images')
    return files


def image_frames(left: str, right=None):
    """Yield Frames from one folder (mono) or two folders paired by sorted order (stereo)."""
    sides = [image_files(left)] + ([image_files(right)] if right else [])
    if len({len(s) for s in sides}) != 1:
        raise ArgError(f'left and right folders differ in size: {[len(s) for s in sides]}')
    for i, paths in enumerate(zip(*sides)):
        grays = tuple(cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths)
        for p, g in zip(paths, grays):
            if g is None:
                raise RuntimeError(f'{p}: cannot read image')
        lossy = any(p.lower().endswith(('.jpg', '.jpeg')) for p in paths)
        yield Frame(0, i, 0.0, i * 100000000, grays, ('mono8',) * len(grays), lossy)


def pair_by_stamp(stream, slop_ns: int):
    """Yield (left, right) messages of a (side, msg) stream with stamps within slop_ns."""
    pending = (deque(maxlen=30), deque(maxlen=30))
    for side, msg in stream:
        t, other = stamp_ns(msg), pending[1 - side]
        best = min(other, key=lambda m: abs(stamp_ns(m) - t), default=None)
        if best is None or abs(stamp_ns(best) - t) > slop_ns:
            pending[side].append(msg)
            continue
        while other.popleft() is not best:
            pass
        pending[side].clear()   # older messages of this side can only pair out of order
        yield (msg, best) if side == 0 else (best, msg)


def bag_frames(path: str, topics: tuple, slop_s: float):
    """Yield Frames from the image topics of a rosbag2 (one topic mono, two stereo)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=''),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    names = []
    for t in topics:
        name = next((n for n in (t, t.rstrip('/') + '/compressed') if n in types), None)
        if name is None or types[name] not in IMAGE_TYPES:
            images = sorted(n for n, ty in types.items() if ty in IMAGE_TYPES)
            raise ArgError(f'{t}: no image topic of that name in the bag; available: {images}')
        names.append(name)
    reader.set_filter(rosbag2_py.StorageFilter(topics=names))
    classes = {n: get_message(types[n]) for n in names}

    def stream():
        while reader.has_next():
            topic, data, _ = reader.read_next()
            yield names.index(topic), deserialize_message(data, classes[topic])

    msgs = ((m,) for _, m in stream()) if len(names) == 1 else \
        pair_by_stamp(stream(), int(slop_s * 1e9))
    for i, m in enumerate(msgs):
        yield convert_raw(RawFrame(0, 0.0, stamp_ns(m[0]), tuple(m)), i)


def _settings(args, spec: dict, base: dict) -> Settings:
    """Merge the settings sources and print the invalid values replaced by defaults."""
    mapping = dict(base)
    if args.params_file:
        mapping.update(load_params_file(args.params_file))
    mapping.update(parse_sets(args.set, spec))
    if args.policy:
        mapping['rectify.policy'] = args.policy
    if args.alpha is not None:
        mapping['rectify.alpha'] = args.alpha
    stereo = args.right_images or args.right_topic or base.get('source.mode') == 'stereo'
    mapping['source.mode'] = 'stereo' if stereo else 'mono'
    settings, warnings = Settings.from_mapping(mapping, spec)
    for key, value in warnings:
        print(f'warning: invalid {key}={value!r}, using {settings[key]!r}', file=sys.stderr)
    return settings


def _collect(settings, frames, check_motion: bool) -> tuple:
    """Run the frames through a non-threaded Session; return (session, reason counts)."""
    session = Session(settings, threaded=False, check_motion=check_motion)
    session.set_capture('auto', False)
    reasons = Counter()
    for frame in frames:
        ov = session.process_frame(frame)
        reasons[ov.verdict.reason] += 1
        if ov.verdict.reason == 'ACCEPTED' and len(session.db) % GUIDANCE_EVERY == 0:
            session.refresh_guidance()
    return session, reasons


def _summary(result, rect, snapshot, reasons, path) -> None:
    """Print the calibration summary."""
    fmt = {'float_kind': lambda x: f'{x:.6g}'}
    if reasons:
        print(f'frames: {sum(reasons.values())}  ' + '  '.join(
            f'{k}: {v}' for k, v in reasons.most_common()))
    print(f'samples: {len(snapshot.samples)}  rejected: {len(result.rejected)}  '
          f'RMS: {result.rms:.4f} px  ({result.duration_s:.2f} s)')
    for i, cam in enumerate(result.cameras):
        print(f'camera {i}: {cam.model}/{cam.distortion_model} {cam.size[0]}x{cam.size[1]} '
              f'used {len(result.used_ids[i])}  RMS {cam.rms:.4f} px')
        print('  K =', np.array2string(np.asarray(cam.K), formatter=fmt).replace('\n', '\n      '))
        print('  D =', np.array2string(np.ravel(cam.D), formatter=fmt))
        print('  P =', np.array2string(np.asarray(rect.P[i]), formatter=fmt)
              .replace('\n', '\n      '))
    if result.stereo is not None:
        print(f'baseline: {np.linalg.norm(result.stereo.T):.6f} m  '
              f'stereo RMS: {result.stereo.rms:.4f} px')
    for code, params in tuple(result.warnings) + tuple(rect.warnings):
        print(f'warning: {code} {params}')
    print(f'output: {path}')


def run(args) -> int:
    """Calibrate from the parsed arguments and return the exit code."""
    require_opencv((4, 8))
    spec = load_spec()
    if args.images is None and args.right_images:
        raise ArgError('--right-images requires --images')
    if args.bag is not None and not args.topic:
        raise ArgError('--bag requires --topic')
    if args.bag is None and (args.topic or args.right_topic):
        raise ArgError('--topic/--right-topic require --bag')
    reasons = Counter()
    if args.dataset:
        d = args.dataset
        if not os.path.exists(os.path.join(d, 'session.yaml')) and \
                os.path.exists(os.path.join(d, 'dataset', 'session.yaml')):
            d = os.path.join(d, 'dataset')   # a run directory was given
        if not os.path.exists(os.path.join(d, 'session.yaml')):
            raise ArgError(f'{args.dataset}: no session.yaml')
        snapshot, board, meta = fileio.read_dataset(d)
        base = dict(meta.get('settings') or {})
        base['source.mode'] = 'stereo' if snapshot.n_cameras == 2 else 'mono'
        settings = _settings(args, spec, base)
    else:
        settings = _settings(args, spec, {})
        board = settings.board()
        if args.images:
            frames, check_motion = image_frames(args.images, args.right_images), False
        else:
            topics = (args.topic,) + ((args.right_topic,) if args.right_topic else ())
            slop = args.slop if args.slop is not None else settings['source.sync_slop']
            frames, check_motion = bag_frames(args.bag, topics, slop), True
        session, reasons = _collect(settings, frames, check_motion)
        snapshot = session.db.snapshot()
    try:
        result = calibration.calibrate(snapshot, board, settings.calib_config())
    except calibration.CalibrationError as e:
        print(f'error: calibration failed: {e.code} {e.params} '
              f'({len(snapshot.samples)} samples; {dict(reasons)})', file=sys.stderr)
        return EXIT_ERROR
    rect = rectification.rectify(result.cameras, result.stereo, settings.rectify_config())
    path = write_run(args.out or settings.output_dir(), settings, result, rect, snapshot, board,
                     dataset=bool(settings['capture.keep_images']) and not args.dataset)
    _summary(result, rect, snapshot, reasons, path)
    return EXIT_OK


def main(argv=None) -> int:
    """Run the offline calibration; exit codes: 0 ok, 1 error, 2 bad arguments."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:        # argparse: --help (0) or bad arguments (2)
        return int(e.code or 0)
    try:
        return run(args)
    except ArgError as e:
        print(f'error: {e}', file=sys.stderr)
        return EXIT_ARGS
    except Exception as e:
        print(f'error: {e}', file=sys.stderr)
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == '__main__':
    sys.exit(main())
