"""
Calibration files: camera_calibration_parsers YAML, run report and dataset folders.

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

from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
import os
from typing import Optional

from dua_camera_calibration.calibration import CameraCalib
from dua_camera_calibration.detection import BoardSpec, Detection
from dua_camera_calibration.samples import DBSnapshot, Sample
import numpy as np
import yaml

_DUMPER = getattr(yaml, 'CSafeDumper', yaml.SafeDumper)
_LOADER = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
_WIDTH = 1 << 16                  # never wrap long flow lists
_SESSION = 'session.yaml'
_DATASET_VERSION = 1


def _plain(x):
    """Return x converted to the plain Python types yaml.safe_dump accepts (floats keep repr)."""
    if is_dataclass(x) and not isinstance(x, type):
        return _plain(asdict(x))
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {_plain(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _dump(path, doc, head='') -> None:
    """Write doc as block YAML with flow scalar lists, preceded by the head text."""
    with open(path, 'w') as f:
        f.write(head)
        yaml.dump(_plain(doc), f, Dumper=_DUMPER, sort_keys=False, default_flow_style=None,
                  width=_WIDTH)


def _load(path):
    """Return the parsed YAML file."""
    with open(path) as f:
        return yaml.load(f, Loader=_LOADER)


def _matrix(a) -> dict:
    """Return the camera_calibration_parsers matrix mapping of a (1-D: one row) array."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    return {'rows': a.shape[0], 'cols': a.shape[1], 'data': a.ravel().tolist()}


def write_camera_yaml(path, name, cam, R, P, comments: Optional[dict] = None) -> None:
    """
    Write a camera_calibration_parsers YAML file (exact keys, repr float precision).

    comments become leading '# key: value' lines, so parsers never see them.
    """
    head = ''
    for k, v in (comments or {}).items():
        head += '# {}: {}\n'.format(k, ' '.join(str(v).splitlines()))
    _dump(path, {'image_width': int(cam.size[0]), 'image_height': int(cam.size[1]),
                 'camera_name': str(name), 'camera_matrix': _matrix(cam.K),
                 'distortion_model': cam.distortion_model,
                 'distortion_coefficients': _matrix(np.ravel(cam.D)),
                 'rectification_matrix': _matrix(R), 'projection_matrix': _matrix(P)}, head)


def read_camera_yaml(path) -> tuple:
    """Read a camera_calibration_parsers YAML file and return (name, CameraCalib, R, P)."""
    d = _load(path)

    def mat(key):
        return np.array(d[key]['data'], np.float64).reshape(d[key]['rows'], d[key]['cols'])

    dm = d.get('distortion_model', 'plumb_bob')
    cam = CameraCalib('fisheye' if dm == 'equidistant' else 'pinhole', dm,
                      (int(d['image_width']), int(d['image_height'])), mat('camera_matrix'),
                      mat('distortion_coefficients').ravel())
    name = str(d.get('camera_name', ''))
    return name, cam, mat('rectification_matrix'), mat('projection_matrix')


def _camera_doc(cam) -> dict:
    """Return the report entry of a CameraCalib."""
    return {'model': cam.model, 'distortion_model': cam.distortion_model, 'size': cam.size,
            'K': cam.K, 'D': np.ravel(cam.D), 'sigmas': cam.sigmas, 'rms': cam.rms,
            'n_views': cam.n_views}


def write_report(path, result, rect, settings: dict) -> None:
    """Write report.yaml: calibration statistics, extrinsics, rectification and settings."""
    cams = []
    for i, cam in enumerate(result.cameras):
        cams.append({**_camera_doc(cam), 'used_ids': result.used_ids[i],
                     'per_view_rms': result.per_view_rms[i]})
    st = result.stereo
    stereo = None if st is None else {
        'R': st.R, 'T': np.ravel(st.T), 'baseline': float(np.linalg.norm(st.T)), 'rms': st.rms,
        'sigmas': st.sigmas, 'n_pairs': st.n_pairs}
    rectification = None if rect is None else {
        'policy': rect.policy, 'alpha': rect.alpha, 'R': rect.R, 'P': rect.P,
        'warnings': [{'code': c, 'params': p} for c, p in rect.warnings]}
    _dump(path, {'rms': result.rms, 'duration_s': result.duration_s, 'cameras': cams,
                 'rejected': result.rejected,
                 'warnings': [{'code': c, 'params': p} for c, p in result.warnings],
                 'stereo': stereo, 'rectification': rectification, 'settings': settings})


def _view_doc(v) -> Optional[dict]:
    """Return the session.yaml entry of a Detection (arrays flattened)."""
    if v is None:
        return None
    return {'corners': v.corners.ravel(), 'ids': v.ids, 'object_points': v.object_points.ravel(),
            'complete': bool(v.complete), 'outline': np.ravel(v.outline),
            'image_size': v.image_size, 'sharpness': v.sharpness, 'window': int(v.window)}


def _view(d) -> Optional[Detection]:
    """Return the Detection of a session.yaml entry."""
    if d is None:
        return None
    return Detection(corners=np.array(d['corners'], np.float32).reshape(-1, 2),
                     ids=np.array(d['ids'], np.int32),
                     object_points=np.array(d['object_points'], np.float32).reshape(-1, 3),
                     complete=bool(d['complete']),
                     outline=np.array(d['outline'], np.float32).reshape(4, 2),
                     image_size=tuple(d['image_size']), sharpness=d['sharpness'],
                     window=int(d['window']))


def _png_path(dirpath, cam, sample_id) -> str:
    """Return the image path of a sample view in a dataset folder."""
    return os.path.join(dirpath, f'cam{cam}', f'{sample_id:04d}.png')


def write_dataset(dirpath, snapshot, board, meta: dict) -> None:
    """Write a dataset folder: session.yaml plus camN/NNNN.png from the stored images."""
    os.makedirs(dirpath, exist_ok=True)
    samples = []
    for s in snapshot.samples:
        samples.append({'id': s.id, 'stamp_ns': s.stamp_ns,
                        'views': [_view_doc(v) for v in s.views], 'features': s.features})
        for i, png in enumerate(s.images_png or ()):
            if png is not None:
                path = _png_path(dirpath, i, s.id)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(png)
    _dump(os.path.join(dirpath, _SESSION),
          {'version': _DATASET_VERSION, 'board': board, 'meta': meta,
           'image_sizes': snapshot.image_sizes, 'samples': samples})


def read_dataset(dirpath) -> tuple:
    """Read a dataset folder and return (DBSnapshot, BoardSpec, meta)."""
    doc = _load(os.path.join(dirpath, _SESSION))
    names = {f.name for f in fields(BoardSpec)}
    board = BoardSpec(**{k: v for k, v in doc['board'].items() if k in names})
    samples = []
    for s in doc['samples']:
        views = tuple(_view(v) for v in s['views'])
        pngs = []
        for i in range(len(views)):
            path = _png_path(dirpath, i, s['id'])
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    pngs.append(f.read())
            else:
                pngs.append(None)
        feats = tuple(s.get('features') or (None,) * len(views))
        samples.append(Sample(s['id'], s['stamp_ns'], views, feats,
                              tuple(pngs) if any(p is not None for p in pngs) else None))
    sizes = tuple(tuple(sz) for sz in doc['image_sizes'])
    return DBSnapshot(1, tuple(samples), sizes), board, doc.get('meta') or {}


def run_dir(base, camera_name, now: Optional[datetime] = None) -> str:
    """Create and return the run directory '<base>/<camera_name>_<YYYYmmdd-HHMMSS>'."""
    stamp = (now or datetime.now()).strftime('%Y%m%d-%H%M%S')
    path = os.path.join(base, '{}_{}'.format(str(camera_name).replace(os.sep, '_'), stamp))
    os.makedirs(path, exist_ok=True)
    return path
