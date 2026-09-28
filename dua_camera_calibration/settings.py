"""
Application settings: parameter spec, defaults, validation and core config builders.

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
import os
from typing import Optional

import cv2
from dua_camera_calibration.calibration import CalibConfig
from dua_camera_calibration.detection import BoardSpec
from dua_camera_calibration.guidance import GateConfig, GuidanceConfig
from dua_camera_calibration.rectification import POLICIES, RectifyConfig
import yaml

SPEC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'dua_camera_calibration_params.yaml')

CHOICES = {
    'board.detector': ('classic', 'sb'),
    'board.type': ('chessboard', 'charuco'),
    'calib.distortion_model': ('plumb_bob', 'rational_polynomial'),
    'camera.model': ('pinhole', 'fisheye'),
    'capture.mode': ('auto', 'manual'),
    'rectify.policy': POLICIES,
    'source.mode': ('mono', 'stereo'),
    'source.transport': ('raw', 'compressed'),
}


def board_error(values) -> Optional[str]:
    """
    Return why a ChArUco board configuration cannot work, or None if it is fine.

    values maps keys to values (dict or Settings). Checks that the marker is smaller than the
    square and that the dictionary exists and holds enough markers for the board.
    """
    if values['board.type'] != 'charuco':
        return None
    if values['board.marker_size'] >= values['board.square_size']:
        return (f"ChArUco marker size ({values['board.marker_size']} m) must be smaller than "
                f"the square size ({values['board.square_size']} m).")
    name = values['board.dictionary']
    if not name.startswith('DICT_') or not hasattr(cv2.aruco, name):
        return f"Unknown ArUco dictionary '{name}': use a cv2.aruco DICT_* name."
    have = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)).bytesList.shape[0]
    need = values['board.cols'] * values['board.rows'] // 2
    if have < need:
        return (f'Dictionary {name} has {have} markers but a {values["board.cols"]}x'
                f'{values["board.rows"]} ChArUco board needs {need}: choose a larger one.')
    return None


def load_spec(path: str = SPEC_FILE) -> dict:
    """Load the parameter spec: key -> PManager entry (type, default_value, ranges, ...)."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)['params']


def _coerce(entry: dict, value):
    """Convert a value to the spec type, clamping numbers to their range."""
    t = entry['type']
    if t == 'bool':
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)
    if t in ('integer', 'double'):
        v = int(value) if t == 'integer' else float(value)
        lo, hi = entry.get('min_value'), entry.get('max_value')
        if lo is not None:
            v = max(v, type(v)(lo))
        if hi is not None:
            v = min(v, type(v)(hi))
        return v
    return str(value)


@dataclass(frozen=True)
class Settings:
    """Flat, immutable application settings keyed like the node parameters (e.g. 'board.cols')."""

    values: tuple   # sorted tuple of (key, value)

    def __getitem__(self, key: str):
        """Return the value of a key."""
        return dict(self.values)[key]

    def as_dict(self) -> dict:
        """Return the settings as a plain dict."""
        return dict(self.values)

    @staticmethod
    def from_mapping(mapping: dict, spec: dict = None) -> tuple:
        """
        Build settings from spec defaults overridden by mapping.

        Returns (Settings, warnings) where warnings lists (key, value) pairs that were invalid
        and replaced by their default.
        """
        spec = spec if spec is not None else load_spec()
        values, warnings = {}, []
        for key, entry in spec.items():
            value = mapping.get(key, entry['default_value'])
            try:
                value = _coerce(entry, value)
            except (TypeError, ValueError):
                warnings.append((key, value))
                value = entry['default_value']
            if key in CHOICES and value not in CHOICES[key]:
                warnings.append((key, value))
                value = entry['default_value']
            values[key] = value
        # invalid ChArUco combinations fall back to the defaults, then to a chessboard
        for key in ('board.marker_size', 'board.dictionary'):
            err = board_error(values)
            if err is not None and board_error({**values, key: spec[key]['default_value']}) != err:
                warnings.append((key, values[key]))
                values[key] = spec[key]['default_value']
        if board_error(values) is not None:
            warnings.append(('board.type', values['board.type']))
            values['board.type'] = 'chessboard'
        return Settings(tuple(sorted(values.items()))), warnings

    def replace(self, **changes) -> 'Settings':
        """Return a copy with some keys changed (double underscores stand for dots)."""
        d = self.as_dict()
        for k, v in changes.items():
            d[k.replace('__', '.')] = v
        return Settings(tuple(sorted(d.items())))

    def updated(self, changes: dict) -> 'Settings':
        """Return a copy with the given {key: value} changes."""
        d = self.as_dict()
        d.update(changes)
        return Settings(tuple(sorted(d.items())))

    @property
    def n_cameras(self) -> int:
        """Return 1 for mono, 2 for stereo."""
        return 2 if self['source.mode'] == 'stereo' else 1

    def board(self) -> BoardSpec:
        """Return the board specification."""
        return BoardSpec(
            type=self['board.type'], cols=self['board.cols'], rows=self['board.rows'],
            square_size=self['board.square_size'], marker_size=self['board.marker_size'],
            dictionary=self['board.dictionary'], legacy_pattern=self['board.legacy_pattern'],
            detector=self['board.detector'], max_pixels=self['detection.max_pixels'])

    def calib_config(self) -> CalibConfig:
        """Return the calibration options."""
        return CalibConfig(
            model=self['camera.model'], distortion_model=self['calib.distortion_model'],
            fix_k3=self['calib.fix_k3'], fix_aspect_ratio=self['calib.fix_aspect_ratio'],
            fix_principal_point=self['calib.fix_principal_point'],
            zero_tangent_dist=self['calib.zero_tangent_dist'], max_views=self['calib.max_views'])

    def gate_config(self, check_motion: bool = True) -> GateConfig:
        """Return the per-frame gate thresholds."""
        return GateConfig(
            max_blur_px=self['gates.max_blur_px'], max_motion_px=self['gates.max_motion_px'],
            max_tilt_deg=self['gates.max_tilt_deg'], min_novelty=self['gates.min_novelty'],
            check_motion=check_motion, model=self['camera.model'])

    def guidance_config(self) -> GuidanceConfig:
        """Return the guidance options."""
        return GuidanceConfig(
            min_samples=self['capture.min_samples'], max_samples=self['capture.max_samples'],
            target_rel_ci=self['guidance.target_rel_ci'])

    def rectify_config(self) -> RectifyConfig:
        """Return the rectification options."""
        return RectifyConfig(
            policy=self['rectify.policy'], alpha=self['rectify.alpha'],
            zero_disparity=self['rectify.zero_disparity'],
            fisheye_max_fov_deg=self['rectify.fisheye_max_fov_deg'])

    def output_dir(self) -> str:
        """Return the base output directory (default: <cwd>/calibrations)."""
        return self['output.dir'] or os.path.join(os.getcwd(), 'calibrations')

    def topics(self) -> tuple:
        """Return the subscribed topics, one per camera ('/compressed' appended if needed)."""
        bases = ((self['source.left_topic'], self['source.right_topic'])
                 if self.n_cameras == 2 else (self['source.topic'],))
        if self['source.transport'] == 'compressed':
            return tuple(b.rstrip('/') + '/compressed' for b in bases)
        return bases
