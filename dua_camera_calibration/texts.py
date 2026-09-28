"""
User-facing strings: indicators, frame verdicts, hints, warnings, errors, policies and help.

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

import string


class _Formatter(string.Formatter):
    """str.format that never raises: missing or None fields render as '?'."""

    def get_value(self, key, args, kwargs):
        """Return the named field, or None if missing."""
        return kwargs.get(key) if isinstance(key, str) else None

    def format_field(self, value, format_spec):
        """Format a field, falling back to str() when the spec does not apply."""
        if value is None:
            return '?'
        try:
            return format(value, format_spec)
        except (TypeError, ValueError):
            return str(value)


_FMT = _Formatter()


def fmt(template: str, params=None) -> str:
    """Format a template with a params dict; missing fields become '?'."""
    return _FMT.vformat(template, (), dict(params or {}))


CAMERA_NAMES = {1: ('',), 2: ('Left', 'Right')}


def camera_prefix(cam: int, n_cameras: int) -> str:
    """Return '' for mono, 'Left: ' / 'Right: ' for stereo."""
    return '' if n_cameras < 2 else CAMERA_NAMES[2][min(cam, 1)] + ': '


# Indicators (guidance.INDICATOR_IDS)

INDICATOR_LABELS = {
    'coverage': 'Image coverage',
    'tilt_x': 'Tilt up/down',
    'tilt_y': 'Tilt left/right',
    'distance': 'Distance',
    'uncertainty': 'Uncertainty',
}

INDICATOR_HELP = {
    'coverage': (
        'Share of the image where board corners were seen (press H for the heatmap: red cells '
        'are still empty). Lens distortion grows toward the edges and is strongest in the '
        'corners, so the model is only trustworthy where you showed the board. Fill every '
        'cell, corners first. Corners close to the edge are useful; only boards touching the '
        'edge are rejected.'),
    'tilt_x': (
        "Board rotated about the camera's horizontal axis (top edge away from or toward the "
        'camera), 15-45 degrees in both directions. Fronto-parallel views cannot tell a long '
        'focal length from a far board; tilted views break that ambiguity and pin down the '
        'principal point. One view is needed in each bin: top away, facing, top toward.'),
    'tilt_y': (
        "Board rotated about the camera's vertical axis (left or right edge away from the "
        'camera), 15-45 degrees in both directions, for the same reason as Tilt up/down. '
        'Bins: left away, facing, right away.'),
    'distance': (
        'Board size in the image, far to close. Close views (board wider than half the image) '
        'constrain distortion; far views (under a quarter of the image) constrain the focal '
        'length. Close, medium and far views are needed.'),
    'uncertainty': (
        '95% confidence interval of focal lengths and principal point, relative to their '
        'values, estimated from the current samples. It needs at least 5 samples. When it '
        'stops shrinking, more samples will not help.'),
}


def indicator_value(ind) -> str:
    """Return the value text of a guidance.Indicator."""
    p = ind.params
    if ind.id == 'coverage':
        pct = 100.0 * p.get('cells_hit', 0) / max(1, p.get('cells', 1))
        return fmt('{pct:.0f}% of cells · corners {c}/4', {'pct': pct, 'c': p.get('corners_hit')})
    if ind.id == 'uncertainty':
        if p.get('rel_ci') is None:
            return 'needs at least 5 samples'
        return fmt('±{v:.2f}% (target {t:.2f}%)',
                   {'v': 100 * p['rel_ci'], 't': 100 * p.get('target', 0.0)})
    return fmt('{n}/3 bins', {'n': sum(bool(b) for b in p.get('bins', ()))})


# Frame verdicts (guidance.REASON_CODES) and GUI-level banner messages

REASONS = {
    'OK': 'Board OK: new pose.',
    'ACCEPTED': 'Captured sample #{id}.',
    'NOT_DETECTED': (
        'Board not detected. The whole board must be visible, sharp and evenly lit. A '
        'chessboard is counted in INNER corners ({cols}×{rows}): check the board settings if '
        'this persists.'),
    'TOO_FEW_CORNERS': 'Only {value} corners detected (minimum {limit}): show more of the board.',
    'NEAR_BORDER': (
        'Board corners too close to the image edge ({value:.0f} px, minimum {limit:.0f} px): '
        'move the board slightly inward.'),
    'BLURRY': (
        'Image too blurry (edge width {value:.1f} px, max {limit:.1f} px): hold still, '
        'refocus, or add light for a shorter exposure.'),
    'MOVING': 'Board moving (max {limit:.1f} px per frame): hold it still for a moment.',
    'TOO_TILTED': (
        'Board tilted {value:.0f}° (max {limit:.0f}°): corners become unreliable, reduce the '
        'tilt.'),
    'TOO_SIMILAR': (
        'Too similar to a stored sample (distance {value:.2f}, min {limit:.2f}): change '
        'position, distance or tilt (see the hint).'),
    'LIMIT': 'Sample limit reached: calibrate, or remove samples to capture new ones.',
}

STATUS = {
    'WAITING': 'Waiting for images on {topic}…',
    'NO_PUBLISHER': (
        'No publisher on {topic}. Check the topic name and transport, then press Refresh.'),
    'NO_DATA': (
        '{n} publisher(s) on {topic} but no frames arrive. Large raw images over Wi-Fi are '
        "often lost with best-effort QoS: try the 'compressed' transport."),
    'DECODE_ERROR': 'Cannot convert frames: {error}.',
    'SETUP_OK': 'Board detected. Press Start capture when ready.',
    'PAUSED': 'Capture paused: press P to resume.',
    'MANUAL_READY': 'Board OK: press Space to capture.',
    'DETECTOR_STALLED': (
        'The detector has been busy for {s:.0f} s. Check the board settings or try the other '
        'detector.'),
    'EXECUTOR_STOPPED': 'ROS executor stopped: {error}. Restart the application.',
    'CALIBRATING': 'Calibrating {n} samples ({model})… {elapsed:.1f} s · {stage}',
    'REVIEW': 'Calibration done: check the results, choose the framing, then Save or Commit.',
    'VERIFY': 'Checking {file}: move the board around and watch the live check metrics.',
    'LOSSY_TRANSPORT': (
        'Frames arrive JPEG-compressed: compression artefacts can bias sub-pixel corners. '
        "Prefer raw, or set the publisher's compressed format to PNG."),
}

# Short tag drawn at the board centroid
REASON_TAGS = {
    'NEAR_BORDER': 'edge', 'BLURRY': 'blurry', 'MOVING': 'moving', 'TOO_TILTED': 'tilted',
    'TOO_SIMILAR': 'similar', 'TOO_FEW_CORNERS': 'few corners', 'LIMIT': 'limit',
}

IMAGE_ERRORS = {
    'UNSUPPORTED_ENCODING': "unsupported encoding '{encoding}'",
    'DECODE_FAILED': "cannot decode a '{format}' compressed image of {size} bytes",
    'BAD_SIZE': 'image buffer of {size} bytes does not match {width}×{height}, step {step}',
}


def reason_text(verdict, extra=None) -> str:
    """Return the banner text of a guidance.FrameVerdict (extra: board/topic fields)."""
    return fmt(REASONS.get(verdict.reason, verdict.reason), {**(extra or {}), **verdict.params})


# Hints (guidance.HINT_CODES). Dict values are variants selected by HINT_VARIANT_KEYS.

REGION_NAMES = {
    'top-left': 'top-left corner', 'top': 'top edge', 'top-right': 'top-right corner',
    'left': 'left edge', 'center': 'center', 'right': 'right edge',
    'bottom-left': 'bottom-left corner', 'bottom': 'bottom edge',
    'bottom-right': 'bottom-right corner',
}

_FACE = 'Hold the board roughly facing the camera (tilt below 15°).'

HINTS = {
    'HOLD_STILL': 'Hold the board still for about a second.',
    'IMPROVE_FOCUS_OR_LIGHT': (
        'Refocus, or add light so that the exposure gets shorter and the edges sharper.'),
    'MOVE_AWAY_FROM_EDGE': (
        'Move the board slightly toward the center: its corners touch the image edge.'),
    'SHOW_WHOLE_BOARD': 'Show the whole board, flat, sharp and evenly lit (no glare).',
    'TILT_LESS': 'Reduce the tilt: turn the board more toward the camera.',
    'COVER_REGION': {
        'corner': (
            'Push the board into the {name}, as close to the edges as possible without '
            'touching them: distortion is strongest there.'),
        'default': 'Move the board toward the {name} of the image (arrow): no data there yet.',
    },
    'TILT_UP_DOWN': {
        'up': 'Tilt the board about 30° with its top edge away from the camera.',
        'down': 'Tilt the board about 30° with its top edge toward the camera.',
        'default': _FACE,
    },
    'TILT_LEFT_RIGHT': {
        'left': 'Turn the board about 30° with its left edge away from the camera.',
        'right': 'Turn the board about 30° with its right edge away from the camera.',
        'default': _FACE,
    },
    'MOVE_CLOSER': {
        'close': 'Bring the board closer until it spans more than half of the image width.',
        'default': 'Bring the board to a medium distance (a quarter to half of the image).',
    },
    'MOVE_FARTHER': {
        'far': 'Move the board farther away (under a quarter of the image), keeping it sharp.',
        'default': 'Move the board to a medium distance (a quarter to half of the image).',
    },
    'KEEP_GOING': 'Keep adding varied views ({samples} samples so far).',
    'READY': 'Enough data: press Calibrate (C).',
}

HINT_VARIANT_KEYS = {'TILT_UP_DOWN': 'direction', 'TILT_LEFT_RIGHT': 'direction',
                     'MOVE_CLOSER': 'bin', 'MOVE_FARTHER': 'bin'}


def hint_text(hint) -> str:
    """Return the text of a guidance.Hint."""
    t = HINTS.get(hint.code, hint.code)
    p = dict(hint.params)
    if hint.code == 'COVER_REGION':
        region = p.get('region', 'center')
        p['name'] = REGION_NAMES.get(region, region)
        t = t['corner' if region.count('-') == 1 else 'default']
    elif isinstance(t, dict):
        t = t.get(p.get(HINT_VARIANT_KEYS[hint.code]), t['default'])
    if hint.code in ('KEEP_GOING', 'READY') and 'pairs' in p:
        t += fmt(' Stereo pairs: {pairs}.', p)
    return fmt(t, p)


# Calibration (calibration.WARNING_CODES, ERROR_CODES, REJECT_CODES, STAGES)
# and rectification (rectification.WARNING_CODES)

WARNINGS = {
    'FISHEYE_CHECK_COND_DISABLED': (
        'Fisheye solver: the condition check was disabled after repeated ill-conditioned '
        'views. The result may be poorly constrained: add sharper views spread over the image '
        'and avoid extreme tilts.'),
    'MODEL_NOT_INVERTIBLE': (
        'The distortion model folds back near the image corners (round trip '
        '{roundtrip_px:.2g} px), so P fell back to [K|0]: add views covering the corners or '
        'use a simpler model. Rectified image corners may be wrong.'),
}

ERRORS = {
    'TOO_FEW_VIEWS': (
        'Too few usable views ({n}, minimum {min}): capture more samples spread over the '
        'image.'),
    'TOO_FEW_PAIRS': (
        'Too few stereo pairs with enough common corners ({n}, minimum {min}): keep the board '
        'inside both views while capturing.'),
    'SOLVE_FAILED': (
        'The solver failed: {msg}. Remove bad views (check the per-view errors) or capture '
        'more varied samples.'),
}

REJECTS = {
    'OUTLIER': 'outlier',
    'ILL_CONDITIONED': 'ill-conditioned',
    'NOT_SELECTED': 'not used',
}

STAGES = {
    'selecting': 'selecting views',
    'solving': 'solving',
    'outliers': 'removing outliers',
    'sigmas': 'estimating uncertainty',
    'stereo': 'solving stereo extrinsics',
}


def warning_text(code: str, params=None) -> str:
    """Return the text of a calibration or rectification warning."""
    return fmt(WARNINGS.get(code, code), params)


def error_text(code: str, params=None) -> str:
    """Return the text of a calibration.CalibrationError."""
    return fmt(ERRORS.get(code, code + ' {msg}'), params)


# P policies (rectification.POLICIES plus the applied 'stereo' and the loaded 'file')

POLICY_NAMES = {
    'square': 'Square pixels (default)',
    'aspect': 'Preserve aspect',
    'opencv': 'OpenCV (legacy)',
    'k': 'P = [K|0]',
    'stereo': 'stereoRectify',
    'file': 'From file',
}

POLICY_TIPS = {
    'square': (
        "fx' = fy', uniform zoom, window centred on the valid region. alpha 0: every output "
        'pixel is valid (cropped); alpha 1: every source pixel is kept (black borders). '
        'Rectified squares stay square.'),
    'aspect': (
        'Uniform zoom that keeps the calibrated fx/fy ratio, same framing as Square pixels. '
        "Use it when consumers expect the sensor's pixel aspect."),
    'opencv': (
        'getOptimalNewCameraMatrix, as in the old camera_calibration: fx and fy are scaled '
        'independently to maximize the valid area, so shapes may be stretched by a few '
        'percent.'),
    'k': 'No zoom: keeps the original intrinsics. Parts of the output may be black or cropped. '
         'alpha is ignored.',
    'stereo': (
        'Stereo pairs always use stereoRectify: both cameras share the rectified focal length '
        'and rows are aligned. alpha 0 crops to valid pixels, 1 keeps every source pixel.'),
    'file': 'Keep the P stored in the loaded file.',
}

# Live validation (rectification.ValidationMetrics keys)

VALIDATION = {
    'reproj_rms_px': ('Reprojection RMS', 'px',
                      'Board fitted with a rigid pose through the calibrated model; values '
                      'near the calibration RMS mean the model explains this new view.'),
    'straightness_rms_px': ('Straightness RMS', 'px',
                            'Distance of undistorted corners from lines fitted to each board '
                            'row and column; residual distortion bends lines.'),
    'epipolar_rms_px': ('Epipolar RMS', 'px',
                        'Offset across the epipolar lines between matching rectified '
                        'corners; should be well below 1 px.'),
    'square_size_m': ('Square size', 'm',
                      'Triangulated square size; compare with the configured one.'),
    'square_size_err': ('Square size error', '%',
                        'Triangulated vs configured square size: checks baseline and scale.'),
}

# Other strings

SAMPLE_ROW = '#{id}  pos {x:.2f},{y:.2f} · size {size:.2f} · tilt {tx:+.0f}°/{ty:+.0f}°'
RESULT_HEADER = '{used} of {total} samples used · RMS {rms:.3f} px · {duration:.2f} s'
COMMIT_OK = 'Committed to {service}: {msg}'
COMMIT_FAILED = 'Commit to {service} failed: {msg}'
COMMIT_TIMEOUT = 'Commit to {service} timed out after {t:.0f} s'
COMMIT_NO_SERVER = ('No SetCameraInfo server at {service} (is the driver running with '
                    'camera_info_manager?)')
CONFIRM_FORCE = ('Not all indicators are full ({missing}). The result may be poorly '
                 'constrained. Calibrate anyway?')
CONFIRM_CLEAR = 'Discard all {n} samples?'
CONFIRM_QUIT = 'The calibration has not been saved. Quit anyway?'
CONFIRM_DISCARD = 'The calibration has not been saved. Discard it?'
CONFIRM_RESET = ('Discard all {n} samples and the current calibration, and restart from '
                 'the setup?')
CONFIRM_TMP = '{path} is under /tmp, which is lost when the container is recreated. Save anyway?'
SIZE_MISMATCH = 'The calibration is for {cw}×{ch} but the images are {w}×{h}.'
SAVED = 'Saved to {path}'

HELP_HTML = """
<h3>How to calibrate</h3>
<ol>
<li><b>Setup:</b> choose topic and transport; enter the board exactly as printed (chessboard:
<i>inner</i> corners; ChArUco: squares, marker size, dictionary). Measure the square size on
the print. The outline turns green when the board is detected. Lock focus, zoom and
aperture.</li>
<li><b>Capture:</b> press Start, then move the board slowly across the whole image and hold it
still for about a second in each pose. Samples are taken automatically when the board is
still, sharp and in a new pose (or with Space in manual mode). Follow the hint line and the
arrow; the bars show what is missing.</li>
<li><b>Calibrate</b> when the bars are full (or earlier, with a warning). A good setup
typically gives an RMS of 0.1-0.5 px (rule of thumb). Check the per-view errors, remove bad
views and recalibrate.</li>
<li><b>Choose the framing</b> of the rectified image (policy, alpha), check it with the
rectified view (R), then Save and/or Commit.</li>
</ol>
<h3>Indicators</h3>
<ul>
<li><b>Image coverage:</b> {coverage}</li>
<li><b>Tilt up/down:</b> {tilt_x}</li>
<li><b>Tilt left/right:</b> {tilt_y}</li>
<li><b>Distance:</b> {distance}</li>
<li><b>Uncertainty:</b> {uncertainty}</li>
</ul>
<h3>Frame gates</h3>
<p>A frame is captured only if it passes, in order: detection; corner count; border distance
(corners farther than the refinement window from the edge); blur (edge width below
<i>gates.max_blur_px</i>); motion (below <i>gates.max_motion_px</i> per frame); tilt (below
<i>gates.max_tilt_deg</i>); novelty (at least <i>gates.min_novelty</i> from every stored pose,
unless the view covers an empty image cell). Space captures the next frame that passes every
gate but novelty.</p>
<h3>Tips</h3>
<p>Use a rigid, flat board; measure the square size; avoid glare; hold perfectly still with
rolling-shutter cameras; prefer the raw transport (JPEG artefacts bias corners).</p>
<h3>Shortcuts</h3>
<p>Space capture · P pause · C calibrate · Backspace remove last · R rectified view ·
H heatmap · Ctrl+S save · Ctrl+O load · Ctrl+R reset · F1 help · F11 full screen ·
Esc cancel ·
Ctrl+Q quit</p>
""".format(**INDICATOR_HELP)
