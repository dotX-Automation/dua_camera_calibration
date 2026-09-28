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

from dua_camera_calibration import calibration, guidance, rectification, texts
from dua_camera_calibration.guidance import FrameVerdict, Hint, Indicator
import pytest

# plausible params of every producer (see guidance, calibration, rectification)
VERDICT_PARAMS = {'camera': 0, 'value': 2.5, 'limit': 1.5, 'id': 3}
HINT_PARAMS = {
    'COVER_REGION': [{'region': r} for r in texts.REGION_NAMES],
    'TILT_UP_DOWN': [{'direction': d} for d in ('up', 'down', 'none')],
    'TILT_LEFT_RIGHT': [{'direction': d} for d in ('left', 'right', 'none')],
    'MOVE_CLOSER': [{'bin': b} for b in ('far', 'mid', 'close')],
    'MOVE_FARTHER': [{'bin': b} for b in ('far', 'mid', 'close')],
    'KEEP_GOING': [{'samples': 7}, {'samples': 7, 'pairs': 5}],
    'READY': [{'samples': 20}, {'samples': 20, 'pairs': 18}],
}
WARNING_PARAMS = {'camera': 0, 'roundtrip_px': 0.3, 'monotonic': False}
ERROR_PARAMS = {'n': 3, 'min': 6, 'camera': 0, 'msg': 'boom'}


def test_every_code_has_text():
    """Every code produced by the core has a user string."""
    assert set(guidance.REASON_CODES) <= set(texts.REASONS)
    assert set(guidance.INDICATOR_IDS) <= set(texts.INDICATOR_LABELS)
    assert set(guidance.INDICATOR_IDS) <= set(texts.INDICATOR_HELP)
    assert set(guidance.HINT_CODES) <= set(texts.HINTS)
    assert set(calibration.WARNING_CODES) <= set(texts.WARNINGS)
    assert set(calibration.ERROR_CODES) <= set(texts.ERRORS)
    assert set(calibration.REJECT_CODES) <= set(texts.REJECTS)
    assert set(calibration.STAGES) <= set(texts.STAGES)
    assert set(rectification.WARNING_CODES) <= set(texts.WARNINGS)
    assert set(rectification.POLICIES) | {'stereo', 'file'} <= set(texts.POLICY_NAMES)
    assert set(texts.POLICY_NAMES) == set(texts.POLICY_TIPS)


@pytest.mark.parametrize('code', guidance.REASON_CODES)
def test_reason_renders(code):
    """Verdict texts format with the gate params and board extras."""
    v = FrameVerdict(False, False, code, VERDICT_PARAMS)
    text = texts.reason_text(v, {'cols': 8, 'rows': 6, 'topic': '/cam'})
    assert '?' not in text and '{' not in text


@pytest.mark.parametrize('code', guidance.HINT_CODES)
def test_hint_renders(code):
    """Hint texts format for every variant, without placeholders left."""
    for params in HINT_PARAMS.get(code, [{}]):
        text = texts.hint_text(Hint(code, params))
        assert text and '?' not in text and '{' not in text, (code, params, text)
    if code in texts.HINT_VARIANT_KEYS:
        assert len({texts.hint_text(Hint(code, p)) for p in HINT_PARAMS[code]}) >= 2


def test_warnings_errors_render():
    """Warnings and errors format with their params."""
    for code in calibration.WARNING_CODES + rectification.WARNING_CODES:
        assert '?' not in texts.warning_text(code, WARNING_PARAMS)
    for code in calibration.ERROR_CODES:
        assert '?' not in texts.error_text(code, ERROR_PARAMS)


def test_indicator_values_and_missing_fields():
    """Indicator values format; missing fields degrade to '?' instead of raising."""
    assert texts.indicator_value(Indicator('coverage', 0.5, {
        'cells_hit': 24, 'cells': 48, 'corners_hit': 2})) == '50% of cells · corners 2/4'
    assert texts.indicator_value(Indicator('tilt_x', 0.6, {'bins': (1, 0, 1)})) == '2/3 bins'
    assert '0.80%' in texts.indicator_value(Indicator('uncertainty', 0.6, {
        'rel_ci': 0.008, 'target': 0.005}))
    assert 'needs' in texts.indicator_value(Indicator('uncertainty', 0.0, {'rel_ci': None}))
    assert texts.fmt('{a:.2f} {b}', {'b': None}) == '? ?'
    assert texts.fmt('{a:.2f}', {'a': 'x'}) == 'x'
