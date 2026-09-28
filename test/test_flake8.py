# Copyright 2017 Open Source Robotics Foundation, Inc.
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
import subprocess
import sys

from ament_flake8 import main as ament_flake8_main
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    # The system flake8 is run with a clean PYTHONPATH and the ament configuration,
    # to avoid plugins registered twice by other environments on the path.
    config = os.path.join(os.path.dirname(ament_flake8_main.__file__),
                          'configuration', 'ament_flake8.ini')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH='')
    python = '/usr/bin/python3' if os.path.exists('/usr/bin/python3') else sys.executable
    res = subprocess.run(
        [python, '-m', 'flake8', '--config', config,
         'dua_camera_calibration', 'test', 'launch', 'setup.py'],
        cwd=root, env=env, capture_output=True, text=True)
    assert res.returncode == 0, \
        'Found code style errors / warnings:\n' + res.stdout + res.stderr
