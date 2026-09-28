"""
Camera calibration GUI standalone application.

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

import os
import signal
import sys
import threading
import traceback

from dua_camera_calibration.detection import require_opencv
from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtWidgets import QApplication
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions


def _spin(executor, errors: list) -> None:
    """Spin the executor, recording the exception that stops it."""
    try:
        executor.spin()
    except Exception as e:
        errors.append(e)
        traceback.print_exc()


def _install_hooks(window) -> None:
    """Route uncaught exceptions to the window (the PyQt default aborts the process)."""
    def excepthook(tp, value, tb):
        traceback.print_exception(tp, value, tb)
        window.report_error('main thread', value)

    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
        window.report_error(getattr(args.thread, 'name', 'thread'), args.exc_value)

    sys.excepthook = excepthook
    threading.excepthook = thread_hook


def _install_signals(window) -> None:
    """First SIGINT/SIGTERM quits cleanly (serviced by the GUI tick), the second exits now."""
    count = [0]

    def handler(signum, frame):
        count[0] += 1
        if count[0] > 1:
            os._exit(130)
        QTimer.singleShot(0, window.quit_now)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main(args=None):
    """Run the calibration GUI."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = executor = spin = window = None
    try:
        require_opencv((4, 8))
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
                or os.environ.get('QT_QPA_PLATFORM')):
            print('No display: set DISPLAY, or use dua_camera_calibration_cli for offline '
                  'calibration.')
            return 1
        # imported here so that the display check runs before Qt and the node load
        from dua_camera_calibration.gui import MainWindow
        from dua_camera_calibration.ros_io import CalibrationNode
        from dua_camera_calibration.settings import Settings
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv[:1])    # never let Qt parse --ros-args
        node = CalibrationNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_errors = []
        spin = threading.Thread(target=_spin, args=(executor, spin_errors), name='ros_spin',
                                daemon=True)
        spin.start()
        settings, warnings = Settings.from_mapping(node.params())
        window = MainWindow(node, settings, warnings)
        window.spin_thread = spin
        _install_hooks(window)
        _install_signals(window)
        window.show()
        return app.exec_()
    except Exception as e:
        print(f'Exception occurred: {e}')
        if 'declare' in str(e).lower() or 'type' in str(e).lower():
            print('Hint: write double parameters with a decimal point (e.g. 1.0, not 1).')
        traceback.print_exc()
        return 1
    finally:
        if window is not None:
            window.shutdown()
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        if spin is not None:
            spin.join(1.0)    # Executor.shutdown() does not wait for spin() to return
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
