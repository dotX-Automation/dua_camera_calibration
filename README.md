# dua_camera_calibration

Interactive and offline calibration of monocular and stereo cameras, pinhole and fisheye, with chessboard and ChArUco targets, on-screen guidance, parameter uncertainties and a configurable rectified projection.

## Contents

- `dua_camera_calibration_app`: Qt GUI that subscribes to live images (raw or compressed), guides the capture, calibrates in the background, previews the rectification, and saves or commits the result.
- `dua_camera_calibration_cli`: headless calibration from image folders, a rosbag2 or a dataset saved by the GUI.
- `synthetic_camera`: node that publishes rendered calibration boards (mono or stereo, raw and/or compressed) and serves `SetCameraInfo`, for tests and demos.
- `launch/dua_camera_calibration.launch.py` and `config/dua_camera_calibration.yaml`: launch file and parameters that pre-fill the GUI.

The package started as a fork of the ROS 2 [`camera_calibration`](https://github.com/ros-perception/image_pipeline/tree/rolling/camera_calibration) package and has been rewritten from scratch. Calibration files keep the original format, so they can be read by `camera_calibration_parsers` and `camera_info_manager`.

## Requirements

- ROS 2 Jazzy, Python 3.12, PyQt5 through `python_qt_binding`.
- OpenCV >= 4.8 in Python, with the `aruco` module (checked at startup). DUA x86 images ship OpenCV 4.11 with contrib modules.
- A display for the GUI; the CLI needs none.

## Usage

### GUI

```bash
ros2 run dua_camera_calibration dua_camera_calibration_app
```

or, with a parameters file (see `config/dua_camera_calibration.yaml`):

```bash
ros2 launch dua_camera_calibration dua_camera_calibration.launch.py cf:=/path/to/params.yaml
```

Parameters only provide the initial values: every setting can be changed in the Setup panel on the left, and invalid values are refused with an explanation. A typical session goes as follows.

1. **Source**: choose mono or stereo, the transport (`raw` or `compressed`) and the topics among the discovered ones (Refresh updates the list). Topics are given as base names: with `compressed`, `/compressed` is appended. JPEG compression can bias corner positions, so prefer raw images or PNG compression; the GUI warns when the received frames are lossy.
2. **Board**: target type and geometry. For a chessboard, columns and rows count the **inner corners** (squares minus one per side); for ChArUco they count the **squares**, and the marker size (smaller than the square) and the dictionary are needed too. Enable *legacy pattern* for ChArUco boards generated with OpenCV < 4.6 that have an even number of rows. Measure the printed square size rather than trusting the nominal one.
3. **Calibration**: camera model (`pinhole` or `fisheye`), distortion model and options. `plumb_bob` with *Fix k3* suits most normal-FOV lenses, `rational_polynomial` suits wide lenses, and `fisheye` (equidistant model) is meant for lenses beyond about 120 degrees. The fisheye model has no distortion model choice, tangential terms or aspect-ratio constraint, so those fields are hidden.
4. **Start capture** and move the board following the hints. In auto mode, frames are captured when they pass every gate and add information; in manual mode, press Space. The banner under the image says why the current frame is rejected and what to do next, and the arrow in the image points to where the board should go.
5. **Calibrate** when the indicators are full; the button also works earlier, after a confirmation. Calibration runs in the background, so the GUI stays responsive and the run can be cancelled.
6. **Review** the results: parameters with standard deviations and 95% confidence intervals, per-view reprojection errors, projection policy and alpha with a live rectified preview, and live check metrics on the current frames. Recalibrate re-runs the solve on the same samples with the current calibration options; Back returns to the capture to add samples.
7. **Save** creates a run directory `<camera name>_<date>-<time>` under the output directory (default `<current directory>/calibrations`). It contains the camera YAML file (`<camera name>.yaml` for mono, `left.yaml` and `right.yaml` for stereo, with camera names `<camera name>_left` and `<camera name>_right`), `report.yaml` with every statistic, and `settings.yaml` in ROS 2 parameters format. **Save dataset** also stores the captured images and corners in `dataset/`, so the CLI can recalibrate them with different options.
8. **Commit** sends the calibration to `sensor_msgs/srv/SetCameraInfo` services, for example those of a driver that uses `camera_info_manager`. The service of each camera is selected in the Results tab among the discovered ones; the default is the service whose name shares the longest path prefix with the image topic, or `commit.service` for mono. Commit is refused when no service is selected, or when both stereo cameras point to the same service.

**Load calibration** opens existing YAML files and shows the rectified live view with the live check metrics, to verify a calibration without recalibrating. The projection policy can be changed there too, and Save writes the camera YAML files only.

Stopping the capture keeps the samples, so the Source and Board settings stay locked until the samples are cleared. **Reset** clears samples and results, resubscribes with the current settings and returns to the initial state, after a confirmation if something would be lost.

Shortcuts:

| Key | Action |
| --- | --- |
| Space | Capture the next valid frame |
| P | Pause or resume the capture |
| C | Calibrate |
| Backspace | Remove the last sample |
| Delete | Remove the selected samples |
| H | Show or hide the coverage heatmap |
| R | Show or hide the rectified view |
| Ctrl+R | Reset |
| Ctrl+S | Save |
| Ctrl+O | Load a calibration |
| F1 | Help |
| F11 | Full screen |
| Esc | Cancel the calibration or exit full screen |
| Ctrl+Q | Quit |

### Indicators

Each indicator is computed per camera and fills up to 100%. Hover it for details, or press "What do these mean?".

- **Image coverage**: share of the image where board corners have been seen, on an 8x6 grid. Lens distortion grows toward the edges and is strongest in the corners, so the model is only trustworthy where the board has been shown. The indicator is full when 90% of the cells and all four corner cells are covered.
- **Tilt up/down** and **Tilt left/right**: board rotation about the horizontal and vertical camera axes, in three bins: below -15 degrees, between -15 and 15 degrees, above 15 degrees. Fronto-parallel views cannot tell a long focal length from a distant board; tilted views remove that ambiguity and pin down the principal point.
- **Distance**: apparent board size, measured as the square root of the fraction of the image the board covers, in three bins: far (below 0.25), mid and close (above 0.5). Close views constrain the distortion, far views constrain the focal length.
- **Uncertainty**: 95% confidence interval of focal lengths and principal point relative to their values, from a quick calibration that runs after every new sample. The indicator is full at the target, 0.5% by default. When it stops shrinking, more samples will not help.

Calibration is ready when at least `capture.min_samples` samples are stored (for stereo, also that many pairs) and every indicator is full, or when the uncertainty improved by less than 10% over the last 5 samples while coverage is at least 80%.

### Frame gates

A frame is captured only if it passes every gate; the thresholds are in the Capture section of the Setup panel.

- The board is detected with at least 6 corners.
- No corner is closer to the image border than the corner refinement window.
- The board edges are sharp: black-to-white transition width below `gates.max_blur_px` (complete chessboards only).
- The board is still: mean corner motion between consecutive frames below `gates.max_motion_px`, scaled with the image diagonal.
- The board is not tilted more than `gates.max_tilt_deg`.
- The view differs enough from every stored sample in position, size and tilt (`gates.min_novelty`), or covers a region of the image that has not been covered yet.

In stereo mode a pair is captured when both views pass. A view seen by one camera only is captured when it covers a new region of that camera, and is used for its intrinsics only.

### Rectified projection policies

For monocular cameras the rectified image is a virtual pinhole camera with `R = I` and `P = [K' | 0]`, and has the same size as the input image. The undistorted image border is sampled densely, and its inner rectangle `I` (valid pixels only) and outer rectangle `O` (all source pixels) are computed in normalized coordinates. With `rho = 1` (square) or `rho = fx / fy` (aspect), and a `W x H` image:

- `s0 = max((W - 1) / (rho |I_x|), (H - 1) / |I_y|)` makes every output pixel valid;
- `s1 = min((W - 1) / (rho |O_x|), (H - 1) / |O_y|)` keeps every source pixel;
- `s = (1 - alpha) s0 + alpha s1`, and the window center is `c = (1 - alpha) c(I) + alpha c(O)`;
- `fy' = s`, `fx' = rho s`, `cx' = (W - 1) / 2 - fx' c_x`, `cy' = (H - 1) / 2 - fy' c_y`.

The available policies are:

- `square` (default): `fx' = fy'`, so squares stay squares in the rectified image.
- `aspect`: `fx' / fy' = fx / fy`, keeping the calibrated pixel aspect ratio.
- `opencv`: legacy `cv2.getOptimalNewCameraMatrix`, which fits `fx'` and `fy'` independently to the valid area. It maximizes the valid area but does not preserve shapes; with `alpha = 0` it reproduces the original `camera_calibration` output.
- `k`: `P = [K | 0]`, keeping the calibrated intrinsics unchanged.

`alpha = 0` keeps only valid pixels, `alpha = 1` keeps every source pixel and adds black borders. If the distortion model cannot be inverted up to the image corners, typically an over-fitted model without samples in the corners, a warning is shown and the `k` policy is applied instead.

Fisheye cameras use the same policies with the equidistant model, limiting the rectified field of view to `rectify.fisheye_max_fov_deg`; there, `opencv` maps to `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify`. Stereo pairs use `cv2.stereoRectify` or its fisheye counterpart, which already produce square pixels on both cameras; alpha and the zero-disparity option apply there.

### Numerical method

- Chessboard corners are detected on a downscaled copy of the image (`detection.max_pixels`) and refined at full resolution with a search window proportional to the corner spacing, which gives about 0.03 px error on synthetic images. ChArUco corners are refined the same way, with a window that stays inside the squares.
- Pinhole calibration uses `cv2.calibrateCamera` with the LU solver, which is orders of magnitude faster than the default SVD with many views and gives the same result. When more than `calib.max_views` samples are stored, the most informative views are selected with a greedy D-optimal design on the per-view information matrices. Views with outlying reprojection errors are rejected and the solve is repeated.
- Parameter standard deviations are computed from the Schur complement of the normal equations, in linear time in the number of views.
- Fisheye cameras use `cv2.fisheye.calibrate`, and ill-conditioned views are dropped automatically.
- For stereo, each camera's intrinsics use every view that camera saw; the extrinsics are then estimated from the pairs with fixed intrinsics (fisheye pairs through normalized coordinates).

### CLI

```bash
ros2 run dua_camera_calibration dua_camera_calibration_cli --images DIR [--right-images DIR] [options]
ros2 run dua_camera_calibration dua_camera_calibration_cli --bag PATH --topic TOPIC [--right-topic TOPIC] [--slop S] [options]
ros2 run dua_camera_calibration dua_camera_calibration_cli --dataset DIR [options]
```

Settings are applied in this order: parameter defaults, the dataset settings (if any), `--params-file` (a ROS 2 parameters file of the node, such as the `settings.yaml` of a previous run), `--set key=value` (repeatable, e.g. `--set board.cols=9 --set board.square_size=0.03`), and finally `--policy` and `--alpha`. Results go to a run directory under `--out` (default `output.dir`). Bag topics may be raw or compressed, and stereo messages are paired by stamp within `--slop`. `--dataset` accepts both a dataset folder and a run directory that contains one. The motion gate is disabled for image folders, whose frames are not consecutive. The exit code is 0 on success, 1 when the calibration fails and 2 for invalid arguments.

### Synthetic camera

```bash
ros2 run dua_camera_calibration synthetic_camera --mode stereo --transport both
```

It publishes on `/synthetic/image_raw` (mono), or `/synthetic/left/image_raw` and `/synthetic/right/image_raw` (stereo), plus the `/compressed` topics, and serves `set_camera_info` in the same namespaces. The board dwells still at each pose for a while, so that the frame gates pass. See `--help` for the camera, board and trajectory options.

## Tests

```bash
colcon test --packages-select dua_camera_calibration
```

The tests cover image conversion, detection accuracy on rendered boards, calibration accuracy and uncertainty on synthetic data, rectification invariants, guidance, file formats, the pipeline, the CLI, an offscreen GUI run and an end-to-end ROS run with the synthetic camera. Set `DCC_SCREENSHOT_DIR` to save screenshots of the offscreen GUI test. The flake8 test runs the system flake8 with a clean `PYTHONPATH` and the ament configuration.

---

## Copyright and License

Copyright 2026 dotX Automation s.r.l.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.
