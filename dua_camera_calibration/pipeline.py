"""
Frame pipeline shared by the GUI and the CLI: mailboxes, worker threads, jobs, run output.

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

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, replace
from datetime import datetime
import os
import queue
import threading
import time
from typing import Callable, Optional

import cv2
from dua_camera_calibration import fileio, guidance, rectification
from dua_camera_calibration.detection import make_detector
from dua_camera_calibration.imageconv import compressed_to_gray, image_to_gray, ImageError
from dua_camera_calibration.samples import SampleDB
import numpy as np
import yaml

VERSION = '4.0.0'
NODE_NAME = 'dua_camera_calibration'

# CONTRACT (do not change names/fields/signatures without updating every user).


class LatestSlot:
    """Single-item mailbox: put() overwrites (drop-oldest), readers see (version, item)."""

    def __init__(self):
        """Create an empty slot."""
        self._cv = threading.Condition()
        self._item = None
        self._ver = 0

    def put(self, item) -> None:
        """Store an item, replacing the previous one."""
        with self._cv:
            self._item = item
            self._ver += 1
            self._cv.notify_all()

    def get(self) -> tuple:
        """Return (version, item)."""
        with self._cv:
            return self._ver, self._item

    def wait_newer(self, ver: int, timeout: float) -> tuple:
        """Wait up to timeout for a version different from ver, then return (version, item)."""
        with self._cv:
            self._cv.wait_for(lambda: self._ver != ver, timeout)
            return self._ver, self._item

    def wake(self) -> None:
        """Wake every waiter without changing the content (used on shutdown)."""
        with self._cv:
            self._cv.notify_all()


@dataclass(frozen=True)
class RawFrame:
    """Messages as received from ROS (or read from a bag), one per camera."""

    gen: int                  # subscription generation (stale frames are dropped)
    recv_time: float          # time.monotonic() at reception
    stamp_ns: int             # header stamp of the first message
    msgs: tuple               # (Image | CompressedImage,) or (left, right)


@dataclass(frozen=True)
class Frame:
    """Converted frame: one uint8 gray image per camera."""

    gen: int
    seq: int
    recv_time: float
    stamp_ns: int
    grays: tuple              # tuple[np.ndarray (H, W) uint8, ...]
    encodings: tuple          # tuple[str, ...]
    lossy: bool               # any camera received lossy (JPEG) data


@dataclass(frozen=True)
class DisplayFrame:
    """Frame prepared for display (downscaled, optionally rectified)."""

    gen: int
    seq: int
    recv_time: float
    images: tuple             # tuple[np.ndarray uint8 gray, C-contiguous, ...]
    scales: tuple             # display px per source px, per camera
    rectified: bool
    src_sizes: tuple = ()     # per camera source (w, h) of the converted frame


@dataclass(frozen=True)
class Overlay:
    """Detection results of one processed frame, drawn over the matching DisplayFrame."""

    gen: int
    seq: int
    recv_time: float
    done_time: float
    detections: tuple         # tuple[Detection | None, ...]
    points: tuple             # tuple[(N, 2) full-res px (rectified if `rectified`) | None, ...]
    rectified: bool
    verdict: object           # guidance.FrameVerdict (reason 'ACCEPTED' when captured)
    hints: tuple              # guidance.frame_hints(verdict)
    validation: object        # rectification.ValidationMetrics | None


@dataclass(frozen=True)
class ViewConfig:
    """Display configuration swapped atomically by the GUI."""

    sizes: tuple              # per camera display (w, h); () means native size
    maps: Optional[tuple] = None   # per camera (map1, map2) from rectification.make_maps, or None


class JobRunner:
    """
    Single background worker for long jobs; used from the GUI thread only.

    The worker is a daemon thread, so a calibration still running at quit does not keep the
    process alive (a ThreadPoolExecutor worker would be joined at interpreter exit).
    """

    def __init__(self):
        """Create the worker."""
        self._queue = queue.SimpleQueue()
        self._pending = {}
        self._closed = False
        self._thread = threading.Thread(target=self._run, name='job', daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Run the queued jobs in order until shutdown."""
        while True:
            item = self._queue.get()
            if item is None:
                return
            f, fn, args = item
            if not f.set_running_or_notify_cancel():
                continue
            try:
                f.set_result(fn(*args))
            except BaseException as e:
                f.set_exception(e)

    def submit(self, kind: str, fn: Callable, *args) -> Optional[Future]:
        """Submit a job unless one of the same kind is still running (then return None)."""
        f = self._pending.get(kind)
        if self._closed or (f is not None and not f.done()):
            return None
        f = Future()
        self._pending[kind] = f
        self._queue.put((f, fn, args))
        return f

    def busy(self, kind: Optional[str] = None) -> bool:
        """Return True if a job (of the given kind) is running."""
        return any(not f.done() for k, f in self._pending.items() if kind in (None, k))

    def pop_done(self) -> list:
        """Remove and return the finished (kind, Future) pairs."""
        done = [(k, f) for k, f in self._pending.items() if f.done()]
        for k, _ in done:
            del self._pending[k]
        return done

    def shutdown(self) -> None:
        """Stop accepting jobs and cancel the queued ones (a running job is abandoned)."""
        self._closed = True
        for f in self._pending.values():
            f.cancel()
        self._queue.put(None)


def stamp_ns(msg) -> int:
    """Return the header stamp of a ROS message in nanoseconds."""
    return int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec)


def convert_raw(raw: RawFrame, seq: int) -> Frame:
    """Convert the messages of a RawFrame (Image or CompressedImage) into a Frame."""
    grays, encodings, lossy = [], [], False
    for m in raw.msgs:
        if hasattr(m, 'format'):      # CompressedImage (duck-typed: no ROS import here)
            g, lz = compressed_to_gray(m.format, m.data)
            lossy = lossy or lz
            encodings.append(m.format)
        else:
            g = image_to_gray(m.encoding, m.width, m.height, m.step, m.data, bool(m.is_bigendian))
            encodings.append(m.encoding)
        grays.append(g)
    return Frame(raw.gen, seq, raw.recv_time, raw.stamp_ns, tuple(grays), tuple(encodings), lossy)


def make_display(frame: Frame, view: ViewConfig) -> DisplayFrame:
    """Downscale (INTER_AREA) or remap the frame for display."""
    maps = view.maps
    rectified = maps is not None and len(maps) == len(frame.grays) and None not in maps
    images, scales = [], []
    for i, g in enumerate(frame.grays):
        h, w = g.shape
        if rectified:
            m1, m2 = maps[i]
            img, s = cv2.remap(g, m1, m2, cv2.INTER_LINEAR), m1.shape[1] / w
        else:
            s = 1.0
            if i < len(view.sizes):
                s = min(1.0, view.sizes[i][0] / w, view.sizes[i][1] / h)
            img = g
            if s < 1.0:
                size = (max(1, round(w * s)), max(1, round(h * s)))
                img, s = cv2.resize(g, size, interpolation=cv2.INTER_AREA), size[0] / w
        images.append(np.ascontiguousarray(img))
        scales.append(s)
    return DisplayFrame(frame.gen, frame.seq, frame.recv_time, tuple(images), tuple(scales),
                        rectified, tuple((g.shape[1], g.shape[0]) for g in frame.grays))


@dataclass(frozen=True)
class _Config:
    """Immutable per-session configuration derived from Settings (swapped atomically)."""

    settings: object
    board: object
    gates: object
    calib: object
    guidance: object
    max_samples: int
    keep_images: bool


def _rate(stamps) -> float:
    """Return the rate [Hz] of a deque of monotonic times (0 if stale or too short)."""
    t = list(stamps)   # atomic copy under the GIL
    if len(t) < 2 or time.monotonic() - t[-1] > 2.0:
        return 0.0
    return (len(t) - 1) / max(t[-1] - t[0], 1e-9)


class Session:
    """
    Calibration session: pipeline threads, sample database, guidance and the active calibration.

    The ROS executor thread calls push(); the GUI thread calls every other public method and
    polls the slots; with threaded=False (CLI) no thread is started and the caller drives
    convert_raw()/process_frame()/refresh_guidance() directly.
    Slots: slot_display (DisplayFrame), slot_overlay (Overlay), slot_guidance (GuidanceState).
    """

    def __init__(self, settings, threaded: bool = True, check_motion: bool = True):
        """Create the session from a settings.Settings; threads start only if threaded."""
        self.slot_in, self.slot_detect = LatestSlot(), LatestSlot()
        self.slot_display, self.slot_overlay, self.slot_guidance = (LatestSlot(), LatestSlot(),
                                                                    LatestSlot())
        self.db = SampleDB(())
        self.image_error = None           # latest ImageError of the converter (None after success)
        self.errors_version = 0           # incremented at every reported error
        self._errors = deque(maxlen=200)
        self._err_lock = threading.Lock()
        self._check_motion = check_motion
        self._cfg = None
        self.configure(settings)
        self._gen = 0
        self._view = ViewConfig(())
        self._active = None
        self._mode, self._paused, self._req_until = 'off', False, 0.0
        # detector-thread state
        self._sizes = None
        self._det_key = self._detector = None
        self._eng_key = self._engine = None
        self._guided = None
        self._prev = None                 # (gen, views) of the previous processed frame
        # stats (each written by one thread)
        self._rx, self._det = deque(maxlen=30), deque(maxlen=30)
        self._dropped, self._latency = [0, 0], 0.0   # [converter, detector]
        self._lossy = False
        self._stop = threading.Event()
        self._threads = ()
        if threaded:
            self._threads = (threading.Thread(target=self._convert_loop, name='converter',
                                              daemon=True),
                             threading.Thread(target=self._detect_loop, name='detector',
                                              daemon=True))
            for t in self._threads:
                t.start()

    @property
    def errors(self) -> list:
        """Return the coalesced errors as (time, where, message, count), oldest first."""
        with self._err_lock:
            return list(self._errors)

    def report(self, where: str, exc: BaseException) -> None:
        """Record an error; identical (where, message) entries are coalesced into a count."""
        msg = f'{type(exc).__name__}: {exc}'
        with self._err_lock:
            count = 1
            for i, (_, w, m, c) in enumerate(self._errors):
                if (w, m) == (where, msg):
                    count = c + 1
                    del self._errors[i]
                    break
            self._errors.append((time.time(), where, msg, count))
            self.errors_version += 1

    # executor thread
    def push(self, raw: RawFrame) -> None:
        """Hand over a received frame (drops it if raw.gen is stale)."""
        if raw.gen != self._gen:
            return
        self._rx.append(time.monotonic())
        self.slot_in.put(raw)

    # GUI thread
    def next_generation(self) -> int:
        """Invalidate in-flight frames (call BEFORE resubscribing) and return the new gen."""
        self._gen += 1
        return self._gen

    def configure(self, settings) -> None:
        """Apply new settings; clear samples if cameras or board change."""
        old = self._cfg
        self._cfg = _Config(settings, settings.board(), settings.gate_config(self._check_motion),
                            settings.calib_config(), settings.guidance_config(),
                            settings['capture.max_samples'], settings['capture.keep_images'])
        if old is not None and (old.board != self._cfg.board
                                or old.settings.topics() != settings.topics()):
            self.db.clear()

    def set_capture(self, mode: str, paused: bool) -> None:
        """Set capture mode: 'off' | 'auto' | 'manual', and pause flag."""
        if mode not in ('off', 'auto', 'manual'):
            raise ValueError(f'unknown capture mode {mode!r}')
        self._mode, self._paused = mode, bool(paused)

    def request_capture(self) -> None:
        """Capture the next frame passing the hard gates (manual capture, within ~1 s)."""
        self._req_until = time.monotonic() + 1.0

    def remove_samples(self, ids) -> None:
        """Remove samples by id."""
        self.db.remove(ids)

    def clear_samples(self) -> None:
        """Remove every sample."""
        self.db.clear()

    def set_active_calibration(self, cameras: Optional[tuple], rect) -> None:
        """Enable live validation and rectified overlay points (None disables)."""
        self._active = None if cameras is None else (tuple(cameras), rect)

    def set_view(self, view: ViewConfig) -> None:
        """Set the display configuration (sizes, rectification maps)."""
        self._view = view

    def stats(self) -> dict:
        """Return rates and counters: rx_hz, det_hz, dropped, latency_s, samples, mem_mb, lossy."""
        snap = self.db.snapshot()
        mem = sum(len(p) for s in snap.samples for p in (s.images_png or ()) if p is not None)
        return {'rx_hz': _rate(self._rx), 'det_hz': _rate(self._det),
                'dropped': sum(self._dropped), 'latency_s': self._latency,
                'samples': len(snap.samples), 'mem_mb': mem / 1e6, 'lossy': self._lossy}

    def alive(self) -> bool:
        """Return True if the worker threads are running (always True when not threaded)."""
        return all(t.is_alive() for t in self._threads)

    def stop(self) -> None:
        """Stop and join the worker threads (idempotent)."""
        self._stop.set()
        for s in (self.slot_in, self.slot_detect):
            s.wake()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(1.0)

    # worker threads
    def _convert_loop(self) -> None:
        """Run the converter thread: slot_in -> convert_raw -> slot_detect and slot_display."""
        ver, seq = 0, 0
        while not self._stop.is_set():
            try:
                v, raw = self.slot_in.wait_newer(ver, 0.2)
                if v == ver or raw is None:
                    continue
                self._dropped[0] += max(0, v - ver - 1)
                ver = v
                if raw.gen != self._gen:
                    continue
                seq += 1
                try:
                    frame = convert_raw(raw, seq)
                    self._lossy = frame.lossy
                except ImageError as e:
                    self.image_error = e
                    raise
                self.image_error = None
                self.slot_detect.put(frame)
                self.slot_display.put(make_display(frame, self._view))
            except Exception as e:
                self.report('converter', e)
                self._stop.wait(0.2)      # back off: a persistent error must not spin

    def _detect_loop(self) -> None:
        """Run the detector thread: guidance, then slot_detect -> process_frame -> slot_overlay."""
        ver = 0
        while not self._stop.is_set():
            try:
                self.refresh_guidance()
                v, frame = self.slot_detect.wait_newer(ver, 0.2)
                if v == ver or frame is None:
                    continue
                self._dropped[1] += max(0, v - ver - 1)
                ver = v
                if frame.gen != self._gen:
                    continue
                ov = self.process_frame(frame)
                self._det.append(time.monotonic())
                self._latency = ov.done_time - ov.recv_time
                self.slot_overlay.put(ov)
            except Exception as e:
                self.report('detector', e)
                self._stop.wait(0.2)      # back off: e.g. make_detector() failing every call

    def _prepare(self, sizes: Optional[tuple]) -> tuple:
        """Return (config, detector, engine), rebuilding what the config or sizes invalidated."""
        cfg = self._cfg
        if sizes is not None and sizes != self._sizes:
            if self.db.snapshot().image_sizes != sizes:
                if len(self.db):
                    raise ValueError(f'image size changed from {self.db.snapshot().image_sizes} '
                                     f'to {sizes}: clear the samples to continue')
                old, self.db = self.db, SampleDB(sizes)
                self.db._version = old.version + 1   # keep db.version monotonic for pollers
            self._sizes = sizes
        if self._det_key != cfg.board:
            self._det_key, self._detector = cfg.board, make_detector(cfg.board)
        key = (cfg.board, cfg.calib, cfg.guidance, self._sizes)
        if self._sizes is not None and self._eng_key != key:
            self._eng_key = key
            self._engine = guidance.GuidanceEngine(cfg.board, cfg.calib, cfg.guidance,
                                                   self._sizes)
        return cfg, self._detector, self._engine

    # shared by the detector thread and the CLI
    def process_frame(self, frame: Frame) -> Overlay:
        """Detect, gate, capture and validate one frame."""
        sizes = tuple((g.shape[1], g.shape[0]) for g in frame.grays)
        cfg, detector, engine = self._prepare(sizes)
        est = engine.estimate if engine is not None else None
        views = tuple(detector.detect(g, est.cams[i] if est is not None else None)
                      for i, g in enumerate(frame.grays))
        snap = self.db.snapshot()
        prev = self._prev[1] if self._prev is not None and self._prev[0] == frame.gen else None
        v = guidance.evaluate_frame(views, sizes, prev, None, snap, est, cfg.gates, cfg.board)
        self._prev = (frame.gen, views)
        mode = self._mode
        req = time.monotonic() < self._req_until
        want = mode != 'off' and ((req and v.hard_ok)
                                  or (mode == 'auto' and not self._paused and v.ok))
        if want and len(snap.samples) >= cfg.max_samples:
            v = replace(v, reason='LIMIT', params={'max': cfg.max_samples})
        elif want:
            self._req_until = 0.0
            pngs = None
            if cfg.keep_images:
                pngs = tuple(cv2.imencode('.png', g, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
                             for g in frame.grays)
            s = self.db.add(v.views, v.features, pngs, frame.stamp_ns)
            v = replace(v, reason='ACCEPTED', params={'id': s.id})
        act, maps = self._active, self._view.maps
        val, rect = None, False
        if act is not None and len(act[0]) == len(views):
            val = rectification.validate_views(views, act[0], act[1], cfg.board)
            rect = act[1] is not None and maps is not None
        if rect:
            pts = tuple(None if d is None else rectification.rectify_points(c, R, P, d.corners)
                        for d, c, R, P in zip(views, act[0], act[1].R, act[1].P))
        else:
            pts = tuple(None if d is None else d.corners for d in views)
        return Overlay(frame.gen, frame.seq, frame.recv_time, time.monotonic(), views, pts, rect,
                       v, guidance.frame_hints(v), val)

    def refresh_guidance(self):
        """Recompute guidance if the database changed; return the new GuidanceState or None."""
        _, _, engine = self._prepare(None)
        if engine is None:
            return None
        snap = self.db.snapshot()
        if self._guided == (engine, snap.version):
            return None
        self._guided = (engine, snap.version)
        state = engine.update(snap)
        self.slot_guidance.put(state)
        return state


def ros_params_doc(settings, node_name: str = NODE_NAME) -> dict:
    """Return the settings as a ROS 2 parameters file document (nested keys)."""
    nested = {}
    for key, value in settings.as_dict().items():
        *head, last = key.split('.')
        d = nested
        for h in head:
            d = d.setdefault(h, {})
        d[last] = value
    return {'/' + node_name: {'ros__parameters': nested}}


def camera_files(settings, n_cameras: int) -> tuple:
    """Return the (file name, camera_name) of each camera YAML of a run."""
    name = settings['camera.name']
    if n_cameras == 1:
        return ((f'{name}.yaml', name),)
    # camera_info_manager checks camera_name against its own driver's name: one per side
    return (('left.yaml', f'{name}_left'), ('right.yaml', f'{name}_right'))


def write_run(out_dir: str, settings, result, rect, snapshot, board,
              dataset: bool = False) -> str:
    """
    Write a calibration run into a new run directory under out_dir and return its path.

    Contents: camera YAML(s) (mono '<camera_name>.yaml', stereo 'left.yaml'/'right.yaml'),
    report.yaml, and the dataset (session.yaml + images) when dataset is True.
    With result None and dataset True, only the dataset is written.
    Stereo camera_name fields are '<camera.name>_left' and '<camera.name>_right'. The settings
    are also saved as settings.yaml, a parameters file usable by the node (cf:=) and the CLI.
    """
    # ponytail: two saves within the same second share the run directory (different files
    # unless the same result is saved twice); add a suffix counter if that ever matters
    path = fileio.run_dir(out_dir, settings['camera.name'])
    date = datetime.now().isoformat(timespec='seconds')
    if result is not None:
        files = camera_files(settings, len(result.cameras))
        for i, (cam, (fname, cname)) in enumerate(zip(result.cameras, files)):
            R = rect.R[i] if rect is not None else np.eye(3)
            P = rect.P[i] if rect is not None else np.hstack([cam.K, np.zeros((3, 1))])
            comments = {'tool': f'dua_camera_calibration {VERSION}', 'date': date,
                        'rms_px': cam.rms, 'n_views': cam.n_views,
                        'policy': rect.policy if rect is not None else 'none',
                        'alpha': rect.alpha if rect is not None else 'none'}
            if result.stereo is not None:
                comments['stereo_rms_px'] = result.stereo.rms
                comments['baseline_m'] = float(np.linalg.norm(result.stereo.T))
            fileio.write_camera_yaml(os.path.join(path, fname), cname, cam, R, P, comments)
        report = os.path.join(path, 'report.yaml')
        fileio.write_report(report, result, rect, settings.as_dict())
        with open(report, 'a') as f:
            yaml.safe_dump({'tool_version': VERSION, 'date': date,
                            'camera_files': dict(files)}, f, sort_keys=False)
    with open(os.path.join(path, 'settings.yaml'), 'w') as f:
        yaml.safe_dump(ros_params_doc(settings), f, sort_keys=True)
    if dataset:
        fileio.write_dataset(os.path.join(path, 'dataset'), snapshot, board,
                             {'tool_version': VERSION, 'date': date,
                              'settings': settings.as_dict()})
    return path
