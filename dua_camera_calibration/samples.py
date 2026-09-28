"""
Thread-safe sample database with immutable snapshots.

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
import threading
from typing import Optional


@dataclass(frozen=True)
class Sample:
    """
    One captured sample: one view per camera (None if that camera did not see the board).

    features[i] is the guidance feature dict of view i (None if the view is None).
    images_png[i] is the PNG-encoded gray image of camera i, or None.
    """

    id: int  # noqa: A003
    stamp_ns: int
    views: tuple                      # tuple[Detection | None, ...]
    features: tuple                   # tuple[dict | None, ...]
    images_png: Optional[tuple] = None  # tuple[bytes | None, ...] | None


@dataclass(frozen=True)
class DBSnapshot:
    """Immutable view of the database at a given version."""

    version: int
    samples: tuple                    # tuple[Sample, ...]
    image_sizes: tuple                # tuple[(w, h), ...] per camera

    @property
    def n_cameras(self) -> int:
        """Return the number of cameras."""
        return len(self.image_sizes)

    def views_of(self, cam: int) -> list:
        """Return (sample_id, Detection) pairs of samples where camera cam saw the board."""
        return [(s.id, s.views[cam]) for s in self.samples if s.views[cam] is not None]

    def pairs(self) -> list:
        """Return (sample_id, left, right) Detections of stereo samples with both views."""
        return [(s.id, s.views[0], s.views[1]) for s in self.samples
                if len(s.views) > 1 and s.views[0] is not None and s.views[1] is not None]


class SampleDB:
    """Sample storage; every method is thread-safe and snapshot() is O(1)."""

    def __init__(self, image_sizes: tuple):
        """Create an empty database; image_sizes is a tuple of (w, h), one per camera."""
        self._lock = threading.Lock()
        self._image_sizes = tuple(tuple(s) for s in image_sizes)
        self._samples = ()
        self._next_id = 0
        self._version = 0

    @property
    def version(self) -> int:
        """Return the version, incremented at every change."""
        return self._version

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._samples)

    def add(self, views: tuple, features: tuple, images_png: Optional[tuple],
            stamp_ns: int) -> Sample:
        """Append a sample and return it."""
        with self._lock:
            s = Sample(self._next_id, int(stamp_ns), tuple(views), tuple(features),
                       tuple(images_png) if images_png is not None else None)
            self._next_id += 1
            self._samples = self._samples + (s,)
            self._version += 1
            return s

    def remove(self, ids) -> None:
        """Remove the samples with the given ids."""
        ids = set(ids)
        with self._lock:
            self._samples = tuple(s for s in self._samples if s.id not in ids)
            self._version += 1

    def clear(self) -> None:
        """Remove every sample."""
        with self._lock:
            self._samples = ()
            self._version += 1

    def snapshot(self) -> DBSnapshot:
        """Return an immutable snapshot."""
        with self._lock:
            return DBSnapshot(self._version, self._samples, self._image_sizes)

    @staticmethod
    def from_samples(samples, image_sizes) -> 'SampleDB':
        """Build a database from existing samples (e.g. a loaded dataset)."""
        db = SampleDB(image_sizes)
        db._samples = tuple(samples)
        db._next_id = max((s.id for s in db._samples), default=-1) + 1
        db._version = 1
        return db
