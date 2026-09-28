"""
ROS 2 side of the calibration app: topic discovery, dynamic subscriptions, SetCameraInfo commit.

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

import os
import time

from ament_index_python.packages import get_package_share_directory
from dua_camera_calibration.pipeline import RawFrame, stamp_ns
from dua_camera_calibration.settings import load_spec
from dua_node_py.dua_node import NodeBase
from dua_qos_py.dua_qos_besteffort import get_image_qos
import message_filters
import numpy as np
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from sensor_msgs.srv import SetCameraInfo

IMAGE_TYPE = 'sensor_msgs/msg/Image'
COMPRESSED_TYPE = 'sensor_msgs/msg/CompressedImage'
SERVICE_TYPE = 'sensor_msgs/srv/SetCameraInfo'


def topic_bases(names_and_types) -> dict:
    """
    Return {'raw': bases, 'compressed': bases} from get_topic_names_and_types() output.

    Compressed topics count only if named '<base>/compressed' (compressedDepth, zstd and theora
    transports use other names or types and cannot be decoded here).
    """
    raw, comp = set(), set()
    for name, types in names_and_types:
        if IMAGE_TYPE in types:
            raw.add(name)
        if COMPRESSED_TYPE in types and name.endswith('/compressed'):
            comp.add(name[:-len('/compressed')])
    return {'raw': tuple(sorted(raw)), 'compressed': tuple(sorted(comp))}


def camera_info(cam, R, P, frame_id: str = '') -> CameraInfo:
    """Return the sensor_msgs/CameraInfo of a calibration.CameraCalib with its R and P."""
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width, info.height = int(cam.size[0]), int(cam.size[1])
    info.distortion_model = cam.distortion_model
    info.d = [float(x) for x in np.ravel(cam.D)]
    info.k = [float(x) for x in np.ravel(cam.K)]
    info.r = [float(x) for x in np.ravel(R)]
    info.p = [float(x) for x in np.ravel(P)]
    return info


class CalibrationNode(NodeBase):
    """
    Calibration app node; every method is meant for the GUI thread while an executor spins.

    Subscriptions and clients are dynamic, so no init_* hook is used; parameters are read-only
    and only pre-fill the GUI.
    """

    _PARAMS_FILE_PATH = os.path.join(get_package_share_directory('dua_camera_calibration'),
                                     'dua_camera_calibration_params.yaml')

    def __init__(self, node_name: str = 'dua_camera_calibration'):
        """Create the node and declare its parameters."""
        # _cc_ prefix: rclpy.Node already uses _clients, _subscriptions, ...
        self._cc_subs = []          # rclpy subscriptions or message_filters.Subscriber
        self._cc_sync = None
        self._cc_clients = {}
        super().__init__(node_name, True)
        self.get_logger().info('Node initialized')

    def params(self) -> dict:
        """Return {key: value} of every parameter of the spec."""
        return {k: self.get_parameter(k).value for k in load_spec(self._PARAMS_FILE_PATH)}

    def discover_topics(self) -> dict:
        """Return the base names of the image topics: {'raw': (...), 'compressed': (...)}."""
        return topic_bases(self.get_topic_names_and_types())

    def discover_services(self) -> list:
        """Return the names of the sensor_msgs/srv/SetCameraInfo services."""
        return sorted(n for n, t in self.get_service_names_and_types() if SERVICE_TYPE in t)

    def set_source(self, settings, gen: int, sink) -> None:
        """
        Subscribe to the image topics of settings; sink receives pipeline.RawFrame objects.

        Mono: one best-effort subscription (depth 1). Stereo: message_filters subscribers
        (depth 5) paired by an ApproximateTimeSynchronizer with slop source.sync_slop.
        """
        self.clear_source()
        msg_type = CompressedImage if settings['source.transport'] == 'compressed' else Image
        topics = settings.topics()

        def deliver(*msgs):
            # Never raise here: message_filters holds its lock without try/finally.
            try:
                sink(RawFrame(gen, time.monotonic(), stamp_ns(msgs[0]), tuple(msgs)))
            except Exception as e:
                self.get_logger().error(f'Frame delivery failed: {e}', throttle_duration_sec=2.0)

        if len(topics) == 1:
            self._cc_subs = [self.dua_create_subscription(msg_type, topics[0], deliver,
                                                          get_image_qos(1))]
            return
        self._cc_subs = [message_filters.Subscriber(self, msg_type, t,
                                                    qos_profile=get_image_qos(5))
                         for t in topics]
        self._cc_sync = message_filters.ApproximateTimeSynchronizer(
            self._cc_subs, 5, settings['source.sync_slop'])
        self._cc_sync.registerCallback(deliver)
        for s in self._cc_subs:
            self.get_logger().info(f"[TOPIC SUB] '{s.sub.topic_name}'")

    def clear_source(self) -> None:
        """Destroy the current image subscriptions."""
        for s in self._cc_subs:
            self.destroy_subscription(getattr(s, 'sub', s))
        self._cc_subs, self._cc_sync = [], None

    def publisher_counts(self, settings) -> tuple:
        """Return the number of publishers of each subscribed topic of settings."""
        return tuple(self.count_publishers(t) for t in settings.topics())

    def _client(self, name: str):
        """Return the (lazily created) SetCameraInfo client of a service."""
        if name not in self._cc_clients:
            self._cc_clients[name] = self.dua_create_service_client(SetCameraInfo, name, False)
        # ponytail: simple_serviceclient.Client lacks service_is_ready/remove_pending_request,
        # so the wrapped rclpy client is used; expose them there if more nodes need them
        return self._cc_clients[name]._client

    def service_ready(self, name: str) -> bool:
        """Return True if the SetCameraInfo service is available."""
        return bool(name) and self._client(name).service_is_ready()

    def commit(self, names: tuple, infos: tuple) -> list:
        """Send one CameraInfo per service name (async); return the rclpy futures."""
        return [self._client(n).call_async(SetCameraInfo.Request(camera_info=i))
                for n, i in zip(names, infos)]

    def cancel(self, futures) -> None:
        """Forget pending commit futures (after a timeout)."""
        for f in futures:
            for c in self._cc_clients.values():
                c._client.remove_pending_request(f)
            f.cancel()
